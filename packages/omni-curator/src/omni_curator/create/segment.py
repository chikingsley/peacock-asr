"""Segment stage: the CPU producer half of the split create pipeline.

Each worker loads the NeMo frame-VAD model **once** (resident for the process), then loops: claim a
video from the queue -> VAD-segment -> cut each span to a 16 kHz FLAC (temp file, then atomic
rename so the labeler never sees a half-written clip) -> enqueue the clip rows in one transaction.
Run several workers as separate processes (``run_segmenters``) to hold the cores near saturation;
they back off when the pending-clip high-watermark is hit so a stalled labeler can't let the
segmenter cut the entire corpus to disk.

The Scribe/label half lives in :mod:`omni_curator.create.labelq`; both share the queue
(:mod:`omni_curator.create.queue`). See ``docs/archive/PIPELINE_SPLIT.md``.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from omni_curator.create.queue import QClip, QueueStore, QVideo
from omni_curator.process.audio import load_16k_mono, write_clip_16k

if TYPE_CHECKING:
    from collections.abc import Sequence

#: Default pending-clip backpressure ceiling. A finite cap (NOT the old ``5_000_000`` that never
#: tripped): once this many clips are unlabeled the segmenter pauses so a stalled labeler can't let
#: it cut the entire corpus to disk.
DEFAULT_PENDING_HWM = 50_000

#: Default free-space floor (GB) on the clips filesystem. The segmenter stops cutting when the disk
#: holding the clips drops below this, so a behind/stalled labeler can never fill the disk.
DEFAULT_MIN_FREE_GB = 50.0


def _free_gb(path: Path) -> float:
    """Free GB on the filesystem holding ``path`` (or its nearest existing parent)."""
    probe = path
    while not probe.exists():
        probe = probe.parent
    return shutil.disk_usage(probe).free / 1_000_000_000


#: Where archived sources live: ``<root>/<lang>/<channel>/<file>`` (see ``create/archive.py``).
DEFAULT_ARCHIVE_ROOT = Path("/mnt/storage/peacock-asr-archive")


def resolve_source_path(
    path: str | Path,
    *,
    channel: str,
    archive_root: Path = DEFAULT_ARCHIVE_ROOT,
) -> Path:
    """Resolve a video's source FLAC, falling back to the storage archive if ``path`` is gone.

    A re-segment runs against a queue whose ``video.path`` still points at the original work-drive
    location (overflow or the SSD). 30k+ sources were since *archived* off overflow to
    ``<archive_root>/<lang>/<channel>/<file>`` and the original deleted, so ``path`` is now gone.
    This finds the source whether it's still on the working drive OR in the archive:

    1. ``path`` itself, if it still exists (overflow / SSD — the common case).
    2. The deterministic archive map ``.../create/<lang>/<channel>/<file>`` ->
       ``<archive_root>/<lang>/<channel>/<file>`` when the create path carries the language.
    3. A scan of the archive root for any ``<lang>/<channel>/<file.name>`` match (handles create
       paths whose layout doesn't expose ``<lang>`` before ``create/``).

    Returns the first hit. Falls back to the original ``path`` (so the caller's existing
    file-not-found error still fires with the path the operator expects) if nothing is found.
    """
    src = Path(path)
    if src.exists():
        return src
    name = src.name
    # (2) deterministic .../create/<rest> -> <archive_root>/<rest>, where <rest> already starts with
    # <lang> in this project's layout (.../<lang>/<channel>/<file>); <rest> = parts after "create".
    parts = src.parts
    if "create" in parts:
        rest = parts[parts.index("create") + 1 :]
        mapped = archive_root.joinpath(*rest)
        if mapped.exists():
            return mapped
    # (3) scan the archive for <lang>/<channel>/<file.name>; lang dirs are the root's children
    if archive_root.is_dir():
        for lang_dir in archive_root.iterdir():
            candidate = lang_dir / channel / name
            if candidate.exists():
                return candidate
    return src


def _cut_clips(
    video: QVideo,
    spans: Sequence[tuple[float, float]],
    *,
    clips_root: Path,
    language: str,
    script: str,
    source: Path | None = None,
) -> list[QClip]:
    """Cut each span to ``clips_root/<channel>/<video_id>/seg_NNNN.flac`` (temp, atomic rename).

    The source is decoded once into a 16 kHz mono array; each span is sliced out of it in memory
    and written as FLAC -- no per-clip ffmpeg spawn, no O(N x file) redundant re-decoding.
    ``source`` is the already-resolved source path (work drive or archive); resolved here if None.
    """
    out_dir = clips_root / video.channel / video.video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    if source is None:
        source = resolve_source_path(video.path, channel=video.channel)
    audio = load_16k_mono(source)  # one decode for the whole video (working drive or archive)
    clips: list[QClip] = []
    for idx, (start, end) in enumerate(spans):
        final = out_dir / f"seg_{idx:04d}.flac"
        tmp = out_dir / f".seg_{idx:04d}.tmp.flac"  # keep .flac so soundfile picks the format
        write_clip_16k(audio, tmp, start, end)
        tmp.replace(final)  # atomic on the same filesystem -> no half-cut clip is ever visible
        clips.append(
            QClip(
                clip_id=f"{video.video_id}_{idx:04d}",
                video_id=video.video_id,
                channel=video.channel,
                clip_index=idx,
                clip_path=str(final),
                start=round(start, 2),
                end=round(end, 2),
                language=language,
                script=script,
                citation=video.citation,
            )
        )
    return clips


def segment_worker(
    queue_path: Path,
    *,
    clips_root: Path,
    language: str,
    script: str,
    worker_id: str,
    max_dur: float = 30.0,
    pending_hwm: int = DEFAULT_PENDING_HWM,
    min_free_gb: float = DEFAULT_MIN_FREE_GB,
    poll_s: float = 5.0,
    idle_exit: bool = True,
    vad_kwargs: dict[str, Any] | None = None,
    device: str | None = None,
    cpu_threads: int = 1,
) -> int:
    """Resident-model loop: claim -> VAD -> cut -> enqueue till the queue drains. Returns videos."""
    import os

    # Cap intra-op threads BEFORE torch imports — `torch.set_num_threads` alone does NOT bound the
    # MKL/OMP/OpenBLAS pools (they size to all cores at import), so N parallel workers otherwise
    # oversubscribe the box and thrash. Set the env vars first; we want workers*threads ~= cores.
    n = str(max(1, cpu_threads))
    for var in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    ):
        os.environ[var] = n
    import torch

    torch.set_num_threads(max(1, cpu_threads))
    from omni_curator.create.vad import load_vad_model, segment_vad_with

    queue = QueueStore(queue_path)
    model = load_vad_model(device=device)
    done = 0
    try:
        while True:
            # Two-sided backpressure: pause if the labeler is behind (pending-clip HWM) OR if the
            # clips disk is running out of space. Either way, idle_exit must NOT fire — there is
            # work, we're just throttled — so always poll-sleep and recheck, never break.
            if queue.pending_clip_count() >= pending_hwm:  # labeler is behind
                time.sleep(poll_s)
                continue
            if _free_gb(clips_root) < min_free_gb:  # clips disk almost full
                time.sleep(poll_s)
                continue
            video = queue.claim_video(worker_id)
            if video is None:
                if idle_exit:
                    break
                time.sleep(poll_s)
                continue
            try:
                # Resolve the source once (working drive OR the storage archive, for re-segments),
                # then VAD + cut both read it — never re-derive the path.
                source = resolve_source_path(video.path, channel=video.channel)
                spans = segment_vad_with(model, source, max_dur=max_dur, **(vad_kwargs or {}))
                clips = _cut_clips(video, spans, clips_root=clips_root, language=language,
                                   script=script, source=source)
                queue.complete_video(video.video_id, clips, claim_token=video.claim_token)
                done += 1
            except torch.cuda.OutOfMemoryError:
                # A GPU OOM is the worker's fault, not the video's — don't fail/retire the clip.
                # Re-raise so this worker exits non-zero (caught by run_segmenters); the video's
                # lease lapses and another worker re-claims it. Swallowing it would silently fail
                # every video on a too-crowded GPU.
                raise
            except Exception as exc:  # noqa: BLE001 — one bad video must never abort the worker
                queue.fail_video(
                    video.video_id, f"{type(exc).__name__}: {exc}", claim_token=video.claim_token
                )
    finally:
        queue.close()
    return done


def run_segmenters(
    queue_path: Path,
    *,
    gpu_procs: int,
    cpu_procs: int,
    clips_root: Path,
    language: str,
    script: str,
    max_dur: float = 30.0,
    pending_hwm: int = DEFAULT_PENDING_HWM,
    min_free_gb: float = DEFAULT_MIN_FREE_GB,
    vad_kwargs: dict[str, Any] | None = None,
) -> None:
    """Drain the queue with ``gpu_procs`` GPU-VAD + ``cpu_procs`` CPU-VAD workers, concurrently.

    Running the GPU and the cores together segments far faster than either alone. Uses the
    ``spawn`` start method — forking after torch/NeMo is imported can deadlock.
    """
    import os

    # GPU workers do VAD on the card (1 CPU thread is plenty); CPU workers split the remaining cores
    # so total intra-op threads ~= core count instead of cores-squared.
    ncpu = os.cpu_count() or 4
    cpu_threads = max(1, (ncpu - gpu_procs) // cpu_procs) if cpu_procs else 1
    specs = [("cuda", i, 1) for i in range(gpu_procs)]
    specs += [("cpu", i, cpu_threads) for i in range(cpu_procs)]
    if not specs:
        return
    common: dict[str, Any] = {
        "clips_root": clips_root, "language": language, "script": script,
        "max_dur": max_dur, "pending_hwm": pending_hwm, "min_free_gb": min_free_gb,
        "vad_kwargs": vad_kwargs,
    }
    if len(specs) == 1:
        dev, _, th = specs[0]
        segment_worker(queue_path, worker_id=f"seg-{dev}-0", device=dev, cpu_threads=th, **common)
        return

    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    workers = [
        ctx.Process(
            target=segment_worker,
            args=(queue_path,),
            kwargs={**common, "worker_id": f"seg-{dev}-{i}", "device": dev, "cpu_threads": th},
        )
        for dev, i, th in specs
    ]
    for w in workers:
        w.start()
    try:
        for w in workers:
            w.join()
    finally:
        # The parent dying (signal, KeyboardInterrupt, an exception above) must NOT leave spawned
        # VAD workers orphaned — they each hold a CUDA context + the resident model (audit: 3
        # orphans pinning 11.5 GB of VRAM). Tear every still-alive child down so it releases the
        # GPU: SIGTERM first (clean), escalate to SIGKILL for any that ignore it.
        _terminate_workers(workers)
    failed = [w.exitcode for w in workers if w.exitcode]
    if failed:  # a worker that died (e.g. CUDA OOM) must not look like a successful drain
        msg = f"{len(failed)}/{len(workers)} segment workers exited non-zero (exit codes {failed})"
        raise RuntimeError(msg)


def _terminate_workers(workers: Sequence[Any], *, grace_s: float = 10.0) -> None:
    """SIGTERM then (after ``grace_s``) SIGKILL every still-alive worker so it releases its VRAM.

    Idempotent: a worker that already exited (normal drain or its own crash) is skipped, so the
    happy path is a no-op. Only the ones still running on an abort are reaped.
    """
    alive = [w for w in workers if w.is_alive()]
    if not alive:
        return
    for w in alive:
        w.terminate()  # SIGTERM
    for w in alive:
        w.join(timeout=grace_s)
    for w in alive:
        if w.is_alive():
            w.kill()  # SIGKILL — it ignored SIGTERM
            w.join(timeout=grace_s)
