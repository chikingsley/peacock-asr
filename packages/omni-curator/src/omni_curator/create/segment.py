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

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from omni_curator.create.queue import QClip, QueueStore, QVideo
from omni_curator.process.audio import load_16k_mono, write_clip_16k

if TYPE_CHECKING:
    from collections.abc import Sequence


def _cut_clips(
    video: QVideo,
    spans: Sequence[tuple[float, float]],
    *,
    clips_root: Path,
    language: str,
    script: str,
) -> list[QClip]:
    """Cut each span to ``clips_root/<channel>/<video_id>/seg_NNNN.flac`` (temp, atomic rename).

    The source is decoded once into a 16 kHz mono array; each span is sliced out of it in memory
    and written as FLAC -- no per-clip ffmpeg spawn, no O(N x file) redundant re-decoding.
    """
    out_dir = clips_root / video.channel / video.video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    audio = load_16k_mono(Path(video.path))  # one decode for the whole video
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
    pending_hwm: int = 50_000,
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
            if queue.pending_clip_count() >= pending_hwm:  # backpressure: labeler is behind
                time.sleep(poll_s)
                continue
            video = queue.claim_video(worker_id)
            if video is None:
                if idle_exit:
                    break
                time.sleep(poll_s)
                continue
            try:
                spans = segment_vad_with(model, Path(video.path), max_dur=max_dur,
                                         **(vad_kwargs or {}))
                clips = _cut_clips(video, spans, clips_root=clips_root, language=language,
                                   script=script)
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
    pending_hwm: int = 50_000,
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
        "max_dur": max_dur, "pending_hwm": pending_hwm, "vad_kwargs": vad_kwargs,
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
    for w in workers:
        w.join()
    failed = [w.exitcode for w in workers if w.exitcode]
    if failed:  # a worker that died (e.g. CUDA OOM) must not look like a successful drain
        msg = f"{len(failed)}/{len(workers)} segment workers exited non-zero (exit codes {failed})"
        raise RuntimeError(msg)
