"""Segment stage: the CPU producer half of the split create pipeline.

Each worker loads the NeMo frame-VAD model **once** (resident for the process), then loops: claim a
video from the queue -> VAD-segment -> cut each span to a 16 kHz FLAC (temp file, then atomic
rename so the labeler never sees a half-written clip) -> enqueue the clip rows in one transaction.
Run several workers as separate processes (``run_segmenters``) to hold the cores near saturation;
they back off when the pending-clip high-watermark is hit so a stalled labeler can't let the
segmenter cut the entire corpus to disk.

The Scribe/label half lives in :mod:`omni_curator.create.labelq`; both share the queue
(:mod:`omni_curator.create.queue`). See ``docs/PIPELINE_SPLIT.md``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from omni_curator.create.pipeline import cut_audio
from omni_curator.create.queue import QClip, QueueStore, QVideo

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
    """Cut each span to ``clips_root/<channel>/<video_id>/seg_NNNN.flac`` (temp, atomic rename)."""
    out_dir = clips_root / video.channel / video.video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    src = Path(video.path)
    clips: list[QClip] = []
    for idx, (start, end) in enumerate(spans):
        final = out_dir / f"seg_{idx:04d}.flac"
        # Temp must keep the .flac suffix: ffmpeg infers the output format from the extension.
        tmp = out_dir / f".seg_{idx:04d}.tmp.flac"
        cut_audio(src, tmp, start, end)
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
) -> int:
    """Resident-model loop: claim -> VAD -> cut -> enqueue till the queue drains. Returns videos."""
    from omni_curator.create.segmenters.vad import load_vad_model, segment_vad_with

    queue = QueueStore(queue_path)
    model = load_vad_model()
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
                queue.complete_video(video.video_id, clips)
                done += 1
            except Exception as exc:  # noqa: BLE001 — one bad video must never abort the worker
                queue.fail_video(video.video_id, f"{type(exc).__name__}: {exc}")
    finally:
        queue.close()
    return done


def run_segmenters(
    queue_path: Path,
    *,
    procs: int,
    clips_root: Path,
    language: str,
    script: str,
    max_dur: float = 30.0,
    pending_hwm: int = 50_000,
    vad_kwargs: dict[str, Any] | None = None,
) -> None:
    """Spawn ``procs`` resident-model segment workers and wait for them to drain the queue.

    Uses the ``spawn`` start method — forking after torch/NeMo is imported can deadlock.
    """
    if procs <= 1:
        segment_worker(
            queue_path, clips_root=clips_root, language=language, script=script,
            worker_id="seg-0", max_dur=max_dur, pending_hwm=pending_hwm, vad_kwargs=vad_kwargs,
        )
        return

    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    workers = [
        ctx.Process(
            target=segment_worker,
            args=(queue_path,),
            kwargs={
                "clips_root": clips_root, "language": language, "script": script,
                "worker_id": f"seg-{i}", "max_dur": max_dur, "pending_hwm": pending_hwm,
                "vad_kwargs": vad_kwargs,
            },
        )
        for i in range(procs)
    ]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
