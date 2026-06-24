"""Audio processing: bring any source audio to the curator standard — 16 kHz mono FLAC."""

from __future__ import annotations

import os
import subprocess
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from omni_curator.data.sample import Sample


def to_16k_flac(
    src: Path, dst: Path, *, start: float | None = None, end: float | None = None
) -> None:
    """Convert any audio to 16 kHz mono FLAC via ffmpeg; optionally cut ``[start, end)``.

    ``start``/``end`` (seconds) are input-seek args placed before ``-i`` (fast seek), so the same
    encoder path serves both a full-file resample and a clip cut.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    seek: list[str] = []
    if start is not None:
        seek += ["-ss", f"{start:.3f}"]
    if end is not None:
        seek += ["-to", f"{end:.3f}"]
    subprocess.run(  # noqa: S603
        [  # noqa: S607
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            *seek, "-i", str(src),
            "-ar", "16000", "-ac", "1", "-c:a", "flac", str(dst),
        ],
        check=True,
    )


def resample_sample(sample: Sample, out_dir: Path) -> Sample | None:
    """Resample a sample's audio to 16 kHz mono FLAC under ``out_dir``; return it updated.

    Skips the ffmpeg pass if the output already exists AND is a valid FLAC, so a re-run resumes.
    A clip left truncated by a killed run is re-encoded rather than poisoning the whole resume.
    Returns ``None`` if the *source* audio can't be decoded (a corrupt clip in a noisy corpus) so
    one bad file is skipped instead of aborting the whole ingest.
    """
    import soundfile as sf

    dst = out_dir / f"{sample.id}.flac"
    info = None
    if dst.exists():
        try:
            info = sf.info(str(dst))  # readable existing clip -> trust it (resume)
        except sf.LibsndfileError:
            info = None  # truncated/corrupt (killed mid-write) -> re-encode below
    if info is None:
        try:
            to_16k_flac(Path(sample.audio_path), dst)
            info = sf.info(str(dst))
        except (subprocess.CalledProcessError, sf.LibsndfileError):
            dst.unlink(missing_ok=True)  # undecodable source -> drop this clip
            return None
    return replace(
        sample,
        audio_path=str(dst),
        sample_rate=16_000,
        duration=float(info.frames) / float(info.samplerate),
    )


def resample_samples(
    samples: Iterable[Sample], out_dir: Path, *, workers: int | None = None
) -> Iterator[Sample]:
    """Resample a stream of samples in parallel (ffmpeg per clip), yielding each as it finishes.

    Streams: at most ~2x ``workers`` clips are in flight, so a multi-million-clip corpus never
    materializes in memory. Already-converted clips are skipped (resume); an undecodable source
    yields nothing. ``workers`` defaults to all cores (each ffmpeg ≈ one core).
    """
    import contextlib

    from tqdm import tqdm

    max_workers = workers or (os.cpu_count() or 8)
    worker = partial(resample_sample, out_dir=out_dir)
    source = iter(samples)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        inflight: deque = deque()
        for _ in range(max_workers * 2):  # prime the pump (bounded look-ahead)
            try:
                inflight.append(pool.submit(worker, next(source)))
            except StopIteration:
                break
        with tqdm(desc=f"resample {out_dir.name}") as pbar:
            while inflight:
                result = inflight.popleft().result()
                pbar.update(1)
                with contextlib.suppress(StopIteration):  # keep the pool full
                    inflight.append(pool.submit(worker, next(source)))
                if result is not None:  # drop clips with undecodable source audio
                    yield result
