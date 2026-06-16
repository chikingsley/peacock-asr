"""Audio processing: bring any source audio to the curator standard — 16 kHz mono FLAC."""

from __future__ import annotations

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from omni_curator.sample import Sample


def to_16k_flac(src: Path, dst: Path) -> None:
    """Convert any audio (mp3/wav/flac/...) at any rate to 16 kHz mono FLAC via ffmpeg."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(  # noqa: S603
        [  # noqa: S607
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
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
) -> list[Sample]:
    """Resample many samples in parallel (ffmpeg per clip, so threads give real speedup), with a
    progress bar. Already-converted clips are skipped, so this resumes cleanly. ``workers`` defaults
    to all CPU cores (each ffmpeg ≈ one core; oversubscribing past core-count doesn't help).
    """
    from tqdm import tqdm

    items = list(samples)
    max_workers = workers or (os.cpu_count() or 8)
    worker = partial(resample_sample, out_dir=out_dir)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        done = tqdm(pool.map(worker, items), total=len(items), desc=f"resample {out_dir.name}")
        return [s for s in done if s is not None]  # drop clips with undecodable source audio
