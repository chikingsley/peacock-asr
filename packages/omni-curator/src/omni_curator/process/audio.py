"""Audio processing: bring any source audio to the curator standard — 16 kHz mono FLAC."""

from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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


def resample_sample(sample: Sample, out_dir: Path) -> Sample:
    """Resample a sample's audio to 16 kHz mono FLAC under ``out_dir``; return it updated."""
    import soundfile as sf

    dst = out_dir / f"{sample.id}.flac"
    to_16k_flac(Path(sample.audio_path), dst)
    info = sf.info(str(dst))
    return replace(
        sample,
        audio_path=str(dst),
        sample_rate=16_000,
        duration=float(info.frames) / float(info.samplerate),
    )
