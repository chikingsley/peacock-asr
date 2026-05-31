"""Audio helpers shared by the segmenters and the pipeline."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def cut_audio(source: Path, output: Path, start: float, end: float) -> None:
    """Cut ``[start, end)`` from ``source`` to a 16 kHz mono FLAC at ``output`` (via ffmpeg)."""
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(  # noqa: S603
        [  # noqa: S607
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(source),
            "-ar", "16000", "-ac", "1", "-c:a", "flac", str(output),
        ],
        check=True,
    )


def audio_duration(audio: Path) -> float:
    """Duration of ``audio`` in seconds."""
    import soundfile as sf

    info = sf.info(str(audio))
    return float(info.frames) / float(info.samplerate)
