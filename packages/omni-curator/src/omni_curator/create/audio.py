"""Create-stage audio primitive: cut a span out of a source recording.

Used by the segment stage (:mod:`omni_curator.create.segment`) to cut each VAD span into a clip.
A standalone module so it doesn't drag in the rest of the create machinery.
"""

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
