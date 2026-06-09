"""Fuse: turn raw ASR variants into a finished transcript.

- ``compile_down`` — fuse one clip's ensemble variants into a consensus label.
- ``transliterate`` — convert text into the target script, content untouched (verify scoring).
"""

from __future__ import annotations

from omni_curator.create.fuse._extract import extract_transcript
from omni_curator.create.fuse.compile_down import compile_down
from omni_curator.create.fuse.transliterate import transliterate

__all__ = ["compile_down", "extract_transcript", "transliterate"]
