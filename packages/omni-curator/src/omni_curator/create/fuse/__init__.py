"""Fuse: turn raw ASR variants into a finished transcript.

- ``compile_down`` — fuse one clip's ensemble variants into a consensus label.
- ``stitch`` — reconcile overlapping-chunk labels into one continuous transcript (chunk path).
- ``polish`` — final pass that repairs obvious machine glitches, preserving natural speech.
- ``transliterate`` — convert text into the target script, content untouched (verify scoring).
"""

from __future__ import annotations

from omni_curator.create.fuse._extract import extract_transcript
from omni_curator.create.fuse.compile_down import compile_down
from omni_curator.create.fuse.polish import polish
from omni_curator.create.fuse.stitch import stitch
from omni_curator.create.fuse.transliterate import transliterate

__all__ = ["compile_down", "extract_transcript", "polish", "stitch", "transliterate"]
