"""Fuse: turn raw ASR variants into a finished transcript.

- ``compile_down`` — fuse one clip's ensemble variants into a consensus label.
- ``stitch`` — reconcile overlapping-chunk labels into one continuous transcript (chunk path).
- ``polish`` — final pass that repairs obvious machine glitches, preserving natural speech.
"""

from __future__ import annotations

from omni_curator.fuse._extract import extract_transcript
from omni_curator.fuse.compile_down import compile_down
from omni_curator.fuse.polish import polish
from omni_curator.fuse.stitch import stitch

__all__ = ["compile_down", "extract_transcript", "polish", "stitch"]
