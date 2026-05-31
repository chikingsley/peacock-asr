"""Segmenters: turn one audio file into a list of (start, end) spans to label.

- ``segment_vad`` — cut at silences, non-overlapping (dense continuous speech).
- ``segment_chunks`` — fixed overlapping windows, 100% coverage (sparse / drill audio).
"""

from __future__ import annotations

from omni_curator.create.segmenters.chunks import segment_chunks
from omni_curator.create.segmenters.vad import SpeechWindow, boolean_windows, segment_vad

__all__ = ["SpeechWindow", "boolean_windows", "segment_chunks", "segment_vad"]
