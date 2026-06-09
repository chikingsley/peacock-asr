"""Segmenters: turn one audio file into a list of (start, end) spans to label.

- ``segment_vad`` — cut at silences, non-overlapping (dense continuous speech).
"""

from __future__ import annotations

from omni_curator.create.segmenters.vad import (
    SpeechWindow,
    boolean_windows,
    load_vad_model,
    segment_vad,
    segment_vad_with,
)

__all__ = [
    "SpeechWindow",
    "boolean_windows",
    "load_vad_model",
    "segment_vad",
    "segment_vad_with",
]
