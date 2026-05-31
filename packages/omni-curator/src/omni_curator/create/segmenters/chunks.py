"""Fixed overlapping-chunk segmenter: tile the whole timeline, blind to speech.

100% coverage (nothing dropped), so it's the safe choice for sparse / drill-style audio where
VAD would drop short utterances. The overlap is transcribed twice, so the chunk path must be
followed by ``fuse.stitch`` to reconcile the seams.
"""

from __future__ import annotations


def segment_chunks(
    duration: float, *, chunk: float = 40.0, overlap: float = 10.0
) -> list[tuple[float, float]]:
    """Fixed windows of ``chunk`` seconds, each sharing ``overlap`` with the next, over [0, dur]."""
    step = max(chunk - overlap, 1.0)
    spans: list[tuple[float, float]] = []
    start = 0.0
    while start < duration:
        spans.append((start, min(start + chunk, duration)))
        if start + chunk >= duration:
            break
        start += step
    return spans
