"""VAD windowing contract: no output span ever exceeds ``hard_max_seconds``.

The core regression: a single continuous-speech island (no internal silence) longer than the cap
was written as ONE over-length clip (dari: 60k clips >30s, up to 777s). ``boolean_windows`` must now
split such islands into contiguous sub-windows, each <= the cap.
"""

from __future__ import annotations

import itertools
import math

from omni_curator.create.vad import SpeechWindow, boolean_windows, split_window

FRAME = 0.02  # 20 ms frames, matching segment_vad_with's default


def _island_flags(seconds: float, *, frame_seconds: float = FRAME) -> list[bool]:
    """A flat mask of `seconds` of continuous speech (no gaps)."""
    return [True] * round(seconds / frame_seconds)


def _windows(seconds: float, *, max_dur: float = 30.0, min_dur: float = 1.0) -> list[SpeechWindow]:
    return boolean_windows(
        _island_flags(seconds),
        frame_seconds=FRAME,
        min_duration_seconds=min_dur,
        merge_gap_seconds=1.5,
        hard_max_seconds=max_dur,
    )


def test_long_island_is_split_into_capped_chunks():
    """~100 s of continuous speech, cap 30 -> 4 windows each <= 30 s, contiguous, lossless."""
    wins = _windows(100.0, max_dur=30.0)
    assert len(wins) == math.ceil(100.0 / 30.0) == 4
    for w in wins:
        assert w.duration <= 30.0 + 1e-9, f"window {w} exceeds cap"
    # contiguous + lossless: chunks abut and span the whole island
    for a, b in itertools.pairwise(wins):
        assert a.end == b.start
    assert wins[0].start == 0.0
    assert abs(wins[-1].end - 100.0) < FRAME


def test_exactly_at_cap_stays_one_window():
    wins = _windows(30.0, max_dur=30.0)
    assert len(wins) == 1
    assert wins[0].duration <= 30.0 + 1e-9


def test_one_over_cap_splits_into_two():
    wins = _windows(31.0, max_dur=30.0)
    assert len(wins) == 2
    for w in wins:
        assert w.duration <= 30.0 + 1e-9


def test_under_min_duration_island_is_dropped():
    """A sub-min island is filtered (existing min-duration rule), not emitted as a tiny clip."""
    assert _windows(0.5, min_dur=1.0) == []


def test_split_window_no_op_within_cap():
    w = SpeechWindow(5.0, 20.0)
    assert split_window(w, hard_max_seconds=30.0, min_duration_seconds=1.0) == [w]


def test_split_window_folds_sub_min_tail_into_previous():
    """An exact-multiple split has no sliver; an awkward length must not leave a < min tail."""
    # 61 s, cap 30 -> ceil = 3 equal chunks of ~20.33 s each (no sliver). Verify all >= min and
    # the boundary case where a naive last-chunk slice would be tiny is folded.
    w = SpeechWindow(0.0, 61.0)
    chunks = split_window(w, hard_max_seconds=30.0, min_duration_seconds=1.0)
    assert all(c.duration <= 30.0 + 1e-9 for c in chunks)
    assert all(c.duration >= 1.0 for c in chunks)
    assert chunks[0].start == 0.0
    assert chunks[-1].end == 61.0
