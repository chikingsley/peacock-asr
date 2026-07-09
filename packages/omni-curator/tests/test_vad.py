"""VAD windowing contract: no output span ever exceeds ``hard_max_seconds``.

The core regression: a single continuous-speech island (no internal silence) longer than the cap
was written as ONE over-length clip (dari: 60k clips >30s, up to 777s). ``boolean_windows`` must now
split such islands into contiguous sub-windows, each <= the cap.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from omni_curator.create.vad import (
    SileroEngine,
    SpeechWindow,
    _pvcobra_device,
    _require_marblenet_v2,
    _resolve_silero_backend,
    boolean_windows,
    build_vad_policy,
    effective_profile_id,
    postprocess_intervals,
    postprocess_profile,
    segment_audio_with,
    split_window,
)

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
    assert split_window(w, hard_max_seconds=30.0) == [w]


def test_split_window_keeps_hard_cap_non_negotiable():
    """Even awkward or impossible min/max cases must not create an over-cap clip."""
    w = SpeechWindow(0.0, 61.0)
    chunks = split_window(w, hard_max_seconds=30.0)
    assert all(c.duration <= 30.0 + 1e-9 for c in chunks)
    assert chunks[0].start == 0.0
    assert chunks[-1].end == 61.0

    awkward = SpeechWindow(0.0, 31.0)
    chunks = split_window(awkward, hard_max_seconds=30.0)
    assert len(chunks) == 2
    assert all(c.duration <= 30.0 + 1e-9 for c in chunks)


def test_shared_postprocessor_sanitizes_clamps_sorts_merges_then_filters():
    profile = postprocess_profile("conservative-v1", max_speech_s=2.0)
    intervals = [
        (2.05, 2.20),  # merges with next after 30 ms padding; combined span clears 250 ms
        (float("nan"), 1.0),
        (2.22, 2.35),
        (5.0, 14.0),  # clamps to 10 s, then splits to the 2 s hard cap
        (-1.0, 0.10),
        (4.0, 3.0),
    ]

    output = postprocess_intervals(intervals, audio_seconds=10.0, profile=profile)

    assert output == sorted(output)
    assert all(0 <= start < end <= 10.0 for start, end in output)
    assert all(end - start <= 2.0 + 1e-9 for start, end in output)
    assert any(start <= 2.02 and end >= 2.38 for start, end in output)
    # The 100 ms edge island remains under 250 ms after clamped padding and is filtered.
    assert not any(start == 0.0 for start, _ in output)


def test_policy_id_is_stable_and_changes_with_effective_configuration():
    first = build_vad_policy(engine="cobra", profile="conservative-v1", max_speech_s=30)
    same = build_vad_policy(engine="cobra", profile="conservative-v1", max_speech_s=30)
    changed = build_vad_policy(engine="cobra", profile="conservative-v1", max_speech_s=40)
    assert first.profile_id == same.profile_id
    assert first.profile_id != changed.profile_id
    assert first.as_dict()["profile_id"] == first.profile_id
    assert effective_profile_id(first, "model-a") != effective_profile_id(first, "model-b")
    assert effective_profile_id(
        first, "model-a", runtime_metadata={"device": "cpu"}
    ) != effective_profile_id(first, "model-a", runtime_metadata={"device": "cuda"})


def test_engine_device_mapping_and_silero_backend_are_explicit():
    assert _pvcobra_device("cpu") == "cpu"
    assert _pvcobra_device("cuda") == "gpu:0"
    assert _pvcobra_device("cuda:2") == "gpu:2"
    assert _resolve_silero_backend(backend="auto", device="cpu") == "onnx"
    assert _resolve_silero_backend(backend="auto", device="cuda") == "jit"
    with pytest.raises(ValueError, match="CPU-only"):
        _resolve_silero_backend(backend="onnx", device="cuda")


def test_legacy_profile_preserves_cap_aware_merge_boundaries():
    profile = postprocess_profile("legacy-marblenet-v1", max_speech_s=30.0)
    assert postprocess_intervals(
        [(0.0, 20.0), (21.0, 40.0)], audio_seconds=40.0, profile=profile
    ) == [(0.0, 20.0), (21.0, 40.0)]


def test_marblenet_v2_hash_is_enforced(tmp_path):
    wrong = tmp_path / "wrong.nemo"
    wrong.write_bytes(b"not the pinned checkpoint")
    with pytest.raises(ValueError, match="hash mismatch"):
        _require_marblenet_v2(wrong)


def test_silero_rejects_non_16k_before_backend_inference():
    engine = object.__new__(SileroEngine)
    with pytest.raises(ValueError, match="16 kHz"):
        engine.predict(np.zeros(8_000, dtype=np.float32), 8_000)


def test_engine_contract_returns_raw_then_one_shared_policy():
    audio = np.zeros(16_000, dtype=np.float32)

    class FakeEngine:
        name = "fake"
        model_revision = "fake-v1"

        def predict(self, received, sample_rate):
            assert received is audio
            assert sample_rate == 16_000
            return [(0.1, 0.2), (0.25, 0.45)]

        def close(self):
            return

    policy = build_vad_policy(engine="marblenet", profile="conservative-v1")
    result = segment_audio_with(FakeEngine(), audio, policy=policy)
    assert result.raw_intervals == ((0.1, 0.2), (0.25, 0.45))
    assert result.intervals == ((0.07, 0.48),)
