from farsi_asr.vad_yield import _quantile_sample, _suspicious_repetition


def test_quantile_sample_spreads_across_duration() -> None:
    rows = [{"clip_id": str(index), "duration": float(index)} for index in range(10)]

    selected = _quantile_sample(rows, 5)

    assert [row["duration"] for row in selected] == [1.0, 3.0, 5.0, 7.0, 9.0]


def test_suspicious_repetition_requires_four_identical_tokens() -> None:
    assert _suspicious_repetition("سلام سلام سلام سلام")
    assert not _suspicious_repetition("سلام سلام سلام دنیا")
