from __future__ import annotations

import sqlite3

from farsi_asr.quality_v4_export import (
    RiskRow,
    _duration_bin,
    _duration_bin_seconds,
    _match_control,
    _normalized_value_ranks,
    _score_group,
    _take_to_target,
)


def _raw_row(
    index: int,
    *,
    text: str,
    duration: float,
    wer: float,
    cer: float,
    edge: float,
    status: str = "aligned",
    coverage_bad: float = 0.0,
    span_bad: float = 0.1,
    margin_ratio: float = 0.1,
) -> sqlite3.Row:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    return connection.execute(
        """
        SELECT
            'part.parquet' AS hub_path,
            ? AS hub_row_index,
            'source' AS source,
            ? AS text,
            'sha' AS audio_sha256,
            ? AS duration,
            ? AS wer,
            ? AS cer,
            ? AS edge_chars,
            ? AS alignment_status,
            ? AS coverage_bad,
            ? AS span_bad,
            ? AS margin_ratio,
            0.0 AS overrun_ratio
        """,
        (
            index,
            text,
            duration,
            wer,
            cer,
            edge,
            status,
            coverage_bad,
            span_bad,
            margin_ratio,
        ),
    ).fetchone()


def _risk_row(index: int, duration: float, risk: float) -> RiskRow:
    return RiskRow(
        hub_path="part.parquet",
        hub_row_index=index,
        source="source",
        text="متن",
        audio_sha256="sha",
        duration=duration,
        duration_bin=_duration_bin(duration),
        has_digit=False,
        aligned=True,
        risk=risk,
    )


def test_normalized_value_ranks_share_ties() -> None:
    assert _normalized_value_ranks([1.0, 1.0, 3.0]) == {1.0: 0.0, 3.0: 1.0}


def test_score_group_omits_asr_agreement_for_digits_and_long_edges() -> None:
    rows = _score_group(
        [
            _raw_row(0, text="سال ۱۴۰۰", duration=8.0, wer=0.8, cer=0.8, edge=8.0),
            _raw_row(1, text="متن", duration=21.0, wer=0.1, cer=0.1, edge=9.0),
        ]
    )
    assert rows[0].agreement_risk is None
    assert rows[0].edge_risk is not None
    assert rows[1].agreement_risk is not None
    assert rows[1].edge_risk is None


def test_unaligned_row_gets_max_alignment_risk() -> None:
    rows = _score_group(
        [
            _raw_row(0, text="متن", duration=8.0, wer=0.1, cer=0.1, edge=0.0),
            _raw_row(
                1,
                text="متن دوم",
                duration=8.0,
                wer=0.1,
                cer=0.1,
                edge=0.0,
                status="not_aligned",
                coverage_bad=1.0,
                span_bad=1.0,
                margin_ratio=1.0,
            ),
        ]
    )
    assert rows[1].alignment_risk == 1.0
    assert rows[1].risk > rows[0].risk


def test_selection_is_deterministic_and_control_is_duration_matched() -> None:
    group = [_risk_row(index, 5.0 + index / 10, index / 20) for index in range(20)]
    lower_half = group[:10]
    cleaned_a = _take_to_target(lower_half, 28.0, seed=7)
    cleaned_b = _take_to_target(lower_half, 28.0, seed=7)
    control_a = _match_control(group, cleaned_a, seed=7)
    control_b = _match_control(group, cleaned_b, seed=7)
    assert [row.identity for row in cleaned_a] == [row.identity for row in cleaned_b]
    assert [row.identity for row in control_a] == [row.identity for row in control_b]
    assert len(control_a) == len(cleaned_a)
    assert len({row.identity for row in control_a}) == len(control_a)
    assert {row.identity for row in control_a}.isdisjoint({row.identity for row in cleaned_a})
    assert all(
        control.identity != cleaned.identity
        for control, cleaned in zip(control_a, cleaned_a, strict=True)
    )


def test_selection_leaves_tiny_duration_bin_empty() -> None:
    assert _take_to_target([_risk_row(0, 22.0, 0.1)], 1.0, seed=7) == []


def test_duration_bin_seconds_keeps_all_four_bins() -> None:
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE quality_rows (source TEXT, duration REAL)")
    connection.executemany(
        "INSERT INTO quality_rows VALUES ('source', ?)",
        [(4.0,), (7.0,), (15.0,), (25.0,)],
    )
    assert _duration_bin_seconds(connection) == {
        ("source", "00-05"): 4.0,
        ("source", "05-10"): 7.0,
        ("source", "10-20"): 15.0,
        ("source", "20-plus"): 25.0,
    }
