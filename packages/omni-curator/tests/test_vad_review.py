from __future__ import annotations

import json
import sqlite3

import numpy as np
import pytest
import soundfile as sf

from omni_curator.create.vad_review import (
    Candidate,
    _prepare_review_dir,
    _subtract_intervals,
    prepare_review,
    review_summary,
    sample_candidates,
)


def _candidate(index: int, *, direction: str, tier: str, duration: float) -> Candidate:
    favored, opposed = (
        ("marblenet", "silero") if direction == "marble_only" else ("silero", "marblenet")
    )
    start = float(index * 2)
    return Candidate(
        source_id=f"{tier}-{index % 5}",
        source_path=f"/{tier}-{index % 5}.flac",
        tier=tier,
        direction=direction,
        favored_engine=favored,
        opposed_engine=opposed,
        start=start,
        end=start + duration,
    )


def test_subtract_intervals_returns_only_disagreement_regions():
    assert _subtract_intervals(
        [(0.0, 2.0), (3.0, 5.0)],
        [(0.5, 1.0), (1.0, 1.5), (4.0, 6.0)],
    ) == [(0.0, 0.5), (1.5, 2.0), (3.0, 4.0)]


def test_sampler_is_exact_balanced_stratified_and_reproducible():
    candidates = []
    for direction in ("marble_only", "silero_only"):
        for tier in ("clean", "noisy"):
            candidates.extend(
                _candidate(index, direction=direction, tier=tier, duration=duration)
                for index, duration in enumerate([0.2] * 20 + [0.5] * 12 + [1.0] * 8)
            )

    selected = sample_candidates(candidates, total_items=80, seed=17)
    repeated = sample_candidates(candidates, total_items=80, seed=17)

    assert [item.candidate_id for item in selected] == [item.candidate_id for item in repeated]
    assert len({item.candidate_id for item in selected}) == 80
    for direction in ("marble_only", "silero_only"):
        for tier in ("clean", "noisy"):
            cell = [item for item in selected if item.direction == direction and item.tier == tier]
            assert len(cell) == 20
            assert sum(item.duration >= 0.75 for item in cell) == 4
            assert sum(0.35 <= item.duration < 0.75 for item in cell) == 6


def _write_pilot(tmp_path):
    rows = []
    for tier in ("clean", "noisy"):
        source = tmp_path / f"{tier}.flac"
        sf.write(source, np.zeros(64_000, dtype=np.float32), 16_000)
        source_id = f"{tier}-source"
        rows.extend(
            [
                {
                    "source_id": source_id,
                    "engine": "marblenet",
                    "path": str(source),
                    "tier": tier,
                    "intervals": [[0.5, 1.0]],
                },
                {
                    "source_id": source_id,
                    "engine": "silero",
                    "path": str(source),
                    "tier": tier,
                    "intervals": [[0.5, 0.75], [2.0, 2.25]],
                },
            ]
        )
    intervals = tmp_path / "intervals.jsonl"
    intervals.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return intervals


def test_prepare_review_writes_blinded_audio_and_resumable_store(tmp_path):
    review_dir = tmp_path / "review"
    summary = prepare_review(
        intervals_path=_write_pilot(tmp_path),
        output_dir=review_dir,
        total_items=4,
        seed=7,
    )

    manifest = json.loads((review_dir / "review_items.json").read_text())
    assert summary["cells"] == {
        "marble_only/clean": 1,
        "marble_only/noisy": 1,
        "silero_only/clean": 1,
        "silero_only/noisy": 1,
    }
    assert len(manifest["items"]) == 4
    assert "favored_engine" not in (review_dir / "index.html").read_text()
    for item in manifest["items"]:
        info = sf.info(review_dir / item["audio"])
        assert info.samplerate == 16_000
        assert info.channels == 1
    assert review_summary(review_dir)["remaining"] == 4

    first = manifest["items"][0]
    with sqlite3.connect(review_dir / "review.sqlite") as connection:
        connection.execute(
            "INSERT INTO votes(item_id,label,replay_count,reviewed_at) VALUES(?,?,?,?)",
            (first["item_id"], "speech", 1, 1.0),
        )
    updated = review_summary(review_dir)
    assert updated["reviewed"] == 1
    assert updated["engine_support"] == {first["favored_engine"]: 1}


def test_overwrite_refuses_unmarked_review_directory(tmp_path):
    output = tmp_path / "not-a-review"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("do not delete")

    with pytest.raises(PermissionError, match="unmarked"):
        _prepare_review_dir(output, overwrite=True)

    assert sentinel.read_text() == "do not delete"
