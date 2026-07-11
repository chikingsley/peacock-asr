from pathlib import Path

import pytest

from asr_benchmark_core.store import BenchmarkStore, NBestCandidate, Prediction


def test_nbest_candidates_round_trip(tmp_path: Path) -> None:
    store = BenchmarkStore(tmp_path / "benchmark.sqlite3")
    store.ensure_run(
        run_id="example",
        adapter="omni",
        model_path=tmp_path / "model",
        benchmark_path=tmp_path / "data.parquet",
        language="Persian",
        config={"nbest": 2},
    )
    candidates = [
        NBestCandidate(0, 0, "best", -1.0, -0.5),
        NBestCandidate(0, 1, "second", -2.0, -1.5),
        NBestCandidate(1, 0, "only", -3.0, -2.5),
    ]
    store.add_nbest("example", candidates)
    assert store.nbest_for_run("example") == candidates
    store.close()


def test_store_is_resumable_and_rejects_run_id_collision(tmp_path: Path) -> None:
    store = BenchmarkStore(tmp_path / "benchmark.sqlite3")
    kwargs = {
        "run_id": "example",
        "adapter": "whisper",
        "model_path": tmp_path / "model",
        "benchmark_path": tmp_path / "data.parquet",
        "language": "Persian",
        "config": {"limit": 10},
    }
    store.ensure_run(**kwargs)
    store.add(
        "example",
        Prediction(0, "reference", "hypothesis", 1.0, 0.1),
    )
    assert store.completed_rows("example") == {0}

    store.ensure_run(**kwargs)
    with pytest.raises(ValueError, match="different configuration"):
        store.ensure_run(**(kwargs | {"language": "English"}))
    store.close()
