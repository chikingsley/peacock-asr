import sqlite3
from argparse import Namespace
from pathlib import Path

import pytest

from farsi_asr.benchmark import run


def test_score_shared_benchmark_store(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    database = tmp_path / "benchmark.sqlite3"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE runs (
            run_id TEXT PRIMARY KEY, adapter TEXT, model_path TEXT, benchmark_path TEXT
        );
        CREATE TABLE predictions (
            run_id TEXT, row_index INTEGER, reference TEXT, hypothesis TEXT,
            audio_seconds REAL, inference_seconds REAL, error TEXT
        );
        INSERT INTO runs VALUES ('smoke', 'whisper', '/model', '/data');
        INSERT INTO predictions VALUES ('smoke', 0, 'سلام دنیا', 'سلام دنیا', 2.0, 0.1, NULL);
        """
    )
    connection.commit()
    connection.close()

    assert run(Namespace(database=database, run_id="smoke")) == 0
    output = capsys.readouterr().out
    assert "wer: 0.00" in output
    assert "rtfx: 20.00" in output
