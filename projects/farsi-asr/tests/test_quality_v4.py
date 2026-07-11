import sqlite3
from argparse import Namespace
from pathlib import Path

import numpy as np

from farsi_asr.quality_v4 import _audio_bytes, cmd_attach


def test_audio_bytes_restores_signed_int8_storage() -> None:
    encoded = bytes([0, 127, 128, 255])
    stored = np.frombuffer(encoded, dtype=np.int8).tolist()

    assert _audio_bytes(stored) == encoded


def test_attach_predictions_preserves_rows(tmp_path: Path) -> None:
    manifest = tmp_path / "sample.jsonl"
    manifest.write_text('{"sample_id":"a","text":"مرسی"}\n', encoding="utf-8")
    database = tmp_path / "bench.sqlite3"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE runs (run_id TEXT PRIMARY KEY, model_path TEXT, benchmark_path TEXT);
        CREATE TABLE predictions (
            run_id TEXT, row_index INTEGER, hypothesis TEXT, error TEXT
        );
        INSERT INTO runs VALUES ('edge', '/model', '/benchmark');
        INSERT INTO predictions VALUES ('edge', 0, 'مرسی', NULL);
        """
    )
    connection.commit()
    connection.close()
    output = tmp_path / "predictions.jsonl"

    assert (
        cmd_attach(
            Namespace(
                input=manifest,
                database=database,
                run_id="edge",
                output=output,
                hypothesis_field="hypothesis",
            )
        )
        == 0
    )
    assert '"hypothesis": "مرسی"' in output.read_text(encoding="utf-8")
