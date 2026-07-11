from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Prediction:
    row_index: int
    reference: str
    hypothesis: str
    audio_seconds: float
    inference_seconds: float
    error: str | None = None


@dataclass(frozen=True)
class NBestCandidate:
    row_index: int
    rank: int
    hypothesis: str
    acoustic_score: float
    lm_score: float


class BenchmarkStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                schema_version INTEGER NOT NULL,
                adapter TEXT NOT NULL,
                model_path TEXT NOT NULL,
                benchmark_path TEXT NOT NULL,
                language TEXT NOT NULL,
                config_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS predictions (
                run_id TEXT NOT NULL REFERENCES runs(run_id),
                row_index INTEGER NOT NULL,
                reference TEXT NOT NULL,
                hypothesis TEXT NOT NULL,
                audio_seconds REAL NOT NULL,
                inference_seconds REAL NOT NULL,
                error TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (run_id, row_index)
            );
            CREATE TABLE IF NOT EXISTS nbest_candidates (
                run_id TEXT NOT NULL REFERENCES runs(run_id),
                row_index INTEGER NOT NULL,
                rank INTEGER NOT NULL,
                hypothesis TEXT NOT NULL,
                acoustic_score REAL NOT NULL,
                lm_score REAL NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (run_id, row_index, rank)
            );
            """
        )

    def ensure_run(  # noqa: PLR0913 - mirrors the persisted run identity fields
        self,
        *,
        run_id: str,
        adapter: str,
        model_path: Path,
        benchmark_path: Path,
        language: str,
        config: dict[str, object],
    ) -> None:
        row = self.connection.execute(
            "SELECT adapter, model_path, benchmark_path, language, config_json FROM runs "
            "WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        values = (
            adapter,
            str(model_path.resolve()),
            str(benchmark_path.resolve()),
            language,
            json.dumps(config, sort_keys=True),
        )
        if row is not None and tuple(row) != values:
            raise ValueError(f"run_id {run_id!r} already exists with different configuration")
        self.connection.execute(
            "INSERT OR IGNORE INTO runs "
            "(run_id, schema_version, adapter, model_path, benchmark_path, language, config_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (run_id, SCHEMA_VERSION, *values),
        )
        self.connection.commit()

    def completed_rows(self, run_id: str) -> set[int]:
        return {
            int(row[0])
            for row in self.connection.execute(
                "SELECT row_index FROM predictions WHERE run_id = ?", (run_id,)
            )
        }

    def add_nbest(self, run_id: str, candidates: list[NBestCandidate]) -> None:
        self.connection.executemany(
            "INSERT OR REPLACE INTO nbest_candidates "
            "(run_id, row_index, rank, hypothesis, acoustic_score, lm_score) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                (run_id, c.row_index, c.rank, c.hypothesis, c.acoustic_score, c.lm_score)
                for c in candidates
            ],
        )
        self.connection.commit()

    def nbest_for_run(self, run_id: str) -> list[NBestCandidate]:
        return [
            NBestCandidate(*row)
            for row in self.connection.execute(
                "SELECT row_index, rank, hypothesis, acoustic_score, lm_score "
                "FROM nbest_candidates WHERE run_id = ? ORDER BY row_index, rank",
                (run_id,),
            )
        ]

    def add(self, run_id: str, prediction: Prediction) -> None:
        self.add_predictions(run_id, [prediction])

    def add_predictions(self, run_id: str, predictions: list[Prediction]) -> None:
        self.connection.executemany(
            "INSERT OR REPLACE INTO predictions "
            "(run_id, row_index, reference, hypothesis, audio_seconds, "
            "inference_seconds, error) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    run_id,
                    p.row_index,
                    p.reference,
                    p.hypothesis,
                    p.audio_seconds,
                    p.inference_seconds,
                    p.error,
                )
                for p in predictions
            ],
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()
