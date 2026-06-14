from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from persian_asr_dataset.paths import DEFAULT_LEDGER

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class LedgerSample:
    sample_id: str
    source: str
    source_split: str
    source_row_id: str
    raw_text: str
    normalized_text: str
    duration_seconds: float | None
    sample_rate: int | None
    audio_ref: str
    storage_kind: str
    metadata: dict[str, Any]
    ingest_version: str


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def connect_ledger(path: Path = DEFAULT_LEDGER) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS samples (
            sample_id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            source_split TEXT NOT NULL,
            source_row_id TEXT NOT NULL,
            raw_text TEXT NOT NULL,
            normalized_text TEXT NOT NULL,
            duration_seconds REAL,
            sample_rate INTEGER,
            audio_ref TEXT NOT NULL,
            storage_kind TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            ingest_version TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_samples_source_split
            ON samples(source, source_split);

        CREATE TABLE IF NOT EXISTS model_scores (
            sample_id TEXT NOT NULL,
            model_card TEXT NOT NULL,
            score_run_id TEXT NOT NULL,
            hypothesis TEXT NOT NULL,
            normalized_hypothesis TEXT NOT NULL,
            wer REAL NOT NULL,
            cer REAL NOT NULL,
            inference_seconds REAL,
            scored_at TEXT NOT NULL,
            PRIMARY KEY (sample_id, model_card, score_run_id),
            FOREIGN KEY(sample_id) REFERENCES samples(sample_id)
        );

        CREATE TABLE IF NOT EXISTS curation_decisions (
            sample_id TEXT NOT NULL,
            decision_version TEXT NOT NULL,
            bucket TEXT NOT NULL,
            reason TEXT NOT NULL,
            decided_at TEXT NOT NULL,
            PRIMARY KEY (sample_id, decision_version),
            FOREIGN KEY(sample_id) REFERENCES samples(sample_id)
        );
        """
    )
    return connection


def upsert_sample(
    connection: sqlite3.Connection,
    sample: LedgerSample,
) -> None:
    now = utc_now()
    connection.execute(
        """
        INSERT INTO samples (
            sample_id, source, source_split, source_row_id, raw_text, normalized_text,
            duration_seconds, sample_rate, audio_ref, storage_kind, metadata_json,
            ingest_version, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(sample_id) DO UPDATE SET
            source = excluded.source,
            source_split = excluded.source_split,
            source_row_id = excluded.source_row_id,
            raw_text = excluded.raw_text,
            normalized_text = excluded.normalized_text,
            duration_seconds = excluded.duration_seconds,
            sample_rate = excluded.sample_rate,
            audio_ref = excluded.audio_ref,
            storage_kind = excluded.storage_kind,
            metadata_json = excluded.metadata_json,
            ingest_version = excluded.ingest_version,
            updated_at = excluded.updated_at
        """,
        (
            sample.sample_id,
            sample.source,
            sample.source_split,
            sample.source_row_id,
            sample.raw_text,
            sample.normalized_text,
            sample.duration_seconds,
            sample.sample_rate,
            sample.audio_ref,
            sample.storage_kind,
            json.dumps(sample.metadata, ensure_ascii=False, sort_keys=True),
            sample.ingest_version,
            now,
            now,
        ),
    )
