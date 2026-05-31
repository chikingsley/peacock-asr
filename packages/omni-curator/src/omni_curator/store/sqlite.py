"""SQLite store for curated samples — point it at a DB and write/read standardized records.

The store is the curator's spine: ingest and create both upsert ``Sample``s here; process updates
them in place; benchmark and the dataset exporters read from here. Pointable anywhere (the DB
usually lives in an artifact folder inside the target language project/experiment).
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from typing import TYPE_CHECKING, Self

from omni_curator.sample import Sample

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (
    id          TEXT PRIMARY KEY,
    source      TEXT NOT NULL,
    language    TEXT NOT NULL,
    text        TEXT NOT NULL,
    audio_path  TEXT NOT NULL,
    duration    REAL NOT NULL,
    sample_rate INTEGER NOT NULL,
    split       TEXT NOT NULL DEFAULT 'train',
    speaker_id  TEXT,
    citation    TEXT,
    meta        TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_samples_source ON samples(source);
CREATE INDEX IF NOT EXISTS idx_samples_split  ON samples(split);
"""

_COLUMNS = (
    "id", "source", "language", "text", "audio_path", "duration",
    "sample_rate", "split", "speaker_id", "citation", "meta",
)


def _to_row(s: Sample) -> tuple[object, ...]:
    return (
        s.id, s.source, s.language, s.text, s.audio_path, s.duration,
        s.sample_rate, s.split, s.speaker_id, s.citation, json.dumps(s.meta, ensure_ascii=False),
    )


def _from_row(row: sqlite3.Row) -> Sample:
    return Sample(
        id=row["id"], source=row["source"], language=row["language"], text=row["text"],
        audio_path=row["audio_path"], duration=row["duration"], sample_rate=row["sample_rate"],
        split=row["split"], speaker_id=row["speaker_id"], citation=row["citation"],
        meta=json.loads(row["meta"]),
    )


class CuratorStore:
    """A SQLite-backed collection of curated :class:`Sample`s."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def upsert(self, samples: list[Sample]) -> int:
        """Insert or replace samples by id; returns the number written."""
        placeholders = ", ".join("?" for _ in _COLUMNS)
        with self._conn:
            self._conn.executemany(
                f"INSERT OR REPLACE INTO samples ({', '.join(_COLUMNS)}) VALUES ({placeholders})",  # noqa: S608 — columns are an internal constant; values are bound params
                [_to_row(s) for s in samples],
            )
        return len(samples)

    def iter_samples(
        self, *, source: str | None = None, split: str | None = None
    ) -> Iterator[Sample]:
        clauses, params = [], []
        if source is not None:
            clauses.append("source = ?")
            params.append(source)
        if split is not None:
            clauses.append("split = ?")
            params.append(split)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"SELECT * FROM samples{where} ORDER BY id"  # noqa: S608 — clauses are fixed strings; params are bound
        for row in self._conn.execute(query, params):
            yield _from_row(row)

    def counts(self) -> dict[str, int]:
        """Sample count + total hours per source."""
        rows = self._conn.execute(
            "SELECT source, COUNT(*) n, COALESCE(SUM(duration), 0) secs "
            "FROM samples GROUP BY source"
        ).fetchall()
        return {r["source"]: r["n"] for r in rows}

    def hours(self) -> float:
        secs = self._conn.execute("SELECT COALESCE(SUM(duration), 0) FROM samples").fetchone()[0]
        return float(secs) / 3600.0


@contextmanager
def open_store(db_path: Path) -> Iterator[CuratorStore]:
    """Context manager around :class:`CuratorStore`."""
    store = CuratorStore(db_path)
    try:
        yield store
    finally:
        store.close()
