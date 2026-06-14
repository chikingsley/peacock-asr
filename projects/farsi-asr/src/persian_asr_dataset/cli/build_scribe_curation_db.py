from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jiwer

from persian_asr_dataset.text_normalization import maybe_normalize

BATCH_SIZE = 5000


@dataclass(frozen=True)
class BuildStats:
    database: str
    rows_seen: int
    rows_written: int
    missing_scribe: int
    empty_scribe: int
    empty_normalized_reference: int
    exact_match: int
    pending_classification: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the SQLite Scribe curation table from persistent job rows.",
    )
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--database", type=Path, default=None)
    parser.add_argument("--rows", type=Path, default=None)
    parser.add_argument("--replace", action="store_true")
    return parser


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def create_table(connection: sqlite3.Connection, *, replace: bool) -> None:
    if replace:
        connection.execute("DROP TABLE IF EXISTS scribe_curation")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS scribe_curation (
            sample_id TEXT NOT NULL,
            job_order INTEGER NOT NULL,
            job_id TEXT,
            source TEXT,
            source_config TEXT,
            canonical_dataset TEXT,
            project_split TEXT,
            original_split TEXT,
            audio_filepath TEXT,
            audio_sha256 TEXT,
            duration_seconds REAL,
            raw_reference_text TEXT,
            raw_scribe_text TEXT,
            normalized_reference TEXT,
            normalized_scribe TEXT,
            wer REAL,
            cer REAL,
            exact_match INTEGER NOT NULL,
            empty_normalized_reference INTEGER NOT NULL,
            missing_scribe INTEGER NOT NULL,
            empty_scribe INTEGER NOT NULL,
            api_eligible INTEGER NOT NULL,
            difference_category TEXT,
            difference_description TEXT,
            likely_cause TEXT,
            scribe_audio_stem TEXT,
            scribe_record_json TEXT,
            source_record_json TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (sample_id, job_order)
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_scribe_curation_job_order
            ON scribe_curation(job_order);
        CREATE INDEX IF NOT EXISTS idx_scribe_curation_category
            ON scribe_curation(difference_category);
        CREATE INDEX IF NOT EXISTS idx_scribe_curation_flags
            ON scribe_curation(api_eligible, exact_match, missing_scribe, empty_scribe);
        CREATE INDEX IF NOT EXISTS idx_scribe_curation_source
            ON scribe_curation(canonical_dataset, project_split);
        """
    )


def audio_stem(row: dict[str, Any]) -> str:
    for key in ("local_job_audio_path", "mac_job_audio_path", "local_cache_path", "mac_cache_path"):
        value = str(row.get(key) or "")
        if value:
            return Path(value).stem
    return str(row.get("audio_sha256") or "")


def scribe_results(connection: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT audio_stem, row_order, audio_path, transcript, record_json
        FROM scribe_results
        """
    ).fetchall()
    return {
        str(audio_stem_value): {
            "audio_stem": audio_stem_value,
            "row_order": row_order,
            "audio_path": audio_path,
            "transcript": transcript,
            "record_json": record_json,
        }
        for audio_stem_value, row_order, audio_path, transcript, record_json in rows
    }


def score(normalized_reference: str, normalized_scribe: str) -> tuple[float | None, float | None]:
    if not normalized_reference:
        return None, None
    wer_value = jiwer.wer(normalized_reference, normalized_scribe)
    cer_value = jiwer.cer(normalized_reference, normalized_scribe)
    if isinstance(cer_value, dict):
        raise TypeError("jiwer.cer returned detailed measurements instead of a scalar")
    return float(wer_value), float(cer_value)


def curation_row(
    source_row: dict[str, Any],
    result: dict[str, Any] | None,
    now: str,
) -> tuple[Any, ...]:
    raw_reference = str(source_row.get("text") or "")
    raw_scribe = str((result or {}).get("transcript") or "").strip()
    normalized_reference = maybe_normalize(raw_reference) or ""
    normalized_scribe = maybe_normalize(raw_scribe) or ""
    wer, cer = score(normalized_reference, normalized_scribe)
    missing_scribe = result is None
    empty_scribe = not bool(raw_scribe)
    empty_reference = not bool(normalized_reference)
    exact = bool(normalized_reference) and normalized_reference == normalized_scribe
    api_eligible = bool(normalized_reference and normalized_scribe and not exact)
    category = "exact_match" if exact else None
    description = "normalized strings are equal" if exact else None
    cause = "deterministic equality" if exact else None
    return (
        str(source_row["sample_id"]),
        int(source_row["job_order"]),
        source_row.get("job_id"),
        source_row.get("source"),
        source_row.get("source_config"),
        source_row.get("canonical_dataset") or source_row.get("source"),
        source_row.get("project_split"),
        source_row.get("original_split"),
        source_row.get("local_job_audio_path") or source_row.get("mac_job_audio_path"),
        source_row.get("audio_sha256"),
        source_row.get("duration_seconds"),
        raw_reference,
        raw_scribe,
        normalized_reference,
        normalized_scribe,
        wer,
        cer,
        int(exact),
        int(empty_reference),
        int(missing_scribe),
        int(empty_scribe),
        int(api_eligible),
        category,
        description,
        cause,
        (result or {}).get("audio_stem"),
        (result or {}).get("record_json"),
        json.dumps(source_row, ensure_ascii=False, sort_keys=True),
        now,
    )


def upsert_rows(connection: sqlite3.Connection, values: list[tuple[Any, ...]]) -> None:
    connection.executemany(
        """
        INSERT INTO scribe_curation (
            sample_id,
            job_order,
            job_id,
            source,
            source_config,
            canonical_dataset,
            project_split,
            original_split,
            audio_filepath,
            audio_sha256,
            duration_seconds,
            raw_reference_text,
            raw_scribe_text,
            normalized_reference,
            normalized_scribe,
            wer,
            cer,
            exact_match,
            empty_normalized_reference,
            missing_scribe,
            empty_scribe,
            api_eligible,
            difference_category,
            difference_description,
            likely_cause,
            scribe_audio_stem,
            scribe_record_json,
            source_record_json,
            updated_at
        )
        VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        ON CONFLICT(sample_id, job_order) DO UPDATE SET
            job_id = excluded.job_id,
            source = excluded.source,
            source_config = excluded.source_config,
            canonical_dataset = excluded.canonical_dataset,
            project_split = excluded.project_split,
            original_split = excluded.original_split,
            audio_filepath = excluded.audio_filepath,
            audio_sha256 = excluded.audio_sha256,
            duration_seconds = excluded.duration_seconds,
            raw_reference_text = excluded.raw_reference_text,
            raw_scribe_text = excluded.raw_scribe_text,
            normalized_reference = excluded.normalized_reference,
            normalized_scribe = excluded.normalized_scribe,
            wer = excluded.wer,
            cer = excluded.cer,
            exact_match = excluded.exact_match,
            empty_normalized_reference = excluded.empty_normalized_reference,
            missing_scribe = excluded.missing_scribe,
            empty_scribe = excluded.empty_scribe,
            api_eligible = excluded.api_eligible,
            difference_category = COALESCE(
                scribe_curation.difference_category,
                excluded.difference_category
            ),
            difference_description = COALESCE(
                scribe_curation.difference_description,
                excluded.difference_description
            ),
            likely_cause = COALESCE(scribe_curation.likely_cause, excluded.likely_cause),
            scribe_audio_stem = excluded.scribe_audio_stem,
            scribe_record_json = excluded.scribe_record_json,
            source_record_json = excluded.source_record_json,
            updated_at = excluded.updated_at
        """,
        values,
    )


def build(database: Path, rows_path: Path, *, replace: bool) -> BuildStats:
    connection = connect(database)
    now = utc_now()
    stats = {
        "rows_seen": 0,
        "rows_written": 0,
        "missing_scribe": 0,
        "empty_scribe": 0,
        "empty_normalized_reference": 0,
        "exact_match": 0,
        "pending_classification": 0,
    }
    try:
        create_table(connection, replace=replace)
        lookup = scribe_results(connection)
        batch: list[tuple[Any, ...]] = []
        with rows_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                source_row = json.loads(line)
                result = lookup.get(audio_stem(source_row))
                values = curation_row(source_row, result, now)
                stats["rows_seen"] += 1
                stats["missing_scribe"] += int(values[19])
                stats["empty_scribe"] += int(values[20])
                stats["empty_normalized_reference"] += int(values[18])
                stats["exact_match"] += int(values[17])
                stats["pending_classification"] += int(values[21])
                batch.append(values)
                if len(batch) >= BATCH_SIZE:
                    with connection:
                        upsert_rows(connection, batch)
                    stats["rows_written"] += len(batch)
                    batch.clear()
        if batch:
            with connection:
                upsert_rows(connection, batch)
            stats["rows_written"] += len(batch)
    finally:
        connection.close()
    return BuildStats(database=str(database), **stats)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    job_dir = args.job_dir.expanduser()
    rows_path = (args.rows or job_dir / "rows.jsonl").expanduser()
    database = (args.database or job_dir / "scribev2.full-20260523.sqlite").expanduser()
    stats = build(database=database, rows_path=rows_path, replace=args.replace)
    print(json.dumps(asdict(stats), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
