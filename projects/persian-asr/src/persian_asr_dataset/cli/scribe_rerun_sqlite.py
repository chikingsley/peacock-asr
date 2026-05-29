from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jiwer
from superwhisper_api.audio.models import audio_model
from superwhisper_api.audio.transcribe import TranscriptResult, create_process_fn

from persian_asr_dataset.vendor.nvidia_stt_fa_fastconformer_hybrid_large import maybe_normalize

DEFAULT_EMPTY_NORMALIZED_SCRIBE_WHERE = """
normalized_scribe = ''
AND raw_scribe_text <> ''
AND empty_normalized_reference = 0
AND missing_scribe = 0
AND empty_scribe = 0
"""


@dataclass(frozen=True)
class BatchStats:
    database: str
    batch_id: int
    name: str
    selected_items: int


@dataclass(frozen=True)
class RunStats:
    database: str
    batch_id: int
    ok: int
    failed: int
    submitted: int
    pending: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage Scribe reruns from SQLite curation rows.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create-batch")
    create.add_argument("--database", type=Path, required=True)
    create.add_argument("--name", required=True)
    create.add_argument("--reason", required=True)
    create.add_argument(
        "--where",
        default=DEFAULT_EMPTY_NORMALIZED_SCRIBE_WHERE,
        help="SQL WHERE clause against scribe_curation.",
    )
    create.add_argument("--limit", type=int, default=0)
    create.add_argument("--model", default="scribe-v2")
    create.add_argument("--language", default="fas")
    create.add_argument("--replace-name", action="store_true")

    run = subparsers.add_parser("run-batch")
    run.add_argument("--database", type=Path, required=True)
    run.add_argument("--batch-id", type=int, required=True)
    run.add_argument("--max-workers", type=int, default=32)
    run.add_argument("--limit", type=int, default=0)
    run.add_argument("--key", default=None)

    summary = subparsers.add_parser("summary")
    summary.add_argument("--database", type=Path, required=True)
    summary.add_argument("--batch-id", type=int, default=0)
    return parser


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=60)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def ensure_tables(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS scribe_rerun_batches (
            batch_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            reason TEXT NOT NULL,
            selection_sql TEXT NOT NULL,
            model TEXT NOT NULL,
            language TEXT,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS scribe_rerun_items (
            batch_id INTEGER NOT NULL,
            sample_id TEXT NOT NULL,
            job_order INTEGER NOT NULL,
            audio_filepath TEXT NOT NULL,
            original_scribe_text TEXT,
            status TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (batch_id, sample_id, job_order),
            FOREIGN KEY (batch_id) REFERENCES scribe_rerun_batches(batch_id)
        );

        CREATE TABLE IF NOT EXISTS scribe_rerun_results (
            result_id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id INTEGER NOT NULL,
            sample_id TEXT NOT NULL,
            job_order INTEGER NOT NULL,
            provider TEXT,
            model_key TEXT,
            model_id TEXT,
            transcript TEXT NOT NULL,
            normalized_transcript TEXT,
            wer REAL,
            cer REAL,
            recording_id TEXT,
            duration REAL,
            processing_time INTEGER,
            raw_response_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (batch_id, sample_id, job_order)
                REFERENCES scribe_rerun_items(batch_id, sample_id, job_order)
        );

        CREATE INDEX IF NOT EXISTS idx_scribe_rerun_items_status
            ON scribe_rerun_items(batch_id, status);
        CREATE INDEX IF NOT EXISTS idx_scribe_rerun_results_row
            ON scribe_rerun_results(batch_id, sample_id, job_order);
        """
    )


def create_batch(args: argparse.Namespace) -> BatchStats:
    database = args.database.expanduser()
    connection = connect(database)
    now = utc_now()
    try:
        ensure_tables(connection)
        if args.replace_name:
            old = connection.execute(
                "SELECT batch_id FROM scribe_rerun_batches WHERE name = ?",
                (args.name,),
            ).fetchone()
            if old:
                old_batch_id = int(old[0])
                with connection:
                    connection.execute(
                        "DELETE FROM scribe_rerun_results WHERE batch_id = ?",
                        (old_batch_id,),
                    )
                    connection.execute(
                        "DELETE FROM scribe_rerun_items WHERE batch_id = ?",
                        (old_batch_id,),
                    )
                    connection.execute(
                        "DELETE FROM scribe_rerun_batches WHERE batch_id = ?",
                        (old_batch_id,),
                    )
        with connection:
            cursor = connection.execute(
                """
                INSERT INTO scribe_rerun_batches (
                    name, reason, selection_sql, model, language, status, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, 'created', ?, ?)
                """,
                (args.name, args.reason, args.where, args.model, args.language, now, now),
            )
            batch_id = int(cursor.lastrowid)
            limit_sql = f"LIMIT {int(args.limit)}" if args.limit else ""
            rows = connection.execute(
                f"""
                SELECT sample_id, job_order, audio_filepath, raw_scribe_text
                FROM scribe_curation
                WHERE {args.where}
                ORDER BY job_order
                {limit_sql}
                """
            ).fetchall()
            connection.executemany(
                """
                INSERT INTO scribe_rerun_items (
                    batch_id, sample_id, job_order, audio_filepath, original_scribe_text,
                    status, attempts, error, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, 'pending', 0, NULL, ?, ?)
                """,
                [
                    (batch_id, sample_id, job_order, audio_path, raw_scribe, now, now)
                    for sample_id, job_order, audio_path, raw_scribe in rows
                ],
            )
        return BatchStats(
            database=str(database),
            batch_id=batch_id,
            name=args.name,
            selected_items=len(rows),
        )
    finally:
        connection.close()


def pending_items(
    connection: sqlite3.Connection,
    batch_id: int,
    limit: int,
) -> list[dict[str, Any]]:
    limit_sql = f"LIMIT {int(limit)}" if limit else ""
    rows = connection.execute(
        f"""
        SELECT item.sample_id, item.job_order, item.audio_filepath, batch.model, batch.language
        FROM scribe_rerun_items item
        JOIN scribe_rerun_batches batch ON batch.batch_id = item.batch_id
        WHERE item.batch_id = ? AND item.status IN ('pending', 'failed')
        ORDER BY item.job_order
        {limit_sql}
        """,
        (batch_id,),
    ).fetchall()
    return [
        {
            "sample_id": sample_id,
            "job_order": job_order,
            "audio_filepath": audio_filepath,
            "model": model,
            "language": language,
        }
        for sample_id, job_order, audio_filepath, model, language in rows
    ]


def score_result(
    connection: sqlite3.Connection,
    sample_id: str,
    job_order: int,
    transcript: str,
) -> tuple[str, float | None, float | None]:
    row = connection.execute(
        """
        SELECT normalized_reference
        FROM scribe_curation
        WHERE sample_id = ? AND job_order = ?
        """,
        (sample_id, job_order),
    ).fetchone()
    normalized_reference = str(row[0] or "") if row else ""
    normalized_transcript = maybe_normalize(transcript) or ""
    if not normalized_reference:
        return normalized_transcript, None, None
    wer_value = jiwer.wer(normalized_reference, normalized_transcript)
    cer_value = jiwer.cer(normalized_reference, normalized_transcript)
    if isinstance(cer_value, dict):
        raise TypeError("jiwer.cer returned detailed measurements instead of a scalar")
    return normalized_transcript, float(wer_value), float(cer_value)


def record_success(
    connection: sqlite3.Connection,
    batch_id: int,
    item: dict[str, Any],
    result: TranscriptResult,
) -> None:
    now = utc_now()
    payload = result.as_dict()
    transcript = str(payload.get("transcript") or "")
    normalized, wer, cer = score_result(
        connection,
        str(item["sample_id"]),
        int(item["job_order"]),
        transcript,
    )
    with connection:
        connection.execute(
            """
            INSERT INTO scribe_rerun_results (
                batch_id, sample_id, job_order, provider, model_key, model_id, transcript,
                normalized_transcript, wer, cer, recording_id, duration, processing_time,
                raw_response_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                batch_id,
                item["sample_id"],
                item["job_order"],
                payload.get("provider"),
                payload.get("model_key"),
                payload.get("model_id"),
                transcript,
                normalized,
                wer,
                cer,
                payload.get("recording_id"),
                payload.get("duration"),
                payload.get("processing_time"),
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
                now,
            ),
        )
        connection.execute(
            """
            UPDATE scribe_rerun_items
            SET status = 'done', attempts = attempts + 1, error = NULL, updated_at = ?
            WHERE batch_id = ? AND sample_id = ? AND job_order = ?
            """,
            (now, batch_id, item["sample_id"], item["job_order"]),
        )


def record_failure(
    connection: sqlite3.Connection,
    batch_id: int,
    item: dict[str, Any],
    result: TranscriptResult,
) -> None:
    now = utc_now()
    payload = result.as_dict()
    error = str(payload.get("error") or "unknown error")
    with connection:
        connection.execute(
            """
            UPDATE scribe_rerun_items
            SET status = 'failed', attempts = attempts + 1, error = ?, updated_at = ?
            WHERE batch_id = ? AND sample_id = ? AND job_order = ?
            """,
            (error, now, batch_id, item["sample_id"], item["job_order"]),
        )


def submit_next(
    pool: ThreadPoolExecutor,
    process_fn,
    items: list[dict[str, Any]],
    futures: dict[Future[TranscriptResult], dict[str, Any]],
    index: list[int],
) -> bool:
    if index[0] >= len(items):
        return False
    item = items[index[0]]
    index[0] += 1
    futures[pool.submit(process_fn, Path(str(item["audio_filepath"])))] = item
    return True


def run_batch(args: argparse.Namespace) -> RunStats:
    database = args.database.expanduser()
    connection = connect(database)
    ok = 0
    failed = 0
    submitted = 0
    try:
        ensure_tables(connection)
        items = pending_items(connection, args.batch_id, args.limit)
        if not items:
            return RunStats(str(database), args.batch_id, ok, failed, submitted, pending=0)
        batch = connection.execute(
            "SELECT model, language FROM scribe_rerun_batches WHERE batch_id = ?",
            (args.batch_id,),
        ).fetchone()
        if not batch:
            raise SystemExit(f"unknown batch id: {args.batch_id}")
        model, language = str(batch[0]), batch[1]
        spec = audio_model(model)
        process_fn = create_process_fn(spec, args.key, language=language)
        max_in_flight = max(args.max_workers * 4, args.max_workers)
        index = [0]
        futures: dict[Future[TranscriptResult], dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            while (
                len(futures) < max_in_flight
                and submit_next(pool, process_fn, items, futures, index)
            ):
                submitted += 1
            while futures:
                done, _ = wait(set(futures), return_when=FIRST_COMPLETED)
                for future in done:
                    item = futures.pop(future)
                    result = future.result()
                    if getattr(result, "error", ""):
                        record_failure(connection, args.batch_id, item, result)
                        failed += 1
                        print(
                            f"FAIL {item['job_order']}: {getattr(result, 'error', '')}",
                            file=sys.stderr,
                        )
                    else:
                        record_success(connection, args.batch_id, item, result)
                        ok += 1
                        text = getattr(result, "transcript", "")[:80].replace("\n", " ")
                        print(f"OK {item['job_order']}: {text}", file=sys.stderr)
                    while (
                        len(futures) < max_in_flight
                        and submit_next(pool, process_fn, items, futures, index)
                    ):
                        submitted += 1
        pending = int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM scribe_rerun_items
                WHERE batch_id = ? AND status IN ('pending', 'failed')
                """,
                (args.batch_id,),
            ).fetchone()[0]
        )
        return RunStats(str(database), args.batch_id, ok, failed, submitted, pending)
    finally:
        connection.close()


def summary(args: argparse.Namespace) -> int:
    connection = connect(args.database.expanduser())
    try:
        ensure_tables(connection)
        if args.batch_id:
            rows = connection.execute(
                """
                SELECT batch.batch_id, batch.name, batch.reason, batch.model, batch.language,
                       item.status, COUNT(*)
                FROM scribe_rerun_batches batch
                LEFT JOIN scribe_rerun_items item ON item.batch_id = batch.batch_id
                WHERE batch.batch_id = ?
                GROUP BY batch.batch_id, item.status
                ORDER BY batch.batch_id, item.status
                """,
                (args.batch_id,),
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT batch.batch_id, batch.name, batch.reason, batch.model, batch.language,
                       item.status, COUNT(*)
                FROM scribe_rerun_batches batch
                LEFT JOIN scribe_rerun_items item ON item.batch_id = batch.batch_id
                GROUP BY batch.batch_id, item.status
                ORDER BY batch.batch_id, item.status
                """
            ).fetchall()
        for row in rows:
            print("\t".join("" if value is None else str(value) for value in row))
    finally:
        connection.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "create-batch":
        stats = create_batch(args)
        print(json.dumps(asdict(stats), ensure_ascii=False, indent=2))
        return 0
    if args.command == "run-batch":
        stats = run_batch(args)
        print(json.dumps(asdict(stats), ensure_ascii=False, indent=2))
        return 0
    if args.command == "summary":
        return summary(args)
    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
