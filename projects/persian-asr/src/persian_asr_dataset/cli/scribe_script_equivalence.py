from __future__ import annotations

import argparse
import sqlite3
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_WHERE = """
curation.normalized_scribe = ''
AND curation.raw_scribe_text <> ''
AND curation.normalized_reference <> ''
AND curation.missing_scribe = 0
AND curation.empty_scribe = 0
"""


@dataclass(frozen=True)
class PrepareStats:
    database: str
    method: str
    selected: int
    inserted_or_updated: int
    pending: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare script-equivalence review rows for Scribe curation."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--database", type=Path, required=True)
    prepare.add_argument(
        "--method",
        default="latin_scribe_review",
        help="Equivalence method name stored in SQLite.",
    )
    prepare.add_argument(
        "--where",
        default=DEFAULT_WHERE,
        help="SQL WHERE clause against scribe_curation aliased as curation.",
    )
    prepare.add_argument("--limit", type=int, default=0)
    prepare.add_argument("--replace-method", action="store_true")

    summary = subparsers.add_parser("summary")
    summary.add_argument("--database", type=Path, required=True)
    summary.add_argument("--method", default=None)
    return parser


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=60)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    return connection


def ensure_table(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS script_equivalence (
            sample_id TEXT NOT NULL,
            job_order INTEGER NOT NULL,
            method TEXT NOT NULL,
            reference_key TEXT NOT NULL,
            scribe_key TEXT NOT NULL,
            is_equivalent INTEGER,
            confidence REAL,
            notes TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (sample_id, job_order, method)
        );

        CREATE INDEX IF NOT EXISTS idx_script_equivalence_method
            ON script_equivalence(method, is_equivalent);
        """
    )


def latest_rerun_sql() -> str:
    return """
    SELECT result.sample_id, result.job_order, result.transcript
    FROM scribe_rerun_results result
    JOIN (
        SELECT sample_id, job_order, MAX(result_id) AS result_id
        FROM scribe_rerun_results
        GROUP BY sample_id, job_order
    ) latest
        ON latest.sample_id = result.sample_id
        AND latest.job_order = result.job_order
        AND latest.result_id = result.result_id
    """


def candidate_rows(
    connection: sqlite3.Connection,
    where_clause: str,
    limit: int,
) -> list[tuple[str, int, str, str]]:
    limit_sql = f"LIMIT {int(limit)}" if limit else ""
    return connection.execute(
        f"""
        WITH latest_rerun AS ({latest_rerun_sql()})
        SELECT
            curation.sample_id,
            curation.job_order,
            curation.normalized_reference,
            COALESCE(NULLIF(latest_rerun.transcript, ''), curation.raw_scribe_text)
        FROM scribe_curation curation
        LEFT JOIN latest_rerun
            ON latest_rerun.sample_id = curation.sample_id
            AND latest_rerun.job_order = curation.job_order
        WHERE {where_clause}
        ORDER BY curation.job_order
        {limit_sql}
        """
    ).fetchall()


def prepare(args: argparse.Namespace) -> PrepareStats:
    database = args.database.expanduser()
    connection = connect(database)
    now = utc_now()
    try:
        ensure_table(connection)
        rows = candidate_rows(connection, args.where, args.limit)
        with connection:
            if args.replace_method:
                connection.execute(
                    "DELETE FROM script_equivalence WHERE method = ?",
                    (args.method,),
                )
            connection.executemany(
                """
                INSERT INTO script_equivalence (
                    sample_id, job_order, method, reference_key, scribe_key,
                    is_equivalent, confidence, notes, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, NULL, NULL, NULL, ?, ?)
                ON CONFLICT(sample_id, job_order, method) DO UPDATE SET
                    reference_key = excluded.reference_key,
                    scribe_key = excluded.scribe_key,
                    updated_at = excluded.updated_at
                """,
                [
                    (sample_id, job_order, args.method, reference, scribe, now, now)
                    for sample_id, job_order, reference, scribe in rows
                ],
            )
        pending = int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM script_equivalence
                WHERE method = ? AND is_equivalent IS NULL
                """,
                (args.method,),
            ).fetchone()[0]
        )
        return PrepareStats(
            database=str(database),
            method=args.method,
            selected=len(rows),
            inserted_or_updated=len(rows),
            pending=pending,
        )
    finally:
        connection.close()


def summary(args: argparse.Namespace) -> list[dict[str, object]]:
    connection = connect(args.database.expanduser())
    try:
        ensure_table(connection)
        if args.method:
            rows = connection.execute(
                """
                SELECT method, is_equivalent, COUNT(*)
                FROM script_equivalence
                WHERE method = ?
                GROUP BY method, is_equivalent
                ORDER BY method, is_equivalent
                """,
                (args.method,),
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT method, is_equivalent, COUNT(*)
                FROM script_equivalence
                GROUP BY method, is_equivalent
                ORDER BY method, is_equivalent
                """
            ).fetchall()
        return [
            {
                "method": method,
                "is_equivalent": is_equivalent,
                "rows": count,
            }
            for method, is_equivalent, count in rows
        ]
    finally:
        connection.close()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        print(asdict(prepare(args)))
    else:
        for row in summary(args):
            print(row)
    return 0
