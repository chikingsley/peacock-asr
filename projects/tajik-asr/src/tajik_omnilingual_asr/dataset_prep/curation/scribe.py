from __future__ import annotations

import argparse
import json
import sqlite3
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from superwhisper_api.audio.models import audio_model
from superwhisper_api.audio.transcribe import create_process_fn

from tajik_omnilingual_asr.dataset_prep.text_normalization import normalize_text

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATASET_DIR = (
    ROOT / "src/tajik_omnilingual_asr/dataset_prep/artifacts/tajik_asr_combined_v0"
)
DEFAULT_DB = DEFAULT_DATASET_DIR / "tajik_asr_combined.sqlite"


@dataclass(frozen=True)
class PendingUtterance:
    id: str
    job_order: int
    audio_path: Path
    audio_filename: str
    split: str
    source: str
    source_split: str
    raw_reference_text: str
    reference_text: str
    source_metadata_json: str


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("pragma journal_mode=wal")
    conn.execute("pragma foreign_keys=on")
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        create table if not exists scribe_runs (
            id text primary key,
            provider text not null,
            model_key text not null,
            model_id text not null,
            language text,
            status text not null,
            created_at text not null,
            completed_at text,
            total_rows integer not null default 0,
            ok_rows integer not null default 0,
            failed_rows integer not null default 0,
            notes text not null default ''
        );

        create table if not exists scribe_transcripts (
            run_id text not null references scribe_runs(id) on delete cascade,
            utterance_id text not null references utterances(id) on delete cascade,
            audio_filename text not null,
            provider text not null,
            model_key text not null,
            model_id text not null,
            transcript text not null,
            normalized_transcript text not null,
            reference_text text not null,
            wer real,
            cer real,
            exact_match integer not null,
            raw_response_json text not null,
            error text not null default '',
            created_at text not null,
            primary key (run_id, utterance_id)
        );

        create index if not exists scribe_transcripts_utterance_idx
            on scribe_transcripts(utterance_id);
        create index if not exists scribe_transcripts_run_error_idx
            on scribe_transcripts(run_id, error);

        create table if not exists scribe_curation (
            sample_id text not null,
            job_id text,
            source text,
            project_split text,
            original_split text,
            audio_filepath text,
            ref_text text,
            pred_text text,
            normalized_ref text,
            normalized_pred_text text,
            wer real,
            cer real,
            primary key (sample_id, job_id)
        );
        create index if not exists idx_scribe_curation_source
            on scribe_curation(source, project_split);
        """
    )
    conn.commit()


def split_words(text: str) -> list[str]:
    return [part for part in text.split() if part]


def edit_distance(left: list[str] | str, right: list[str] | str) -> int:
    previous = list(range(len(right) + 1))
    for i, left_item in enumerate(left, start=1):
        current = [i]
        for j, right_item in enumerate(right, start=1):
            cost = 0 if left_item == right_item else 1
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + cost,
                )
            )
        previous = current
    return previous[-1]


def error_rate(reference: list[str] | str, hypothesis: list[str] | str) -> float:
    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0
    return edit_distance(reference, hypothesis) / len(reference)


def compute_wer(reference: str, hypothesis: str) -> float:
    return error_rate(split_words(reference), split_words(hypothesis))


def compute_cer(reference: str, hypothesis: str) -> float:
    return error_rate(reference, hypothesis)


def create_run(
    conn: sqlite3.Connection,
    *,
    model_key: str,
    language: str | None,
    notes: str,
) -> str:
    spec = audio_model(model_key)
    run_id = f"scribe-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    conn.execute(
        """
        insert into scribe_runs (
            id, provider, model_key, model_id, language, status, created_at, notes
        ) values (?, ?, ?, ?, ?, 'running', ?, ?)
        """,
        (run_id, spec.provider, spec.key, spec.model_id, language, now_iso(), notes),
    )
    conn.commit()
    return run_id


def pending_rows(
    conn: sqlite3.Connection,
    dataset_dir: Path,
    *,
    run_id: str | None,
    split: str | None,
    source: str | None,
    limit: int | None,
) -> list[PendingUtterance]:
    filters = []
    params: list[Any] = []
    if split:
        filters.append("u.split = ?")
        params.append(split)
    if source:
        filters.append("u.source = ?")
        params.append(source)
    if run_id:
        filters.append(
            """
            not exists (
                select 1
                from scribe_transcripts st
                where st.run_id = ? and st.utterance_id = u.id
            )
            """
        )
        params.append(run_id)
    where = f"where {' and '.join(filters)}" if filters else ""
    limit_sql = "limit ?" if limit is not None else ""
    if limit is not None:
        params.append(limit)
    rows = conn.execute(
        f"""
        with ordered as (
            select
                u.*,
                row_number() over (order by u.split, u.source, u.source_id, u.id) as job_order
            from utterances u
        )
        select
            u.id,
            u.job_order,
            u.audio_filename,
            u.split,
            u.source,
            u.source_split,
            u.raw_transcription,
            u.normalized_text,
            u.source_metadata_json
        from ordered u
        {where}
        order by u.job_order
        {limit_sql}
        """,
        params,
    ).fetchall()
    return [
        PendingUtterance(
            id=str(row["id"]),
            job_order=int(row["job_order"]),
            audio_path=dataset_dir / str(row["audio_filename"]),
            audio_filename=str(row["audio_filename"]),
            split=str(row["split"]),
            source=str(row["source"]),
            source_split=str(row["source_split"]),
            raw_reference_text=str(row["raw_transcription"]),
            reference_text=str(row["normalized_text"]),
            source_metadata_json=str(row["source_metadata_json"]),
        )
        for row in rows
    ]


def insert_result(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    utterance: PendingUtterance,
    provider: str,
    model_key: str,
    model_id: str,
    transcript: str,
    raw_response: dict[str, Any],
    error: str,
) -> None:
    normalized = normalize_text(transcript)
    exact = int(normalized == utterance.reference_text)
    wer = None if error else compute_wer(utterance.reference_text, normalized)
    cer = None if error else compute_cer(utterance.reference_text, normalized)
    raw_json = json.dumps(raw_response, ensure_ascii=False, sort_keys=True)
    conn.execute(
        """
        insert or replace into scribe_transcripts (
            run_id, utterance_id, audio_filename, provider, model_key, model_id,
            transcript, normalized_transcript, reference_text, wer, cer, exact_match,
            raw_response_json, error, created_at
        ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            utterance.id,
            str(utterance.audio_path.relative_to(DEFAULT_DATASET_DIR)),
            provider,
            model_key,
            model_id,
            transcript,
            normalized,
            utterance.reference_text,
            wer,
            cer,
            exact,
            raw_json,
            error,
            now_iso(),
        ),
    )
    conn.execute(
        """
        insert or replace into scribe_curation (
            sample_id, job_id, source, project_split, original_split, audio_filepath,
            ref_text, pred_text, normalized_ref, normalized_pred_text, wer, cer
        ) values (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
        (
            utterance.id,
            run_id,
            utterance.source,
            utterance.split,
            utterance.source_split,
            utterance.audio_filename,
            utterance.raw_reference_text,
            transcript,
            utterance.reference_text,
            normalized,
            wer,
            cer,
        ),
    )
    conn.commit()


def update_run_counts(conn: sqlite3.Connection, run_id: str, status: str) -> None:
    counts = conn.execute(
        """
        select
            count(*) total,
            sum(case when error = '' then 1 else 0 end) ok,
            sum(case when error != '' then 1 else 0 end) failed
        from scribe_transcripts
        where run_id = ?
        """,
        (run_id,),
    ).fetchone()
    conn.execute(
        """
        update scribe_runs
        set status = ?,
            completed_at = case when ? in ('completed', 'failed') then ? else completed_at end,
            total_rows = ?,
            ok_rows = ?,
            failed_rows = ?
        where id = ?
        """,
        (
            status,
            status,
            now_iso(),
            int(counts["total"] or 0),
            int(counts["ok"] or 0),
            int(counts["failed"] or 0),
            run_id,
        ),
    )
    conn.commit()


def cmd_init(args: argparse.Namespace) -> int:
    with connect(args.db) as conn:
        ensure_schema(conn)
    print(f"initialized\t{args.db}")
    return 0


def cmd_pending(args: argparse.Namespace) -> int:
    with connect(args.db) as conn:
        ensure_schema(conn)
        rows = pending_rows(
            conn,
            args.dataset_dir,
            run_id=args.run_id,
            split=args.split,
            source=args.source,
            limit=args.limit,
        )
    print(f"pending\t{len(rows)}")
    return 0


def process_one(process: Any, utterance: PendingUtterance) -> tuple[PendingUtterance, Any]:
    return utterance, process(utterance.audio_path)


def cmd_run(args: argparse.Namespace) -> int:
    spec = audio_model(args.model)
    process = create_process_fn(spec, args.key, language=args.language)
    with connect(args.db) as conn:
        ensure_schema(conn)
        run_id = args.run_id or create_run(
            conn,
            model_key=args.model,
            language=args.language,
            notes=args.notes,
        )
        rows = pending_rows(
            conn,
            args.dataset_dir,
            run_id=run_id,
            split=args.split,
            source=args.source,
            limit=args.limit,
        )
        print(f"run_id\t{run_id}")
        print(f"pending\t{len(rows)}")
        if args.dry_run or not rows:
            update_run_counts(conn, run_id, "dry_run" if args.dry_run else "completed")
            return 0

        ok = 0
        failed = 0
        max_in_flight = max(args.max_workers * 4, args.max_workers)
        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            pending = iter(rows)
            futures = set()
            while len(futures) < max_in_flight:
                try:
                    utterance = next(pending)
                except StopIteration:
                    break
                futures.add(pool.submit(process_one, process, utterance))

            while futures:
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    utterance, result = future.result()
                    payload = result.as_dict()
                    error = str(payload.get("error") or "")
                    transcript = str(payload.get("transcript") or "")
                    raw_response = payload.get("raw_response")
                    insert_result(
                        conn,
                        run_id=run_id,
                        utterance=utterance,
                        provider=spec.provider,
                        model_key=spec.key,
                        model_id=spec.model_id,
                        transcript=transcript,
                        raw_response=raw_response if isinstance(raw_response, dict) else payload,
                        error=error,
                    )
                    if error:
                        failed += 1
                    else:
                        ok += 1
                    print(f"done\tok={ok}\tfailed={failed}\tid={utterance.id}")
                    try:
                        next_utterance = next(pending)
                    except StopIteration:
                        continue
                    futures.add(pool.submit(process_one, process, next_utterance))

        update_run_counts(conn, run_id, "completed")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Curate Tajik ASR data with Scribe into SQLite.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init-db", help="Create curation tables.")
    init.set_defaults(func=cmd_init)

    pending = subparsers.add_parser("pending", help="Count pending rows.")
    pending.add_argument("--run-id")
    pending.add_argument("--split")
    pending.add_argument("--source")
    pending.add_argument("--limit", type=int)
    pending.set_defaults(func=cmd_pending)

    run = subparsers.add_parser("run-scribe", help="Run Scribe and write results into SQLite.")
    run.add_argument("--run-id")
    run.add_argument("--model", default="scribe-v2")
    run.add_argument("--language", default="tgk")
    run.add_argument("--key")
    run.add_argument("--split")
    run.add_argument("--source")
    run.add_argument("--limit", type=int)
    run.add_argument("--max-workers", type=int, default=8)
    run.add_argument("--notes", default="")
    run.add_argument("--dry-run", action="store_true")
    run.set_defaults(func=cmd_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
