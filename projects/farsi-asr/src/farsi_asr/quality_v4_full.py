from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sqlite3
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from asr_benchmark_core.data import Example
from omni_curator.audit.benchmark import score_pair
from omni_curator.audit.quality import asr_edge_mismatch

if TYPE_CHECKING:
    from asr_benchmark_core.adapters import Adapter

SAMPLE_RATE = 16_000


def _audio_bytes(value: bytes | list[int]) -> bytes:
    if isinstance(value, bytes):
        return value
    return np.asarray(value, dtype=np.int8).tobytes()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _example(row_index: int, text: str, encoded: bytes) -> Example:
    audio, sampling_rate = sf.read(io.BytesIO(encoded), dtype="float32", always_2d=False)
    if audio.ndim == 2:  # noqa: PLR2004 - audio channel dimension
        audio = audio.mean(axis=1)
    return Example(
        row_index=row_index,
        audio=np.asarray(audio, dtype=np.float32),
        sampling_rate=int(sampling_rate),
        reference=text,
    )


def _source(path: Path) -> str:
    for part in path.parts:
        if part.startswith("corpus="):
            return part.removeprefix("corpus=")
    raise ValueError(f"corpus partition missing from {path}")


def _relative_hub_path(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=300)
    connection.execute("PRAGMA busy_timeout=300000")
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS run_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS quality_rows (
            hub_path TEXT NOT NULL,
            hub_row_index INTEGER NOT NULL,
            source TEXT NOT NULL,
            text TEXT NOT NULL,
            audio_size INTEGER NOT NULL,
            duration REAL NOT NULL,
            audio_sha256 TEXT NOT NULL,
            hypothesis TEXT NOT NULL,
            inference_seconds REAL NOT NULL,
            error TEXT,
            asr_edge_json TEXT NOT NULL,
            asr_agreement_json TEXT NOT NULL,
            PRIMARY KEY (hub_path, hub_row_index)
        );
        CREATE INDEX IF NOT EXISTS quality_rows_source ON quality_rows(source);
        CREATE TABLE IF NOT EXISTS ctc_alignments (
            hub_path TEXT NOT NULL,
            hub_row_index INTEGER NOT NULL,
            status TEXT NOT NULL,
            alignment_json TEXT NOT NULL,
            preflight_json TEXT,
            PRIMARY KEY (hub_path, hub_row_index),
            FOREIGN KEY (hub_path, hub_row_index)
                REFERENCES quality_rows(hub_path, hub_row_index)
        );
        CREATE INDEX IF NOT EXISTS ctc_alignments_status ON ctc_alignments(status);
        """
    )
    return connection


def _ensure_metadata(connection: sqlite3.Connection, metadata: dict[str, Any]) -> None:
    existing = dict(connection.execute("SELECT key, value FROM run_metadata"))
    encoded = {key: json.dumps(value, sort_keys=True) for key, value in metadata.items()}
    conflicts = {
        key: (existing[key], value)
        for key, value in encoded.items()
        if key in existing and existing[key] != value
    }
    if conflicts:
        raise RuntimeError(f"quality ledger metadata mismatch: {conflicts}")
    connection.executemany(
        "INSERT OR IGNORE INTO run_metadata(key, value) VALUES (?, ?)", encoded.items()
    )
    connection.commit()


def _score_batch(
    adapter: Adapter,
    examples: list[Example],
) -> tuple[list[str], list[str | None], float]:
    started = time.perf_counter()
    try:
        hypotheses = adapter.transcribe_batch(examples)
        errors: list[str | None] = [None] * len(examples)
    except Exception as exc:  # noqa: BLE001 - retain failures and keep the resumable job moving
        hypotheses = [""] * len(examples)
        errors = [f"{type(exc).__name__}: {exc}"] * len(examples)
    seconds = (time.perf_counter() - started) / len(examples)
    return hypotheses, errors, seconds


def _insert_batch(
    connection: sqlite3.Connection,
    *,
    hub_path: str,
    source: str,
    rows: list[tuple[int, str, int, bytes]],
    examples: list[Example],
    hypotheses: list[str],
    errors: list[str | None],
    inference_seconds: float,
) -> None:
    payloads = []
    for (row_index, text, audio_size, encoded), example, hypothesis, error in zip(
        rows, examples, hypotheses, errors, strict=True
    ):
        payloads.append(
            (
                hub_path,
                row_index,
                source,
                text,
                audio_size,
                example.audio_seconds,
                hashlib.sha256(encoded).hexdigest(),
                hypothesis,
                inference_seconds,
                error,
                json.dumps(asdict(asr_edge_mismatch(text, hypothesis)), ensure_ascii=False),
                json.dumps(score_pair(text, hypothesis), ensure_ascii=False),
            )
        )
    connection.executemany(
        """
        INSERT OR REPLACE INTO quality_rows(
            hub_path, hub_row_index, source, text, audio_size, duration, audio_sha256,
            hypothesis, inference_seconds, error, asr_edge_json, asr_agreement_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        payloads,
    )


def _score_shard(
    connection: sqlite3.Connection,
    adapter: Adapter,
    *,
    root: Path,
    shard: Path,
    batch_size: int,
    remaining_rows: int | None,
) -> int:
    hub_path = _relative_hub_path(root, shard)
    source = _source(shard)
    completed = {
        int(row[0])
        for row in connection.execute(
            "SELECT hub_row_index FROM quality_rows WHERE hub_path = ?", (hub_path,)
        )
    }
    parquet = pq.ParquetFile(shard)
    required = {"text", "audio_bytes", "audio_size"}
    if not required.issubset(parquet.schema_arrow.names):
        raise RuntimeError(f"unsupported V4 schema in {shard}: {parquet.schema_arrow.names}")
    processed = 0
    row_offset = 0
    pending_rows: list[tuple[int, str, int, bytes]] = []
    pending_examples: list[Example] = []
    for batch in parquet.iter_batches(
        batch_size=128, columns=["text", "audio_bytes", "audio_size"]
    ):
        values = zip(
            batch.column(0).to_pylist(),
            batch.column(1).to_pylist(),
            batch.column(2).to_pylist(),
            strict=True,
        )
        for batch_index, (text_raw, audio_raw, audio_size_raw) in enumerate(values):
            row_index = row_offset + batch_index
            if row_index in completed:
                continue
            if remaining_rows is not None and processed + len(pending_examples) >= remaining_rows:
                break
            text = str(text_raw)
            encoded = _audio_bytes(audio_raw)
            pending_rows.append((row_index, text, int(audio_size_raw), encoded))
            pending_examples.append(_example(row_index, text, encoded))
            if len(pending_examples) < batch_size:
                continue
            hypotheses, errors, seconds = _score_batch(adapter, pending_examples)
            _insert_batch(
                connection,
                hub_path=hub_path,
                source=source,
                rows=pending_rows,
                examples=pending_examples,
                hypotheses=hypotheses,
                errors=errors,
                inference_seconds=seconds,
            )
            processed += len(pending_examples)
            pending_rows = []
            pending_examples = []
        row_offset += len(batch)
        if remaining_rows is not None and processed + len(pending_examples) >= remaining_rows:
            break
    if pending_examples and (remaining_rows is None or processed < remaining_rows):
        hypotheses, errors, seconds = _score_batch(adapter, pending_examples)
        _insert_batch(
            connection,
            hub_path=hub_path,
            source=source,
            rows=pending_rows,
            examples=pending_examples,
            hypotheses=hypotheses,
            errors=errors,
            inference_seconds=seconds,
        )
        processed += len(pending_examples)
    connection.commit()
    return processed


def score_asr(args: argparse.Namespace, *, adapter: Adapter | None = None) -> int:
    from asr_benchmark_core.adapters import load_adapter

    shards = sorted(
        args.dataset_root.glob("version=0/corpus=*/split=train/language=fas_Arab/part-*.parquet")
    )
    if args.expected_shards and len(shards) != args.expected_shards:
        raise SystemExit(
            f"expected {args.expected_shards} V4 train shards under {args.dataset_root}; "
            f"found {len(shards)}"
        )
    if args.limit_shards:
        shards = shards[: args.limit_shards]
    if not shards:
        raise SystemExit(f"no V4 train shards found under {args.dataset_root}")
    connection = _connect(args.database)
    _ensure_metadata(
        connection,
        {
            "hub_repo": args.hub_repo,
            "hub_revision": args.hub_revision,
            "dataset_root": str(args.dataset_root.resolve()),
            "model_path": str(args.model.resolve()),
            "model_sha256": args.model_sha256,
            "adapter": args.adapter,
            "language": args.language,
            "batch_size": args.batch_size,
        },
    )
    recognizer = adapter or load_adapter(
        args.adapter, args.model, language=args.language, device=args.device
    )
    started = time.perf_counter()
    new_rows = 0
    for shard_index, shard in enumerate(shards, start=1):
        remaining = None if args.limit_rows == 0 else max(0, args.limit_rows - new_rows)
        if remaining == 0:
            break
        shard_rows = _score_shard(
            connection,
            recognizer,
            root=args.dataset_root,
            shard=shard,
            batch_size=args.batch_size,
            remaining_rows=remaining,
        )
        new_rows += shard_rows
        total_rows = int(connection.execute("SELECT COUNT(*) FROM quality_rows").fetchone()[0])
        errors = int(
            connection.execute(
                "SELECT COUNT(*) FROM quality_rows WHERE error IS NOT NULL"
            ).fetchone()[0]
        )
        print(
            f"shard={shard_index}/{len(shards)} new={shard_rows} total={total_rows} "
            f"errors={errors} elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
    connection.close()
    print(f"ASR quality ledger added {new_rows} rows -> {args.database}")
    return 0


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _pending_alignment_rows(
    connection: sqlite3.Connection, *, limit: int
) -> list[tuple[str, int, str, str, float]]:
    return cast(
        "list[tuple[str, int, str, str, float]]",
        connection.execute(
            """
            SELECT q.hub_path, q.hub_row_index, q.source, q.text, q.duration
            FROM quality_rows AS q
            LEFT JOIN ctc_alignments AS c
              ON c.hub_path = q.hub_path AND c.hub_row_index = q.hub_row_index
            WHERE c.hub_path IS NULL
            ORDER BY q.hub_path, q.hub_row_index
            LIMIT ?
            """,
            (limit,),
        ).fetchall(),
    )


def _materialize_alignment_chunk(
    *,
    dataset_root: Path,
    work_dir: Path,
    pending: list[tuple[str, int, str, str, float]],
) -> Path:
    grouped: dict[str, list[tuple[int, str, str, float]]] = defaultdict(list)
    for hub_path, row_index, source, text, duration in pending:
        grouped[hub_path].append((row_index, source, text, duration))
    audio_dir = work_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    for hub_path, requested in grouped.items():
        wanted = {
            row_index: (source, text, duration) for row_index, source, text, duration in requested
        }
        parquet = pq.ParquetFile(dataset_root / hub_path)
        row_offset = 0
        found: set[int] = set()
        for batch in parquet.iter_batches(batch_size=128, columns=["audio_bytes"]):
            for batch_index, audio_raw in enumerate(batch.column(0).to_pylist()):
                row_index = row_offset + batch_index
                if row_index not in wanted:
                    continue
                source, text, duration = wanted[row_index]
                identity = hashlib.sha256(f"{hub_path}:{row_index}".encode()).hexdigest()[:20]
                audio_path = (audio_dir / f"{source}-{identity}.flac").resolve()
                audio_path.write_bytes(_audio_bytes(audio_raw))
                manifest_rows.append(
                    {
                        "sample_id": f"{source}-{identity}",
                        "hub_path": hub_path,
                        "hub_row_index": row_index,
                        "source": source,
                        "audio_filepath": str(audio_path),
                        "text": text,
                        "duration": duration,
                    }
                )
                found.add(row_index)
            row_offset += len(batch)
            if len(found) == len(wanted):
                break
        missing = set(wanted) - found
        if missing:
            raise RuntimeError(f"missing V4 rows in {hub_path}: {sorted(missing)[:10]}")
    manifest_rows.sort(key=lambda row: (str(row["hub_path"]), int(row["hub_row_index"])))
    manifest = work_dir / "raw.jsonl"
    _write_jsonl(manifest, manifest_rows)
    if len(manifest_rows) != len(pending):
        raise RuntimeError(
            f"materialized {len(manifest_rows)} alignment rows; expected {len(pending)}"
        )
    return manifest


def _run_alignment_chunk(args: argparse.Namespace, *, work_dir: Path, manifest: Path) -> Path:
    from omni_curator.audit.quality_cli import (
        cmd_nfa_prepare,
        cmd_nfa_run,
        cmd_nfa_summarize,
    )

    prepared = work_dir / "nfa-input.jsonl"
    rejected = work_dir / "nfa-rejected.jsonl"
    cmd_nfa_prepare(
        argparse.Namespace(
            input=manifest,
            output=prepared,
            summary=work_dir / "nfa-prepare-summary.json",
            language="fas_Arab",
            reference_field="text",
            tokenizer_model=args.ctc_model,
            rejected_output=rejected,
        )
    )
    nfa_output = work_dir / "nfa-output"
    cmd_nfa_run(
        argparse.Namespace(
            input=prepared,
            output_dir=nfa_output,
            model=args.ctc_model,
            nemo_root=args.nemo_root,
            batch_size=args.nfa_batch_size,
            device=args.device,
            viterbi_device=args.viterbi_device,
            log=work_dir / "nfa.log",
        )
    )
    aligned_manifest = nfa_output / "nfa-input_with_output_file_paths.json"
    scored = work_dir / "scored.jsonl"
    cmd_nfa_summarize(
        argparse.Namespace(
            input=manifest,
            aligned_manifest=aligned_manifest,
            output=scored,
            summary=work_dir / "nfa-summary.json",
            audio_field="audio_filepath",
            reference_field="text",
            duration_field="duration",
            run_metadata=nfa_output / "omni-quality-nfa-run.json",
            rejected_input=rejected,
        )
    )
    return scored


def _store_alignment_chunk(
    connection: sqlite3.Connection, *, scored: Path, expected_rows: int
) -> int:
    rows = _read_jsonl(scored)
    if len(rows) != expected_rows:
        raise RuntimeError(f"alignment output has {len(rows)} rows; expected {expected_rows}")
    payloads = []
    for row in rows:
        quality = dict(row.get("quality") or {})
        alignment = dict(quality["ctc_alignment"])
        alignment.pop("provenance", None)
        preflight = quality.get("ctc_alignment_preflight")
        payloads.append(
            (
                str(row["hub_path"]),
                int(row["hub_row_index"]),
                str(alignment["status"]),
                json.dumps(alignment, ensure_ascii=False),
                json.dumps(preflight, ensure_ascii=False) if preflight else None,
            )
        )
    connection.executemany(
        """
        INSERT OR REPLACE INTO ctc_alignments(
            hub_path, hub_row_index, status, alignment_json, preflight_json
        ) VALUES (?, ?, ?, ?, ?)
        """,
        payloads,
    )
    connection.commit()
    return len(payloads)


def align_ctc(args: argparse.Namespace) -> int:
    connection = _connect(args.database)
    asr_rows = int(connection.execute("SELECT COUNT(*) FROM quality_rows").fetchone()[0])
    asr_shards = int(
        connection.execute("SELECT COUNT(DISTINCT hub_path) FROM quality_rows").fetchone()[0]
    )
    asr_errors = int(
        connection.execute("SELECT COUNT(*) FROM quality_rows WHERE error IS NOT NULL").fetchone()[
            0
        ]
    )
    if args.expected_rows and asr_rows != args.expected_rows:
        raise SystemExit(f"expected {args.expected_rows} ASR rows; found {asr_rows}")
    if args.expected_shards and asr_shards != args.expected_shards:
        raise SystemExit(f"expected {args.expected_shards} ASR shards; found {asr_shards}")
    if asr_errors > args.max_asr_errors:
        raise SystemExit(f"ASR ledger has {asr_errors} errors; maximum is {args.max_asr_errors}")
    actual_ctc_sha256 = _sha256_file(args.ctc_model)
    if actual_ctc_sha256 != args.ctc_model_sha256:
        raise SystemExit(
            f"CTC model SHA256 mismatch: expected {args.ctc_model_sha256}, "
            f"found {actual_ctc_sha256}"
        )
    _ensure_metadata(
        connection,
        {
            "ctc_model_path": str(args.ctc_model.resolve()),
            "ctc_model_sha256": args.ctc_model_sha256,
            "nemo_root": str(args.nemo_root.resolve()),
            "nemo_revision": args.nemo_revision,
            "ctc_nfa_batch_size": args.nfa_batch_size,
        },
    )
    completed = int(connection.execute("SELECT COUNT(*) FROM ctc_alignments").fetchone()[0])
    chunks = 0
    while True:
        pending = _pending_alignment_rows(connection, limit=args.chunk_rows)
        if not pending or (args.limit_chunks and chunks >= args.limit_chunks):
            break
        work_dir = args.work_dir / f"chunk-{completed:06d}"
        if work_dir.exists():
            shutil.rmtree(work_dir)
        manifest = _materialize_alignment_chunk(
            dataset_root=args.dataset_root,
            work_dir=work_dir,
            pending=pending,
        )
        scored = _run_alignment_chunk(args, work_dir=work_dir, manifest=manifest)
        added = _store_alignment_chunk(connection, scored=scored, expected_rows=len(pending))
        completed += added
        chunks += 1
        statuses = dict(
            connection.execute(
                "SELECT status, COUNT(*) FROM ctc_alignments GROUP BY status"
            ).fetchall()
        )
        print(
            f"ctc_chunk={chunks} added={added} total={completed}/{asr_rows} statuses={statuses}",
            flush=True,
        )
        shutil.rmtree(work_dir)
    connection.close()
    print(f"CTC quality ledger contains {completed} rows -> {args.database}")
    return 0


def add_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("score-asr", help="score every pinned V4 train row into SQLite")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--adapter", choices=("whisper", "qwen", "omni"), default="whisper")
    parser.add_argument("--language", default="Persian")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit-shards", type=int, default=0)
    parser.add_argument("--limit-rows", type=int, default=0)
    parser.add_argument("--expected-shards", type=int, default=0)
    parser.add_argument("--hub-repo", default="Peacockery/farsi-asr-corpus-v4")
    parser.add_argument("--hub-revision", default="564d41da9e5b935c0fe2bf2443e205ca7b747c96")
    parser.set_defaults(func=score_asr)

    align = subparsers.add_parser(
        "align-ctc", help="add resumable pinned CTC alignment signals to the V4 ledger"
    )
    align.add_argument("--dataset-root", type=Path, required=True)
    align.add_argument("--database", type=Path, required=True)
    align.add_argument("--work-dir", type=Path, required=True)
    align.add_argument("--ctc-model", type=Path, required=True)
    align.add_argument("--ctc-model-sha256", required=True)
    align.add_argument("--nemo-root", type=Path, required=True)
    align.add_argument("--nemo-revision", required=True)
    align.add_argument("--chunk-rows", type=int, default=5000)
    align.add_argument("--nfa-batch-size", type=int, default=4)
    align.add_argument("--device", default="cuda")
    align.add_argument("--viterbi-device", default="cpu")
    align.add_argument("--expected-rows", type=int, default=0)
    align.add_argument("--expected-shards", type=int, default=0)
    align.add_argument("--max-asr-errors", type=int, default=0)
    align.add_argument("--limit-chunks", type=int, default=0)
    align.set_defaults(func=align_ctc)
