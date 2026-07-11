from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from asr_benchmark_core.data import Example
from omni_curator.audit.benchmark import score_pair
from omni_curator.audit.quality import asr_edge_mismatch

if TYPE_CHECKING:
    import argparse

    from asr_benchmark_core.adapters import Adapter

SAMPLE_RATE = 16_000


def _audio_bytes(value: bytes | list[int]) -> bytes:
    if isinstance(value, bytes):
        return value
    return np.asarray(value, dtype=np.int8).tobytes()


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
    connection = sqlite3.connect(path)
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
