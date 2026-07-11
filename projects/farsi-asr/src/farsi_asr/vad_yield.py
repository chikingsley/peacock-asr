from __future__ import annotations

import argparse
import json
import math
import sqlite3
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import soundfile as sf
from asr_benchmark_core.adapters import load_adapter
from asr_benchmark_core.data import Example

if TYPE_CHECKING:
    from collections.abc import Sequence

ARABIC_RANGES = (
    (0x0600, 0x06FF),
    (0x0750, 0x077F),
    (0x08A0, 0x08FF),
    (0xFB50, 0xFDFF),
    (0xFE70, 0xFEFF),
)
AUDIO_CHANNEL_DIMENSIONS = 2


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Run the bounded local-ASR yield gate over retained VAD pilot clips."
    )
    sub = result.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="freeze a source-balanced clip selection")
    prepare.add_argument("--clips", type=Path, required=True)
    prepare.add_argument("--database", type=Path, required=True)
    prepare.add_argument("--engine", action="append", required=True)
    prepare.add_argument("--clips-per-source", type=int, default=5)
    prepare.set_defaults(func=prepare_selection)

    run = sub.add_parser("run", help="transcribe the frozen selection with one local model")
    run.add_argument("--database", type=Path, required=True)
    run.add_argument("--model-name", required=True)
    run.add_argument("--adapter", choices=("omni", "whisper", "qwen"), required=True)
    run.add_argument("--model", type=Path, required=True)
    run.add_argument("--language", default="Persian")
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--batch-size", type=int, default=1)
    run.set_defaults(func=run_model)

    summarize = sub.add_parser("summarize", help="write and print aggregate yield metrics")
    summarize.add_argument("--database", type=Path, required=True)
    summarize.add_argument("--output", type=Path, required=True)
    summarize.set_defaults(func=summarize_results)
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _quantile_sample(rows: Sequence[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (float(row["duration"]), str(row["clip_id"])))
    if len(ordered) < count:
        raise ValueError(f"need {count} clips but found {len(ordered)}")
    indices = [
        min(len(ordered) - 1, math.floor((index + 0.5) * len(ordered) / count))
        for index in range(count)
    ]
    return [ordered[index] for index in indices]


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS selection (
            clip_id TEXT PRIMARY KEY,
            engine TEXT NOT NULL,
            tier TEXT NOT NULL,
            source_id TEXT NOT NULL,
            path TEXT NOT NULL,
            duration REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS models (
            model_name TEXT PRIMARY KEY,
            adapter TEXT NOT NULL,
            model_path TEXT NOT NULL,
            language TEXT NOT NULL,
            device TEXT NOT NULL,
            batch_size INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS predictions (
            model_name TEXT NOT NULL REFERENCES models(model_name),
            clip_id TEXT NOT NULL REFERENCES selection(clip_id),
            transcript TEXT NOT NULL,
            inference_seconds REAL NOT NULL,
            error TEXT,
            PRIMARY KEY (model_name, clip_id)
        );
        """
    )
    return connection


def prepare_selection(args: argparse.Namespace) -> int:
    if args.clips_per_source <= 0:
        raise SystemExit("--clips-per-source must be positive")
    rows = _read_jsonl(args.clips)
    engines = tuple(dict.fromkeys(args.engine))
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        engine = str(row["engine"])
        if engine in engines:
            grouped[(engine, str(row["source_id"]))].append(row)
    selected: list[dict[str, Any]] = []
    sources_by_engine: dict[str, set[str]] = defaultdict(set)
    for (engine, source_id), group in sorted(grouped.items()):
        selected.extend(_quantile_sample(group, args.clips_per_source))
        sources_by_engine[engine].add(source_id)
    source_counts = {engine: len(sources_by_engine[engine]) for engine in engines}
    if len(set(source_counts.values())) != 1 or not next(iter(source_counts.values()), 0):
        raise SystemExit(f"engines do not cover the same nonzero source count: {source_counts}")

    connection = _connect(args.database)
    existing = connection.execute("SELECT COUNT(*) FROM selection").fetchone()[0]
    if existing:
        connection.close()
        raise SystemExit(f"selection already frozen with {existing} rows: {args.database}")
    connection.executemany(
        "INSERT INTO selection VALUES (?, ?, ?, ?, ?, ?)",
        [
            (
                str(row["clip_id"]),
                str(row["engine"]),
                str(row["tier"]),
                str(row["source_id"]),
                str(Path(str(row["path"])).resolve()),
                float(row["duration"]),
            )
            for row in selected
        ],
    )
    connection.commit()
    connection.close()
    print(
        json.dumps(
            {
                "database": str(args.database.resolve()),
                "engines": list(engines),
                "sources_per_engine": source_counts,
                "clips_per_source": args.clips_per_source,
                "selected_clips": len(selected),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _load_example(row_index: int, path: Path) -> Example:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == AUDIO_CHANNEL_DIMENSIONS:
        audio = audio.mean(axis=1)
    return Example(row_index, np.asarray(audio, dtype=np.float32), int(sample_rate), "")


def run_model(args: argparse.Namespace) -> int:
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if not args.model.exists():
        raise SystemExit(f"model path does not exist: {args.model}")
    connection = _connect(args.database)
    model_row = (
        args.model_name,
        args.adapter,
        str(args.model.resolve()),
        args.language,
        args.device,
        args.batch_size,
    )
    connection.execute("INSERT OR IGNORE INTO models VALUES (?, ?, ?, ?, ?, ?)", model_row)
    stored = connection.execute(
        "SELECT adapter, model_path, language, device, batch_size FROM models WHERE model_name = ?",
        (args.model_name,),
    ).fetchone()
    if stored != model_row[1:]:
        connection.close()
        raise SystemExit(f"model name already has different configuration: {args.model_name}")
    completed = {
        row[0]
        for row in connection.execute(
            "SELECT clip_id FROM predictions WHERE model_name = ?", (args.model_name,)
        )
    }
    rows = connection.execute(
        "SELECT clip_id, path FROM selection ORDER BY engine, tier, source_id, duration, clip_id"
    ).fetchall()
    pending = [(clip_id, path) for clip_id, path in rows if clip_id not in completed]
    print(f"model={args.model_name} completed={len(completed)} pending={len(pending)}", flush=True)
    if not pending:
        connection.close()
        return 0

    adapter = load_adapter(args.adapter, args.model, language=args.language, device=args.device)
    processed = 0
    for start in range(0, len(pending), args.batch_size):
        batch_rows = pending[start : start + args.batch_size]
        examples = [
            _load_example(start + index, Path(path)) for index, (_, path) in enumerate(batch_rows)
        ]
        began = time.perf_counter()
        error: str | None = None
        try:
            transcripts = adapter.transcribe_batch(examples)
        except Exception as exc:  # noqa: BLE001 - persist model failures for the gate
            transcripts = [""] * len(examples)
            error = f"{type(exc).__name__}: {exc}"
        elapsed_per_clip = (time.perf_counter() - began) / len(examples)
        connection.executemany(
            "INSERT INTO predictions VALUES (?, ?, ?, ?, ?)",
            [
                (args.model_name, clip_id, str(transcript), elapsed_per_clip, error)
                for (clip_id, _), transcript in zip(batch_rows, transcripts, strict=True)
            ],
        )
        connection.commit()
        processed += len(batch_rows)
        print(
            f"model={args.model_name} processed={processed}/{len(pending)} "
            f"error={error is not None}",
            flush=True,
        )
    connection.close()
    return 0


def _is_arabic_character(character: str) -> bool:
    codepoint = ord(character)
    return any(start <= codepoint <= end for start, end in ARABIC_RANGES)


def _suspicious_repetition(text: str) -> bool:
    tokens = text.split()
    return any(len(set(tokens[index : index + 4])) == 1 for index in range(len(tokens) - 3))


def _metrics(rows: Sequence[sqlite3.Row]) -> dict[str, Any]:
    valid = [row for row in rows if row["error"] is None]
    transcripts = [str(row["transcript"]).strip() for row in valid]
    audio_seconds = sum(float(row["duration"]) for row in valid)
    inference_seconds = sum(float(row["inference_seconds"]) for row in rows)
    letters = [character for text in transcripts for character in text if character.isalpha()]
    arabic_letters = sum(_is_arabic_character(character) for character in letters)
    return {
        "clips": len(rows),
        "errors": len(rows) - len(valid),
        "empty_transcripts": sum(not text for text in transcripts),
        "nonempty_yield": sum(bool(text) for text in transcripts) / len(valid) if valid else None,
        "perso_arabic_letter_rate": arabic_letters / len(letters) if letters else None,
        "characters_per_audio_second": sum(len(text) for text in transcripts) / audio_seconds
        if audio_seconds
        else None,
        "words_per_audio_second": sum(len(text.split()) for text in transcripts) / audio_seconds
        if audio_seconds
        else None,
        "suspicious_repetitions": sum(_suspicious_repetition(text) for text in transcripts),
        "audio_seconds": audio_seconds,
        "inference_seconds": inference_seconds,
        "rtfx": audio_seconds / inference_seconds if inference_seconds else None,
    }


def summarize_results(args: argparse.Namespace) -> int:
    connection = _connect(args.database)
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        "SELECT p.model_name, s.engine, s.tier, s.duration, p.transcript, "
        "p.inference_seconds, p.error FROM predictions p JOIN selection s USING (clip_id) "
        "ORDER BY p.model_name, s.engine, s.tier, s.source_id, s.clip_id"
    ).fetchall()
    grouped: dict[tuple[str, str, str], list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        grouped[(row["model_name"], row["engine"], row["tier"])].append(row)
    model_rows = {
        row["model_name"]: dict(row)
        for row in connection.execute("SELECT * FROM models ORDER BY model_name")
    }
    selected = connection.execute("SELECT COUNT(*) FROM selection").fetchone()[0]
    connection.close()
    summary = {
        "database": str(args.database.resolve()),
        "selected_clips": selected,
        "models": model_rows,
        "metrics": [
            {"model": model, "engine": engine, "tier": tier, **_metrics(group)}
            for (model, engine, tier), group in sorted(grouped.items())
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def main() -> int:
    args = parser().parse_args()
    return int(args.func(args))
