from __future__ import annotations

import argparse
import hashlib
import json
import random
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import HfApi, hf_hub_download

V4_REPO = "Peacockery/farsi-asr-corpus-v4"
V4_REVISION = "564d41da9e5b935c0fe2bf2443e205ca7b747c96"
V4_CORPORA = (
    "asr_farsi_youtube",
    "common_voice_25_0",
    "fleurs",
    "mana_tts",
    "neyshekar",
    "thomcles_persian_farsi_speech",
    "worldspeech",
)
SAMPLE_RATE = 16_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _stable_seed(seed: int, value: str) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _audio_bytes(value: bytes | list[int]) -> bytes:
    if isinstance(value, bytes):
        return value
    return np.asarray(value, dtype=np.int8).tobytes()


def _training_shards() -> dict[str, list[str]]:
    info = HfApi().dataset_info(V4_REPO, revision=V4_REVISION)
    if info.sha != V4_REVISION:
        raise RuntimeError(f"Hub revision mismatch: requested {V4_REVISION}, resolved {info.sha}")
    grouped: dict[str, list[str]] = defaultdict(list)
    for sibling in info.siblings or []:
        name = sibling.rfilename
        for corpus in V4_CORPORA:
            prefix = f"version=0/corpus={corpus}/split=train/language=fas_Arab/"
            if name.startswith(prefix) and name.endswith(".parquet"):
                grouped[corpus].append(name)
                break
    missing = set(V4_CORPORA) - grouped.keys()
    if missing:
        raise RuntimeError(f"V4 train partitions missing for: {sorted(missing)}")
    return {corpus: sorted(grouped[corpus]) for corpus in V4_CORPORA}


def _sample_shard(
    shard: Path,
    *,
    corpus: str,
    hub_path: str,
    limit: int,
    seed: int,
    audio_dir: Path,
) -> list[dict[str, Any]]:
    parquet = pq.ParquetFile(shard)
    columns = set(parquet.schema_arrow.names)
    required = {"text", "audio_bytes", "audio_size"}
    if not required.issubset(columns):
        raise RuntimeError(f"unsupported V4 schema in {shard}: {sorted(columns)}")
    total = parquet.metadata.num_rows
    selected = set(
        random.Random(_stable_seed(seed, f"{corpus}:{hub_path}")).sample(  # noqa: S311
            range(total), min(limit, total)
        )
    )
    rows: list[dict[str, Any]] = []
    row_offset = 0
    audio_dir.mkdir(parents=True, exist_ok=True)
    for batch in parquet.iter_batches(
        batch_size=256, columns=["text", "audio_bytes", "audio_size"]
    ):
        texts = batch.column(0).to_pylist()
        audio_values = batch.column(1).to_pylist()
        audio_sizes = batch.column(2).to_pylist()
        for batch_index, (text, audio_value, audio_size) in enumerate(
            zip(texts, audio_values, audio_sizes, strict=True)
        ):
            row_index = row_offset + batch_index
            if row_index not in selected:
                continue
            encoded = _audio_bytes(audio_value)
            sample_id = f"{corpus}-{Path(hub_path).stem}-{row_index:07d}"
            audio_path = (audio_dir / f"{sample_id}.flac").resolve()
            audio_path.write_bytes(encoded)
            info = sf.info(str(audio_path))
            rows.append(
                {
                    "sample_id": sample_id,
                    "source": corpus,
                    "audio_filepath": str(audio_path),
                    "text": str(text),
                    "duration": float(info.duration),
                    "audio_size": int(audio_size),
                    "audio_sha256": hashlib.sha256(encoded).hexdigest(),
                    "hub_repo": V4_REPO,
                    "hub_revision": V4_REVISION,
                    "hub_path": hub_path,
                    "hub_row_index": row_index,
                }
            )
        row_offset += len(texts)
    if len(rows) != min(limit, total):
        raise RuntimeError(f"sampled {len(rows)} rows from {shard}; expected {min(limit, total)}")
    return rows


def _write_benchmark(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "audio": [Path(str(row["audio_filepath"])).read_bytes() for row in rows],
            "transcription": [str(row["text"]) for row in rows],
        },
        schema=pa.schema([("audio", pa.binary()), ("transcription", pa.string())]),
    )
    pq.write_table(table, path, compression="zstd")


def cmd_sample(args: argparse.Namespace) -> int:
    if args.per_corpus < 1:
        raise SystemExit("--per-corpus must be at least 1")
    shards = _training_shards()
    source_dir = args.output_dir / "source"
    audio_dir = args.output_dir / "audio"
    all_rows: list[dict[str, Any]] = []
    selected_shards: dict[str, dict[str, Any]] = {}
    for corpus in V4_CORPORA:
        candidates = shards[corpus]
        rng = random.Random(_stable_seed(args.seed, corpus))  # noqa: S311
        hub_path = rng.choice(candidates)
        alternatives = [candidate for candidate in candidates if candidate != hub_path]
        rng.shuffle(alternatives)
        for candidate in [hub_path, *alternatives]:
            print(f"{corpus}: downloading {candidate}", flush=True)
            local = Path(
                hf_hub_download(
                    repo_id=V4_REPO,
                    filename=candidate,
                    repo_type="dataset",
                    revision=V4_REVISION,
                    local_dir=source_dir,
                )
            )
            if pq.ParquetFile(local).metadata.num_rows >= args.per_corpus:
                hub_path = candidate
                break
            print(
                f"{corpus}: {candidate} has fewer than {args.per_corpus} rows; trying another",
                flush=True,
            )
        else:
            raise RuntimeError(f"no {corpus} shard contains {args.per_corpus} rows")
        rows = _sample_shard(
            local,
            corpus=corpus,
            hub_path=hub_path,
            limit=args.per_corpus,
            seed=args.seed,
            audio_dir=audio_dir / corpus,
        )
        all_rows.extend(rows)
        selected_shards[corpus] = {
            "hub_path": hub_path,
            "local_path": str(local.resolve()),
            "sha256": _sha256(local),
            "available_train_shards": len(candidates),
            "sample_rows": len(rows),
        }
        print(f"{corpus}: materialized {len(rows)} rows", flush=True)
    manifest = args.output_dir / "sample.jsonl"
    benchmark = args.output_dir / "benchmark.parquet"
    _write_jsonl(manifest, all_rows)
    _write_benchmark(benchmark, all_rows)
    _write_json(
        args.output_dir / "sample-summary.json",
        {
            "hub_repo": V4_REPO,
            "hub_revision": V4_REVISION,
            "seed": args.seed,
            "per_corpus": args.per_corpus,
            "rows": len(all_rows),
            "audio_hours": sum(float(row["duration"]) for row in all_rows) / 3600,
            "manifest": str(manifest.resolve()),
            "manifest_sha256": _sha256(manifest),
            "benchmark": str(benchmark.resolve()),
            "benchmark_sha256": _sha256(benchmark),
            "selected_shards": selected_shards,
        },
    )
    print(f"wrote {len(all_rows)} rows -> {args.output_dir}")
    return 0


def cmd_attach(args: argparse.Namespace) -> int:
    rows = _read_jsonl(args.input)
    connection = sqlite3.connect(f"file:{args.database}?mode=ro", uri=True)
    run = connection.execute(
        "SELECT model_path, benchmark_path FROM runs WHERE run_id = ?", (args.run_id,)
    ).fetchone()
    if run is None:
        raise SystemExit(f"run_id does not exist: {args.run_id}")
    predictions = connection.execute(
        "SELECT row_index, hypothesis, error FROM predictions WHERE run_id = ? ORDER BY row_index",
        (args.run_id,),
    ).fetchall()
    connection.close()
    if len(predictions) != len(rows):
        raise SystemExit(f"prediction count {len(predictions)} does not match manifest {len(rows)}")
    for expected, (row_index, hypothesis, error) in enumerate(predictions):
        if row_index != expected:
            raise SystemExit(f"prediction row sequence breaks at {expected}: got {row_index}")
        rows[expected][args.hypothesis_field] = str(hypothesis)
        rows[expected]["draft_asr"] = {
            "run_id": args.run_id,
            "model_path": str(run[0]),
            "benchmark_path": str(run[1]),
            "error": error,
        }
    _write_jsonl(args.output, rows)
    print(f"attached {len(rows)} predictions -> {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and score a source-balanced audit of the pinned Farsi V4 corpus."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample = subparsers.add_parser("sample", help="materialize one deterministic shard sample")
    sample.add_argument("--output-dir", type=Path, required=True)
    sample.add_argument("--per-corpus", type=int, default=200)
    sample.add_argument("--seed", type=int, default=20260711)
    sample.set_defaults(func=cmd_sample)

    attach = subparsers.add_parser("attach", help="attach shared benchmark predictions to JSONL")
    attach.add_argument("--input", type=Path, required=True)
    attach.add_argument("--database", type=Path, required=True)
    attach.add_argument("--run-id", required=True)
    attach.add_argument("--output", type=Path, required=True)
    attach.add_argument("--hypothesis-field", default="hypothesis")
    attach.set_defaults(func=cmd_attach)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
