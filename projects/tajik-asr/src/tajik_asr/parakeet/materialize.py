"""Materialize an Omni Parquet evaluation split for NeMo/Parakeet evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow.parquet as pq
from omni_finetune_core.parquet import SAMPLE_RATE

from tajik_asr import DATA

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import TextIO

DEFAULT_DATASET_ROOT = DATA / "datasets" / "v3" / "version=0"
DEFAULT_OUTPUT_ROOT = DATA / "parakeet" / "eval" / "tajik-v3-youtube-test"


@dataclass
class MaterializationStats:
    rows: int = 0
    excluded: int = 0
    duration_seconds: float = 0.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write embedded Omni Parquet audio plus a NeMo evaluation manifest."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--split", default="test")
    parser.add_argument("--language", default="tgk_Cyrl")
    parser.add_argument("--corpus-prefix", default="youtube-")
    parser.add_argument(
        "--max-duration",
        type=float,
        default=40.0,
        help="Exclude rows longer than the evaluation model accepts; 0 disables the filter.",
    )
    parser.add_argument("--dataset-repo", default="Peacockery/tajik-asr-corpus-v3")
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _write_exact(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"existing audio differs from source row: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _matching_parquets(dataset_root: Path, args: argparse.Namespace) -> list[Path]:
    paths = sorted(
        dataset_root.glob(
            f"corpus={args.corpus_prefix}*/split={args.split}/language={args.language}/*.parquet"
        )
    )
    if not paths:
        raise SystemExit(
            f"no matching Parquet files under {dataset_root} for "
            f"corpus={args.corpus_prefix}* split={args.split} language={args.language}"
        )
    return paths


def _keep_duration(size: int, max_duration: float | None, stats: MaterializationStats) -> bool:
    duration = size / SAMPLE_RATE
    if max_duration is not None and duration > max_duration:
        stats.excluded += 1
        return False
    stats.rows += 1
    stats.duration_seconds += duration
    return True


def _scan_rows(parquet_paths: Iterable[Path], max_duration: float | None) -> MaterializationStats:
    stats = MaterializationStats()
    for parquet_path in parquet_paths:
        for batch in pq.ParquetFile(parquet_path).iter_batches(
            batch_size=1024, columns=["audio_size"]
        ):
            for size in batch.column("audio_size").to_pylist():
                _keep_duration(size, max_duration, stats)
    return stats


def _write_rows(
    parquet_paths: Iterable[Path],
    audio_root: Path,
    handle: TextIO,
    max_duration: float | None,
    args: argparse.Namespace,
) -> MaterializationStats:
    stats = MaterializationStats()
    for parquet_path in parquet_paths:
        corpus = next(
            part.split("=", 1)[1] for part in parquet_path.parts if part.startswith("corpus=")
        )
        for batch in pq.ParquetFile(parquet_path).iter_batches(
            batch_size=128, columns=["text", "audio_bytes", "audio_size"]
        ):
            texts = batch.column("text").to_pylist()
            blobs = batch.column("audio_bytes").to_pylist()
            sizes = batch.column("audio_size").to_pylist()
            for text, blob, size in zip(texts, blobs, sizes, strict=True):
                if not _keep_duration(size, max_duration, stats):
                    continue
                payload = array("b", blob).tobytes()
                digest = hashlib.sha256(payload).hexdigest()
                audio_path = audio_root / corpus / f"{digest[:24]}.flac"
                _write_exact(audio_path, payload)
                record = {
                    "audio_filepath": str(audio_path),
                    "text": text,
                    "duration": round(size / SAMPLE_RATE, 6),
                    "corpus": corpus,
                    "audio_sha256": digest,
                    "dataset_repo": args.dataset_repo,
                    "dataset_revision": args.dataset_revision,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return stats


def _write_materialization_summary(
    path: Path,
    dataset_root: Path,
    manifest: Path,
    max_duration: float | None,
    stats: MaterializationStats,
    args: argparse.Namespace,
) -> None:
    summary = {
        "dataset_root": str(dataset_root),
        "dataset_repo": args.dataset_repo,
        "dataset_revision": args.dataset_revision,
        "split": args.split,
        "corpus_prefix": args.corpus_prefix,
        "language": args.language,
        "max_duration": max_duration,
        "rows": stats.rows,
        "excluded_over_max_duration": stats.excluded,
        "hours": stats.duration_seconds / 3600,
        "manifest": str(manifest),
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def materialize(args: argparse.Namespace) -> tuple[int, int, float, Path]:
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    manifest = output_root / "manifest.jsonl"
    max_duration = args.max_duration or None
    if not dataset_root.is_dir():
        raise SystemExit(f"dataset root does not exist: {dataset_root}")
    parquet_paths = _matching_parquets(dataset_root, args)
    if args.dry_run:
        stats = _scan_rows(parquet_paths, max_duration)
        return stats.rows, stats.excluded, stats.duration_seconds, manifest

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_tmp = manifest.with_suffix(".jsonl.tmp")
    with manifest_tmp.open("w", encoding="utf-8") as handle:
        stats = _write_rows(parquet_paths, output_root / "audio", handle, max_duration, args)
    manifest_tmp.replace(manifest)
    _write_materialization_summary(
        output_root / "materialization.json",
        dataset_root,
        manifest,
        max_duration,
        stats,
        args,
    )
    return stats.rows, stats.excluded, stats.duration_seconds, manifest


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows, excluded, duration_seconds, manifest = materialize(args)
    mode = "would materialize" if args.dry_run else "materialized"
    print(
        f"{mode} rows={rows} excluded={excluded} hours={duration_seconds / 3600:.2f} "
        f"manifest={manifest}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
