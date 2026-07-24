"""Materialize an Omni Parquet export as deterministic NeMo audio manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow.parquet as pq

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import TextIO

    from parakeet_finetune_core.project import ParakeetProject

SAMPLE_RATE = 16_000


@dataclass
class MaterializationStats:
    """Counts and duration for one materialized split."""

    rows: int = 0
    excluded_empty: int = 0
    excluded_duration: int = 0
    duration_seconds: float = 0.0
    rows_by_corpus: dict[str, int] = field(default_factory=dict)

    @property
    def hours(self) -> float:
        return self.duration_seconds / 3600.0


def build_parser(project: ParakeetProject) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize an omni-curator export for NeMo/Parakeet training."
    )
    parser.add_argument("--dataset-root", type=Path, default=project.default_dataset_root)
    parser.add_argument("--output-root", type=Path, default=project.default_materialized_root)
    parser.add_argument("--split", default="train")
    parser.add_argument("--language", default=project.language)
    parser.add_argument(
        "--corpus",
        action="append",
        default=None,
        help="Exact corpus partition to include; repeat for multiple. Default: every corpus.",
    )
    parser.add_argument("--min-duration", type=float, default=0.5)
    parser.add_argument("--max-duration", type=float, default=30.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dataset-repo", default=None)
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _partition_value(path: Path, key: str) -> str:
    prefix = f"{key}="
    return next(part.removeprefix(prefix) for part in path.parts if part.startswith(prefix))


def matching_parquets(
    dataset_root: Path,
    *,
    split: str,
    language: str,
    corpora: set[str] | None,
) -> list[Path]:
    paths = sorted(dataset_root.glob(f"corpus=*/split={split}/language={language}/*.parquet"))
    if corpora is not None:
        paths = [path for path in paths if _partition_value(path, "corpus") in corpora]
    if not paths:
        corpus_note = "all corpora" if corpora is None else ", ".join(sorted(corpora))
        raise SystemExit(
            f"no Parquet rows under {dataset_root} for split={split} "
            f"language={language} corpora={corpus_note}"
        )
    return paths


def _payload(blob: object) -> bytes:
    if isinstance(blob, bytes):
        return blob
    if isinstance(blob, bytearray):
        return bytes(blob)
    if isinstance(blob, list):
        return array("b", blob).tobytes()
    raise TypeError(f"unsupported audio_bytes value: {type(blob).__name__}")


def _write_exact(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"existing materialized audio differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _record(
    *,
    audio_path: Path,
    text: str,
    duration: float,
    corpus: str,
    digest: str,
    args: argparse.Namespace,
) -> dict[str, object]:
    record: dict[str, object] = {
        "audio_filepath": str(audio_path),
        "text": text,
        "duration": round(duration, 6),
        "corpus": corpus,
        "audio_sha256": digest,
    }
    if args.dataset_repo:
        record["dataset_repo"] = args.dataset_repo
    if args.dataset_revision:
        record["dataset_revision"] = args.dataset_revision
    return record


def _rows(
    paths: Iterable[Path],
    args: argparse.Namespace,
    *,
    audio_root: Path | None,
    manifest: TextIO | None,
) -> MaterializationStats:
    stats = MaterializationStats()
    for parquet_path in paths:
        corpus = _partition_value(parquet_path, "corpus")
        for batch in pq.ParquetFile(parquet_path).iter_batches(
            batch_size=128, columns=["text", "audio_bytes", "audio_size"]
        ):
            for text_value, blob, size_value in zip(
                batch.column("text").to_pylist(),
                batch.column("audio_bytes").to_pylist(),
                batch.column("audio_size").to_pylist(),
                strict=True,
            ):
                text = str(text_value or "").strip()
                if not text:
                    stats.excluded_empty += 1
                    continue
                duration = int(size_value) / SAMPLE_RATE
                if duration < args.min_duration or (
                    args.max_duration > 0 and duration > args.max_duration
                ):
                    stats.excluded_duration += 1
                    continue
                payload = _payload(blob)
                digest = hashlib.sha256(payload).hexdigest()
                if audio_root is not None and manifest is not None:
                    audio_path = audio_root / corpus / digest[:2] / f"{digest}.flac"
                    _write_exact(audio_path, payload)
                    manifest.write(
                        json.dumps(
                            _record(
                                audio_path=audio_path,
                                text=text,
                                duration=duration,
                                corpus=corpus,
                                digest=digest,
                                args=args,
                            ),
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                stats.rows += 1
                stats.duration_seconds += duration
                stats.rows_by_corpus[corpus] = stats.rows_by_corpus.get(corpus, 0) + 1
                if args.limit and stats.rows >= args.limit:
                    return stats
    return stats


def _summary(
    path: Path,
    *,
    dataset_root: Path,
    manifest: Path,
    args: argparse.Namespace,
    stats: MaterializationStats,
) -> None:
    payload = {
        "dataset_root": str(dataset_root),
        "dataset_repo": args.dataset_repo,
        "dataset_revision": args.dataset_revision,
        "split": args.split,
        "language": args.language,
        "corpora": sorted(args.corpus) if args.corpus else None,
        "min_duration": args.min_duration,
        "max_duration": args.max_duration,
        "rows": stats.rows,
        "rows_by_corpus": stats.rows_by_corpus,
        "excluded_empty": stats.excluded_empty,
        "excluded_duration": stats.excluded_duration,
        "hours": stats.hours,
        "manifest": str(manifest),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def materialize(args: argparse.Namespace) -> tuple[MaterializationStats, Path]:
    if args.dataset_root is None:
        raise SystemExit("--dataset-root is required")
    if args.output_root is None:
        raise SystemExit("--output-root is required")
    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not dataset_root.is_dir():
        raise SystemExit(f"dataset root does not exist: {dataset_root}")
    paths = matching_parquets(
        dataset_root,
        split=args.split,
        language=args.language,
        corpora=set(args.corpus) if args.corpus else None,
    )
    manifest = output_root / "manifests" / f"{args.split}.jsonl"
    if args.dry_run:
        return _rows(paths, args, audio_root=None, manifest=None), manifest

    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest_tmp = manifest.with_suffix(".jsonl.tmp")
    with manifest_tmp.open("w", encoding="utf-8") as handle:
        stats = _rows(paths, args, audio_root=output_root / "audio", manifest=handle)
    manifest_tmp.replace(manifest)
    _summary(
        output_root / f"materialization-{args.split}.json",
        dataset_root=dataset_root,
        manifest=manifest,
        args=args,
        stats=stats,
    )
    return stats, manifest


def materialize_main(project: ParakeetProject, argv: list[str] | None = None) -> int:
    args = build_parser(project).parse_args(argv)
    stats, manifest = materialize(args)
    verb = "would materialize" if args.dry_run else "materialized"
    print(
        f"{verb} rows={stats.rows} hours={stats.hours:.2f} "
        f"excluded_empty={stats.excluded_empty} excluded_duration={stats.excluded_duration} "
        f"manifest={manifest}",
        flush=True,
    )
    print(f"  by corpus: {stats.rows_by_corpus}", flush=True)
    return 0
