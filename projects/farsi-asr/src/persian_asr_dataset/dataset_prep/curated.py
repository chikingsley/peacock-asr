from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from persian_asr_dataset.dataset_prep.text_audit import (
    DEFAULT_TOKENIZER,
    TokenizerCoverageError,
    assert_tokenizer_clean,
)
from persian_asr_dataset.paths import PROJECT_ROOT
from persian_asr_dataset.text_normalization import maybe_normalize

OMNI_SAMPLE_RATE = 16_000
# Curation/selection inputs live in the core package; the final Omni parquet export
# lands inside the finetune_omni package. Both gitignored in place.
CORE_DATA_ROOT = PROJECT_ROOT / "src" / "persian_asr_dataset" / "data"
OMNI_DATA_ROOT = PROJECT_ROOT / "src" / "finetune_omni" / "data"
DEFAULT_DATASET_ROOT = (
    CORE_DATA_ROOT / "selection" / "candidate-manifests" / "persian-asr-clean-100h-filter"
)
DEFAULT_OUTPUT_ROOT = OMNI_DATA_ROOT / "training" / "omnilingual" / "persian_asr_clean_100h_filter"
DEV_SOURCE_SPLITS = {"dev", "validation"}
SKIPPED_SOURCE_SPLITS = {"test"}
PARQUET_SCHEMA = pa.schema(
    [
        ("text", pa.string()),
        ("audio_bytes", pa.list_(pa.int8())),
        ("audio_size", pa.int64()),
    ]
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a curated Persian ASR manifest to Omnilingual parquet."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_DATASET_ROOT / "manifests/train_manifest.jsonl",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--version", default="0")
    parser.add_argument("--dataset-name", default="persian_asr_clean_100h_filter")
    parser.add_argument("--language", default="fas_Arab")
    parser.add_argument("--rows-per-file", type=int, default=1000)
    parser.add_argument("--row-group-size", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--include-source-test",
        action="store_true",
        help="Route source test rows into train instead of skipping them.",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=DEFAULT_TOKENIZER,
        help="SentencePiece tokenizer used for the export coverage gate.",
    )
    parser.add_argument(
        "--allow-unknown-tokens",
        action="store_true",
        help="Skip the tokenizer-coverage gate (allow <unk>-producing text). Unsafe.",
    )
    return parser


def target_split(source_split: str, *, include_source_test: bool) -> str | None:
    split = source_split.lower()
    if split in DEV_SOURCE_SPLITS:
        return "dev"
    if split in SKIPPED_SOURCE_SPLITS and not include_source_test:
        return None
    return "train"


def audio_size(row: dict[str, Any]) -> int:
    duration = float(row["duration"])
    sample_rate = int(row.get("sample_rate") or OMNI_SAMPLE_RATE)
    return round(duration * sample_rate)


def read_audio_array(row: dict[str, Any]) -> np.ndarray:
    path = Path(str(row["audio_filepath"]))
    return np.frombuffer(path.read_bytes(), dtype=np.int8)


def shard_dir(args: argparse.Namespace, corpus: str, split: str) -> Path:
    return (
        args.output_root
        / f"version={args.version}"
        / f"corpus={corpus}"
        / f"split={split}"
        / f"language={args.language}"
    )


def write_shard(
    columns: dict[str, list[Any]],
    out_dir: Path,
    shard_index: int,
    row_group_size: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_arrays(
        [
            pa.array(columns["text"], type=pa.string()),
            pa.array(columns["audio_bytes"], type=pa.list_(pa.int8())),
            pa.array(columns["audio_size"], type=pa.int64()),
        ],
        schema=PARQUET_SCHEMA,
    )
    pq.write_table(
        table,
        out_dir / f"part-{shard_index:05d}.parquet",
        row_group_size=row_group_size,
    )
    for values in columns.values():
        values.clear()


def new_columns() -> dict[str, list[Any]]:
    return {"text": [], "audio_bytes": [], "audio_size": []}


def flush_ready_shards(
    shards: dict[tuple[str, str], dict[str, list[Any]]],
    shard_indices: dict[tuple[str, str], int],
    args: argparse.Namespace,
) -> None:
    for (corpus, split), columns in list(shards.items()):
        if len(columns["text"]) < args.rows_per_file:
            continue
        write_shard(
            columns=columns,
            out_dir=shard_dir(args, corpus, split),
            shard_index=shard_indices[(corpus, split)],
            row_group_size=args.row_group_size,
        )
        shard_indices[(corpus, split)] += 1


def flush_remaining_shards(
    shards: dict[tuple[str, str], dict[str, list[Any]]],
    shard_indices: dict[tuple[str, str], int],
    args: argparse.Namespace,
) -> None:
    for (corpus, split), columns in sorted(shards.items()):
        if columns["text"]:
            write_shard(
                columns=columns,
                out_dir=shard_dir(args, corpus, split),
                shard_index=shard_indices[(corpus, split)],
                row_group_size=args.row_group_size,
            )


def write_stats(
    args: argparse.Namespace,
    hours_by_corpus: dict[str, float],
    summary: dict[str, Any],
) -> None:
    stats_path = args.output_root / f"language_distribution_{args.version}.tsv"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8") as handle:
        handle.write("corpus\tlanguage\thours\n")
        for corpus, hours in sorted(hours_by_corpus.items()):
            handle.write(f"{corpus}\t{args.language}\t{hours:.8f}\n")

    summary_path = args.output_root / "export_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {stats_path}")
    print(f"wrote {summary_path}")


def prepare_output_root(args: argparse.Namespace) -> None:
    version_root = args.output_root / f"version={args.version}"
    if args.overwrite and version_root.exists():
        shutil.rmtree(version_root)


def manifest_total(manifest: Path) -> int | None:
    verification = manifest.parents[1] / "verification.json"
    if not verification.exists():
        return None
    data = json.loads(verification.read_text(encoding="utf-8"))
    rows = data.get("manifest_rows")
    return int(rows) if rows is not None else None


def export(args: argparse.Namespace) -> dict[str, Any]:
    prepare_output_root(args)

    shards: dict[tuple[str, str], dict[str, list[Any]]] = defaultdict(new_columns)
    shard_indices: dict[tuple[str, str], int] = defaultdict(int)
    rows_by_split: dict[str, int] = defaultdict(int)
    rows_by_corpus: dict[str, int] = defaultdict(int)
    hours_by_split: dict[str, float] = defaultdict(float)
    hours_by_corpus: dict[str, float] = defaultdict(float)
    skipped_by_source_split: dict[str, int] = defaultdict(int)
    kept_rows = 0

    total = manifest_total(args.manifest)
    with args.manifest.open(encoding="utf-8") as handle:
        progress = tqdm(handle, total=total, desc="curated Persian -> omni parquet", unit="utt")
        for line in progress:
            row = json.loads(line)
            split = target_split(
                str(row["source_split"]), include_source_test=args.include_source_test
            )
            if split is None:
                skipped_by_source_split[str(row["source_split"])] += 1
                continue

            corpus = str(row["source"])
            size = audio_size(row)
            columns = shards[(corpus, split)]
            columns["text"].append(maybe_normalize(str(row["text"])) or "")
            columns["audio_bytes"].append(read_audio_array(row))
            columns["audio_size"].append(size)

            seconds = size / OMNI_SAMPLE_RATE
            rows_by_split[split] += 1
            rows_by_corpus[corpus] += 1
            hours_by_split[split] += seconds / 3600
            hours_by_corpus[corpus] += seconds / 3600
            kept_rows += 1
            flush_ready_shards(shards, shard_indices, args)

            if args.limit and kept_rows >= args.limit:
                break

    flush_remaining_shards(shards, shard_indices, args)

    summary = {
        "dataset": args.dataset_name,
        "manifest": str(args.manifest),
        "output_root": str(args.output_root),
        "version": args.version,
        "language": args.language,
        "rows": kept_rows,
        "rows_by_split": dict(sorted(rows_by_split.items())),
        "rows_by_corpus": dict(sorted(rows_by_corpus.items())),
        "hours_by_split": dict(sorted(hours_by_split.items())),
        "hours_by_corpus": dict(sorted(hours_by_corpus.items())),
        "skipped_by_source_split": dict(sorted(skipped_by_source_split.items())),
        "source_test_policy": "train" if args.include_source_test else "skip",
        "rows_per_file": args.rows_per_file,
        "row_group_size": args.row_group_size,
    }
    write_stats(args, hours_by_corpus, summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = export(args)
    version_root = args.output_root / f"version={args.version}"
    print(f"dataset root: {version_root}")
    print(json.dumps(summary["rows_by_split"], indent=2, sort_keys=True))

    if args.allow_unknown_tokens:
        print("tokenizer-coverage gate: SKIPPED (--allow-unknown-tokens)")
        return 0
    try:
        stats = assert_tokenizer_clean(version_root, args.tokenizer)
    except TokenizerCoverageError as error:
        print(f"tokenizer-coverage gate: FAILED\n{error}")
        return 1
    print(f"tokenizer-coverage gate: PASSED (rows={stats.rows}, unk_rows=0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
