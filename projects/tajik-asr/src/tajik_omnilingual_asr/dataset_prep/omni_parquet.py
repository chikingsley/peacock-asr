from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = ROOT / "src/tajik_omnilingual_asr/dataset_prep/artifacts/tajik_asr_combined_v0"
# The omni parquet lives inside its source artifact (sibling of omni_manifest/).
# Versioning is by artifact dir (tajik_asr_combined_v0, _v1, ...); the parquet's own
# version=0 partition is a fairseq2 layout requirement, not the version axis.
DEFAULT_OUTPUT_ROOT = DEFAULT_DATASET / "omni_parquet"
LANGUAGE = "tgk_Cyrl"
SAMPLE_RATE = 16_000
PARQUET_SCHEMA = pa.schema(
    [
        ("text", pa.string()),
        ("audio_bytes", pa.list_(pa.int8())),
        ("audio_size", pa.int64()),
    ]
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export the Tajik ASR combined dataset to Omnilingual mixture parquet."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--version", default="0")
    parser.add_argument("--language", default=LANGUAGE)
    parser.add_argument("--rows-per-file", type=int, default=1000)
    parser.add_argument("--row-group-size", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def fetch_rows(db_path: Path) -> list[sqlite3.Row]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute(
            """
            select id, split, source, audio_filename, normalized_text
            from utterances
            order by split, source, source_id, id
            """
        ).fetchall()


def shard_dir(
    output_root: Path, version: str, corpus: str, split: str, language: str
) -> Path:
    return (
        output_root
        / f"version={version}"
        / f"corpus={corpus}"
        / f"split={split}"
        / f"language={language}"
    )


def new_columns() -> dict[str, list[Any]]:
    return {"text": [], "audio_bytes": [], "audio_size": []}


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


def flush_ready_shards(
    shards: dict[tuple[str, str], dict[str, list[Any]]],
    shard_indices: dict[tuple[str, str], int],
    output_root: Path,
    version: str,
    language: str,
    rows_per_file: int,
    row_group_size: int,
) -> None:
    for (corpus, split), columns in list(shards.items()):
        if len(columns["text"]) < rows_per_file:
            continue
        write_shard(
            columns,
            shard_dir(output_root, version, corpus, split, language),
            shard_indices[(corpus, split)],
            row_group_size,
        )
        shard_indices[(corpus, split)] += 1


def flush_remaining_shards(
    shards: dict[tuple[str, str], dict[str, list[Any]]],
    shard_indices: dict[tuple[str, str], int],
    output_root: Path,
    version: str,
    language: str,
    row_group_size: int,
) -> None:
    for (corpus, split), columns in sorted(shards.items()):
        if columns["text"]:
            write_shard(
                columns,
                shard_dir(output_root, version, corpus, split, language),
                shard_indices[(corpus, split)],
                row_group_size,
            )


def write_stats(
    output_root: Path,
    version: str,
    language: str,
    hours_by_corpus: dict[str, float],
    summary: dict[str, Any],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    stats_path = output_root / f"language_distribution_{version}.tsv"
    with stats_path.open("w", encoding="utf-8") as handle:
        handle.write("corpus\tlanguage\thours\n")
        for corpus, hours in sorted(hours_by_corpus.items()):
            handle.write(f"{corpus}\t{language}\t{hours:.8f}\n")

    summary_path = output_root / "export_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote\t{stats_path}")
    print(f"wrote\t{summary_path}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset_dir = args.dataset_dir.resolve()
    output_root = args.output_root.resolve()
    version_root = output_root / f"version={args.version}"
    db_path = dataset_dir / "tajik_asr_combined.sqlite"

    if args.overwrite and version_root.exists():
        shutil.rmtree(version_root)
    if version_root.exists():
        raise SystemExit(f"output exists, rerun with --overwrite: {version_root}")

    rows = fetch_rows(db_path)
    shards: dict[tuple[str, str], dict[str, list[Any]]] = defaultdict(new_columns)
    shard_indices: dict[tuple[str, str], int] = defaultdict(int)
    rows_by_split: dict[str, int] = defaultdict(int)
    rows_by_corpus: dict[str, int] = defaultdict(int)
    hours_by_split: dict[str, float] = defaultdict(float)
    hours_by_corpus: dict[str, float] = defaultdict(float)

    for row in rows:
        split = str(row["split"])
        corpus = str(row["source"])
        audio_path = dataset_dir / str(row["audio_filename"])
        audio = np.frombuffer(audio_path.read_bytes(), dtype=np.int8)
        audio_size = max(1, round(librosa.get_duration(path=audio_path) * SAMPLE_RATE))

        columns = shards[(corpus, split)]
        columns["text"].append(str(row["normalized_text"]))
        columns["audio_bytes"].append(audio)
        columns["audio_size"].append(audio_size)

        hours = audio_size / SAMPLE_RATE / 3600
        rows_by_split[split] += 1
        rows_by_corpus[corpus] += 1
        hours_by_split[split] += hours
        hours_by_corpus[corpus] += hours
        flush_ready_shards(
            shards,
            shard_indices,
            output_root,
            args.version,
            args.language,
            args.rows_per_file,
            args.row_group_size,
        )

    flush_remaining_shards(
        shards,
        shard_indices,
        output_root,
        args.version,
        args.language,
        args.row_group_size,
    )

    summary = {
        "dataset": "tajik_asr_corpus",
        "dataset_dir": str(dataset_dir),
        "output_root": str(output_root),
        "version": args.version,
        "language": args.language,
        "rows": len(rows),
        "rows_by_split": dict(sorted(rows_by_split.items())),
        "rows_by_corpus": dict(sorted(rows_by_corpus.items())),
        "hours_by_split": dict(sorted(hours_by_split.items())),
        "hours_by_corpus": dict(sorted(hours_by_corpus.items())),
        "rows_per_file": args.rows_per_file,
        "row_group_size": args.row_group_size,
    }
    write_stats(output_root, args.version, args.language, hours_by_corpus, summary)

    for split, count in sorted(rows_by_split.items()):
        print(f"{split}\t{count}")
    print(f"parquet_root\t{output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
