from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from farsi_asr_dataset.canonical import build_all
from farsi_asr_dataset.paths import DEFAULT_DATA_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build canonical Persian ASR Parquet datasets.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--rows-per-file", type=int, default=2_000)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--splits", nargs="*", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summaries = build_all(
        args.data_root,
        args.rows_per_file,
        datasets=set(args.datasets) if args.datasets else None,
        project_splits=set(args.splits) if args.splits else None,
    )
    print(json.dumps([asdict(summary) for summary in summaries], ensure_ascii=False, indent=2))
    return 0
