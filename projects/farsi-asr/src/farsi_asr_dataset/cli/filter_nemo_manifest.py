from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FilterStats:
    rows_seen: int
    rows_written: int
    rows_dropped_max_chars: int
    rows_dropped_max_chars_per_second: int
    rows_dropped_any: int
    hours_seen: float
    hours_written: float
    hours_dropped: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Filter a NeMo ASR manifest by transcript length and transcript/audio ratio."
    )
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--metadata-path", type=Path, default=None)
    parser.add_argument("--max-chars", type=int, default=400)
    parser.add_argument("--max-chars-per-second", type=float, default=60.0)
    return parser


def should_drop(
    row: dict[str, Any],
    max_chars: int,
    max_chars_per_second: float,
) -> tuple[bool, bool]:
    text = str(row["text"])
    duration = float(row["duration"])
    too_many_chars = len(text) > max_chars
    too_dense = len(text) / max(duration, 1e-9) > max_chars_per_second
    return too_many_chars, too_dense


def filter_manifest(args: argparse.Namespace) -> FilterStats:
    rows_seen = 0
    rows_written = 0
    rows_dropped_max_chars = 0
    rows_dropped_max_chars_per_second = 0
    rows_dropped_any = 0
    hours_seen = 0.0
    hours_written = 0.0
    hours_dropped = 0.0

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with (
        args.input_manifest.open(encoding="utf-8") as source,
        args.output_manifest.open("w", encoding="utf-8") as target,
    ):
        for line in source:
            row = json.loads(line)
            rows_seen += 1
            duration_hours = float(row["duration"]) / 3600
            hours_seen += duration_hours
            too_many_chars, too_dense = should_drop(
                row,
                max_chars=args.max_chars,
                max_chars_per_second=args.max_chars_per_second,
            )
            if too_many_chars:
                rows_dropped_max_chars += 1
            if too_dense:
                rows_dropped_max_chars_per_second += 1
            if too_many_chars or too_dense:
                rows_dropped_any += 1
                hours_dropped += duration_hours
                continue
            target.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            rows_written += 1
            hours_written += duration_hours

    return FilterStats(
        rows_seen=rows_seen,
        rows_written=rows_written,
        rows_dropped_max_chars=rows_dropped_max_chars,
        rows_dropped_max_chars_per_second=rows_dropped_max_chars_per_second,
        rows_dropped_any=rows_dropped_any,
        hours_seen=hours_seen,
        hours_written=hours_written,
        hours_dropped=hours_dropped,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    stats = filter_manifest(args)
    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "input_manifest": str(args.input_manifest),
        "output_manifest": str(args.output_manifest),
        "filters": {
            "max_chars": args.max_chars,
            "max_chars_per_second": args.max_chars_per_second,
        },
        "stats": asdict(stats),
    }
    metadata_path = args.metadata_path or args.output_manifest.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
