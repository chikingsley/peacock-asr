#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge one or more Lhotse JSONL cut manifests into a single output "
            "manifest. Duplicate cut IDs are rejected."
        )
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        required=True,
        help="Input manifest paths (.jsonl or .jsonl.gz).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output manifest path (.jsonl or .jsonl.gz).",
    )
    return parser.parse_args()


def _open_text(path: Path, mode: str):
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def main() -> int:
    args = parse_args()
    seen_ids: set[str] = set()
    total_lines = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with _open_text(args.output, "wt") as sink:
        for manifest_path in args.inputs:
            if not manifest_path.is_file():
                msg = f"manifest not found: {manifest_path}"
                raise FileNotFoundError(msg)
            with _open_text(manifest_path, "rt") as source:
                for line in source:
                    payload = json.loads(line)
                    cut_id = str(payload["id"])
                    if cut_id in seen_ids:
                        msg = f"duplicate cut id detected while merging manifests: {cut_id}"
                        raise ValueError(msg)
                    seen_ids.add(cut_id)
                    sink.write(line)
                    total_lines += 1

    summary = {
        "output": str(args.output),
        "inputs": [str(path) for path in args.inputs],
        "merged_cuts": total_lines,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
