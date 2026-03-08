#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import TextIO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite the audio source prefix inside a Lhotse JSONL(.gz) manifest."
        )
    )
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--old-prefix", type=str, required=True)
    parser.add_argument("--new-prefix", type=str, required=True)
    return parser.parse_args()


def _open_text(path: Path, mode: str) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8")
    return path.open(mode + "t", encoding="utf-8")


def main() -> int:
    args = parse_args()
    old_prefix = args.old_prefix.rstrip("/")
    new_prefix = args.new_prefix.rstrip("/")
    rewritten = 0
    total = 0

    with _open_text(args.input_manifest, "r") as src, _open_text(
        args.output_manifest, "w"
    ) as dst:
        for line in src:
            payload = json.loads(line)
            source = str(payload["recording"]["sources"][0]["source"])
            total += 1
            if source.startswith(old_prefix):
                payload["recording"]["sources"][0]["source"] = (
                    new_prefix + source[len(old_prefix) :]
                )
                rewritten += 1
            dst.write(json.dumps(payload, ensure_ascii=True) + "\n")

    print(
        json.dumps(
            {
                "input_manifest": str(args.input_manifest),
                "output_manifest": str(args.output_manifest),
                "total_rows": total,
                "rewritten_rows": rewritten,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
