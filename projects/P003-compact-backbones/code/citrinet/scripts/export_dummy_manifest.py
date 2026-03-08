#!/usr/bin/env python3
"""Build a tiny NeMo-style manifest for Citrinet preflight work."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import wave
from pathlib import Path

EXPECTED_TSV_COLUMNS = 2


def _wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        frames = handle.getnframes()
        rate = handle.getframerate()
    return frames / rate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a NeMo JSONL manifest from a TSV of wav path + text."
    )
    parser.add_argument(
        "--input-tsv",
        type=Path,
        required=True,
        help="TSV with columns: audio_filepath<TAB>text",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL manifest path.",
    )
    args = parser.parse_args()

    rows: list[str] = []
    with args.input_tsv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for line_no, row in enumerate(reader, start=1):
            if len(row) != EXPECTED_TSV_COLUMNS:
                msg = f"{args.input_tsv}:{line_no}: expected 2 tab-separated columns"
                raise SystemExit(msg)
            audio_path = Path(row[0]).expanduser().resolve()
            text = row[1].strip()
            if not audio_path.exists():
                msg = f"{args.input_tsv}:{line_no}: missing audio file: {audio_path}"
                raise SystemExit(msg)
            record = {
                "audio_filepath": str(audio_path),
                "duration": _wav_duration_seconds(audio_path),
                "text": text,
            }
            rows.append(json.dumps(record, sort_keys=True))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(rows) + "\n", encoding="utf-8")
    sys.stdout.write(f"{args.output}\n")


if __name__ == "__main__":
    main()
