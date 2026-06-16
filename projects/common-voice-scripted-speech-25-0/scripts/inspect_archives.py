#!/usr/bin/env python3
"""Inspect Common Voice tar archives without extracting them."""

from __future__ import annotations

import argparse
import json
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT / "manifests" / "datasets.jsonl"
DEFAULT_ARCHIVES = ROOT / "data" / "common-voice-scripted-speech-25-0" / "raw" / "archives"
DEFAULT_REPORT = ROOT / "data" / "common-voice-scripted-speech-25-0" / "reports" / "archive_inventory.json"


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def inspect_archive(path: Path, max_tsv_headers: int) -> dict[str, Any]:
    suffix_counts: Counter[str] = Counter()
    top_level: Counter[str] = Counter()
    tsv_headers: list[dict[str, str]] = []
    total_members = 0
    regular_files = 0
    total_uncompressed_bytes = 0

    with tarfile.open(path, "r:*") as archive:
        for member in archive:
            total_members += 1
            parts = Path(member.name).parts
            if parts:
                top_level[parts[0]] += 1
            if not member.isfile():
                continue
            regular_files += 1
            total_uncompressed_bytes += member.size
            suffix_counts[Path(member.name).suffix.lower() or "<none>"] += 1
            if member.name.endswith(".tsv") and len(tsv_headers) < max_tsv_headers:
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                header = extracted.readline().decode("utf-8", errors="replace").strip()
                tsv_headers.append({"path": member.name, "header": header})

    return {
        "archive": str(path.relative_to(ROOT)),
        "size_bytes": path.stat().st_size,
        "total_members": total_members,
        "regular_files": regular_files,
        "total_uncompressed_bytes": total_uncompressed_bytes,
        "suffix_counts": dict(sorted(suffix_counts.items())),
        "top_level_counts": dict(top_level.most_common(20)),
        "tsv_headers": tsv_headers,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVES)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-tsv-headers", type=int, default=25)
    args = parser.parse_args()

    reports = []
    for row in load_manifest(args.manifest):
        path = args.archive_dir / row["filename"]
        if not path.exists():
            print(f"missing: {path}")
            continue
        print(f"inspecting: {path}")
        reports.append({**row, **inspect_archive(path, args.max_tsv_headers)})

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(reports, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
