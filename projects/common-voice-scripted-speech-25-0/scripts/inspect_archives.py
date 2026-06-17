"""Inspect Common Voice tar archives without extracting them."""

from __future__ import annotations

import argparse
import json
import logging
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT / "manifests" / "datasets.jsonl"
DEFAULT_ARCHIVES = PROJECT / "data" / "raw" / "archives"
DEFAULT_REPORT = PROJECT / "data" / "reports" / "archive_inventory.json"

LOGGER = logging.getLogger("inspect_archives")


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def load_manifest(path: Path) -> list[dict[str, Any]]:
    """Load MDC archive manifest rows from JSONL."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped_line = raw_line.strip()
            if stripped_line:
                rows.append(json.loads(stripped_line))
    return rows


def inspect_archive(path: Path, max_tsv_headers: int) -> dict[str, Any]:
    """Inspect one archive and return member counts and TSV headers."""
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
    """Inspect locally downloaded archives and write an inventory report."""
    _configure_logging()
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
            LOGGER.info("missing: %s", path)
            continue
        LOGGER.info("inspecting: %s", path)
        reports.append({**row, **inspect_archive(path, args.max_tsv_headers)})

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(reports, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("wrote %s", args.report)


if __name__ == "__main__":
    main()
