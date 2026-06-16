#!/usr/bin/env python3
"""Stage raw Common Voice Scripted Speech 25.0 archives for Hugging Face upload."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
PROJECT = Path(__file__).resolve().parents[1]
REPO_ID = "Peacockery/common-voice-scripted-speech-25-0"
DEFAULT_MANIFEST = PROJECT / "manifests" / "datasets.jsonl"
DEFAULT_ARCHIVES = ROOT / "data" / "common-voice-scripted-speech-25-0" / "raw" / "archives"
DEFAULT_REPORT = ROOT / "data" / "common-voice-scripted-speech-25-0" / "reports" / "archive_inventory.json"
DEFAULT_OUT = ROOT / "data" / "hf-upload" / "common-voice-scripted-speech-25-0"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def read_inventory(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {row["filename"]: row for row in rows}


def render_readme(rows: list[dict[str, Any]], archive_rows: list[dict[str, Any]]) -> str:
    table = [
        "| Language | Locale | MDC dataset ID | Archive | Size | SHA-256 |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for row in archive_rows:
        table.append(
            "| {language} | `{locale}` | `{dataset_id}` | `{filename}` | {size_bytes} | `{sha256}` |".format(
                language=row.get("language", ""),
                locale=row.get("locale", ""),
                dataset_id=row.get("dataset_id", ""),
                filename=row.get("filename", ""),
                size_bytes=row.get("size_bytes", ""),
                sha256=row.get("sha256", ""),
            )
        )
    languages = sorted({str(row.get("locale", "")) for row in rows if row.get("locale")})
    return "\n".join(
        [
            "---",
            "license: cc0-1.0",
            "language:",
            *[f"- {language}" for language in languages],
            "task_categories:",
            "- automatic-speech-recognition",
            "tags:",
            "- common-voice",
            "- mozilla-data-collective",
            "- scripted-speech",
            "- asr",
            "pretty_name: Common Voice Scripted Speech 25.0",
            "---",
            "",
            "# Common Voice Scripted Speech 25.0",
            "",
            "This dataset mirrors Mozilla Data Collective Common Voice Scripted Speech 25.0 archives",
            "for ASR training and corpus preparation. Archives are preserved unchanged from their",
            "presigned Mozilla Data Collective downloads.",
            "",
            "Rows can be separated later by `locale`, `language`, `dataset_id`, and archive filename.",
            "",
            "## Archives",
            "",
            *table,
            "",
            "## License",
            "",
            "The source listing for these archives reports Creative Commons Zero v1.0 Universal",
            "(`CC0-1.0`). See https://spdx.org/licenses/CC0-1.0.html.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVES)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    rows = load_jsonl(args.manifest)
    inventory = read_inventory(args.inventory)
    staged_rows: list[dict[str, Any]] = []
    archive_out = args.out_dir / "archives"

    for row in rows:
        source = args.archive_dir / row["filename"]
        if not source.exists():
            print(f"missing: {source}")
            continue
        destination = archive_out / row["filename"]
        print(f"stage: {source} -> {destination}")
        link_or_copy(source, destination)
        digest_path = source.with_suffix(source.suffix + ".sha256")
        digest = digest_path.read_text(encoding="utf-8").split()[0] if digest_path.exists() else sha256_file(source)
        staged_rows.append(
            {
                **row,
                "path": f"archives/{row['filename']}",
                "size_bytes": source.stat().st_size,
                "sha256": digest,
                "inventory": inventory.get(row["filename"], {}),
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "README.md").write_text(render_readme(rows, staged_rows), encoding="utf-8")
    (args.out_dir / "summary.json").write_text(
        json.dumps(
            {
                "repo_id": REPO_ID,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source_collection": "Common Voice Scripted Speech 25.0",
                "archive_count": len(staged_rows),
                "archives": staged_rows,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"staged {len(staged_rows)} archives in {args.out_dir}")


if __name__ == "__main__":
    main()
