"""Stage Common Voice Scripted Speech 25.0 files for Hugging Face."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
REPO_ID = "Peacockery/common-voice-scripted-speech-25-0"
DEFAULT_MANIFEST = PROJECT / "manifests" / "datasets.jsonl"
DEFAULT_LOCAL_README = PROJECT / "HF_README.md"
DEFAULT_ARCHIVES = PROJECT / "data" / "raw" / "archives"
DEFAULT_INVENTORY = PROJECT / "data" / "reports" / "archive_inventory.json"
DEFAULT_OUT = (
    Path("/mnt/overflow/peacock-asr")
    / "common-voice-scripted-speech-25-0"
    / "hf-staging"
)
CHUNK_SIZE = 1024 * 1024
LICENSE_TEXT = """SPDX-License-Identifier: CC0-1.0

Common Voice Scripted Speech 25.0 is reported by Mozilla Data Collective
as Creative Commons Zero v1.0 Universal.

Upstream source: https://mozilladatacollective.com/
SPDX license record: https://spdx.org/licenses/CC0-1.0.html
Creative Commons deed: https://creativecommons.org/publicdomain/zero/1.0/

This repository preserves upstream source dataset IDs, locales, archive
filenames, and checksums so downstream users can trace every normalized row
back to its Mozilla Data Collective source archive.
"""

LOGGER = logging.getLogger("stage_hf_release")


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped_line = raw_line.strip()
            if stripped_line:
                rows.append(json.loads(stripped_line))
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _read_inventory(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {row["filename"]: row for row in rows}


def _digest_for_archive(source: Path) -> str:
    digest_path = source.with_suffix(source.suffix + ".sha256")
    if digest_path.exists():
        return digest_path.read_text(encoding="utf-8").split()[0]
    return _sha256_file(source)


def _language_metadata(rows: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    locales = sorted({str(row["locale"]) for row in rows if row.get("locale")})
    regular = [locale for locale in locales if "-" not in locale]
    bcp47 = [locale for locale in locales if "-" in locale]
    return regular, bcp47


def _archive_table(archive_rows: list[dict[str, Any]]) -> list[str]:
    table = [
        "| Language | Locale | MDC dataset ID | Archive | Size | SHA-256 |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    table.extend(
        (
            "| {language} | `{locale}` | `{dataset_id}` | `{filename}` | "
            "{size_bytes} | `{sha256}` |".format(
                language=row.get("language", ""),
                locale=row.get("locale", ""),
                dataset_id=row.get("dataset_id", ""),
                filename=row.get("filename", ""),
                size_bytes=row.get("size_bytes", ""),
                sha256=row.get("sha256", ""),
            )
        )
        for row in archive_rows
    )
    return table


def _render_readme(
    manifest_rows: list[dict[str, Any]],
    archive_rows: list[dict[str, Any]],
) -> str:
    languages, language_bcp47 = _language_metadata(manifest_rows)
    metadata = [
        "---",
        "license: cc0-1.0",
        "language:",
        *[f"- {language}" for language in languages],
    ]
    if language_bcp47:
        metadata.extend(["language_bcp47:", *[f"- {tag}" for tag in language_bcp47]])
    metadata.extend(
        [
            "task_categories:",
            "- automatic-speech-recognition",
            "tags:",
            "- common-voice",
            "- mozilla-data-collective",
            "- scripted-speech",
            "- multilingual-asr",
            "- asr",
            "pretty_name: Common Voice Scripted Speech 25.0",
            "---",
            "",
        ],
    )
    missing_count = len(manifest_rows) - len(archive_rows)
    return "\n".join(
        [
            *metadata,
            "# Common Voice Scripted Speech 25.0",
            "",
            "This repository is being prepared as a row-normalized multilingual",
            "ASR dataset built from Mozilla Data Collective Common Voice",
            "Scripted Speech 25.0. The normalized rows are the main deliverable:",
            "`audio`, `sentence`, `locale`, `language`, `split`,",
            "`source_dataset_id`, `source_archive`, and upstream Common Voice",
            "metadata where present.",
            "",
            "During staging, the original MDC `.tar.gz` archives are preserved",
            "with checksums as source material. They provide provenance and make",
            "the normalization reproducible; the final user-facing dataset should",
            "load like other Hugging Face ASR datasets, with language and split",
            "fields available for filtering.",
            "",
            "## Status",
            "",
            f"- Manifest entries: {len(manifest_rows)}",
            f"- Archives staged: {len(archive_rows)}",
            f"- Archives still downloading or missing locally: {missing_count}",
            "- Current staging layer: original MDC `.tar.gz` archives plus",
            "  manifest metadata.",
            "- Target dataset layer: normalized rows across all downloaded",
            "  languages, preserving upstream `train`, `dev`, `test`,",
            "  `validated`, `invalidated`, `other`, and reporting files where",
            "  present.",
            "",
            "## Why this exists",
            "",
            "Mozilla moved recent Common Voice downloads to Mozilla Data Collective.",
            "This mirror does the collection and bookkeeping work once, with",
            "checksums and source dataset IDs, giving ASR users one documented",
            "place to retrieve the combined collection.",
            "",
            "## Prior Art",
            "",
            "- `mozilla-foundation/common_voice_17_0` points users to Mozilla Data",
            "  Collective for",
            "  post-October-2025 Common Voice access.",
            "- `legacy-datasets/common_voice` is the older Hugging Face packaged",
            "  Common Voice dataset and uses `cc0-1.0` metadata.",
            "- Many community repos publish per-language or processed Common",
            "  Voice variants. This repo is focused on one combined, traceable,",
            "  row-normalized Common Voice Scripted Speech 25.0 dataset.",
            "",
            "## Dataset Shape",
            "",
            "The normalized table should use one row per audio clip and keep the",
            "upstream split name in a `split` or `original_split` field. Expected",
            "columns are:",
            "",
            "- `audio`",
            "- `sentence`",
            "- `locale`",
            "- `language`",
            "- `split` / `original_split`",
            "- `source_dataset_id`",
            "- `source_archive`",
            "- upstream Common Voice metadata such as `client_id`, `sentence_id`,",
            "  `sentence_domain`, votes, age, gender, accents, variant, and",
            "  segment when present",
            "- `duration_ms` from `clip_durations.tsv` when available",
            "",
            "## Quality Views",
            "",
            "Quality filtering should be published as optional views or manifests",
            "layered on top of the canonical normalized rows. The local",
            "`omni-curator` notes treat scripted Common Voice like read or",
            "broadcast speech: Scribe WER tiers of excellent <= 5%, good <= 15%,",
            "and acceptable <= 25%, plus duration and characters-per-second",
            "backstops for obvious audio/text mismatch. Those filters are useful",
            "for training recipes, while the base dataset should keep upstream",
            "split and validation state visible.",
            "",
            "## Archives",
            "",
            *_archive_table(archive_rows),
            "",
            "## License",
            "",
            "The Mozilla Data Collective records for these archives report",
            "Creative Commons Zero v1.0 Universal (`CC0-1.0`). See `LICENSE`",
            "and https://spdx.org/licenses/CC0-1.0.html.",
            "",
        ],
    )


def _stage_rows(
    rows: list[dict[str, Any]],
    archive_dir: Path,
    archive_out: Path,
    inventory: dict[str, Any],
) -> list[dict[str, Any]]:
    staged_rows: list[dict[str, Any]] = []
    for row in rows:
        source = archive_dir / row["filename"]
        if not source.exists():
            LOGGER.info("missing: %s", source)
            continue
        destination = archive_out / row["filename"]
        LOGGER.info("stage: %s -> %s", source, destination)
        _link_or_copy(source, destination)
        staged_rows.append(
            {
                **row,
                "path": f"archives/{row['filename']}",
                "size_bytes": source.stat().st_size,
                "sha256": _digest_for_archive(source),
                "inventory": inventory.get(row["filename"], {}),
            },
        )
    return staged_rows


def main() -> None:
    """Stage the current local dataset state for a Hub upload."""
    _configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVES)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--local-readme", type=Path, default=DEFAULT_LOCAL_README)
    args = parser.parse_args()

    rows = _load_jsonl(args.manifest)
    inventory = _read_inventory(args.inventory)
    archive_out = args.out_dir / "archives"
    staged_rows = _stage_rows(rows, args.archive_dir, archive_out, inventory)
    readme = _render_readme(rows, staged_rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "README.md").write_text(readme, encoding="utf-8")
    args.local_readme.parent.mkdir(parents=True, exist_ok=True)
    args.local_readme.write_text(readme, encoding="utf-8")
    (args.out_dir / "LICENSE").write_text(LICENSE_TEXT, encoding="utf-8")
    (args.out_dir / "summary.json").write_text(
        json.dumps(
            {
                "repo_id": REPO_ID,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source_collection": "Common Voice Scripted Speech 25.0",
                "manifest_count": len(rows),
                "archive_count": len(staged_rows),
                "missing_count": len(rows) - len(staged_rows),
                "archives": staged_rows,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    LOGGER.info("staged %s archives in %s", len(staged_rows), args.out_dir)


if __name__ == "__main__":
    main()
