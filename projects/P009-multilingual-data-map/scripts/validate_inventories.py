#!/usr/bin/env python3
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INVENTORIES_DIR = PROJECT_ROOT / "inventories"
DRAFTS_DIR = INVENTORIES_DIR / "drafts"
LANGUAGE_NOTES_DIR = PROJECT_ROOT / "docs" / "languages"

DATASET_HEADER = [
    "language",
    "dataset_name",
    "hours_used_in_nvidia_recipe",
    "estimated_total_public_hours",
    "access_class",
    "role",
    "source_url",
    "notes",
]
VENDOR_HEADER = ["vendor", "access_class", "languages", "scope", "url", "notes"]

ALLOWED_ACCESS_CLASSES = {
    "public_open",
    "licensed",
    "commercial",
    "needs_audit",
}
ALLOWED_ROLES = {
    "nvidia_seed_recipe",
    "public_replacement_candidate",
    "public_scale_candidate",
    "eval_only",
    "commercial_option",
    "licensed_option",
}


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"{path} is missing a header row")
        rows = [{key: (value or "").strip() for key, value in row.items()} for row in reader]
        return list(reader.fieldnames), rows


def is_numeric_or_blank(value: str) -> bool:
    if not value:
        return True
    try:
        float(value)
    except ValueError:
        return False
    return True


def is_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def validate_dataset_file(path: Path, expected_language: str | None = None) -> list[str]:
    errors: list[str] = []
    header, rows = read_tsv(path)
    if header != DATASET_HEADER:
        errors.append(
            f"{path.relative_to(PROJECT_ROOT)} header mismatch: expected {DATASET_HEADER}, got {header}"
        )
        return errors

    seen = Counter()
    for index, row in enumerate(rows, start=2):
        row_id = f"{path.relative_to(PROJECT_ROOT)}:{index}"
        language = row["language"]
        dataset_name = row["dataset_name"]
        key = (language, dataset_name)
        seen[key] += 1

        if not language:
            errors.append(f"{row_id} missing language")
        if expected_language and language != expected_language:
            errors.append(f"{row_id} expected language {expected_language!r}, got {language!r}")
        if not dataset_name:
            errors.append(f"{row_id} missing dataset_name")
        if row["access_class"] not in ALLOWED_ACCESS_CLASSES:
            errors.append(f"{row_id} invalid access_class {row['access_class']!r}")
        if row["role"] not in ALLOWED_ROLES:
            errors.append(f"{row_id} invalid role {row['role']!r}")
        if not is_numeric_or_blank(row["hours_used_in_nvidia_recipe"]):
            errors.append(f"{row_id} non-numeric hours_used_in_nvidia_recipe {row['hours_used_in_nvidia_recipe']!r}")
        if not is_numeric_or_blank(row["estimated_total_public_hours"]):
            errors.append(
                f"{row_id} non-numeric estimated_total_public_hours {row['estimated_total_public_hours']!r}"
            )
        if not row["source_url"] or not is_http_url(row["source_url"]):
            errors.append(f"{row_id} invalid source_url {row['source_url']!r}")
        if not row["notes"]:
            errors.append(f"{row_id} missing notes")

    for key, count in seen.items():
        if count > 1:
            errors.append(f"{path.relative_to(PROJECT_ROOT)} duplicate dataset row for {key}")

    return errors


def validate_vendor_file(path: Path) -> list[str]:
    errors: list[str] = []
    header, rows = read_tsv(path)
    if header != VENDOR_HEADER:
        errors.append(
            f"{path.relative_to(PROJECT_ROOT)} header mismatch: expected {VENDOR_HEADER}, got {header}"
        )
        return errors

    seen = Counter()
    for index, row in enumerate(rows, start=2):
        row_id = f"{path.relative_to(PROJECT_ROOT)}:{index}"
        vendor = row["vendor"]
        seen[vendor] += 1

        if not vendor:
            errors.append(f"{row_id} missing vendor")
        if row["access_class"] not in ALLOWED_ACCESS_CLASSES - {"public_open", "needs_audit"}:
            errors.append(f"{row_id} invalid vendor access_class {row['access_class']!r}")
        if not row["languages"]:
            errors.append(f"{row_id} missing languages")
        if not row["scope"]:
            errors.append(f"{row_id} missing scope")
        if not row["url"] or not is_http_url(row["url"]):
            errors.append(f"{row_id} invalid url {row['url']!r}")
        if not row["notes"]:
            errors.append(f"{row_id} missing notes")

    for vendor, count in seen.items():
        if count > 1:
            errors.append(f"{path.relative_to(PROJECT_ROOT)} duplicate vendor row for {vendor}")

    return errors


def validate_seed_subset(seed_rows: list[dict[str, str]], draft_rows: list[dict[str, str]]) -> list[str]:
    errors: list[str] = []
    draft_map = {(row["language"], row["dataset_name"]): row for row in draft_rows}

    for row in seed_rows:
        key = (row["language"], row["dataset_name"])
        draft_row = draft_map.get(key)
        if draft_row is None:
            errors.append(f"seed row {key} is missing from draft inventories")
            continue
        for field in DATASET_HEADER:
            if row[field] != draft_row[field]:
                errors.append(
                    f"seed row {key} disagrees with draft field {field!r}: "
                    f"seed={row[field]!r} draft={draft_row[field]!r}"
                )
    return errors


def main() -> int:
    errors: list[str] = []
    draft_rows: list[dict[str, str]] = []

    seed_header, seed_rows = read_tsv(INVENTORIES_DIR / "seed_datasets.tsv")
    if seed_header != DATASET_HEADER:
        errors.append(
            "inventories/seed_datasets.tsv header mismatch: "
            f"expected {DATASET_HEADER}, got {seed_header}"
        )
    else:
        errors.extend(validate_dataset_file(INVENTORIES_DIR / "seed_datasets.tsv"))

    draft_files = sorted(path for path in DRAFTS_DIR.glob("*.tsv") if path.name != ".gitkeep")
    if not draft_files:
        errors.append("inventories/drafts has no language TSV files")

    for draft_file in draft_files:
        language = draft_file.stem
        errors.extend(validate_dataset_file(draft_file, expected_language=language))
        _, rows = read_tsv(draft_file)
        draft_rows.extend(rows)

        note_path = LANGUAGE_NOTES_DIR / f"{language}.md"
        if not note_path.exists():
            errors.append(f"missing language note for draft {draft_file.name}: {note_path.relative_to(PROJECT_ROOT)}")

    errors.extend(validate_vendor_file(INVENTORIES_DIR / "vendor_sources.tsv"))
    if seed_header == DATASET_HEADER:
        errors.extend(validate_seed_subset(seed_rows, draft_rows))

    if errors:
        print("Inventory validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print(
        "Inventory validation passed "
        f"({len(seed_rows)} seed rows, {len(draft_files)} draft files, {len(draft_rows)} draft rows)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
