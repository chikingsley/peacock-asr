"""Queue MDC dataset snippets, refresh HF metadata, and upload lightweight files.

# /// script
# dependencies = ["huggingface-hub>=1.0.0"]
# ///
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import re
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
PROJECT = Path(__file__).resolve().parents[1]
REPO_ID = "Peacockery/common-voice-scripted-speech-25-0"
MANIFEST = PROJECT / "manifests" / "datasets.jsonl"
STAGE_SCRIPT = PROJECT / "scripts" / "stage_hf_release.py"
STAGE_ROOT = (
    Path("/mnt/overflow/peacock-asr")
    / "common-voice-scripted-speech-25-0"
    / "hf-staging"
)
ARCHIVE_DIR = PROJECT / "data" / "raw" / "archives"
DOTENV_PATHS = (ROOT / ".env", PROJECT / ".env")
MDC_API_KEY_NAMES = ("MDC_API_KEY", "MOZILLA_DATA_COLLECTIVE_API_KEY")
HF_TOKEN_ENV = "HF_TOKEN"  # noqa: S105
CC0 = "CC0-1.0"
DATASET_RE = re.compile(r"/api/datasets/([^/]+)/download")
FILENAME_RE = re.compile(r"curl\s+-o\s+[\"']([^\"']+\.tar\.gz)[\"']")
LOGGER = logging.getLogger("queue_mdc_datasets")


@dataclass(frozen=True, slots=True)
class QueueItem:
    """A user-provided MDC dataset ID and expected archive filename."""

    dataset_id: str
    filename: str


@dataclass(frozen=True, slots=True)
class QueueResult:
    """Result of checking one MDC dataset against the manifest."""

    item: QueueItem
    row: dict[str, str]
    status: str


class QueueError(RuntimeError):
    """Raised when a queued snippet cannot be safely added."""


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


def _dotenv_values() -> dict[str, str]:
    values: dict[str, str] = {}
    for path in DOTENV_PATHS:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                values[key.strip().removeprefix("export ").strip()] = (
                    value.strip().strip("'\"")
                )
    return values


def _env_value(names: tuple[str, ...] | str) -> str:
    if isinstance(names, str):
        names = (names,)
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    dotenv = _dotenv_values()
    for name in names:
        value = dotenv.get(name)
        if value:
            return value
    return ""


def _read_input(path: Path | None) -> str:
    if path is not None:
        return path.read_text(encoding="utf-8")
    return sys.stdin.read()


def _parse_items(text: str) -> list[QueueItem]:
    dataset_ids = DATASET_RE.findall(text)
    filenames = FILENAME_RE.findall(text)
    if len(dataset_ids) != len(filenames):
        message = (
            f"Found {len(dataset_ids)} dataset IDs and {len(filenames)} filenames; "
            "paste complete curl snippets so they can be paired safely."
        )
        raise QueueError(message)
    if not dataset_ids:
        message = "No MDC dataset snippets found."
        raise QueueError(message)
    return [
        QueueItem(dataset_id=dataset_id, filename=filename)
        for dataset_id, filename in zip(dataset_ids, filenames, strict=True)
    ]


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped_line = raw_line.strip()
            if stripped_line:
                rows.append(json.loads(stripped_line))
    return rows


def _row_for_payload(payload: dict[str, Any]) -> dict[str, str]:
    name = str(payload["name"])
    language = name.rsplit(" - ", 1)[-1]
    dataset_id = str(payload["id"])
    return {
        "dataset_id": dataset_id,
        "name": name,
        "collection": "Common Voice Scripted Speech 25.0",
        "locale": str(payload["locale"]),
        "language": language,
        "filename": str(payload["filename"]),
        "license": CC0,
        "license_url": "https://spdx.org/licenses/CC0-1.0.html",
        "source": "Mozilla Data Collective",
        "download_api": (
            f"https://mozilladatacollective.com/api/datasets/{dataset_id}/download"
        ),
    }


def _fetch_metadata(item: QueueItem, token: str) -> dict[str, Any]:
    request = urllib.request.Request(
        f"https://mozilladatacollective.com/api/datasets/{item.dataset_id}",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def _validate_metadata(item: QueueItem, payload: dict[str, Any]) -> None:
    filename = payload.get("filename")
    if filename != item.filename:
        message = (
            f"{item.dataset_id} filename mismatch: pasted {item.filename}, "
            f"MDC returned {filename}"
        )
        raise QueueError(message)
    license_id = payload.get("licenseAbbreviation")
    if license_id != CC0:
        message = f"{item.dataset_id} license mismatch: MDC returned {license_id}"
        raise QueueError(message)


def _append_rows(items: list[QueueItem], token: str) -> list[QueueResult]:
    rows = _load_manifest(MANIFEST)
    by_id = {str(row["dataset_id"]): row for row in rows}
    by_filename = {str(row["filename"]): row for row in rows}
    results: list[QueueResult] = []
    new_rows: list[dict[str, str]] = []

    for item in items:
        payload = _fetch_metadata(item, token)
        _validate_metadata(item, payload)
        row = _row_for_payload(payload)

        existing_id = by_id.get(item.dataset_id)
        existing_filename = by_filename.get(item.filename)
        if existing_id is not None or existing_filename is not None:
            if existing_id != existing_filename:
                message = (
                    f"{item.dataset_id} conflicts with existing manifest state "
                    f"for filename {item.filename}"
                )
                raise QueueError(message)
            results.append(QueueResult(item=item, row=row, status="exists"))
            continue

        new_rows.append(row)
        by_id[item.dataset_id] = row
        by_filename[item.filename] = row
        results.append(QueueResult(item=item, row=row, status="added"))

    if new_rows:
        with MANIFEST.open("a", encoding="utf-8") as handle:
            for row in new_rows:
                handle.write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n",
                )
    return results


def _validate_manifest() -> tuple[int, int, int]:
    rows = _load_manifest(MANIFEST)
    ids = [str(row["dataset_id"]) for row in rows]
    filenames = [str(row["filename"]) for row in rows]
    duplicate_ids = len(ids) - len(set(ids))
    duplicate_filenames = len(filenames) - len(set(filenames))
    if duplicate_ids or duplicate_filenames:
        message = (
            f"Manifest duplicates after append: ids={duplicate_ids}, "
            f"filenames={duplicate_filenames}"
        )
        raise QueueError(message)
    return len(rows), duplicate_ids, duplicate_filenames


def _stage_metadata() -> dict[str, Any]:
    subprocess.run(  # noqa: S603
        [sys.executable, str(STAGE_SCRIPT)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    summary_path = STAGE_ROOT / "summary.json"
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _upload_metadata(token: str) -> list[str]:
    hub = importlib.import_module("huggingface_hub")
    api = hub.HfApi(token=token)
    uploaded: list[str] = []
    for local, remote in [
        (STAGE_ROOT / "README.md", "README.md"),
        (STAGE_ROOT / "LICENSE", "LICENSE"),
        (STAGE_ROOT / "summary.json", "summary.json"),
        (MANIFEST, "manifests/datasets.jsonl"),
    ]:
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=remote,
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message="Refresh Common Voice Scripted Speech 25.0 metadata",
        )
        uploaded.append(remote)
    return uploaded


def _print_results(
    results: list[QueueResult],
    manifest_count: int,
    summary: dict[str, Any],
    uploaded: list[str],
) -> None:
    for result in results:
        row = result.row
        LOGGER.info(
            "%s: %s %s %s %s",
            result.status,
            row["dataset_id"],
            row["locale"],
            row["language"],
            row["filename"],
        )
    LOGGER.info("manifest_rows=%s", manifest_count)
    LOGGER.info("archives_staged=%s", summary["archive_count"])
    LOGGER.info("missing_archives=%s", summary["missing_count"])
    if uploaded:
        LOGGER.info("uploaded=%s", ", ".join(uploaded))


def main() -> None:
    """Queue MDC snippets and refresh local/HF metadata."""
    _configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", type=Path)
    parser.add_argument("--no-upload", action="store_true")
    args = parser.parse_args()

    items = _parse_items(_read_input(args.input))
    mdc_token = _env_value(MDC_API_KEY_NAMES)
    if not mdc_token:
        message = "MDC_API_KEY missing from environment or .env"
        raise SystemExit(message)

    results = _append_rows(items, mdc_token)
    manifest_count, _, _ = _validate_manifest()
    summary = _stage_metadata()

    uploaded: list[str] = []
    if not args.no_upload:
        hf_token = _env_value(HF_TOKEN_ENV)
        if hf_token:
            uploaded = _upload_metadata(hf_token)
        else:
            LOGGER.info("HF_TOKEN missing; skipped upload")

    _print_results(results, manifest_count, summary, uploaded)


if __name__ == "__main__":
    try:
        main()
    except QueueError as exc:
        raise SystemExit(str(exc)) from exc
