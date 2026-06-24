"""Add languages to the manifest from pasted Mozilla Data Collective download snippets.

Paste one or more MDC ``curl`` download snippets (each carrying a ``/api/datasets/<id>/download``
URL and a ``-o <filename>.tar.gz``); this pairs them, fetches+validates each dataset's metadata
from MDC, and appends new rows to the manifest. Existing rows are left untouched.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.request
from dataclasses import dataclass
from typing import TYPE_CHECKING

from cv26 import config
from cv26.manifest import ManifestRow, append_rows, load_manifest

if TYPE_CHECKING:
    from pathlib import Path

LOGGER = logging.getLogger("cv26.mdc.queue")

CC0 = "CC0-1.0"
CC0_URL = "https://spdx.org/licenses/CC0-1.0.html"
_DATASET_RE = re.compile(r"/api/datasets/([^/]+)/download")
_FILENAME_RE = re.compile(r"curl\s+-o\s+[\"']([^\"']+\.tar\.gz)[\"']")


class QueueError(RuntimeError):
    """Raised when a pasted snippet cannot be safely added to the manifest."""


@dataclass(frozen=True, slots=True)
class QueueItem:
    """A pasted MDC dataset ID and its expected archive filename."""

    dataset_id: str
    filename: str


def parse_items(text: str) -> list[QueueItem]:
    """Pair dataset IDs with archive filenames from pasted curl snippets."""
    dataset_ids = _DATASET_RE.findall(text)
    filenames = _FILENAME_RE.findall(text)
    if not dataset_ids:
        raise QueueError("No MDC dataset snippets found.")
    if len(dataset_ids) != len(filenames):
        raise QueueError(
            f"Found {len(dataset_ids)} dataset IDs and {len(filenames)} filenames; "
            "paste complete curl snippets so they can be paired safely.",
        )
    return [QueueItem(d, f) for d, f in zip(dataset_ids, filenames, strict=True)]


def _fetch_metadata(item: QueueItem, token: str) -> dict[str, object]:
    request = urllib.request.Request(
        f"https://mozilladatacollective.com/api/datasets/{item.dataset_id}",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 — fixed https MDC endpoint
        return json.loads(response.read().decode("utf-8"))


def _row_from_metadata(item: QueueItem, payload: dict[str, object]) -> ManifestRow:
    if payload.get("filename") != item.filename:
        raise QueueError(
            f"{item.dataset_id} filename mismatch: pasted {item.filename}, "
            f"MDC returned {payload.get('filename')}",
        )
    license_id = payload.get("licenseAbbreviation")
    if license_id != CC0:
        raise QueueError(f"{item.dataset_id} license mismatch: MDC returned {license_id}")
    name = str(payload["name"])
    collection, _, language = name.rpartition(" - ")
    return ManifestRow(
        dataset_id=item.dataset_id,
        name=name,
        collection=collection or name,
        locale=str(payload["locale"]),
        language=language or name,
        filename=item.filename,
        license=CC0,
        license_url=CC0_URL,
        source="Mozilla Data Collective",
        download_api=f"https://mozilladatacollective.com/api/datasets/{item.dataset_id}/download",
    )


def add_from_text(text: str, *, manifest: Path | None = None) -> list[ManifestRow]:
    """Append new manifest rows for pasted snippets; return the rows actually added."""
    manifest = manifest or config.MANIFEST
    token = config.require_mdc_key()
    existing = load_manifest(manifest)
    known_ids = {row.dataset_id for row in existing}
    known_files = {row.filename for row in existing}

    added: list[ManifestRow] = []
    for item in parse_items(text):
        if item.dataset_id in known_ids or item.filename in known_files:
            LOGGER.info("exists: %s %s", item.dataset_id, item.filename)
            continue
        row = _row_from_metadata(item, _fetch_metadata(item, token))
        added.append(row)
        known_ids.add(row.dataset_id)
        known_files.add(row.filename)
        LOGGER.info("added: %s %s %s", row.dataset_id, row.locale, row.language)

    if added:
        append_rows(manifest, added)
    LOGGER.info("manifest rows: %s (+%s)", len(existing) + len(added), len(added))
    return added
