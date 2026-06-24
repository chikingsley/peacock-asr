"""The manifest model and its loader — one definition shared by every stage.

``manifests/datasets.jsonl`` holds one Mozilla Data Collective dataset per line, written by
:mod:`cv26.queue`. Each line carries all of :class:`ManifestRow`'s fields.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

# These components are used to build (and later delete) filesystem paths, so they are validated by
# strict allowlist + `fullmatch` (not `$`, which would admit a trailing newline). This rejects
# separators, `..`, leading dots, NUL/control chars, and anything that could escape the roots.
_ID_RE = re.compile(r"[A-Za-z0-9_-]+")
_FILENAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\.tar\.gz")


class ManifestError(ValueError):
    """Raised when a manifest line is malformed, unsafe, or duplicates another."""


@dataclass(frozen=True, slots=True)
class ManifestRow:
    """One Mozilla Data Collective Common Voice archive."""

    dataset_id: str
    name: str
    collection: str
    locale: str
    language: str
    filename: str
    license: str
    license_url: str
    source: str
    download_api: str

    def __post_init__(self) -> None:
        """Reject path components that could escape the archive/parquet roots."""
        if not _FILENAME_RE.fullmatch(self.filename) or ".." in self.filename:
            raise ManifestError(f"unsafe archive filename: {self.filename!r}")
        if not _ID_RE.fullmatch(self.dataset_id):
            raise ManifestError(f"unsafe dataset_id: {self.dataset_id!r}")
        if not _ID_RE.fullmatch(self.locale):
            raise ManifestError(f"unsafe locale: {self.locale!r}")


_FIELD_NAMES = frozenset(f.name for f in fields(ManifestRow))


def load_manifest(path: Path) -> list[ManifestRow]:
    """Parse the JSONL manifest into rows, raising on missing/extra fields."""
    rows: list[ManifestRow] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            data = json.loads(raw_line)
            extra = data.keys() - _FIELD_NAMES
            missing = _FIELD_NAMES - data.keys()
            if extra or missing:
                msg = f"{path}:{line_number} bad manifest row (missing={missing}, extra={extra})"
                raise ManifestError(msg)
            rows.append(ManifestRow(**data))
    return rows


def select_rows(
    rows: Iterable[ManifestRow],
    *,
    dataset_ids: Iterable[str] = (),
    locales: Iterable[str] = (),
    limit: int = 0,
) -> list[ManifestRow]:
    """Filter rows by dataset ID and/or locale, preserving manifest order."""
    wanted_ids = set(dataset_ids)
    wanted_locales = set(locales)
    selected = [
        row
        for row in rows
        if (not wanted_ids or row.dataset_id in wanted_ids)
        and (not wanted_locales or row.locale in wanted_locales)
    ]
    if wanted_ids:
        missing = sorted(wanted_ids - {row.dataset_id for row in selected})
        if missing:
            raise ManifestError(f"Dataset IDs not in manifest: {', '.join(missing)}")
    return selected[:limit] if limit else selected


def append_rows(path: Path, rows: Iterable[ManifestRow]) -> None:
    """Append rows to the manifest as compact JSONL."""
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            payload = {f.name: getattr(row, f.name) for f in fields(row)}
            handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
