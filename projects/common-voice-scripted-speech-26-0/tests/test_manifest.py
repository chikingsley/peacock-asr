"""Manifest path-component validation: unsafe filenames/ids/locales are rejected at construction."""

from __future__ import annotations

import pytest

from cv26.manifest import ManifestError, ManifestRow


def _kwargs(**overrides: str) -> dict[str, str]:
    base = {
        "dataset_id": "ds123",
        "name": "Common Voice - Abkhaz",
        "collection": "Common Voice",
        "locale": "ab",
        "language": "Abkhaz",
        "filename": "cv.tar.gz",
        "license": "CC0-1.0",
        "license_url": "https://spdx.org/licenses/CC0-1.0.html",
        "source": "Mozilla Data Collective",
        "download_api": "https://example/api",
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize(
    "overrides",
    [
        {"filename": "../evil.tar.gz"},
        {"filename": "/abs/evil.tar.gz"},
        {"filename": "sub/dir.tar.gz"},
        {"filename": "evil.zip"},
        {"filename": "..tar.gz"},
        {"filename": "evil..tar.gz"},  # consecutive dots inside a basename
        {"filename": ".hidden.tar.gz"},
        {"filename": "sub\\evil.tar.gz"},
        {"filename": "evil\x00.tar.gz"},  # NUL byte
        {"filename": "evil\n.tar.gz"},  # newline (the fullmatch-vs-$ trap)
        {"filename": "evil\r.tar.gz"},
        {"dataset_id": "../x"},
        {"dataset_id": "ds\n"},
        {"locale": "a/b"},
        {"locale": "ab\n"},
    ],
)
def test_unsafe_components_rejected(overrides: dict[str, str]) -> None:
    """Path-escaping filenames, ids, and locales raise ManifestError."""
    with pytest.raises(ManifestError):
        ManifestRow(**_kwargs(**overrides))


def test_valid_row_accepted() -> None:
    """A normal row (including a hyphenated BCP-47 locale) constructs fine."""
    ManifestRow(**_kwargs(locale="zh-CN"))
