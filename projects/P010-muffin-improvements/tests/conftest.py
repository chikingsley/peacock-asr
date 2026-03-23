"""Shared test fixtures for P010.

Data fixture reads P010_FEATURES_DIR from .env or environment, downloads data
if needed, and fails clearly if the path is not configured at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _sentinel(features_dir: Path) -> Path:
    return features_dir / "seq_data_librispeech_v4" / "tr_feat.npy"


@pytest.fixture(scope="session")
def features_dir() -> Path:
    """Return the SpeechOcean762 features directory, downloading if needed.

    Reads P010_FEATURES_DIR from the environment or .env file (via pydantic-settings).
    If data is not downloaded yet, triggers download automatically.
    Fails clearly if P010_FEATURES_DIR is not configured.
    """
    try:
        from p010.settings import Settings
        d = Settings().features_dir  # type: ignore[call-arg]  # reads P010_FEATURES_DIR from .env
    except Exception:
        pytest.fail(
            "P010_FEATURES_DIR is not configured.\n"
            "Add it to .env or set the environment variable, then run:\n"
            "    uv run p010 download",
            pytrace=False,
        )

    if not _sentinel(d).exists():
        print(f"\nData not found at {d} — downloading...")
        from p010.data import download_features
        download_features(d)

    assert _sentinel(d).exists(), f"Download completed but sentinel not found: {_sentinel(d)}"
    return d
