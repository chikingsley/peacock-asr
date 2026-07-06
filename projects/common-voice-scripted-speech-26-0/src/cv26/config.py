"""Shared paths, secrets, and logging — the one place each was previously reimplemented.

Every other module imports paths and env lookups from here instead of redefining ``.env``
parsing, API-key resolution, or the data-dir layout. Paths hang off the ``data`` symlink so the
project stays portable: it currently points at ``/mnt/tiny-2t/peacock-asr/...``.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

CHUNK_SIZE = 1024 * 1024

PROJECT = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parents[4]

MANIFEST = PROJECT / "manifests" / "datasets.jsonl"
DATA = PROJECT / "data"
ARCHIVES = DATA / "raw" / "archives"
PARQUET_ROOT = DATA / "hf-parquet"
REPORTS = DATA / "reports"
DOWNLOAD_REPORT = REPORTS / "downloads.jsonl"
PROCESSING_STATE = PARQUET_ROOT / "processing-state.jsonl"

# Target Hub dataset (public; Common Voice Scripted Speech 26.0).
REPO_ID = "Peacockery/common-voice-scripted-speech-26"

_DOTENV_PATHS = (ROOT / ".env", PROJECT / ".env")
MDC_API_KEY_NAMES = ("MDC_API_KEY", "MOZILLA_DATA_COLLECTIVE_API_KEY")
HF_TOKEN_NAME = "HF_TOKEN"  # noqa: S105


def configure_logging() -> None:
    """Configure concise, message-only logs and quiet the noisy HTTP libraries."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


def _dotenv_values() -> dict[str, str]:
    values: dict[str, str] = {}
    for path in _DOTENV_PATHS:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                values[key.strip().removeprefix("export ").strip()] = value.strip().strip("'\"")
    return values


def env_value(*names: str) -> str:
    """Resolve the first non-empty value across env vars then ``.env`` files (empty if none)."""
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


def require_mdc_key() -> str:
    """Return the Mozilla Data Collective API key or exit with guidance."""
    key = env_value(*MDC_API_KEY_NAMES)
    if not key:
        raise SystemExit(
            "Set MDC_API_KEY or MOZILLA_DATA_COLLECTIVE_API_KEY in the environment or an "
            "ignored .env file.",
        )
    return key


def require_hf_token() -> str:
    """Return the Hugging Face token or exit with guidance."""
    token = env_value(HF_TOKEN_NAME)
    if not token:
        raise SystemExit("HF_TOKEN missing from environment or .env")
    return token


def sha256_file(path: Path) -> str:
    """Stream a file through SHA-256 and return the hex digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()
