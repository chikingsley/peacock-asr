"""Download Mozilla Data Collective archives listed in the manifest.

POST to a dataset's ``/download`` endpoint with the Bearer key for a short-lived presigned URL,
then stream the tarball with resume support. Rate limits (429) and transient 5xx/network errors
back off and retry; the resolved download URL is refreshed on each archive attempt because it
expires. Each completed archive is recorded in :data:`cv26.config.DOWNLOAD_REPORT`.
"""

from __future__ import annotations

import email.utils
import json
import logging
import os
import shutil
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from cv26 import config

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from cv26.manifest import ManifestRow

LOGGER = logging.getLogger("cv26.download")

CHUNK_SIZE = config.CHUNK_SIZE
DEFAULT_WORKERS = 2
# Pause downloading below this much free space so the process loop can drain archives first
# (archives are multi-GB; this leaves room for the processor's transient shards too).
MIN_FREE_BYTES = 50 * 1024**3
DEFAULT_RETRIES = 20
DEFAULT_MAX_SLEEP_SECONDS = 24 * 60 * 60
IDLE_SLEEP_SECONDS = 30
REQUEST_TIMEOUT_SECONDS = 60
DOWNLOAD_TIMEOUT_SECONDS = 120
HTTP_PARTIAL_CONTENT = 206
HTTP_TOO_MANY_REQUESTS = 429
HTTP_NOT_FOUND = 404
HTTP_FORBIDDEN = 403
HTTP_SERVER_ERRORS = frozenset({500, 502, 503, 504})
RATE_LIMIT_BASE_SECONDS = 60.0
RATE_LIMIT_BUFFER_SECONDS = 5.0
RETRY_BASE_SECONDS = 30.0


class DownloadError(RuntimeError):
    """Raised when an MDC request or archive download cannot be completed."""


@dataclass(frozen=True, slots=True)
class DownloadConfig:
    """Settings shared by download worker threads."""

    token: str
    out_dir: Path
    report_path: Path
    retries: int = DEFAULT_RETRIES
    max_sleep: int = DEFAULT_MAX_SLEEP_SECONDS
    workers: int = DEFAULT_WORKERS
    force: bool = False


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    stripped = value.strip()
    if stripped.isdigit():
        return float(stripped)
    try:
        parsed = email.utils.parsedate_to_datetime(stripped)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return max(0.0, (parsed - datetime.now(UTC)).total_seconds())


def _parse_resets_at(body: str) -> float | None:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return None
    resets_at = payload.get("limit", {}).get("resetsAt")
    if not isinstance(resets_at, str):
        return None
    try:
        parsed = datetime.fromisoformat(resets_at)
    except ValueError:
        return None
    return max(0.0, (parsed - datetime.now(UTC)).total_seconds())


def _retry_delay(
    exc: urllib.error.HTTPError,
    body: str,
    attempt: int,
    max_sleep: int,
) -> float | None:
    if exc.code == HTTP_TOO_MANY_REQUESTS:
        delay = _parse_resets_at(body) or _parse_retry_after(exc.headers.get("Retry-After"))
        if delay is None:
            delay = min(RATE_LIMIT_BASE_SECONDS * attempt, float(max_sleep))
        return min(delay + RATE_LIMIT_BUFFER_SECONDS, float(max_sleep))
    if exc.code in HTTP_SERVER_ERRORS:
        return min(RETRY_BASE_SECONDS * attempt, float(max_sleep))
    return None


def _resolve_url(dataset_id: str, cfg: DownloadConfig) -> str:
    endpoint = f"https://mozilladatacollective.com/api/datasets/{dataset_id}/download"
    payload: dict[str, object] | None = None
    for attempt in range(1, cfg.retries + 2):
        request = urllib.request.Request(  # noqa: S310 — fixed https MDC endpoint
            endpoint,
            data=b"{}",
            headers={"Authorization": f"Bearer {cfg.token}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:  # noqa: S310
                payload = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            delay = _retry_delay(exc, body, attempt, cfg.max_sleep)
            if delay is None or attempt > cfg.retries:
                msg = f"MDC download URL request failed for {dataset_id}: {exc.code} {body}"
                raise DownloadError(msg) from exc
            LOGGER.info(
                "  MDC returned %s; sleeping %.0fs before retry (%s/%s)",
                exc.code,
                delay,
                attempt,
                cfg.retries,
            )
            time.sleep(delay)

    url = (payload or {}).get("downloadUrl")
    if not isinstance(url, str) or not url:
        raise DownloadError(f"MDC response for {dataset_id} did not include downloadUrl")
    if urllib.parse.urlparse(url).scheme != "https":
        raise DownloadError(f"MDC response for {dataset_id} returned a non-HTTPS URL")
    return url


def _download_once(url: str, destination: Path) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.parent / f".{destination.name}.part"
    resume_at = tmp_path.stat().st_size if tmp_path.exists() else 0
    headers = {"Range": f"bytes={resume_at}-"} if resume_at else {}
    request = urllib.request.Request(url, headers=headers)  # noqa: S310 — presigned https URL
    with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:  # noqa: S310
        status = getattr(response, "status", response.getcode())
        mode = "ab" if resume_at and status == HTTP_PARTIAL_CONTENT else "wb"
        if resume_at and status != HTTP_PARTIAL_CONTENT:
            LOGGER.info("  server ignored Range; restarting partial file")
        with tmp_path.open(mode) as tmp:
            while chunk := response.read(CHUNK_SIZE):
                tmp.write(chunk)
            tmp.flush()
            os.fsync(tmp.fileno())
    size = tmp_path.stat().st_size
    tmp_path.replace(destination)
    return size


def _download_with_retries(row: ManifestRow, cfg: DownloadConfig) -> tuple[int, str]:
    destination = cfg.out_dir / row.filename
    for attempt in range(1, cfg.retries + 2):
        try:
            url = _resolve_url(row.dataset_id, cfg)
            size = _download_once(url, destination)
            digest = config.sha256_file(destination)
            destination.with_suffix(destination.suffix + ".sha256").write_text(
                f"{digest}  {destination.name}\n",
                encoding="utf-8",
            )
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            delay = _retry_delay(exc, body, attempt, cfg.max_sleep)
            if delay is None and exc.code in {HTTP_FORBIDDEN, HTTP_NOT_FOUND}:
                delay = min(RETRY_BASE_SECONDS * attempt, float(cfg.max_sleep))
            if delay is None or attempt > cfg.retries:
                raise
            LOGGER.info(
                "  download returned %s; refreshing URL after %.0fs (%s/%s)",
                exc.code,
                delay,
                attempt,
                cfg.retries,
            )
            time.sleep(delay)
        except (TimeoutError, urllib.error.URLError) as exc:
            if attempt > cfg.retries:
                raise
            delay = min(RETRY_BASE_SECONDS * attempt, float(cfg.max_sleep))
            LOGGER.info(
                "  download interrupted: %s; refreshing URL after %.0fs (%s/%s)",
                exc,
                delay,
                attempt,
                cfg.retries,
            )
            time.sleep(delay)
        else:
            return size, digest
    raise DownloadError(f"exhausted retries for {row.dataset_id}")


def _append_report(path: Path, event: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")


def _process_row(row: ManifestRow, cfg: DownloadConfig, report_lock: threading.Lock) -> str:
    archive = cfg.out_dir / row.filename
    LOGGER.info("%s -> %s", row.dataset_id, archive.name)
    if archive.exists() and not cfg.force:
        LOGGER.info("  exists; skipping")
        return "skipped"
    try:
        size, digest = _download_with_retries(row, cfg)
    except (DownloadError, OSError) as exc:
        # One dataset failing (e.g. 403 access-not-granted, 404, exhausted retries) must not kill
        # the whole watcher — log it, record it, and let the loop move on to the rest.
        LOGGER.warning("  FAILED %s (%s); skipping", row.dataset_id, exc)
        with report_lock:
            _append_report(
                cfg.report_path,
                {
                    "dataset_id": row.dataset_id,
                    "filename": row.filename,
                    "status": "failed",
                    "error": str(exc)[:300],
                    "failed_at": datetime.now(UTC).isoformat(),
                },
            )
        return "failed"
    with report_lock:
        _append_report(
            cfg.report_path,
            {
                "dataset_id": row.dataset_id,
                "filename": row.filename,
                "locale": row.locale,
                "language": row.language,
                "downloaded_at": datetime.now(UTC).isoformat(),
                "size_bytes": size,
                "sha256": digest,
            },
        )
    LOGGER.info("  downloaded %s bytes sha256=%s", size, digest)
    return "downloaded"


def free_bytes(path: Path) -> int:
    """Return free bytes on the filesystem holding ``path``."""
    return shutil.disk_usage(path).free


def pending_rows(rows: Iterable[ManifestRow], out_dir: Path, *, force: bool) -> list[ManifestRow]:
    """Rows whose archive is not yet on disk (or all rows when ``force``)."""
    if force:
        return list(rows)
    return [row for row in rows if not (out_dir / row.filename).exists()]


def run_batch(rows: list[ManifestRow], cfg: DownloadConfig) -> set[str]:
    """Download a batch of rows (thread pool when ``workers`` > 1); return failed dataset IDs."""
    report_lock = threading.Lock()
    failed: set[str] = set()
    if cfg.workers == 1:
        for row in rows:
            if _process_row(row, cfg, report_lock) == "failed":
                failed.add(row.dataset_id)
        return failed
    with ThreadPoolExecutor(max_workers=cfg.workers) as executor:
        future_to_row = {executor.submit(_process_row, row, cfg, report_lock): row for row in rows}
        for future in as_completed(future_to_row):
            if future.result() == "failed":
                failed.add(future_to_row[future].dataset_id)
    return failed
