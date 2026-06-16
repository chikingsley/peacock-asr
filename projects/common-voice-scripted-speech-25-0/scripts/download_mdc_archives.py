"""Download Mozilla Data Collective archives from a JSONL manifest."""

from __future__ import annotations

import argparse
import email.utils
import hashlib
import json
import logging
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT / "manifests" / "datasets.jsonl"
DEFAULT_OUT = ROOT / "data" / "common-voice-scripted-speech-25-0" / "raw" / "archives"
DEFAULT_REPORT = (
    ROOT / "data" / "common-voice-scripted-speech-25-0" / "reports" / "downloads.jsonl"
)
API_KEY_NAMES = ("MDC_API_KEY", "MOZILLA_DATA_COLLECTIVE_API_KEY")
DOTENV_PATHS = (ROOT / ".env", PROJECT / ".env")

CHUNK_SIZE = 1024 * 1024
DEFAULT_WORKERS = 2
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

LOGGER = logging.getLogger("download_mdc_archives")


class ManifestError(ValueError):
    """Raised when the manifest is malformed."""


class DownloadError(RuntimeError):
    """Raised when MDC or an archive download fails."""


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Runtime settings shared by worker functions."""

    token: str
    out_dir: Path
    report_path: Path
    retries: int
    max_sleep: int
    workers: int
    force: bool = False


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            stripped_line = raw_line.strip()
            if not stripped_line:
                continue
            row = json.loads(stripped_line)
            for key in ("dataset_id", "filename"):
                if not row.get(key):
                    message = f"{path}:{line_number} missing {key}"
                    raise ManifestError(message)
            rows.append(row)
    return rows


def _dotenv_values(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip().removeprefix("export ").strip()
            if key in API_KEY_NAMES:
                values[key] = value.strip().strip("'\"")
    return values


def _api_key() -> str:
    for name in API_KEY_NAMES:
        key = os.environ.get(name)
        if key:
            return key
    for path in DOTENV_PATHS:
        values = _dotenv_values(path)
        for name in API_KEY_NAMES:
            key = values.get(name)
            if key:
                return key
    message = (
        "Set MDC_API_KEY or MOZILLA_DATA_COLLECTIVE_API_KEY in the environment "
        "or an ignored .env file."
    )
    raise SystemExit(message)


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    stripped_value = value.strip()
    if stripped_value.isdigit():
        return float(stripped_value)
    try:
        parsed = email.utils.parsedate_to_datetime(stripped_value)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return max(0.0, (parsed - datetime.now(timezone.utc)).total_seconds())


def _parse_resets_at(body: str) -> float | None:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return None
    resets_at = payload.get("limit", {}).get("resetsAt")
    if not isinstance(resets_at, str):
        return None
    try:
        parsed = datetime.fromisoformat(resets_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    return max(0.0, (parsed - datetime.now(timezone.utc)).total_seconds())


def _retry_delay(
    exc: urllib.error.HTTPError,
    body: str,
    attempt: int,
    max_sleep: int,
) -> float | None:
    if exc.code == HTTP_TOO_MANY_REQUESTS:
        delay = _parse_resets_at(body) or _parse_retry_after(
            exc.headers.get("Retry-After"),
        )
        if delay is None:
            delay = min(RATE_LIMIT_BASE_SECONDS * attempt, float(max_sleep))
        return min(delay + RATE_LIMIT_BUFFER_SECONDS, float(max_sleep))
    if exc.code in HTTP_SERVER_ERRORS:
        return min(RETRY_BASE_SECONDS * attempt, float(max_sleep))
    return None


def _post_download_url(
    dataset_id: str,
    token: str,
    retries: int,
    max_sleep: int,
) -> str:
    endpoint = f"https://mozilladatacollective.com/api/datasets/{dataset_id}/download"
    payload: dict[str, Any] | None = None
    for attempt in range(1, retries + 2):
        request = urllib.request.Request(  # noqa: S310
            endpoint,
            data=b"{}",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(  # noqa: S310
                request,
                timeout=REQUEST_TIMEOUT_SECONDS,
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            delay = _retry_delay(exc, body, attempt, max_sleep)
            if delay is None or attempt > retries:
                message = (
                    "MDC download URL request failed for "
                    f"{dataset_id}: {exc.code} {body}"
                )
                raise DownloadError(message) from exc
            LOGGER.info(
                "  MDC returned %s; sleeping %.0fs before retry (%s/%s)",
                exc.code,
                delay,
                attempt,
                retries,
            )
            time.sleep(delay)

    if payload is None:
        message = f"MDC response for {dataset_id} was empty"
        raise DownloadError(message)
    url = payload.get("downloadUrl")
    if not isinstance(url, str) or not url:
        message = f"MDC response for {dataset_id} did not include downloadUrl"
        raise DownloadError(message)
    parsed_url = urllib.parse.urlparse(url)
    if parsed_url.scheme != "https":
        message = f"MDC response for {dataset_id} returned a non-HTTPS URL"
        raise DownloadError(message)
    return url


def _download_once(url: str, destination: Path) -> tuple[int, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.parent / f".{destination.name}.part"

    resume_at = tmp_path.stat().st_size if tmp_path.exists() else 0
    headers: dict[str, str] = {}
    if resume_at:
        headers["Range"] = f"bytes={resume_at}-"
    request = urllib.request.Request(url, headers=headers)  # noqa: S310
    with urllib.request.urlopen(  # noqa: S310
        request,
        timeout=DOWNLOAD_TIMEOUT_SECONDS,
    ) as response:
        status = getattr(response, "status", response.getcode())
        mode = "ab" if resume_at and status == HTTP_PARTIAL_CONTENT else "wb"
        if resume_at and status != HTTP_PARTIAL_CONTENT:
            LOGGER.info("  server ignored Range; restarting partial file")
            resume_at = 0
        with tmp_path.open(mode) as tmp:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                tmp.write(chunk)
            tmp.flush()
            os.fsync(tmp.fileno())

    sha256 = hashlib.sha256()
    size = 0
    with tmp_path.open("rb") as handle:
        while True:
            chunk = handle.read(CHUNK_SIZE)
            if not chunk:
                break
            sha256.update(chunk)
            size += len(chunk)
    digest = sha256.hexdigest()
    tmp_path.replace(destination)
    destination.with_suffix(destination.suffix + ".sha256").write_text(
        f"{digest}  {destination.name}\n",
        encoding="utf-8",
    )
    return size, digest


def _download_with_retries(row: dict[str, Any], config: RunConfig) -> tuple[int, str]:
    for attempt in range(1, config.retries + 2):
        try:
            url = _post_download_url(
                row["dataset_id"],
                config.token,
                config.retries,
                config.max_sleep,
            )
            return _download_once(url, config.out_dir / row["filename"])
        except urllib.error.HTTPError as exc:  # noqa: PERF203
            body = exc.read().decode("utf-8", errors="replace")
            delay = _retry_delay(exc, body, attempt, config.max_sleep)
            if delay is None and exc.code in {HTTP_FORBIDDEN, HTTP_NOT_FOUND}:
                delay = min(RETRY_BASE_SECONDS * attempt, float(config.max_sleep))
            if delay is None or attempt > config.retries:
                raise
            LOGGER.info(
                "  download returned %s; refreshing URL after %.0fs (%s/%s)",
                exc.code,
                delay,
                attempt,
                config.retries,
            )
            time.sleep(delay)
        except (TimeoutError, urllib.error.URLError) as exc:
            if attempt > config.retries:
                raise
            delay = min(RETRY_BASE_SECONDS * attempt, float(config.max_sleep))
            LOGGER.info(
                "  download interrupted: %s; refreshing URL after %.0fs (%s/%s)",
                exc,
                delay,
                attempt,
                config.retries,
            )
            time.sleep(delay)
    message = f"exhausted retries for {row['dataset_id']}"
    raise DownloadError(message)


def _append_report(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _select_rows(
    rows: list[dict[str, Any]],
    dataset_ids: list[str],
) -> list[dict[str, Any]]:
    if not dataset_ids:
        return rows
    requested_ids = set(dataset_ids)
    selected_rows = [row for row in rows if row["dataset_id"] in requested_ids]
    found_ids = {row["dataset_id"] for row in selected_rows}
    missing_ids = sorted(requested_ids - found_ids)
    if missing_ids:
        message = f"Dataset IDs not found in manifest: {', '.join(missing_ids)}"
        raise SystemExit(message)
    return selected_rows


def _pending_rows(
    rows: list[dict[str, Any]],
    out_dir: Path,
    *,
    force: bool,
) -> list[dict[str, Any]]:
    if force:
        return rows
    return [row for row in rows if not (out_dir / row["filename"]).exists()]


def _process_row(
    row: dict[str, Any],
    config: RunConfig,
    report_lock: threading.Lock,
) -> str:
    archive = config.out_dir / row["filename"]
    LOGGER.info("%s -> %s", row["dataset_id"], archive)
    if archive.exists() and not config.force:
        LOGGER.info("  exists; skipping")
        return "skipped"

    size, digest = _download_with_retries(row, config)
    report = {
        **row,
        "archive_path": str(archive.relative_to(ROOT)),
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "size_bytes": size,
        "sha256": digest,
    }
    with report_lock:
        _append_report(config.report_path, report)
    LOGGER.info("  downloaded %s bytes sha256=%s", size, digest)
    return "downloaded"


def _run_candidates(rows: list[dict[str, Any]], config: RunConfig) -> None:
    report_lock = threading.Lock()
    if config.workers == 1:
        for row in rows:
            _process_row(row, config, report_lock)
        return

    with ThreadPoolExecutor(max_workers=config.workers) as executor:
        futures = [
            executor.submit(_process_row, row, config, report_lock) for row in rows
        ]
        for future in as_completed(futures):
            future.result()


def _dry_run(rows: list[dict[str, Any]], out_dir: Path, max_downloads: int) -> None:
    for index, row in enumerate(rows, 1):
        LOGGER.info("%s -> %s", row["dataset_id"], out_dir / row["filename"])
        if max_downloads and index >= max_downloads:
            break


def main() -> None:
    """Run the MDC archive downloader."""
    _configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument("--max-sleep", type=int, default=DEFAULT_MAX_SLEEP_SECONDS)
    parser.add_argument("--max-downloads", type=int, default=0, help="0 means no limit")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--dataset-id",
        action="append",
        default=[],
        help="Only process this dataset ID; may be repeated",
    )
    args = parser.parse_args()

    token = "" if args.dry_run else _api_key()
    workers = max(1, args.workers)
    config = RunConfig(
        token=token,
        out_dir=args.out_dir,
        report_path=args.report,
        retries=args.retries,
        max_sleep=args.max_sleep,
        workers=workers,
        force=args.force,
    )
    watch_manifest = not (
        args.dry_run or args.dataset_id or args.max_downloads or args.force
    )

    while True:
        rows = _select_rows(_load_manifest(args.manifest), args.dataset_id)
        if args.dry_run:
            _dry_run(rows, args.out_dir, args.max_downloads)
            return

        candidates = _pending_rows(rows, args.out_dir, force=args.force)
        if args.max_downloads:
            candidates = candidates[: args.max_downloads]
        if watch_manifest:
            candidates = candidates[:workers]

        if not candidates:
            if not watch_manifest:
                return
            LOGGER.info(
                "no missing archives; watching %s every %ss",
                args.manifest,
                IDLE_SLEEP_SECONDS,
            )
            time.sleep(IDLE_SLEEP_SECONDS)
            continue

        _run_candidates(candidates, config)

        if not watch_manifest:
            return


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
