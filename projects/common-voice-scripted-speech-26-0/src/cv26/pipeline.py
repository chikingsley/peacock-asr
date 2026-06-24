"""Steady-state loop: convert downloaded archives, upload shards, verify, delete the raw archive.

Completion is tracked in an append-only JSONL log (:data:`cv26.config.PROCESSING_STATE`); each
archive moves through ``converted`` -> ``uploaded`` -> ``complete``. A ``complete`` archive is
skipped on the next pass, so the loop is safe to restart and safe to run with ``--watch`` while the
downloader is still landing new archives.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from cv26 import config
from cv26.convert import convert_archive
from cv26.manifest import ManifestRow, load_manifest, select_rows
from cv26.upload import ensure_repo, hf_api, upload_shards, verify_remote

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from huggingface_hub import HfApi

LOGGER = logging.getLogger("cv26.pipeline")
IDLE_SLEEP_SECONDS = 30
_GB = 1024**3
# Terminal states a row is never reprocessed from: uploaded+verified, or no ASR rows to upload.
_DONE = frozenset({"complete", "no_shards"})


class PipelineError(RuntimeError):
    """Raised when the convert/upload/delete pipeline cannot continue safely."""


def _assert_within(path: Path, root: Path) -> None:
    """Guard against a manifest-derived path escaping its configured root before delete/write."""
    if not path.resolve().is_relative_to(root.resolve()):
        raise PipelineError(f"path escapes root {root}: {path}")


def _archive_sha256(archive: Path) -> str:
    """Return the archive hash, preferring the download-time sidecar (avoids a full re-read)."""
    sidecar = archive.with_suffix(archive.suffix + ".sha256")
    if sidecar.exists():
        parts = sidecar.read_text(encoding="utf-8").split()
        if parts:
            return parts[0]
    return config.sha256_file(archive)


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Settings for one processing run."""

    archive_dir: Path
    parquet_root: Path
    state_path: Path
    repo_id: str
    upload: bool = False
    create_repo: bool = False
    delete_after_upload: bool = False
    overwrite: bool = False
    dry_run: bool = False
    limit: int = 0


def load_state(path: Path) -> dict[str, str]:
    """Return the latest status per dataset ID from the append-only state log."""
    if not path.exists():
        return {}
    status: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if raw_line.strip():
                event = json.loads(raw_line)
                status[str(event["dataset_id"])] = str(event["status"])
    return status


def _append_state(path: Path, event: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {"created_at": datetime.now(UTC).isoformat(), **event}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")


def select_candidates(rows: Iterable[ManifestRow], cfg: RunConfig) -> list[ManifestRow]:
    """Return downloaded rows not yet marked ``complete``, in manifest order."""
    status = load_state(cfg.state_path)
    candidates = [
        row
        for row in rows
        if status.get(row.dataset_id) not in _DONE and (cfg.archive_dir / row.filename).exists()
    ]
    return candidates[: cfg.limit] if cfg.limit else candidates


def process_one(row: ManifestRow, cfg: RunConfig, api: HfApi | None) -> None:
    """Convert one archive, then (when uploading) upload, verify, and only then delete it.

    With ``--delete-after-upload`` both the raw archive AND the local parquet shards are removed
    once the shards are uploaded and verified on the Hub (they live on the Hub now), to keep disk
    bounded. An archive that yields zero shards is recorded ``no_shards`` and left on disk — never
    deleted. A convert-only run (no ``--upload``) stops at ``converted`` so a later upload run takes
    it.
    """
    archive = cfg.archive_dir / row.filename
    _assert_within(archive, cfg.archive_dir)
    if cfg.dry_run:
        LOGGER.info("would_process: %s %s", row.dataset_id, archive.name)
        return

    archive_sha256 = _archive_sha256(archive)
    shards = convert_archive(row, cfg.archive_dir, cfg.parquet_root, overwrite=cfg.overwrite)
    base = {
        "dataset_id": row.dataset_id,
        "filename": row.filename,
        "archive_sha256": archive_sha256,
        "parquet_paths": [str(path.relative_to(cfg.parquet_root)) for path in shards],
    }
    if not shards:
        _append_state(cfg.state_path, {**base, "status": "no_shards"})
        LOGGER.warning("no ASR rows; keeping archive, nothing to upload: %s", row.dataset_id)
        return
    LOGGER.info("converted %s -> %d shard(s)", row.dataset_id, len(shards))
    _append_state(cfg.state_path, {**base, "status": "converted"})
    if not cfg.upload:
        return

    if api is None:
        raise PipelineError("upload requested without an HF API client")
    uploaded = upload_shards(api, cfg.repo_id, cfg.parquet_root, shards)
    verify_remote(api, cfg.repo_id, uploaded)
    _append_state(
        cfg.state_path,
        {**base, "status": "uploaded", "repo_id": cfg.repo_id, "uploaded_paths": uploaded},
    )

    if cfg.delete_after_upload:
        _assert_within(archive, cfg.archive_dir)
        archive.unlink(missing_ok=True)
        archive.with_suffix(archive.suffix + ".sha256").unlink(missing_ok=True)
        for shard in shards:  # verified on the Hub above; drop the local copies to free disk
            _assert_within(shard, cfg.parquet_root)
            shard.unlink(missing_ok=True)
        LOGGER.info("deleted archive + %d local shards: %s", len(shards), archive.name)

    _append_state(
        cfg.state_path,
        {
            **base,
            "status": "complete",
            "repo_id": cfg.repo_id,
            "uploaded_paths": uploaded,
            "archive_deleted": cfg.delete_after_upload,
        },
    )


def _archive_size(cfg: RunConfig, row: ManifestRow) -> int:
    archive = cfg.archive_dir / row.filename
    return archive.stat().st_size if archive.exists() else 0


def _attempt(row: ManifestRow, cfg: RunConfig, api: HfApi | None, failed: set[str]) -> bool:
    """Process one archive; return whether it ran (False = deferred for space, or failed)."""
    size = _archive_size(cfg, row)
    free = shutil.disk_usage(cfg.archive_dir).free
    if size > free:  # its shard (~archive size) would not fit; drain smaller archives first
        LOGGER.warning(
            "deferring %s: %d GB archive needs more free space (%d GB)",
            row.dataset_id,
            size // _GB,
            free // _GB,
        )
        return False
    try:
        process_one(row, cfg, api)
    except Exception as exc:  # one bad archive must not stop the whole drain
        LOGGER.exception("FAILED processing %s; skipping", row.dataset_id)
        failed.add(row.dataset_id)
        _append_state(
            cfg.state_path,
            {
                "dataset_id": row.dataset_id,
                "filename": row.filename,
                "status": "failed",
                "error": str(exc)[:300],
            },
        )
        return False
    return True


def run(cfg: RunConfig, *, watch: bool) -> None:
    """Process pending archives once, or loop until interrupted when ``watch`` is set.

    Smallest archives are processed first so deletes free disk before the giant archives are
    reached; an archive that cannot fit its shard in free space is deferred and retried later.
    """
    api: HfApi | None = None
    if cfg.upload:
        api = hf_api(config.require_hf_token())
        ensure_repo(api, cfg.repo_id, create=cfg.create_repo)

    failed: set[str] = set()  # archives that errored this session (bad TSV, corrupt, …) — skip
    while True:
        rows = select_rows(load_manifest(config.MANIFEST))
        candidates = sorted(
            (r for r in select_candidates(rows, cfg) if r.dataset_id not in failed),
            key=lambda r: _archive_size(cfg, r),
        )
        processed = sum(_attempt(row, cfg, api, failed) for row in candidates)
        if not watch:
            return
        if not processed:
            time.sleep(IDLE_SLEEP_SECONDS)
