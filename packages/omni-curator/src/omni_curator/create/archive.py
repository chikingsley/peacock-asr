"""Archive (or delete) segmented videos' source FLACs to reclaim space on the working drive.

After ``segment`` sets a video's status to ``segmented``, NOTHING downstream reads its source FLAC —
labelq / harvest / verify / export all use the cut clips (``clips.clip_path``), never
``videos.path``. So once a video is segmented its source ``create/<slug>/<id>.flac`` is redundant
for the pipeline; keep it only to re-segment with different VAD params (hence default to MOVE, not
delete). yt-dlp's download ledger (``downloaded.txt``) is keyed by video id, so moving or deleting a
source never re-triggers its download.

Safety:
- Only touches ``status='segmented'`` rows — disjoint from what ``segment`` (pending/segmenting) and
  ``download`` claim, so it is safe to run alongside a live pipeline.
- A cross-filesystem move (overflow ext4 -> storage exfat) is copy -> size-verify -> atomic-rename
  into place -> unlink the original (``Path.replace`` is NOT atomic across filesystems). A
  same-filesystem move is a single atomic ``os.replace``. The original is never unlinked until its
  copy is on disk at full size.
- Every archived/deleted file is appended to a JSONL manifest, so the action is auditable.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Callable

_BYTES_PER_GB = 1_000_000_000


@dataclass
class ArchiveStats:
    """Outcome of one archive run."""

    archived: int = 0  # sources moved or deleted
    bytes_freed: int = 0
    missing: int = 0  # source already gone (idempotent re-run)
    errors: list[tuple[str, str]] = field(default_factory=list)  # (video_id, message)


def _segmented_sources(queue_path: Path) -> list[tuple[str, str, str]]:
    """``(video_id, channel, source_path)`` for every segmented video, ordered by id (read-only)."""
    conn = sqlite3.connect(queue_path)  # SELECT-only; WAL allows this beside a live writer
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT video_id, channel, path FROM videos WHERE status='segmented' ORDER BY video_id"
        ).fetchall()
    finally:
        conn.close()
    return [(r["video_id"], r["channel"], r["path"]) for r in rows]


def _free_gb(path: Path) -> float:
    """Free GB on the filesystem holding ``path`` (or its nearest existing parent)."""
    probe = path
    while not probe.exists():
        probe = probe.parent
    stat = os.statvfs(probe)
    return stat.f_bavail * stat.f_frsize / _BYTES_PER_GB


def _sha256(path: Path) -> str:
    """Streamed SHA-256 of a file — content identity, since equal byte-size is NOT proof of
    identical content (a truncated/corrupt copy can match size). Gates source-audio deletion."""
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_verify(src: Path, dst: Path) -> int:
    """Copy ``src`` -> ``dst`` and verify the copied byte-size matches; return the size.

    Copies to a ``.partial`` sibling first and atomic-renames it into place, so ``dst`` is never a
    half-written file. Raises ``OSError`` (leaving ``src`` untouched) if the copy is short.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    size = src.stat().st_size
    tmp = dst.with_name(dst.name + ".partial")
    shutil.copy2(src, tmp)
    with tmp.open("rb") as fh:
        os.fsync(fh.fileno())  # flush the copied bytes to disk BEFORE the caller unlinks the source
    copied = tmp.stat().st_size
    if copied != size or _sha256(tmp) != _sha256(src):
        tmp.unlink(missing_ok=True)
        msg = f"copy verify FAILED for {src} (size {copied} vs {size}, or checksum mismatch)"
        raise OSError(msg)
    tmp.replace(dst)
    dir_fd = os.open(dst.parent, os.O_RDONLY)  # make the rename itself durable (exFAT/ext4)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
    return size


def _move(src: Path, dst: Path) -> int:
    """Move ``src`` -> ``dst``: atomic rename if same filesystem, else copy-verify-unlink."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    size = src.stat().st_size
    if dst.exists() and dst.stat().st_size == size and _sha256(dst) == _sha256(src):
        src.unlink()  # idempotent: a prior run already copied an IDENTICAL file (content-verified)
        return size
    if src.stat().st_dev == dst.parent.stat().st_dev:
        src.replace(dst)  # same filesystem -> atomic rename
        return size
    copy_verify(src, dst)  # cross filesystem -> copy, size-verify, atomic-finalize
    src.unlink()
    return size


def archive_source_audio(
    queue_path: Path,
    *,
    archive_root: Path | None = None,
    delete: bool = False,
    only_if_free_gb: float | None = None,
    on_event: Callable[[str], None] | None = None,
) -> ArchiveStats:
    """Move (or ``delete``) the source FLAC of every segmented video; return the run stats.

    ``archive_root`` is required unless ``delete``; sources move to ``<archive_root>/<channel>/``.
    ``only_if_free_gb`` gates the whole run on the working drive (where the sources live): if it
    already has at least that many GB free, nothing is done. Idempotent — a source already gone is
    counted ``missing`` and skipped.
    """
    notify = on_event or (lambda _m: None)
    stats = ArchiveStats()
    if not delete and archive_root is None:
        msg = "archive_root is required unless delete=True"
        raise ValueError(msg)
    sources = _segmented_sources(queue_path)
    if not sources:
        notify("archive: no segmented videos to archive")
        return stats

    if only_if_free_gb is not None:
        free = _free_gb(Path(sources[0][2]).parent)
        if free >= only_if_free_gb:
            notify(f"archive: {free:.0f} GB free >= {only_if_free_gb:.0f} threshold; skipping")
            return stats

    manifest = (archive_root or queue_path.parent) / "archive_manifest.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("a", encoding="utf-8") as log:
        for video_id, channel, path_str in sources:
            src = Path(path_str)
            if not src.exists():
                stats.missing += 1
                continue
            try:
                size = src.stat().st_size
                if delete:
                    dst_str = None
                    src.unlink()
                else:
                    # archive_root is non-None here (validated above when delete is False)
                    dst = cast("Path", archive_root) / channel / src.name
                    _move(src, dst)
                    dst_str = str(dst)
            except OSError as exc:
                stats.errors.append((video_id, str(exc)))
                continue
            stats.archived += 1
            stats.bytes_freed += size
            log.write(
                json.dumps(
                    {
                        "ts": time.time(),
                        "video_id": video_id,
                        "channel": channel,
                        "src": str(src),
                        "dst": dst_str,
                        "size": size,
                        "action": "delete" if delete else "move",
                    }
                )
                + "\n"
            )
    notify(
        f"archive: {stats.archived} sources "
        f"{'deleted' if delete else 'moved'}, {stats.bytes_freed / _BYTES_PER_GB:.1f} GB freed, "
        f"{stats.missing} already gone, {len(stats.errors)} errors"
    )
    return stats
