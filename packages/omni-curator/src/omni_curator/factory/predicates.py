"""Per-stage "claimable now" predicates derived from the DBs (factory_plan §2, the two v0 rows).

A trigger must mean **work claimable right now**, not just "rows in a transient state" — otherwise a
stage launches, finds nothing to claim, exits, and gets relaunched in a tight loop. The two v0
predicates:

- ``enqueue`` (one-shot): the create-root (SSD) holds ``*.flac`` whose ``video_id`` is absent from
  ``queue.videos`` -> there is new source to seed.
- ``segment`` (daemon): ``queue.videos.status='pending'`` OR (``'segmenting'`` with a **stale**
  lease) > 0 -> there is a claimable video. The stale window mirrors
  :meth:`QueueStore.claim_video`'s ``stale_after_s`` so the predicate counts exactly the rows
  ``claim_video`` would hand out.

Counts are read through :class:`QueueStore` / its sqlite connection — not reinvented.
"""

from __future__ import annotations

import time
from pathlib import Path

from omni_curator.create.queue import QueueStore

#: Mirrors ``QueueStore.claim_video(stale_after_s=1800.0)``: a ``segmenting`` row older than this is
#: reclaimable, so it counts as claimable work for the segment predicate.
SEGMENT_STALE_AFTER_S = 1800.0


def _video_ids(queue: QueueStore) -> set[str]:
    """All ``video_id``s currently in ``queue.videos`` (any status)."""
    rows = queue._conn.execute("SELECT video_id FROM videos").fetchall()  # noqa: SLF001
    return {r["video_id"] for r in rows}


def flac_video_ids(create_root: Path) -> set[str]:
    """The ``video_id`` (``<channel>_<stem>``) of every ``*.flac`` under ``create_root/<channel>/``.

    Matches :func:`omni_curator.project.cmd_enqueue`: it scans ``create_root/<slug>/*.flac`` and
    forms ``f"{slug}_{flac.stem}"``. The factory does not know the channel registry, so it derives
    the channel slug from the FLAC's parent directory name — the same value ``enqueue`` uses.
    """
    if not create_root.exists():
        return set()
    ids: set[str] = set()
    for flac in create_root.glob("*/*.flac"):
        ids.add(f"{flac.parent.name}_{flac.stem}")
    return ids


def enqueue_needed(queue_path: Path, create_root: Path) -> bool:
    """``True`` if the create-root has FLACs not yet present in ``queue.videos``."""
    on_disk = flac_video_ids(create_root)
    if not on_disk:
        return False
    queue = QueueStore(queue_path)
    try:
        enqueued = _video_ids(queue)
    finally:
        queue.close()
    return bool(on_disk - enqueued)


def segment_backlog(queue_path: Path, *, now: float | None = None) -> int:
    """Count of claimable videos: ``pending`` plus stale-``segmenting`` (per ``claim_video``)."""
    if not queue_path.exists():
        return 0
    moment = time.time() if now is None else now
    queue = QueueStore(queue_path)
    try:
        row = queue._conn.execute(  # noqa: SLF001
            "SELECT count(*) AS n FROM videos WHERE status='pending' "
            "OR (status='segmenting' AND locked_at < ?)",
            (moment - SEGMENT_STALE_AFTER_S,),
        ).fetchone()
    finally:
        queue.close()
    return int(row["n"])


def segment_needed(queue_path: Path, *, now: float | None = None) -> bool:
    """``True`` if there is at least one claimable video to segment."""
    return segment_backlog(queue_path, now=now) > 0


# -- v1 predicates ------------------------------------------------------------------------------

#: Mirrors ``run_labeler``'s ``lease_s=900.0``: a ``labeling`` clip older than this is reclaimable,
#: so it counts as claimable work for the labelq predicate (just like ``reclaim_stale_clips``).
LABELQ_STALE_AFTER_S = 900.0


def labelq_backlog(queue_path: Path, *, now: float | None = None) -> int:
    """Count of claimable clips: ``pending`` plus stale-``labeling`` (mirrors ``claim_clips``)."""
    if not queue_path.exists():
        return 0
    moment = time.time() if now is None else now
    queue = QueueStore(queue_path)
    try:
        row = queue._conn.execute(  # noqa: SLF001
            "SELECT count(*) AS n FROM clips WHERE status='pending' "
            "OR (status='labeling' AND locked_at < ?)",
            (moment - LABELQ_STALE_AFTER_S,),
        ).fetchone()
    finally:
        queue.close()
    return int(row["n"])


def labelq_needed(queue_path: Path, *, now: float | None = None) -> bool:
    """``True`` if there is at least one claimable clip to label."""
    return labelq_backlog(queue_path, now=now) > 0


def harvest_backlog(queue_path: Path) -> int:
    """Count of ``status='done' AND harvested_at IS NULL`` clips — harvestable right now."""
    if not queue_path.exists():
        return 0
    queue = QueueStore(queue_path)
    try:
        row = queue._conn.execute(  # noqa: SLF001
            "SELECT count(*) AS n FROM clips WHERE status='done' AND harvested_at IS NULL"
        ).fetchone()
    finally:
        queue.close()
    return int(row["n"])


def harvest_needed(queue_path: Path) -> bool:
    """``True`` if there are done-but-unharvested clips to fold into the channel stores."""
    return harvest_backlog(queue_path) > 0


def archive_needed(queue_path: Path) -> bool:
    """``True`` if a ``status='segmented'`` video still has its source FLAC on disk.

    Archive drains a processed video's now-redundant source off the working drive. The predicate is
    "at least one segmented video whose ``path`` still exists" — so a drained queue (every source
    already moved) makes it false and archive isn't relaunched. Read-only, beside a live writer.
    """
    if not queue_path.exists():
        return False
    queue = QueueStore(queue_path)
    try:
        rows = queue._conn.execute(  # noqa: SLF001
            "SELECT path FROM videos WHERE status='segmented'"
        ).fetchall()
    finally:
        queue.close()
    return any(Path(r["path"]).exists() for r in rows)
