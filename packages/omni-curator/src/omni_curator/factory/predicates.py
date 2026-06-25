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
from typing import TYPE_CHECKING

from omni_curator.create.queue import QueueStore

if TYPE_CHECKING:
    from pathlib import Path

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
