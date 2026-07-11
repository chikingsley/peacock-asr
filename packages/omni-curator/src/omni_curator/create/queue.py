"""SQLite work queue decoupling segmentation (CPU producer) from labeling (I/O consumer).

Running VAD then Scribe sequentially per video in one process saturates neither the CPU (VAD) nor
the network (Scribe). This queue splits the two stages so each runs flat out:

- ``segment`` workers claim a *video* (rare, coarse claims), VAD-segment it, cut the clips to disk,
  and enqueue ``clips`` rows.
- ``labelq`` (one dispatcher + an HTTP thread pool) batch-claims *clips*, Scribe-labels them, and
  writes the labels back here.
- ``harvest`` folds done clips into the canonical per-channel ``CuratorStore``.

The queue DB (``data/queue.sqlite``) is transient work-state, deliberately separate from the
canonical store. Concurrency safety: WAL mode, ``BEGIN IMMEDIATE`` for the (rare) claim writes, a
lease (``locked_at``) for crash recovery, and a per-claim ``claim_token`` so a result from a
reclaimed lease can never overwrite the retry that replaced it.
See ``docs/CURATION_FACTORY.md`` for the current stage contracts.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import uuid4

from omni_curator.data.store import connect_wal

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    video_id   TEXT PRIMARY KEY,
    channel    TEXT NOT NULL,
    path       TEXT NOT NULL,
    tier       TEXT NOT NULL,
    citation   TEXT,
    category   TEXT NOT NULL DEFAULT 'uncategorized',
    meta       TEXT NOT NULL DEFAULT '{}',
    status     TEXT NOT NULL DEFAULT 'pending',   -- pending|segmenting|segmented|failed
    n_clips    INTEGER,
    attempts   INTEGER NOT NULL DEFAULT 0,
    locked_by  TEXT,
    locked_at  REAL,
    claim_token TEXT,
    last_error TEXT,
    updated_at REAL
);
CREATE INDEX IF NOT EXISTS idx_videos_status ON videos(status);

CREATE TABLE IF NOT EXISTS clips (
    clip_id     TEXT PRIMARY KEY,
    video_id    TEXT NOT NULL,
    channel     TEXT NOT NULL,
    clip_index  INTEGER NOT NULL,
    clip_path   TEXT NOT NULL,
    tier        TEXT NOT NULL DEFAULT 'unknown',
    start       REAL NOT NULL,
    end         REAL NOT NULL,
    language    TEXT NOT NULL,
    script      TEXT NOT NULL,
    citation    TEXT,
    category    TEXT NOT NULL DEFAULT 'uncategorized',
    meta        TEXT NOT NULL DEFAULT '{}',
    status      TEXT NOT NULL DEFAULT 'pending',   -- pending|labeling|done|failed
    attempts    INTEGER NOT NULL DEFAULT 0,
    claim_token TEXT,
    locked_at   REAL,
    label       TEXT,
    variants    TEXT,
    last_error  TEXT,
    done_at     REAL,
    harvested_at REAL,
    updated_at  REAL
);
CREATE INDEX IF NOT EXISTS idx_clips_status  ON clips(status);
CREATE INDEX IF NOT EXISTS idx_clips_harvest ON clips(status, harvested_at);
"""


@dataclass(frozen=True)
class QVideo:
    """A video claimed for segmentation."""

    video_id: str
    channel: str
    path: str
    tier: str
    citation: str | None
    claim_token: str | None = None  # set by claim_video; guards complete_video/fail_video
    category: str = "uncategorized"
    meta: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class QClip:
    """A clip cut by the segmenter, enqueued for labeling."""

    clip_id: str
    video_id: str
    channel: str
    clip_index: int
    clip_path: str
    start: float
    end: float
    language: str
    script: str
    citation: str | None
    tier: str = "unknown"
    category: str = "uncategorized"
    meta: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class QLabeledClip:
    """A labeled clip ready to harvest into the canonical store."""

    clip_id: str
    video_id: str
    channel: str
    clip_index: int
    clip_path: str
    start: float
    end: float
    language: str
    citation: str | None
    label: str
    variants: str | None
    tier: str = "unknown"
    category: str = "uncategorized"
    meta: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class QueueRepairResult:
    """Counts from refreshing existing queue row metadata."""

    matched: int
    changed: int
    updated: int
    skipped_missing: int


def _meta_json(meta: dict[str, object]) -> str:
    return json.dumps(meta, ensure_ascii=False, sort_keys=True)


def _load_meta(raw: str | None) -> dict[str, object]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


class QueueStore:
    """SQLite-backed segment/label work queue. One instance per process (own connection)."""

    def __init__(self, db_path: Path, *, busy_timeout_ms: int = 60_000) -> None:
        self.path = db_path
        self._conn = connect_wal(db_path, _SCHEMA, busy_timeout_ms=busy_timeout_ms, manual_tx=True)
        self._migrate()

    def _migrate(self) -> None:
        """Add columns a pre-existing queue predates (idempotent; CREATE TABLE never alters)."""
        self._add_missing_columns(
            "videos",
            {
                "claim_token": "TEXT",
                "category": "TEXT NOT NULL DEFAULT 'uncategorized'",
                "meta": "TEXT NOT NULL DEFAULT '{}'",
            },
        )
        self._add_missing_columns(
            "clips",
            {
                "tier": "TEXT NOT NULL DEFAULT 'unknown'",
                "category": "TEXT NOT NULL DEFAULT 'uncategorized'",
                "meta": "TEXT NOT NULL DEFAULT '{}'",
            },
        )

    def _add_missing_columns(self, table: str, columns: dict[str, str]) -> None:
        have = {row["name"] for row in self._conn.execute(f"PRAGMA table_info({table})")}
        for name, definition in columns.items():
            if name not in have:
                self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def _tx(self, *, immediate: bool = False) -> Iterator[None]:
        self._conn.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
        try:
            yield
        except BaseException:
            self._conn.rollback()
            raise
        self._conn.commit()

    # -- enqueue (idempotent by PK) ----------------------------------------------------------

    def enqueue_videos(self, videos: list[QVideo]) -> int:
        """Insert pending video rows; existing ``video_id``s untouched. Returns rows inserted."""
        now = time.time()
        with self._tx():
            cur = self._conn.executemany(
                "INSERT OR IGNORE INTO videos "
                "(video_id, channel, path, tier, citation, category, meta, updated_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                [
                    (
                        v.video_id,
                        v.channel,
                        v.path,
                        v.tier,
                        v.citation,
                        v.category,
                        _meta_json(v.meta),
                        now,
                    )
                    for v in videos
                ],
            )
            return cur.rowcount

    def video_records(self) -> list[dict[str, object]]:
        """Return current video rows as plain dicts for audit/repair commands."""
        return [dict(row) for row in self._conn.execute("SELECT * FROM videos ORDER BY video_id")]

    def repair_video_metadata(
        self, videos: list[QVideo], *, dry_run: bool = False
    ) -> QueueRepairResult:
        """Refresh existing video/clip tier, citation, category, and metadata by ``video_id``."""
        current = {
            row["video_id"]: row
            for row in self._conn.execute(
                "SELECT video_id, tier, citation, category, meta FROM videos"
            )
        }
        now = time.time()
        matched = 0
        changed = 0
        updated = 0
        missing = 0
        updates: list[tuple[str, str | None, str, str, float, str]] = []
        clip_updates: list[tuple[str, str | None, str, dict[str, object], float, str]] = []
        for video in videos:
            row = current.get(video.video_id)
            if row is None:
                missing += 1
                continue
            matched += 1
            meta = _meta_json(video.meta)
            values = (video.tier, video.citation, video.category, meta)
            old = (row["tier"], row["citation"], row["category"], row["meta"])
            if values == old:
                continue
            changed += 1
            updates.append((*values, now, video.video_id))
            clip_updates.append(
                (video.tier, video.citation, video.category, dict(video.meta), now, video.video_id)
            )
        if updates and not dry_run:
            with self._tx(immediate=True):
                cur = self._conn.executemany(
                    "UPDATE videos SET tier=?, citation=?, category=?, meta=?, updated_at=? "
                    "WHERE video_id=?",
                    updates,
                )
                updated = cur.rowcount
                for tier, citation, category, source_meta, changed_at, video_id in clip_updates:
                    rows = self._conn.execute(
                        "SELECT clip_id, meta FROM clips WHERE video_id=?", (video_id,)
                    ).fetchall()
                    repaired: list[tuple[str, str | None, str, str, float, str]] = []
                    for clip in rows:
                        meta = dict(source_meta)
                        old_meta = _load_meta(clip["meta"])
                        if "segmentation" in old_meta:
                            meta["segmentation"] = old_meta["segmentation"]
                        repaired.append(
                            (
                                tier,
                                citation,
                                category,
                                _meta_json(meta),
                                changed_at,
                                clip["clip_id"],
                            )
                        )
                    self._conn.executemany(
                        "UPDATE clips SET tier=?, citation=?, category=?, meta=?, updated_at=? "
                        "WHERE clip_id=?",
                        repaired,
                    )
        return QueueRepairResult(matched, changed, updated, missing)

    # -- segment stage -----------------------------------------------------------------------

    def claim_video(self, worker_id: str, *, stale_after_s: float = 1800.0) -> QVideo | None:
        """Claim the next video under a lease — stale-``segmenting`` FIRST, then ``pending``.

        A stale ``segmenting`` row is one whose lease lapsed (its worker died / the parent was
        killed): it must be reclaimed BEFORE fresh ``pending`` work, otherwise a long video that
        lapsed is starved behind the never-emptying pending pile and is never retried. We sort
        ``segmenting`` ahead of ``pending`` and break ties by ``locked_at`` (oldest lapse first).
        ``None`` if nothing is claimable.
        """
        now = time.time()
        with self._tx(immediate=True):
            row = self._conn.execute(
                "SELECT * FROM videos WHERE status='pending' "
                "OR (status='segmenting' AND locked_at < ?) "
                # status='segmenting' < 'pending' lexically, so plain status ASC already puts stale
                # segmenting first; the explicit CASE makes that intent unambiguous and stable, and
                # the locked_at tiebreak retries the longest-lapsed video first.
                "ORDER BY CASE status WHEN 'segmenting' THEN 0 ELSE 1 END, locked_at LIMIT 1",
                (now - stale_after_s,),
            ).fetchone()
            if row is None:
                return None
            token = uuid4().hex
            self._conn.execute(
                "UPDATE videos SET status='segmenting', locked_by=?, locked_at=?, claim_token=?, "
                "attempts=attempts+1, updated_at=? WHERE video_id=?",
                (worker_id, now, token, now, row["video_id"]),
            )
        return QVideo(
            row["video_id"],
            row["channel"],
            row["path"],
            row["tier"],
            row["citation"],
            claim_token=token,
            category=row["category"],
            meta=_load_meta(row["meta"]),
        )

    def complete_video(
        self,
        video_id: str,
        clips: list[QClip],
        *,
        claim_token: str | None = None,
        publish: Callable[[], None] | None = None,
        video_meta: dict[str, object] | None = None,
    ) -> bool:
        """Enqueue a segmented video's clips and mark it done — one txn (no half-cut clip view).

        Guarded by ``claim_token``: if the video was reclaimed (a fresh token) the marking matches
        zero rows and the clips are NOT enqueued, so a stale segmenter can't double-process.
        ``publish`` runs only after the token-guarded row update succeeds and while this writer
        holds the transaction, so stale segmenters can cut staging files but cannot publish final
        clip files after SQLite ownership moved on.
        """
        now = time.time()
        video_meta_json = _meta_json(video_meta) if video_meta is not None else None
        cols = (
            "clip_id, video_id, channel, clip_index, clip_path, tier, "
            "start, end, language, script, citation, category, meta, updated_at"
        )
        with self._tx():
            cur = self._conn.execute(
                "UPDATE videos SET status='segmented', n_clips=?, locked_by=NULL, locked_at=NULL, "
                "claim_token=NULL, meta=COALESCE(?, meta), updated_at=? "
                "WHERE video_id=? AND claim_token IS ?",
                (len(clips), video_meta_json, now, video_id, claim_token),
            )
            if not cur.rowcount:
                return False
            if publish is not None:
                publish()
            self._conn.executemany(
                f"INSERT OR IGNORE INTO clips ({cols}) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",  # noqa: S608 — fixed column list
                [
                    (
                        c.clip_id,
                        c.video_id,
                        c.channel,
                        c.clip_index,
                        c.clip_path,
                        c.tier,
                        c.start,
                        c.end,
                        c.language,
                        c.script,
                        c.citation,
                        c.category,
                        _meta_json(c.meta),
                        now,
                    )
                    for c in clips
                ],
            )
            return True

    def fail_video(
        self, video_id: str, error: str, *, claim_token: str | None = None, max_attempts: int = 3
    ) -> None:
        """Record a segmentation failure; requeue until ``max_attempts``, then mark failed.

        Guarded by ``claim_token``: a stale segmenter (its row already reclaimed) does nothing, so
        its late failure can't flip a successfully-reclaimed/segmented row back to pending/failed.
        """
        now = time.time()
        with self._tx():
            row = self._conn.execute(
                "SELECT attempts FROM videos WHERE video_id=? AND claim_token IS ?",
                (video_id, claim_token),
            ).fetchone()
            if row is None:
                return  # reclaimed by another worker; our stale failure must not touch its row
            status = "failed" if row["attempts"] >= max_attempts else "pending"
            self._conn.execute(
                "UPDATE videos SET status=?, last_error=?, locked_by=NULL, locked_at=NULL, "
                "claim_token=NULL, updated_at=? WHERE video_id=? AND claim_token IS ?",
                (status, error[:500], now, video_id, claim_token),
            )

    def pending_clip_count(self) -> int:
        """Clips not yet done — the backpressure signal for the segmenter's high-watermark."""
        return self._conn.execute(
            "SELECT count(*) FROM clips WHERE status IN ('pending', 'labeling')"
        ).fetchone()[0]

    # -- re-segment --------------------------------------------------------------------------

    def resegment_preview(self) -> dict[str, int]:
        """Counts of what :meth:`reset_for_resegment` would touch (read-only, for confirmation).

        ``{'videos': N videos that will go back to pending, 'clips': N clip rows that will be
        deleted}`` — only ``segmented``/``failed`` videos and their clips are in scope.
        """
        videos = self._conn.execute(
            "SELECT count(*) FROM videos WHERE status IN ('segmented', 'failed')"
        ).fetchone()[0]
        clips = self._conn.execute("SELECT count(*) FROM clips").fetchone()[0]
        return {"videos": videos, "clips": clips}

    def reset_for_resegment(self) -> dict[str, int]:
        """Reset ``segmented``/``failed`` videos to ``pending`` and DELETE all clip rows.

        Idempotent: re-running on an already-reset queue is a no-op (no segmented/failed rows, no
        clips). Returns ``{'videos': reset, 'clips': deleted}``. The clip *files* are removed
        separately by the caller (it holds the paths via :meth:`all_clip_paths`); this only clears
        the queue's bookkeeping so :func:`segment` re-cuts every video from scratch.
        """
        now = time.time()
        with self._tx(immediate=True):
            cur = self._conn.execute(
                "UPDATE videos SET status='pending', n_clips=NULL, locked_by=NULL, locked_at=NULL, "
                "claim_token=NULL, last_error=NULL, attempts=0, updated_at=? "
                "WHERE status IN ('segmented', 'failed')",
                (now,),
            )
            videos = cur.rowcount
            deleted = self._conn.execute("DELETE FROM clips").rowcount
        return {"videos": videos, "clips": deleted}

    def all_clip_paths(self) -> list[str]:
        """Every clip's on-disk path — so the caller can delete orphaned files before a reset."""
        return [row["clip_path"] for row in self._conn.execute("SELECT clip_path FROM clips")]

    # -- label stage -------------------------------------------------------------------------

    def claim_clips(self, batch: int, claim_token: str, *, lease_s: float = 900.0) -> list[QClip]:
        """Batch-claim pending clips under ``claim_token`` + lease; returns the claimed clips."""
        now = time.time()
        with self._tx(immediate=True):
            rows = self._conn.execute(
                "UPDATE clips SET status='labeling', claim_token=?, locked_at=?, "
                "attempts=attempts+1, updated_at=? WHERE clip_id IN "
                "(SELECT clip_id FROM clips WHERE status='pending' ORDER BY clip_id LIMIT ?) "
                "RETURNING *",
                (claim_token, now, now, batch),
            ).fetchall()
        _ = lease_s  # lease enforced by reclaim_stale_clips; kept explicit for callers
        return [
            QClip(
                r["clip_id"],
                r["video_id"],
                r["channel"],
                r["clip_index"],
                r["clip_path"],
                r["start"],
                r["end"],
                r["language"],
                r["script"],
                r["citation"],
                tier=r["tier"],
                category=r["category"],
                meta=_load_meta(r["meta"]),
            )
            for r in rows
        ]

    def complete_clips(self, claim_token: str, results: list[tuple[str, str, str]]) -> int:
        """Write labels back, guarded by ``status='labeling' AND claim_token`` (reclaim-safe)."""
        now = time.time()
        with self._tx():
            cur = self._conn.executemany(
                "UPDATE clips SET status='done', label=?, variants=?, done_at=?, updated_at=? "
                "WHERE clip_id=? AND status='labeling' AND claim_token=?",
                [(label, variants, now, now, cid, claim_token) for cid, label, variants in results],
            )
            return cur.rowcount

    def fail_clips(
        self, claim_token: str, clip_ids: list[str], error: str, *, max_attempts: int = 3
    ) -> None:
        """Requeue failed clips (guarded by token) until ``max_attempts``, then mark failed."""
        now = time.time()
        with self._tx():
            for cid in clip_ids:
                self._conn.execute(
                    "UPDATE clips SET status=CASE WHEN attempts >= ? THEN 'failed' ELSE 'pending' "
                    "END, claim_token=NULL, locked_at=NULL, last_error=?, updated_at=? "
                    "WHERE clip_id=? AND status='labeling' AND claim_token=?",
                    (max_attempts, error[:500], now, cid, claim_token),
                )

    def release_clips(self, claim_token: str, clip_ids: list[str]) -> None:
        """Return claimed clips to ``pending`` WITHOUT charging an attempt (guarded by token).

        For run-level events that are not the clip's fault — a dead key (auth outage), an abort,
        clips claimed but never processed. ``claim_clips`` charges ``attempts`` at claim time, so
        a key outage would otherwise burn every clip's retry budget and permanently fail the head
        of the queue before the run-level renewal/abort machinery resolves it.
        """
        now = time.time()
        with self._tx():
            for cid in clip_ids:
                self._conn.execute(
                    "UPDATE clips SET status='pending', attempts=attempts-1, claim_token=NULL, "
                    "locked_at=NULL, updated_at=? "
                    "WHERE clip_id=? AND status='labeling' AND claim_token=?",
                    (now, cid, claim_token),
                )

    def reclaim_stale_clips(self, lease_s: float) -> int:
        """Expired-lease clips back to ``pending``, token cleared (so late writes are rejected)."""
        now = time.time()
        with self._tx():
            cur = self._conn.execute(
                "UPDATE clips SET status='pending', claim_token=NULL, locked_at=NULL, updated_at=? "
                "WHERE status='labeling' AND locked_at < ?",
                (now, now - lease_s),
            )
            return cur.rowcount

    # -- harvest -----------------------------------------------------------------------------

    def harvestable(self, limit: int = 2000) -> list[QLabeledClip]:
        """Done-but-unharvested clips, oldest first."""
        rows = self._conn.execute(
            "SELECT * FROM clips WHERE status='done' AND harvested_at IS NULL "
            "ORDER BY clip_id LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            QLabeledClip(
                r["clip_id"],
                r["video_id"],
                r["channel"],
                r["clip_index"],
                r["clip_path"],
                r["start"],
                r["end"],
                r["language"],
                r["citation"],
                r["label"] or "",
                r["variants"],
                tier=r["tier"],
                category=r["category"],
                meta=_load_meta(r["meta"]),
            )
            for r in rows
        ]

    def mark_harvested(self, clip_ids: list[str]) -> None:
        now = time.time()
        with self._tx():
            self._conn.executemany(
                "UPDATE clips SET harvested_at=? WHERE clip_id=?", [(now, c) for c in clip_ids]
            )

    # -- progress ----------------------------------------------------------------------------

    def status_counts(self) -> dict[str, dict[str, int]]:
        """``{'videos': {status: n}, 'clips': {status: n}}`` for progress display."""
        out: dict[str, dict[str, int]] = {}
        for table in ("videos", "clips"):
            rows = self._conn.execute(
                f"SELECT status, count(*) AS n FROM {table} GROUP BY status"  # noqa: S608 — fixed names
            ).fetchall()
            out[table] = {r["status"]: r["n"] for r in rows}
        return out
