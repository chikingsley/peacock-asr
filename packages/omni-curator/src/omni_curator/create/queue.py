"""SQLite work queue decoupling segmentation (CPU producer) from labeling (I/O consumer).

The fused create path (``label_to_store`` -> ``vad_path``) runs VAD then Scribe sequentially per
video in one process, so neither the CPU (VAD) nor the network (Scribe) is ever saturated. This
queue splits the two stages:

- ``segment`` workers claim a *video* (rare, coarse claims), VAD-segment it, cut the clips to disk,
  and enqueue ``clips`` rows.
- ``labelq`` (one dispatcher + an HTTP thread pool) batch-claims *clips*, Scribe-labels them, and
  writes the labels back here.
- ``harvest`` folds done clips into the canonical per-channel ``CuratorStore``.

The queue DB (``data/queue.sqlite``) is transient work-state, deliberately separate from the
canonical store. Concurrency safety: WAL mode, ``BEGIN IMMEDIATE`` for the (rare) claim writes, a
lease (``locked_at``) for crash recovery, and a per-claim ``claim_token`` so a result from a
reclaimed lease can never overwrite the retry that replaced it. See ``docs/PIPELINE_SPLIT.md``.
"""

from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    video_id   TEXT PRIMARY KEY,
    channel    TEXT NOT NULL,
    path       TEXT NOT NULL,
    tier       TEXT NOT NULL,
    citation   TEXT,
    status     TEXT NOT NULL DEFAULT 'pending',   -- pending|segmenting|segmented|failed
    n_clips    INTEGER,
    attempts   INTEGER NOT NULL DEFAULT 0,
    locked_by  TEXT,
    locked_at  REAL,
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
    start       REAL NOT NULL,
    end         REAL NOT NULL,
    language    TEXT NOT NULL,
    script      TEXT NOT NULL,
    citation    TEXT,
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


class QueueStore:
    """SQLite-backed segment/label work queue. One instance per process (own connection)."""

    def __init__(self, db_path: Path, *, busy_timeout_ms: int = 60_000) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.path = db_path
        self._conn = sqlite3.connect(db_path, timeout=busy_timeout_ms / 1000)
        self._conn.row_factory = sqlite3.Row
        self._conn.isolation_level = None  # explicit BEGIN/COMMIT (see _tx)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
        self._conn.executescript(_SCHEMA)

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
                "(video_id, channel, path, tier, citation, updated_at) VALUES (?,?,?,?,?,?)",
                [(v.video_id, v.channel, v.path, v.tier, v.citation, now) for v in videos],
            )
            return cur.rowcount

    # -- segment stage -----------------------------------------------------------------------

    def claim_video(self, worker_id: str, *, stale_after_s: float = 1800.0) -> QVideo | None:
        """Claim the next pending (or stale-segmenting) video under a lease. ``None`` if drained."""
        now = time.time()
        with self._tx(immediate=True):
            row = self._conn.execute(
                "SELECT * FROM videos WHERE status='pending' "
                "OR (status='segmenting' AND locked_at < ?) ORDER BY status LIMIT 1",
                (now - stale_after_s,),
            ).fetchone()
            if row is None:
                return None
            self._conn.execute(
                "UPDATE videos SET status='segmenting', locked_by=?, locked_at=?, "
                "attempts=attempts+1, updated_at=? WHERE video_id=?",
                (worker_id, now, now, row["video_id"]),
            )
        return QVideo(row["video_id"], row["channel"], row["path"], row["tier"], row["citation"])

    def complete_video(self, video_id: str, clips: list[QClip]) -> None:
        """Enqueue a segmented video's clips and mark it done — one txn (no half-cut clip view)."""
        now = time.time()
        cols = (
            "clip_id, video_id, channel, clip_index, clip_path, "
            "start, end, language, script, citation, updated_at"
        )
        with self._tx():
            self._conn.executemany(
                f"INSERT OR IGNORE INTO clips ({cols}) VALUES (?,?,?,?,?,?,?,?,?,?,?)",  # noqa: S608 — fixed column list
                [
                    (c.clip_id, c.video_id, c.channel, c.clip_index, c.clip_path, c.start, c.end,
                     c.language, c.script, c.citation, now)
                    for c in clips
                ],
            )
            self._conn.execute(
                "UPDATE videos SET status='segmented', n_clips=?, locked_by=NULL, locked_at=NULL, "
                "updated_at=? WHERE video_id=?",
                (len(clips), now, video_id),
            )

    def fail_video(self, video_id: str, error: str, *, max_attempts: int = 3) -> None:
        """Record a segmentation failure; requeue until ``max_attempts``, then mark failed."""
        now = time.time()
        with self._tx():
            row = self._conn.execute(
                "SELECT attempts FROM videos WHERE video_id=?", (video_id,)
            ).fetchone()
            status = "failed" if row and row["attempts"] >= max_attempts else "pending"
            self._conn.execute(
                "UPDATE videos SET status=?, last_error=?, locked_by=NULL, locked_at=NULL, "
                "updated_at=? WHERE video_id=?",
                (status, error[:500], now, video_id),
            )

    def pending_clip_count(self) -> int:
        """Clips not yet done — the backpressure signal for the segmenter's high-watermark."""
        return self._conn.execute(
            "SELECT count(*) FROM clips WHERE status IN ('pending', 'labeling')"
        ).fetchone()[0]

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
                r["clip_id"], r["video_id"], r["channel"], r["clip_index"], r["clip_path"],
                r["start"], r["end"], r["language"], r["script"], r["citation"],
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

    def fail_clips(self, claim_token: str, clip_ids: list[str], error: str,
                   *, max_attempts: int = 3) -> None:
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
                r["clip_id"], r["video_id"], r["channel"], r["clip_index"], r["clip_path"],
                r["start"], r["end"], r["language"], r["citation"], r["label"] or "", r["variants"],
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
