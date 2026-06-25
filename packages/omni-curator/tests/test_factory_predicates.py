"""The v0 enqueue/segment predicates against temp queue sqlite DBs (factory_plan §2)."""

from __future__ import annotations

import time

from omni_curator.create.queue import QueueStore, QVideo
from omni_curator.factory import predicates


def _enqueue(queue_path, videos):
    q = QueueStore(queue_path)
    q.enqueue_videos(videos)
    q.close()


# -- enqueue predicate -------------------------------------------------------------------------


def test_enqueue_false_when_create_root_empty(tmp_path):
    create = tmp_path / "create"
    create.mkdir()
    assert predicates.enqueue_needed(tmp_path / "queue.sqlite", create) is False


def test_enqueue_true_when_flac_absent_from_queue(tmp_path):
    create = tmp_path / "create"
    (create / "chan").mkdir(parents=True)
    (create / "chan" / "vid001.flac").touch()
    # queue has no videos yet -> the on-disk FLAC is new
    assert predicates.enqueue_needed(tmp_path / "queue.sqlite", create) is True


def test_enqueue_false_when_all_flacs_already_enqueued(tmp_path):
    create = tmp_path / "create"
    (create / "chan").mkdir(parents=True)
    (create / "chan" / "vid001.flac").touch()
    qpath = tmp_path / "queue.sqlite"
    # video_id is "<channel>_<stem>" == "chan_vid001" (matches cmd_enqueue)
    _enqueue(qpath, [QVideo("chan_vid001", "chan", "/x.flac", "noisy", None)])
    assert predicates.enqueue_needed(qpath, create) is False


def test_enqueue_true_when_one_new_flac_among_enqueued(tmp_path):
    create = tmp_path / "create"
    (create / "chan").mkdir(parents=True)
    (create / "chan" / "vid001.flac").touch()
    (create / "chan" / "vid002.flac").touch()
    qpath = tmp_path / "queue.sqlite"
    _enqueue(qpath, [QVideo("chan_vid001", "chan", "/x.flac", "noisy", None)])
    assert predicates.enqueue_needed(qpath, create) is True  # vid002 still new


def test_flac_video_ids_derives_channel_from_parent_dir(tmp_path):
    create = tmp_path / "create"
    (create / "alpha").mkdir(parents=True)
    (create / "beta").mkdir(parents=True)
    (create / "alpha" / "a1.flac").touch()
    (create / "beta" / "b1.flac").touch()
    assert predicates.flac_video_ids(create) == {"alpha_a1", "beta_b1"}


# -- segment predicate -------------------------------------------------------------------------


def test_segment_false_when_no_queue(tmp_path):
    assert predicates.segment_needed(tmp_path / "missing.sqlite") is False
    assert predicates.segment_backlog(tmp_path / "missing.sqlite") == 0


def test_segment_true_with_pending_videos(tmp_path):
    qpath = tmp_path / "queue.sqlite"
    _enqueue(qpath, [QVideo("chan_v1", "chan", "/v1.flac", "noisy", None)])
    assert predicates.segment_backlog(qpath) == 1
    assert predicates.segment_needed(qpath) is True


def test_segment_false_when_all_segmented(tmp_path):
    qpath = tmp_path / "queue.sqlite"
    q = QueueStore(qpath)
    q.enqueue_videos([QVideo("chan_v1", "chan", "/v1.flac", "noisy", None)])
    claimed = q.claim_video("w1")
    assert claimed is not None
    q.complete_video("chan_v1", [], claim_token=claimed.claim_token)
    q.close()
    assert predicates.segment_needed(qpath) is False  # status='segmented', not claimable


def test_segment_fresh_segmenting_lease_is_not_claimable(tmp_path):
    qpath = tmp_path / "queue.sqlite"
    q = QueueStore(qpath)
    q.enqueue_videos([QVideo("chan_v1", "chan", "/v1.flac", "noisy", None)])
    q.claim_video("w1")  # now status='segmenting', locked_at=now (fresh)
    q.close()
    assert predicates.segment_needed(qpath) is False  # a live segmenter owns it


def test_segment_stale_segmenting_lease_is_claimable(tmp_path):
    qpath = tmp_path / "queue.sqlite"
    q = QueueStore(qpath)
    q.enqueue_videos([QVideo("chan_v1", "chan", "/v1.flac", "noisy", None)])
    q.claim_video("w1")
    q.close()
    # evaluate "now" far enough in the future that the lease is stale (mirrors claim_video)
    future = time.time() + predicates.SEGMENT_STALE_AFTER_S + 1.0
    assert predicates.segment_backlog(qpath, now=future) == 1
    assert predicates.segment_needed(qpath, now=future) is True
