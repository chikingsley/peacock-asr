"""The enqueue/segment predicates against temp queue sqlite DBs."""

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


# -- v1 predicates: labelq / harvest / archive -------------------------------------------------


def _segment_into_clips(qpath, video_id="chan_v1", n=2):
    """Drive a video to 'segmented' with ``n`` pending clips enqueued."""
    from omni_curator.create.queue import QClip

    q = QueueStore(qpath)
    q.enqueue_videos([QVideo(video_id, "chan", "/v1.flac", "noisy", None)])
    claimed = q.claim_video("w1")
    assert claimed is not None
    clips = [
        QClip(
            f"{video_id}_{i:04d}",
            video_id,
            "chan",
            i,
            f"/clips/{i}.flac",
            i * 5.0,
            i * 5.0 + 5.0,
            "tgk_Cyrl",
            "Cyrillic",
            None,
        )
        for i in range(n)
    ]
    q.complete_video(video_id, clips, claim_token=claimed.claim_token)
    return q


def test_labelq_true_with_pending_clips(tmp_path):
    qpath = tmp_path / "queue.sqlite"
    _segment_into_clips(qpath, n=3).close()
    assert predicates.labelq_backlog(qpath) == 3
    assert predicates.labelq_needed(qpath) is True


def test_labelq_false_with_no_queue(tmp_path):
    assert predicates.labelq_needed(tmp_path / "missing.sqlite") is False


def test_labelq_fresh_labeling_lease_not_claimable(tmp_path):
    qpath = tmp_path / "queue.sqlite"
    q = _segment_into_clips(qpath, n=2)
    q.claim_clips(10, "tok")  # status='labeling', fresh lease
    q.close()
    assert predicates.labelq_needed(qpath) is False


def test_labelq_stale_labeling_lease_is_claimable(tmp_path):
    qpath = tmp_path / "queue.sqlite"
    q = _segment_into_clips(qpath, n=2)
    q.claim_clips(10, "tok")
    q.close()
    future = time.time() + predicates.LABELQ_STALE_AFTER_S + 1.0
    assert predicates.labelq_backlog(qpath, now=future) == 2


def test_harvest_true_for_done_unharvested(tmp_path):
    qpath = tmp_path / "queue.sqlite"
    q = _segment_into_clips(qpath, n=2)
    q.claim_clips(10, "tok")
    q.complete_clips("tok", [("chan_v1_0000", "a", ""), ("chan_v1_0001", "b", "")])
    q.close()
    assert predicates.harvest_backlog(qpath) == 2
    assert predicates.harvest_needed(qpath) is True


def test_harvest_false_after_harvested(tmp_path):
    qpath = tmp_path / "queue.sqlite"
    q = _segment_into_clips(qpath, n=1)
    q.claim_clips(10, "tok")
    q.complete_clips("tok", [("chan_v1_0000", "a", "")])
    q.mark_harvested(["chan_v1_0000"])
    q.close()
    assert predicates.harvest_needed(qpath) is False


def test_archive_true_when_segmented_source_exists(tmp_path):
    src = tmp_path / "v1.flac"
    src.touch()
    qpath = tmp_path / "queue.sqlite"
    q = QueueStore(qpath)
    q.enqueue_videos([QVideo("chan_v1", "chan", str(src), "noisy", None)])
    claimed = q.claim_video("w1")
    assert claimed is not None
    q.complete_video("chan_v1", [], claim_token=claimed.claim_token)
    q.close()
    assert predicates.archive_needed(qpath) is True


def test_archive_false_when_source_missing(tmp_path):
    qpath = tmp_path / "queue.sqlite"
    q = QueueStore(qpath)
    q.enqueue_videos([QVideo("chan_v1", "chan", "/gone/v1.flac", "noisy", None)])
    claimed = q.claim_video("w1")
    assert claimed is not None
    q.complete_video("chan_v1", [], claim_token=claimed.claim_token)
    q.close()
    assert predicates.archive_needed(qpath) is False


def test_archive_false_when_only_pending(tmp_path):
    src = tmp_path / "v1.flac"
    src.touch()
    qpath = tmp_path / "queue.sqlite"
    _enqueue(qpath, [QVideo("chan_v1", "chan", str(src), "noisy", None)])
    assert predicates.archive_needed(qpath) is False  # not segmented yet
