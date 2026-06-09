"""QueueStore contract: claims, token-guarded writebacks, leases, attempt fairness, harvest.

These encode the invariants the split pipeline depends on (see docs/PIPELINE_SPLIT.md):
a reclaimed lease's late write can never land, and run-level outages (a dead key) never
burn a clip's retry budget.
"""

from __future__ import annotations

import pytest

from omni_curator.create.queue import QClip, QueueStore, QVideo


@pytest.fixture
def queue(tmp_path):
    q = QueueStore(tmp_path / "queue.sqlite")
    yield q
    q.close()


def _videos(n: int) -> list[QVideo]:
    return [
        QVideo(f"chan_v{i:03d}", "chan", f"/audio/v{i:03d}.flac", "noisy", None)
        for i in range(n)
    ]


def _clips(video_id: str, n: int) -> list[QClip]:
    return [
        QClip(f"{video_id}_{i:04d}", video_id, "chan", i, f"/clips/{video_id}/{i:04d}.flac",
              i * 10.0, i * 10.0 + 5.0, "tgk_Cyrl", "Cyrillic", None)
        for i in range(n)
    ]


def test_enqueue_is_idempotent(queue):
    assert queue.enqueue_videos(_videos(3)) == 3
    assert queue.enqueue_videos(_videos(3)) == 0  # same PKs -> no new rows


def test_video_claim_complete_cycle(queue):
    queue.enqueue_videos(_videos(1))
    video = queue.claim_video("seg-0")
    assert video is not None
    assert video.video_id == "chan_v000"
    assert queue.claim_video("seg-1") is None  # claimed -> nothing pending

    queue.complete_video(video.video_id, _clips(video.video_id, 4))
    counts = queue.status_counts()
    assert counts["videos"] == {"segmented": 1}
    assert counts["clips"] == {"pending": 4}


def test_clip_writeback_requires_matching_token(queue):
    queue.enqueue_videos(_videos(1))
    queue.complete_video("chan_v000", _clips("chan_v000", 2))

    claimed = queue.claim_clips(10, "token-A")
    assert len(claimed) == 2
    # A stale/foreign token writes nothing — the reclaim-safety guarantee.
    assert queue.complete_clips("token-B", [(c.clip_id, "label", "[]") for c in claimed]) == 0
    assert queue.complete_clips("token-A", [(c.clip_id, "label", "[]") for c in claimed]) == 2
    assert queue.status_counts()["clips"] == {"done": 2}


def test_reclaim_clears_token_so_late_write_is_rejected(queue):
    queue.enqueue_videos(_videos(1))
    queue.complete_video("chan_v000", _clips("chan_v000", 1))
    claimed = queue.claim_clips(1, "token-old")

    assert queue.reclaim_stale_clips(lease_s=0.0) == 1  # lease expired immediately
    # The original claimant comes back after the reclaim: its write must not land.
    assert queue.complete_clips("token-old", [(claimed[0].clip_id, "late", "[]")]) == 0
    assert queue.status_counts()["clips"] == {"pending": 1}


def test_release_does_not_charge_attempts_but_fail_does(queue):
    queue.enqueue_videos(_videos(1))
    queue.complete_video("chan_v000", _clips("chan_v000", 1))

    # Three release cycles (e.g. dead-key outages): no attempt burn, still claimable.
    for round_ in range(3):
        claimed = queue.claim_clips(1, f"tok-{round_}")
        assert len(claimed) == 1, f"clip not claimable on round {round_}"
        queue.release_clips(f"tok-{round_}", [claimed[0].clip_id])

    # Genuine failures ARE capped: claim+fail until the queue marks it failed.
    for round_ in range(10):
        claimed = queue.claim_clips(1, f"fail-{round_}")
        if not claimed:
            break
        queue.fail_clips(f"fail-{round_}", [claimed[0].clip_id], "boom", max_attempts=3)
    assert queue.status_counts()["clips"] == {"failed": 1}


def test_harvest_marks_and_excludes(queue):
    queue.enqueue_videos(_videos(1))
    queue.complete_video("chan_v000", _clips("chan_v000", 2))
    claimed = queue.claim_clips(2, "tok")
    queue.complete_clips("tok", [(c.clip_id, f"label {c.clip_index}", "[]") for c in claimed])

    ready = queue.harvestable(limit=10)
    assert [c.label for c in ready] == ["label 0", "label 1"]
    queue.mark_harvested([c.clip_id for c in ready])
    assert queue.harvestable(limit=10) == []
