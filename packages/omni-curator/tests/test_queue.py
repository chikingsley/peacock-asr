"""QueueStore contract: claims, token-guarded writebacks, leases, attempt fairness, harvest.

These encode the invariants the split pipeline depends on:
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


def test_repair_video_metadata_updates_existing_rows(queue):
    assert queue.enqueue_videos(_videos(1)) == 1

    result = queue.repair_video_metadata(
        [
            QVideo(
                "chan_v000",
                "chan",
                "/audio/v000.flac",
                "clean",
                "https://example.test/channel",
                category="news",
                meta={"webpage_url": "https://www.youtube.com/watch?v=v000"},
            )
        ]
    )
    assert result.matched == 1
    assert result.changed == 1
    assert result.updated == 1

    video = queue.claim_video("seg-0")
    assert video is not None
    assert video.tier == "clean"
    assert video.category == "news"
    assert video.meta["webpage_url"] == "https://www.youtube.com/watch?v=v000"


def test_video_claim_complete_cycle(queue):
    queue.enqueue_videos(_videos(1))
    video = queue.claim_video("seg-0")
    assert video is not None
    assert video.video_id == "chan_v000"
    assert queue.claim_video("seg-1") is None  # claimed -> nothing pending

    queue.complete_video(video.video_id, _clips(video.video_id, 4), claim_token=video.claim_token)
    counts = queue.status_counts()
    assert counts["videos"] == {"segmented": 1}
    assert counts["clips"] == {"pending": 4}


def test_source_metadata_survives_queue_lifecycle(queue):
    video = QVideo(
        "chan_v000",
        "chan",
        "/audio/v000.flac",
        "clean",
        "https://example.test/channel",
        category="news",
        meta={"title": "Video title", "upload_date": "20250102"},
    )
    assert queue.enqueue_videos([video]) == 1

    claimed_video = queue.claim_video("seg-0")
    assert claimed_video is not None
    assert claimed_video.category == "news"
    assert claimed_video.meta["title"] == "Video title"

    clips = [
        QClip(
            "chan_v000_0000",
            "chan_v000",
            "chan",
            0,
            "/clips/chan_v000/seg_0000.flac",
            0.0,
            5.0,
            "tgk_Cyrl",
            "Cyrillic",
            "https://example.test/channel",
            tier=claimed_video.tier,
            category=claimed_video.category,
            meta=dict(claimed_video.meta),
        )
    ]
    assert queue.complete_video(
        claimed_video.video_id, clips, claim_token=claimed_video.claim_token
    )

    claimed_clip = queue.claim_clips(1, "label-token")[0]
    assert claimed_clip.tier == "clean"
    assert claimed_clip.category == "news"
    assert claimed_clip.meta["upload_date"] == "20250102"

    assert queue.complete_clips("label-token", [(claimed_clip.clip_id, "label", "[]")]) == 1
    ready = queue.harvestable(limit=1)
    assert ready[0].tier == "clean"
    assert ready[0].category == "news"
    assert ready[0].meta["title"] == "Video title"


def test_video_completion_requires_matching_token(queue):
    """A stale segmenter whose video was reclaimed must not double-complete or flip the row."""
    queue.enqueue_videos(_videos(1))
    first = queue.claim_video("seg-0")
    assert first is not None
    second = queue.claim_video("seg-1", stale_after_s=0.0)  # lease stale -> reclaimed, fresh token
    assert second is not None
    assert second.claim_token != first.claim_token

    # the original worker returns late: its stale token writes nothing
    queue.complete_video(first.video_id, _clips(first.video_id, 3), claim_token=first.claim_token)
    assert queue.status_counts()["videos"] == {"segmenting": 1}  # still owned by second
    assert queue.status_counts().get("clips", {}) == {}

    # the current owner completes successfully
    queue.complete_video(
        second.video_id, _clips(second.video_id, 2), claim_token=second.claim_token
    )
    assert queue.status_counts()["videos"] == {"segmented": 1}
    assert queue.status_counts()["clips"] == {"pending": 2}


def test_video_publish_callback_requires_matching_token(queue):
    """Stale segmenters must not publish clip files after their SQLite claim was reclaimed."""
    queue.enqueue_videos(_videos(1))
    first = queue.claim_video("seg-0")
    assert first is not None
    second = queue.claim_video("seg-1", stale_after_s=0.0)
    assert second is not None

    published: list[str] = []
    stale_ok = queue.complete_video(
        first.video_id,
        _clips(first.video_id, 1),
        claim_token=first.claim_token,
        publish=lambda: published.append("stale"),
    )
    assert stale_ok is False
    assert published == []

    current_ok = queue.complete_video(
        second.video_id,
        _clips(second.video_id, 1),
        claim_token=second.claim_token,
        publish=lambda: published.append("current"),
    )
    assert current_ok is True
    assert published == ["current"]


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


def test_claim_prefers_stale_segmenting_over_pending(queue):
    """A lapsed-lease segmenting video must be reclaimed BEFORE fresh pending work (anti-starve)."""
    # v000 claimed then left to go stale; v001 is fresh pending.
    queue.enqueue_videos(_videos(2))
    stale = queue.claim_video("seg-0")  # claims v000 (status -> segmenting)
    assert stale is not None
    assert stale.video_id == "chan_v000"

    # Next claim with an already-expired stale window: the stale segmenting v000 must win over the
    # pending v001, even though v001 is pending and would sort first under a naive ORDER BY status.
    reclaimed = queue.claim_video("seg-1", stale_after_s=-1.0)
    assert reclaimed is not None
    assert reclaimed.video_id == "chan_v000"  # stale segmenting reclaimed first
    assert reclaimed.claim_token != stale.claim_token

    # With v000 freshly leased again (NOT yet stale under a real 30-min window), the only claimable
    # left is the pending v001 — proving the reclaim didn't steal it from genuine pending work.
    nxt = queue.claim_video("seg-2")
    assert nxt is not None
    assert nxt.video_id == "chan_v001"


def test_resegment_resets_videos_and_clears_clips(queue):
    """resegment sends segmented/failed videos back to pending and wipes clip rows (idempotent)."""
    queue.enqueue_videos(_videos(2))
    # v000 -> segmented (with clips); v001 -> failed.
    queue.complete_video("chan_v000", _clips("chan_v000", 3))
    f = queue.claim_video("seg-x")
    assert f is not None
    for _ in range(5):  # exhaust attempts -> failed
        queue.fail_video(f.video_id, "boom", claim_token=f.claim_token, max_attempts=1)
        f2 = queue.claim_video("seg-x", stale_after_s=0.0)
        if f2 is None:
            break
        f = f2

    preview = queue.resegment_preview()
    assert preview["clips"] == 3
    assert preview["videos"] >= 1  # at least the segmented one; failed too if it reached failed

    paths = queue.all_clip_paths()
    assert len(paths) == 3

    result = queue.reset_for_resegment()
    assert result["clips"] == 3
    counts = queue.status_counts()
    assert counts.get("clips", {}) == {}  # all clip rows gone
    assert "segmented" not in counts["videos"]
    assert "failed" not in counts["videos"]
    assert counts["videos"]["pending"] == 2  # both back to pending

    # Idempotent: a second reset on the clean queue is a no-op.
    again = queue.reset_for_resegment()
    assert again == {"videos": 0, "clips": 0}


def test_harvest_marks_and_excludes(queue):
    queue.enqueue_videos(_videos(1))
    queue.complete_video("chan_v000", _clips("chan_v000", 2))
    claimed = queue.claim_clips(2, "tok")
    queue.complete_clips("tok", [(c.clip_id, f"label {c.clip_index}", "[]") for c in claimed])

    ready = queue.harvestable(limit=10)
    assert [c.label for c in ready] == ["label 0", "label 1"]
    queue.mark_harvested([c.clip_id for c in ready])
    assert queue.harvestable(limit=10) == []
