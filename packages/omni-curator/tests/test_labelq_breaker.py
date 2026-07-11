"""Unit tests for the labelq dispatcher breaker (_Breaker): ride through blips, then abort.

These pin the decision logic that drives the release-vs-fail choice in run_labeler: ``record``
returns True on a backoff pause (an outage round, whose clips the loop then releases instead of
burning), resets the streak on any success, and aborts only after ``max_pauses`` pauses with no
success in between. ``time.sleep`` is patched so the backoff is instant.
"""

from __future__ import annotations

from unittest import mock

import httpx
import pytest

from omni_curator.create.labelq import _Breaker, _is_retryable, run_labeler
from omni_curator.create.queue import QClip, QueueStore, QVideo


def _errs(n: int) -> list[tuple[object, Exception]]:
    return [(object(), RuntimeError("502 Bad Gateway")) for _ in range(n)]


def _seed_clips(qpath, n: int) -> None:
    """Seed a fresh queue with ``n`` pending clips under one video."""
    q = QueueStore(qpath)
    q.enqueue_videos([QVideo("chan_v000", "chan", "/audio/v000.flac", "noisy", None)])
    q.complete_video(
        "chan_v000",
        [
            QClip(
                f"chan_v000_{i:04d}",
                "chan_v000",
                "chan",
                i,
                f"/clips/{i:04d}.flac",
                i * 10.0,
                i * 10.0 + 5.0,
                "tgk_Cyrl",
                "Cyrillic",
                None,
            )
            for i in range(n)
        ],
    )
    q.close()


def _clip_counts(qpath) -> dict[str, int]:
    after = QueueStore(qpath)
    counts = after.status_counts()["clips"]
    after.close()
    return counts


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (RuntimeError("502 Bad Gateway"), True),
        (RuntimeError("503 Service Unavailable"), True),
        (RuntimeError("Server disconnected without sending a response."), True),
        (RuntimeError("401 Unauthorized"), True),
        (ConnectionError("connection reset by peer"), True),
        (TimeoutError(), True),
        (FileNotFoundError("Source file not found: /clips/0001.flac"), False),
        (RuntimeError("ffmpeg exploded decoding the clip"), False),
        (ValueError("label normalized to nothing"), False),
    ],
)
def test_is_retryable_classifies(exc, expected):
    assert _is_retryable(exc) is expected


def _status_error(code: int) -> httpx.HTTPStatusError:
    req = httpx.Request("POST", "http://svc/v1/transcriptions")
    return httpx.HTTPStatusError("e", request=req, response=httpx.Response(code, request=req))


def test_is_retryable_httpx_status_codes():
    assert _is_retryable(_status_error(503)) is True  # server error -> retry
    assert _is_retryable(_status_error(429)) is True  # rate limit -> retry
    assert _is_retryable(_status_error(408)) is True  # request timeout -> retry
    assert _is_retryable(_status_error(400)) is False  # bad request (bad clip) -> charge + retire
    assert _is_retryable(httpx.ConnectError("connect failed")) is True


def test_record_returns_false_below_threshold():
    b = _Breaker(50, max_pauses=3, base_backoff=0.0)
    assert b.record(0, _errs(10), 0) is False


def test_record_pauses_at_threshold_and_returns_true():
    b = _Breaker(50, max_pauses=3, base_backoff=0.0)
    with mock.patch("omni_curator.create.labelq.time.sleep"):
        assert b.record(0, _errs(50), 0) is True


def test_success_resets_pause_streak():
    """A success between pauses resets the streak; it then takes max_pauses again to abort."""
    b = _Breaker(50, max_pauses=2, base_backoff=0.0)
    with mock.patch("omni_curator.create.labelq.time.sleep"):
        assert b.record(0, _errs(50), 0) is True  # pause 1
        assert b.record(0, _errs(50), 0) is True  # pause 2
        b.record(5, [], 5)  # success -> resets the pause streak
        assert b.record(0, _errs(50), 0) is True  # pause 1 again (would abort without the reset)
        assert b.record(0, _errs(50), 0) is True  # pause 2 again
        with pytest.raises(RuntimeError, match="consecutive failures"):
            b.record(0, _errs(50), 0)  # only NOW aborts


def test_aborts_after_max_pauses():
    b = _Breaker(50, max_pauses=2, base_backoff=0.0)
    with mock.patch("omni_curator.create.labelq.time.sleep"):
        assert b.record(0, _errs(50), 0) is True  # pause 1
        assert b.record(0, _errs(50), 0) is True  # pause 2
        with pytest.raises(RuntimeError, match="consecutive failures"):
            b.record(0, _errs(50), 0)  # 3rd trip with max_pauses=2 -> abort


def test_outage_releases_clips_instead_of_failing_them(tmp_path):
    """End-to-end: a persistent outage aborts but must NOT permanently fail recoverable clips.

    Every clip fails (service down). The breaker eventually aborts, but because each round makes no
    progress, run_labeler releases (un-charges) the clips rather than burning attempts toward the
    cap — so they all stay retryable for the next run, not marked ``failed``.
    """
    qpath = tmp_path / "queue.sqlite"
    _seed_clips(qpath, 30)

    def outage(clip, *, langs, runs, instruction):
        raise RuntimeError("502 Bad Gateway")

    with (
        mock.patch("omni_curator.create.labelq._label_clip", outage),
        mock.patch("omni_curator.create.labelq.time.sleep"),
        pytest.raises(RuntimeError, match="aborted"),
    ):
        run_labeler(qpath, workers=5, breaker_threshold=10, max_pauses=2)

    counts = _clip_counts(qpath)
    assert counts.get("failed", 0) == 0, f"outage permanently failed clips: {counts}"
    assert counts.get("pending", 0) == 30, f"clips not all retryable: {counts}"


def test_corrupt_clips_are_failed_not_released(tmp_path):
    """The counterpart: clip-specific faults (missing/corrupt audio) MUST charge attempts and be
    retired at the cap — never released forever (which would wedge the queue on a bad video).
    """
    qpath = tmp_path / "queue.sqlite"
    _seed_clips(qpath, 30)

    def missing(clip, *, langs, runs, instruction):
        raise FileNotFoundError(f"Source file not found: {clip.clip_path}")

    with (
        mock.patch("omni_curator.create.labelq._label_clip", missing),
        mock.patch("omni_curator.create.labelq.time.sleep"),
        pytest.raises(RuntimeError, match="aborted"),
    ):
        run_labeler(qpath, workers=5, breaker_threshold=10, max_pauses=2)

    counts = _clip_counts(qpath)
    assert counts.get("failed", 0) > 0, f"corrupt clips never retired: {counts}"
