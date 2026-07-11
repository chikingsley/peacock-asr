"""Verify breaker: ride through transient blips, but never spray a genuinely-dead service.

The deployed superwhisper service owns ASR key rotation, so the curator no longer renews
ElevenLabs keys in-run (the old dead-key circuit-breaker is gone). What remains, and what these
tests pin: on a consecutive-failure streak the run backs off and RETRIES (so an intermittent
500/502 blip no longer kills a multi-hour run), but if the service stays down across
``max_pauses`` pauses with no success in between, the run still aborts after far fewer calls
than the full pool. A label that normalizes to nothing never burns a call at all.
"""

from __future__ import annotations

import threading
from unittest import mock

import pytest

import omni_curator.audit.verify as V
from omni_curator.data.sample import Sample
from omni_curator.data.store import CuratorStore


def _seed(tmp_path, name: str, n: int) -> CuratorStore:
    store = CuratorStore(tmp_path / name)
    store.upsert(
        [
            Sample(
                id=f"c{i:04d}",
                source="x",
                language="tgk_Cyrl",
                text="салом дӯстон",
                audio_path=f"/nonexistent/{i:04d}.flac",
                duration=2.0,
                sample_rate=16_000,
            )
            for i in range(n)
        ]
    )
    return store


def test_healthy_run_scores_everything(tmp_path):
    """A service that always succeeds scores every pending clip, no aborts."""

    def good_fn(path):
        return {"transcript": "салом дӯстон"}

    store = _seed(tmp_path, "good.sqlite", 120)
    with mock.patch.object(V, "make_scribe_fns", return_value={"auto": good_fn}):
        stats = V.verify_store(store, workers=10)
    assert stats.scored == 120
    assert stats.failed == 0
    store.close()


@pytest.mark.parametrize("error", ["503 Service Unavailable", "ffmpeg exploded"])
def test_persistent_failure_aborts_with_bounded_calls(tmp_path, error):
    """A service that never recovers aborts after a bounded number of calls (no spray) — whether
    the failure is a server-side error or a per-clip decode error."""
    calls = {"n": 0}

    def broken_fn(path):
        calls["n"] += 1
        return {"error": error}

    store = _seed(tmp_path, "broken.sqlite", 400)
    with (
        mock.patch.object(V, "make_scribe_fns", return_value={"auto": broken_fn}),
        pytest.raises(RuntimeError, match="consecutive failures"),
    ):
        V.verify_store(store, workers=10, breaker_threshold=50, pause_s=0.0, max_pauses=2)
    assert calls["n"] < 400, f"sprayed {calls['n']} calls at a dead service"
    store.close()


def test_transient_blip_recovers_without_aborting(tmp_path):
    """A burst of failures that then recovers must NOT abort — the breaker rides through it."""
    lock = threading.Lock()
    calls = {"n": 0}

    def flaky_fn(path):
        with lock:
            calls["n"] += 1
            n = calls["n"]
        if n <= 60:
            return {"error": "502 Bad Gateway"}
        return {"transcript": "салом дӯстон"}

    store = _seed(tmp_path, "flaky.sqlite", 200)
    with mock.patch.object(V, "make_scribe_fns", return_value={"auto": flaky_fn}):
        stats = V.verify_store(store, workers=10, breaker_threshold=50, pause_s=0.0, max_pauses=5)
    assert stats.scored == 140
    assert stats.failed == 60
    store.close()


def test_blind_cap_limits_calls_before_first_success(tmp_path):
    """A dead service must not get a full (large) window of calls before the breaker trips."""
    calls = {"n": 0}

    def broken_fn(path):
        calls["n"] += 1
        return {"error": "500 Internal Server Error"}

    store = _seed(tmp_path, "blind.sqlite", 400)
    # Large window (workers=200) but breaker_threshold=50; max_pauses=0 aborts on the first trip.
    with (
        mock.patch.object(V, "make_scribe_fns", return_value={"auto": broken_fn}),
        pytest.raises(RuntimeError, match="consecutive failures"),
    ):
        V.verify_store(store, workers=200, breaker_threshold=50, pause_s=0.0, max_pauses=0)
    assert calls["n"] < 120, f"submitted {calls['n']} before the breaker — blind cap regressed"
    store.close()


def test_unscoreable_labels_skipped_without_any_call(tmp_path):
    """Labels that normalize to nothing ('♪', '...') must not burn a Scribe call."""
    calls = {"n": 0}

    def fn(path):
        calls["n"] += 1
        return {"transcript": "x"}

    store = CuratorStore(tmp_path / "junk.sqlite")
    store.upsert(
        [
            Sample(
                id="j0",
                source="x",
                language="tgk_Cyrl",
                text="...",
                audio_path="/nonexistent/j0.flac",
                duration=2.0,
                sample_rate=16_000,
            ),
            Sample(
                id="j1",
                source="x",
                language="tgk_Cyrl",
                text="♪",
                audio_path="/nonexistent/j1.flac",
                duration=2.0,
                sample_rate=16_000,
            ),
        ]
    )
    with mock.patch.object(V, "make_scribe_fns", return_value={"auto": fn}):
        stats = V.verify_store(store, workers=2)
    assert calls["n"] == 0
    assert stats.unscoreable == 2
    store.close()
