"""Verify breaker: a failing service must never become a request spray.

The deployed superwhisper service owns ASR key rotation, so the curator no longer renews
ElevenLabs keys in-run (the old dead-key circuit-breaker is gone). What remains, and what these
tests pin, is the generic consecutive-failure abort: a persistent failure (e.g. a service
outage) trips ``breaker_threshold`` and aborts the run after FAR fewer calls than the full
pool, and a label that normalizes to nothing never burns a call at all.
"""

from __future__ import annotations

from unittest import mock

import pytest

import omni_curator.verify as V
from omni_curator.sample import Sample
from omni_curator.store import CuratorStore


def _seed(tmp_path, name: str, n: int) -> CuratorStore:
    store = CuratorStore(tmp_path / name)
    store.upsert(
        [
            Sample(id=f"c{i:04d}", source="x", language="tgk_Cyrl", text="салом дӯстон",
                   audio_path=f"/nonexistent/{i:04d}.flac", duration=2.0, sample_rate=16_000)
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


def test_persistent_failure_aborts_with_bounded_calls(tmp_path):
    """A failing service trips the consecutive-failure breaker after far fewer calls than rows."""
    calls = {"n": 0}

    def broken_fn(path):
        calls["n"] += 1
        return {"error": "503 Service Unavailable"}

    store = _seed(tmp_path, "broken.sqlite", 400)
    with mock.patch.object(V, "make_scribe_fns", return_value={"auto": broken_fn}), \
         pytest.raises(RuntimeError, match="consecutive failures"):
        V.verify_store(store, workers=10, breaker_threshold=50)
    assert calls["n"] < 400, f"sprayed {calls['n']} calls at a dead service"
    store.close()


def test_consecutive_nonauth_failures_trip_the_breaker(tmp_path):
    calls = {"n": 0}

    def broken_fn(path):
        calls["n"] += 1
        return {"error": "ffmpeg exploded"}

    store = _seed(tmp_path, "ffmpeg.sqlite", 400)
    with mock.patch.object(V, "make_scribe_fns", return_value={"auto": broken_fn}), \
         pytest.raises(RuntimeError, match="consecutive failures"):
        V.verify_store(store, workers=10, breaker_threshold=50)
    assert calls["n"] < 400
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
            Sample(id="j0", source="x", language="tgk_Cyrl", text="...",
                   audio_path="/nonexistent/j0.flac", duration=2.0, sample_rate=16_000),
            Sample(id="j1", source="x", language="tgk_Cyrl", text="♪",
                   audio_path="/nonexistent/j1.flac", duration=2.0, sample_rate=16_000),
        ]
    )
    with mock.patch.object(V, "make_scribe_fns", return_value={"auto": fn}):
        stats = V.verify_store(store, workers=2)
    assert calls["n"] == 0
    assert stats.unscoreable == 2
    store.close()
