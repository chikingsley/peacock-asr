"""Verify breaker + key renewal: a dead key must never become a request spray.

Re-creates (as durable tests) the no-network simulations that validated the breaker
(commits 584e2006, c5b4cedf): bounded calls on a dead key, in-run recovery when renewal
fixes it, and the generic consecutive-failure abort.
"""

from __future__ import annotations

from unittest import mock

import pytest

import omni_curator.verify as V
from omni_curator.sample import Sample
from omni_curator.store import CuratorStore


class FakeResult:
    def __init__(self, payload: dict):
        self._payload = payload

    def as_dict(self) -> dict:
        return self._payload


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


def test_dead_key_aborts_with_bounded_calls(tmp_path):
    """Renewal never helps -> abort after max_renewals, having called FAR fewer than all rows."""
    calls = {"n": 0}

    def dead_fn(path):
        calls["n"] += 1
        return FakeResult({"error": "Client error '401 Unauthorized' for url x"})

    store = _seed(tmp_path, "dead.sqlite", 400)
    with mock.patch.object(V, "make_scribe_fns", return_value={"auto": dead_fn}), \
         mock.patch.object(V, "renew_scribe_key", return_value="sk_new"), \
         mock.patch.object(V, "default_key", return_value="sk_old"), \
         pytest.raises(RuntimeError, match="renewal exhausted"):
        V.verify_store(store, workers=10, max_renewals=2)
    assert calls["n"] < 400, f"sprayed {calls['n']} calls at a dead key"
    store.close()


def test_renewal_recovers_mid_run(tmp_path):
    """Key dies, renewal fixes it -> the run completes and everything gets scored."""
    state = {"key": "dead"}

    def flaky_fn(path):
        if state["key"] == "dead":
            return FakeResult({"error": "401 Unauthorized"})
        return FakeResult({"transcript": "салом дӯстон"})

    def renew():
        state["key"] = "fresh"
        return "sk_new"

    store = _seed(tmp_path, "flaky.sqlite", 120)
    with mock.patch.object(V, "make_scribe_fns", return_value={"auto": flaky_fn}), \
         mock.patch.object(V, "renew_scribe_key", renew), \
         mock.patch.object(V, "default_key", return_value="sk_old"):
        stats = V.verify_store(store, workers=10, max_renewals=3)
    assert stats.renewals == 1
    assert stats.scored > 0
    assert stats.scored + stats.failed == 120  # every row accounted for
    store.close()


def test_consecutive_nonauth_failures_trip_the_breaker(tmp_path):
    calls = {"n": 0}

    def broken_fn(path):
        calls["n"] += 1
        return FakeResult({"error": "ffmpeg exploded"})

    store = _seed(tmp_path, "broken.sqlite", 400)
    with mock.patch.object(V, "make_scribe_fns", return_value={"auto": broken_fn}), \
         mock.patch.object(V, "default_key", return_value="sk"), \
         pytest.raises(RuntimeError, match="consecutive failures"):
        V.verify_store(store, workers=10, breaker_threshold=50)
    assert calls["n"] < 400
    store.close()


def test_unscoreable_labels_skipped_without_any_call(tmp_path):
    """Labels that normalize to nothing ('♪', '...') must not burn a Scribe call."""
    calls = {"n": 0}

    def fn(path):
        calls["n"] += 1
        return FakeResult({"transcript": "x"})

    store = CuratorStore(tmp_path / "junk.sqlite")
    store.upsert(
        [
            Sample(id="j0", source="x", language="tgk_Cyrl", text="...",
                   audio_path="/nonexistent/j0.flac", duration=2.0, sample_rate=16_000),
            Sample(id="j1", source="x", language="tgk_Cyrl", text="♪",
                   audio_path="/nonexistent/j1.flac", duration=2.0, sample_rate=16_000),
        ]
    )
    with mock.patch.object(V, "make_scribe_fns", return_value={"auto": fn}), \
         mock.patch.object(V, "default_key", return_value="sk"):
        stats = V.verify_store(store, workers=2)
    assert calls["n"] == 0
    assert stats.unscoreable == 2
    store.close()
