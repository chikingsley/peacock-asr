"""Tests for the abortable-download guard (P4).

The yt-dlp subprocess loop itself needs the network and is exercised by the live drain; the guard
that decides *whether to start the next chunk* is pure and tested here against a real filesystem
(real ``statvfs``), no mocks.
"""

from __future__ import annotations

from omni_curator.create.youtube import _download_blocked


def test_guard_passes_when_unconstrained(tmp_path):
    # neither a floor nor an abort signal -> never blocked (legacy single-pass behavior)
    assert _download_blocked(tmp_path, min_free_gb=None, abort=None) is None


def test_abort_signal_blocks(tmp_path):
    assert _download_blocked(tmp_path, min_free_gb=None, abort=lambda: True) == "hard-halt signal"
    # an abort callback that stays False does not block
    assert _download_blocked(tmp_path, min_free_gb=None, abort=lambda: False) is None


def test_disk_floor_blocks_below_threshold(tmp_path):
    # a floor of 0 GB is always satisfied (free space is positive) -> not blocked
    assert _download_blocked(tmp_path, min_free_gb=0.0, abort=None) is None
    # an absurd floor can never be met -> blocked, with the free/floor numbers in the reason
    reason = _download_blocked(tmp_path, min_free_gb=1e12, abort=None)
    assert reason is not None
    assert "disk floor" in reason


def test_abort_takes_precedence_over_disk(tmp_path):
    # hard-halt is checked first: it wins even when disk is also fine
    assert _download_blocked(tmp_path, min_free_gb=0.0, abort=lambda: True) == "hard-halt signal"
