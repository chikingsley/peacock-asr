"""The single-writer flock primitive: acquire, contention, release-on-exit (factory_plan §1)."""

from __future__ import annotations

import multiprocessing as mp
import time

import pytest

from omni_curator.factory import flock


def test_acquire_then_release(tmp_path):
    path = flock.lock_path(tmp_path, "segment")
    assert not flock.is_locked(path)  # free
    fd = flock.try_acquire(path)
    assert fd is not None
    assert flock.is_locked(path)  # now held (probe sees a live owner)
    flock.release(fd)
    assert not flock.is_locked(path)  # released


def test_contention_second_acquire_fails_while_held(tmp_path):
    path = flock.lock_path(tmp_path, "enqueue")
    with flock.hold(path):
        assert flock.try_acquire(path) is None  # a second owner can't take it
        with pytest.raises(BlockingIOError), flock.hold(path):
            pass  # unreachable: hold() raises before the body


def test_hold_releases_on_block_exit(tmp_path):
    path = flock.lock_path(tmp_path, "segment")
    with flock.hold(path):
        assert flock.is_locked(path)
    assert not flock.is_locked(path)  # auto-released


def _hold_then_sleep(path_str: str, hold_s: float) -> None:
    from pathlib import Path

    from omni_curator.factory import flock as fl

    with fl.hold(Path(path_str)):
        time.sleep(hold_s)


def test_lock_released_when_owner_process_dies(tmp_path):
    """A child holding the lock that is killed (kill -9 analogue) frees it for the supervisor."""
    path = flock.lock_path(tmp_path, "segment")
    ctx = mp.get_context("spawn")
    proc = ctx.Process(target=_hold_then_sleep, args=(str(path), 30.0))
    proc.start()
    try:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and not flock.is_locked(path):
            time.sleep(0.05)
        assert flock.is_locked(path)  # child owns it
        proc.kill()  # SIGKILL: no chance to clean up; the kernel must drop the flock
        proc.join(10.0)
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and flock.is_locked(path):
            time.sleep(0.05)
        assert not flock.is_locked(path)  # freed on death
    finally:
        if proc.is_alive():
            proc.kill()
            proc.join(5.0)
