"""The single-writer ``flock`` primitive — replaces pidfiles.

Each long-running stage is owned by a per-(project, stage) exclusive ``flock`` on
``<project>/data/.lock.<stage>``. The stage process acquires it at startup and holds it for its
lifetime; the kernel releases it automatically on exit/crash/``kill -9`` (no stale-PID problem, no
PID reuse). The supervisor decides "is stage X already running?" by a **non-blocking** try-acquire:

- can't acquire -> a live owner exists (manual OR factory-launched); leave it alone.
- can acquire   -> no one owns the stage; it is free to launch.

A child *launched by* the factory acquires the lock itself (the curate stage owns the lock for its
own lifetime — see :func:`hold`). The supervisor only ever *probes* (:func:`is_locked`); it never
holds a stage lock, so probing it (try-acquire then immediately release) does not race a concurrent
launch — the child re-acquires on its next attempt.
"""

from __future__ import annotations

import fcntl
import os
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


def lock_path(data_dir: Path, stage: str) -> Path:
    """The lock file for a (project, stage): ``<data_dir>/.lock.<stage>``."""
    return data_dir / f".lock.{stage}"


def try_acquire(path: Path) -> int | None:
    """Non-blocking exclusive ``flock`` on ``path``.

    Returns the held fd (caller must :func:`release` it) if the lock was free, or ``None`` if a
    live owner holds it. Used both by the supervisor (probe: acquire then release at once) and by
    :func:`hold` (a stage takes the lock for its lifetime).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:  # BlockingIOError -> a live owner holds it
        os.close(fd)
        return None
    return fd


def release(fd: int) -> None:
    """Release a held lock fd (closing the fd drops the ``flock``)."""
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def is_locked(path: Path) -> bool:
    """``True`` if a live owner holds the lock (the supervisor's "stage X running?" probe).

    Implemented as try-acquire + immediate release: if we could take it, no one owned it (free);
    if we couldn't, a live owner exists. Probing never holds the lock, so it cannot starve a real
    owner.
    """
    fd = try_acquire(path)
    if fd is None:
        return True
    release(fd)
    return False


@contextmanager
def hold(path: Path) -> Iterator[int]:
    """Acquire the lock for the duration of the ``with`` block (a stage's lifetime).

    Raises :class:`BlockingIOError` if a live owner already holds it (exactly-one-writer). The lock
    releases on block exit and, as a backstop, when the process dies.
    """
    fd = try_acquire(path)
    if fd is None:
        msg = f"stage lock already held: {path}"
        raise BlockingIOError(msg)
    try:
        yield fd
    finally:
        release(fd)
