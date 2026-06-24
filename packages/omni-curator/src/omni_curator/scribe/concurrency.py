"""Live, file-driven concurrency control for the Scribe jobs (labelq + verify).

Both Scribe jobs hold a fixed-size thread pool (sized to ``--pool-max``) but throttle how many
requests are actually in flight via a *window* re-read from a small text file every batch — the
same idea as GNU parallel's ``--jobs <procfile>``: edit the number in the file and the running
job retunes itself (raise -> submit more; lower -> let in-flight finish, start no new ones until
it is back under target). No restart, no signal, no lost work.

``split_budget`` divides a single Scribe-API budget across the currently-active jobs so a balancer
(the factory) can keep their *sum* within what the SuperWhisper service can take — one knob (the
budget) instead of hand-tuning each job.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


def read_window(window_file: Path | None, *, default: int, cap: int) -> int:
    """Return the desired in-flight window, clamped to ``[1, cap]``.

    Reads a single integer from ``window_file`` (cheap; meant to be called every batch so a live
    edit takes effect on the next round). Falls back to ``default`` when the file is missing,
    empty, or unparseable — a garbled or absent file must never stall or crash a running job.
    ``cap`` is the thread pool's hard ceiling; the window can never exceed it.
    """
    cap = max(1, cap)
    fallback = max(1, min(default, cap))
    if window_file is None:
        return fallback
    try:
        text = window_file.read_text(encoding="utf-8").strip()
    except OSError:
        return fallback
    try:
        value = int(text)
    except ValueError:
        return fallback
    return max(1, min(value, cap))


def write_window(window_file: Path, value: int) -> None:
    """Atomically write the window integer (temp + ``replace``).

    A reader (:func:`read_window`) calls this file every batch, so a plain truncate-then-write
    would expose a moment where the file is empty and the reader falls back to its default —
    silently jumping a job to a window the balancer never intended. Writing a sibling temp file
    and ``replace``-ing it makes the swap atomic on the same filesystem, so the reader only ever
    sees the old value or the new one.
    """
    window_file.parent.mkdir(parents=True, exist_ok=True)
    # Unique temp per writer (pid) so two racing balancers never clobber each other's temp file.
    tmp = window_file.parent / f"{window_file.name}.{os.getpid()}.tmp"
    tmp.write_text(f"{value}\n", encoding="utf-8")
    tmp.replace(window_file)


def split_budget(budget: int, jobs: Sequence[str]) -> dict[str, int]:
    """Divide ``budget`` Scribe slots across ``jobs`` as evenly as possible, at least 1 each.

    Remainder is handed to the first jobs (deterministic by input order), so e.g.
    ``split_budget(301, ["a", "b", "c"]) == {"a": 101, "b": 100, "c": 100}``. With no jobs the
    result is empty. If ``budget`` is smaller than the job count, every job still gets 1 (the
    service is shared; a slow slice each beats starving a job outright) — so the returned sum may
    exceed a budget below the job count, which the caller treats as "as low as it goes."
    """
    keys = list(jobs)
    if not keys:
        return {}
    base, extra = divmod(max(budget, len(keys)), len(keys))
    return {key: base + (1 if i < extra else 0) for i, key in enumerate(keys)}
