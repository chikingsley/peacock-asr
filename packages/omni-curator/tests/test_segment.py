"""Segment-stage helpers: orphan-worker teardown, source resolution, free-space backpressure.

These cover the failure modes the audit found: a killed parent orphaning VRAM-holding VAD workers,
and a re-segment whose queue path points at a since-archived source.
"""

from __future__ import annotations

import multiprocessing as mp
import time

import pytest

from omni_curator.create.segment import (
    DEFAULT_PENDING_HWM,
    _free_gb,
    _terminate_workers,
    resolve_source_path,
)


def _sleep_forever() -> None:
    while True:  # pragma: no cover - child process body
        time.sleep(3600)


def _noop() -> None:  # pragma: no cover - child process body
    return


def test_terminate_workers_reaps_alive_children():
    """The finally path must SIGTERM/-KILL still-running workers so they release their VRAM."""
    ctx = mp.get_context("spawn")
    workers = [ctx.Process(target=_sleep_forever) for _ in range(3)]
    for w in workers:
        w.start()
    assert all(w.is_alive() for w in workers)

    _terminate_workers(workers, grace_s=5.0)

    assert all(not w.is_alive() for w in workers)
    assert all(w.exitcode is not None for w in workers)


def test_terminate_workers_is_a_noop_when_all_exited():
    """Happy path (workers already drained) must not error and must touch nothing."""
    ctx = mp.get_context("spawn")
    w = ctx.Process(target=_noop)
    w.start()
    w.join()
    assert not w.is_alive()
    _terminate_workers([w])  # no exception, no hang
    assert w.exitcode == 0


def test_resolve_source_returns_path_when_present(tmp_path):
    src = tmp_path / "create" / "chan" / "vid.flac"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"x")
    assert resolve_source_path(src, channel="chan", archive_root=tmp_path / "arch") == src


def test_resolve_source_maps_deterministically_to_archive(tmp_path):
    """A gone create path maps to <archive_root>/<rest-after-create>."""
    archive = tmp_path / "archive"
    # archive layout: <root>/<lang>/<channel>/<file>; create path carries <lang>/<channel>/<file>
    archived = archive / "dari" / "tolonews" / "vid.flac"
    archived.parent.mkdir(parents=True)
    archived.write_bytes(b"x")
    gone = tmp_path / "overflow" / "create" / "dari" / "tolonews" / "vid.flac"
    resolved = resolve_source_path(gone, channel="tolonews", archive_root=archive)
    assert resolved == archived


def test_resolve_source_scans_archive_when_layout_lacks_lang(tmp_path):
    """If the create path doesn't expose <lang>, scan <root>/<lang>/<channel>/<file.name>."""
    archive = tmp_path / "archive"
    archived = archive / "georgian" / "gpb" / "abc.flac"
    archived.parent.mkdir(parents=True)
    archived.write_bytes(b"x")
    # create path with no recoverable <lang> segment before the file
    gone = tmp_path / "ssd" / "gpb" / "abc.flac"
    resolved = resolve_source_path(gone, channel="gpb", archive_root=archive)
    assert resolved == archived


def test_resolve_source_falls_back_to_original_when_nowhere(tmp_path):
    gone = tmp_path / "create" / "dari" / "chan" / "missing.flac"
    out = resolve_source_path(gone, channel="chan", archive_root=tmp_path / "empty")
    assert out == gone  # so the caller's file-not-found fires with the expected path


def test_free_gb_probes_existing_parent(tmp_path):
    # a path that doesn't exist yet still reports the free space of its parent fs
    val = _free_gb(tmp_path / "does" / "not" / "exist")
    assert val > 0


def test_default_pending_hwm_is_finite_and_sane():
    assert DEFAULT_PENDING_HWM == 50_000  # not the old effectively-infinite 5_000_000


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-q"])
