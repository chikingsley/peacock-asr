"""Real-filesystem tests for the source-audio archiver — no mocks: real queue, real files."""

from __future__ import annotations

import json
from pathlib import Path

from omni_curator.create.archive import archive_source_audio, copy_verify
from omni_curator.create.queue import QClip, QueueStore, QVideo

_FLAC = b"FLAC0123456789" * 10


def _video(create: Path, vid: str, channel: str = "chan") -> QVideo:
    return QVideo(vid, channel, str(create / channel / f"{vid}.flac"), "noisy", None)


def _clip(vid: str) -> QClip:
    return QClip(
        f"{vid}_0000",
        vid,
        "chan",
        0,
        f"/clips/{vid}/0.flac",
        0.0,
        5.0,
        "tgk_Cyrl",
        "Cyrillic",
        None,
    )


def _seed_segmented(queue: QueueStore, create: Path, vids: list[str]) -> None:
    """Enqueue + mark segmented + write the real source FLAC for each video."""
    queue.enqueue_videos([_video(create, v) for v in vids])
    for v in vids:
        src = Path(_video(create, v).path)
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_bytes(_FLAC)
        queue.complete_video(v, [_clip(v)])  # -> status='segmented'


def test_archive_moves_segmented_sources(tmp_path):
    qpath, create, archive = tmp_path / "q.sqlite", tmp_path / "create", tmp_path / "arch"
    queue = QueueStore(qpath)
    _seed_segmented(queue, create, ["v0", "v1", "v2"])
    queue.close()

    stats = archive_source_audio(qpath, archive_root=archive)

    assert stats.archived == 3
    assert stats.bytes_freed == 3 * len(_FLAC)
    for v in ("v0", "v1", "v2"):
        assert not (create / "chan" / f"{v}.flac").exists()  # source moved out of the working dir
        moved = archive / "chan" / f"{v}.flac"
        assert moved.exists()
        assert moved.read_bytes() == _FLAC  # content intact
    manifest = [
        json.loads(line) for line in (archive / "archive_manifest.jsonl").read_text().splitlines()
    ]
    assert {m["video_id"] for m in manifest} == {"v0", "v1", "v2"}
    assert all(m["action"] == "move" for m in manifest)


def test_archive_leaves_unsegmented_sources(tmp_path):
    qpath, create = tmp_path / "q.sqlite", tmp_path / "create"
    queue = QueueStore(qpath)
    _seed_segmented(queue, create, ["done"])
    queue.enqueue_videos([_video(create, "pending")])  # enqueued but NOT segmented
    pending_src = Path(_video(create, "pending").path)
    pending_src.parent.mkdir(parents=True, exist_ok=True)
    pending_src.write_bytes(b"x" * 50)
    queue.close()

    stats = archive_source_audio(qpath, archive_root=tmp_path / "arch")

    assert stats.archived == 1  # only the segmented one
    assert not (create / "chan" / "done.flac").exists()
    assert pending_src.exists()  # the pending video's source is untouched


def test_archive_delete_mode(tmp_path):
    qpath, create = tmp_path / "q.sqlite", tmp_path / "create"
    queue = QueueStore(qpath)
    _seed_segmented(queue, create, ["v0", "v1"])
    queue.close()

    stats = archive_source_audio(qpath, delete=True)

    assert stats.archived == 2
    assert not any((create / "chan" / f"{v}.flac").exists() for v in ("v0", "v1"))


def test_archive_is_idempotent(tmp_path):
    qpath, create, archive = tmp_path / "q.sqlite", tmp_path / "create", tmp_path / "arch"
    queue = QueueStore(qpath)
    _seed_segmented(queue, create, ["v0"])
    queue.close()

    archive_source_audio(qpath, archive_root=archive)
    stats2 = archive_source_audio(qpath, archive_root=archive)  # re-run: source already gone

    assert stats2.archived == 0
    assert stats2.missing == 1


def test_only_if_free_gb_skips_when_space_is_fine(tmp_path):
    qpath, create = tmp_path / "q.sqlite", tmp_path / "create"
    queue = QueueStore(qpath)
    _seed_segmented(queue, create, ["v0"])
    queue.close()

    # tmp has far more than 1 MB free, so a 0.001 GB threshold means "space is fine, skip".
    stats = archive_source_audio(qpath, archive_root=tmp_path / "arch", only_if_free_gb=0.001)

    assert stats.archived == 0
    assert (create / "chan" / "v0.flac").exists()  # untouched


def test_copy_verify_roundtrip(tmp_path):
    src = tmp_path / "a.flac"
    src.write_bytes(b"hello world" * 100)
    dst = tmp_path / "sub" / "a.flac"

    size = copy_verify(src, dst)

    assert size == len(b"hello world" * 100)
    assert dst.read_bytes() == src.read_bytes()
    assert not (tmp_path / "sub" / "a.flac.partial").exists()  # temp cleaned up by the rename
