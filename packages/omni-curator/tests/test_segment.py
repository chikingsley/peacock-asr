"""Segment-stage helpers: orphan-worker teardown, source resolution, free-space backpressure.

These cover the failure modes the audit found: a killed parent orphaning VRAM-holding VAD workers,
and a re-segment whose queue path points at a since-archived source.
"""

from __future__ import annotations

import multiprocessing as mp
import time

import numpy as np
import pytest

from omni_curator.create.queue import QVideo
from omni_curator.create.segment import (
    DEFAULT_PENDING_HWM,
    _free_gb,
    _process_segment_video,
    _publish_staged_clip_dir,
    _staging_root,
    _terminate_workers,
    resolve_source_path,
    run_segmenters,
)
from omni_curator.create.vad import build_vad_policy


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


def test_publish_staged_clip_dir_moves_output_and_cleans_staging(tmp_path):
    clips_root = tmp_path / "clips"
    staging_root = _staging_root(clips_root, "claim-token")
    staging_dir = staging_root / "chan" / "video1"
    final_dir = clips_root / "chan" / "video1"
    staging_dir.mkdir(parents=True)
    (staging_dir / "seg_0000.flac").write_bytes(b"clip")

    _publish_staged_clip_dir(staging_root, staging_dir, final_dir)

    assert (final_dir / "seg_0000.flac").read_bytes() == b"clip"
    assert not staging_root.exists()


def test_process_decodes_once_reuses_audio_and_stamps_policy(tmp_path, monkeypatch):
    source = tmp_path / "source.flac"
    source.write_bytes(b"placeholder")
    audio = np.zeros(16_000, dtype=np.float32)
    decode_calls = []
    written_arrays = []

    def fake_decode(path):
        decode_calls.append(path)
        return audio

    def fake_write(received, path, start, end):
        written_arrays.append(received)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"{start}-{end}".encode())

    monkeypatch.setattr("omni_curator.create.segment.load_16k_mono", fake_decode)
    monkeypatch.setattr("omni_curator.create.segment.write_clip_16k", fake_write)

    class FakeEngine:
        name = "marblenet"
        model_revision = "model-sha"

        def __init__(self):
            self.runtime_metadata = {"device": "cpu"}

        def predict(self, received, sample_rate):
            assert received is audio
            assert sample_rate == 16_000
            return [(0.0, 1.0)]

        def close(self):
            return

    class FakeQueue:
        clips = None

        def complete_video(self, video_id, clips, *, claim_token, publish, video_meta):
            assert video_id == "video1"
            assert claim_token == tmp_path.name
            self.clips = clips
            self.video_meta = video_meta
            publish()
            return True

    queue = FakeQueue()
    video = QVideo(
        "video1", "channel", str(source), "clean", None,
        claim_token=tmp_path.name, meta={"x": 1},
    )
    policy = build_vad_policy(profile="conservative-v1")

    assert _process_segment_video(
        queue,
        FakeEngine(),
        policy,
        video,
        clips_root=tmp_path / "clips",
        language="fas_Arab",
        script="Perso-Arabic",
    )
    assert decode_calls == [source]
    assert len(written_arrays) == 1
    assert written_arrays[0] is audio
    assert queue.clips[0].meta["x"] == 1
    segmentation = queue.clips[0].meta["segmentation"]
    assert segmentation["policy_id"] == policy.profile_id
    assert segmentation["profile_id"] != policy.profile_id
    assert segmentation["model_revision"] == "model-sha"
    assert queue.video_meta["segmentation"] == segmentation


def test_mixed_silero_auto_backends_fail_before_queue_claims(tmp_path):
    policy = build_vad_policy(engine="silero", profile="conservative-v1")
    with pytest.raises(ValueError, match="mix ONNX CPU and JIT CUDA"):
        run_segmenters(
            tmp_path / "does-not-exist.sqlite",
            gpu_procs=1,
            cpu_procs=1,
            clips_root=tmp_path / "clips",
            language="fas_Arab",
            script="Perso-Arabic",
            policy=policy,
        )
    assert not (tmp_path / "does-not-exist.sqlite").exists()


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-q"])
