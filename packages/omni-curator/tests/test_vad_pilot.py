from __future__ import annotations

import json

import numpy as np
import pytest

from omni_curator.create.vad_pilot import _scribe_sample, read_pilot_manifest, run_vad_pilot


def _manifest(tmp_path):
    clean = tmp_path / "clean.flac"
    noisy = tmp_path / "noisy.flac"
    clean.write_bytes(b"clean")
    noisy.write_bytes(b"noisy")
    manifest = tmp_path / "pilot.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps(
                    {"id": "clean1", "path": str(clean), "tier": "clean", "channel": "book"}
                ),
                json.dumps(
                    {"id": "noisy1", "path": str(noisy), "tier": "noisy", "channel": "news"}
                ),
            ]
        )
        + "\n"
    )
    return manifest


def test_manifest_is_exact_bounded_selector(tmp_path):
    rows = read_pilot_manifest(_manifest(tmp_path))
    assert [(row.source_id, row.tier) for row in rows] == [
        ("clean1", "clean"),
        ("noisy1", "noisy"),
    ]


def test_manifest_rejects_duplicate_source_ids(tmp_path):
    manifest = _manifest(tmp_path)
    first = manifest.read_text().splitlines()[0]
    manifest.write_text(f"{first}\n{first}\n")
    with pytest.raises(ValueError, match="duplicate"):
        read_pilot_manifest(manifest)


def test_manifest_rejects_path_traversal_source_id(tmp_path):
    manifest = _manifest(tmp_path)
    row = json.loads(manifest.read_text().splitlines()[0])
    row["id"] = "../../production-clips"
    manifest.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="safe single path component"):
        read_pilot_manifest(manifest)


def test_pilot_writes_isolated_reproducible_artifacts(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path)
    output = tmp_path / "pilot-output"
    audio = np.zeros(32_000, dtype=np.float32)

    class FakeEngine:
        name = "marblenet"
        model_revision = "fake-model-sha"

        def __init__(self):
            self.runtime_metadata = {"device": "cpu"}

        def predict(self, received, sample_rate):
            assert received is audio
            assert sample_rate == 16_000
            return [(0.1, 1.0)]

        def close(self):
            return

    monkeypatch.setattr(
        "omni_curator.create.vad_pilot.load_vad_engine", lambda *_a, **_k: FakeEngine()
    )
    monkeypatch.setattr("omni_curator.create.vad_pilot.load_16k_mono", lambda _path: audio)

    summary = run_vad_pilot(
        manifest=manifest,
        output_dir=output,
        engines=["marblenet"],
    )

    assert summary["production_queue_touched"] is False
    assert summary["source_count"] == 2
    assert (output / "run.json").is_file()
    records = [json.loads(line) for line in (output / "intervals.jsonl").read_text().splitlines()]
    assert len(records) == 2
    assert {row["tier"] for row in records} == {"clean", "noisy"}
    assert all(row["model_revision"] == "fake-model-sha" for row in records)
    with pytest.raises(FileExistsError):
        run_vad_pilot(manifest=manifest, output_dir=output, engines=["marblenet"])
    repeated = run_vad_pilot(
        manifest=manifest, output_dir=output, engines=["marblenet"], overwrite=True
    )
    assert repeated["source_count"] == 2


def test_overwrite_refuses_unmarked_directory(tmp_path):
    manifest = _manifest(tmp_path)
    output = tmp_path / "not-a-pilot"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("do not delete")
    with pytest.raises(PermissionError, match="unmarked"):
        run_vad_pilot(
            manifest=manifest,
            output_dir=output,
            engines=["marblenet"],
            overwrite=True,
        )
    assert sentinel.read_text() == "do not delete"


def test_scribe_sample_stops_after_three_consecutive_service_failures(tmp_path, monkeypatch):
    rows = []
    for index in range(10):
        clip = tmp_path / f"{index}.flac"
        clip.write_bytes(b"clip")
        rows.append(
            {
                "clip_id": str(index), "source_id": "source", "tier": "clean",
                "channel": "channel", "engine": "cobra", "profile_id": "profile",
                "model_revision": "model", "path": str(clip), "start": 0.0,
                "end": 1.0, "duration": 1.0,
            }
        )

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("service unavailable")

    monkeypatch.setattr("omni_curator.scribe.swservice.transcribe_file", unavailable)
    output = _scribe_sample(rows, engines=["cobra"], limit=10, model="scribe-v2", language=None)
    assert len(output) == 3
    assert all(row["error"] for row in output)


def test_scribe_sample_counts_structured_service_errors(tmp_path, monkeypatch):
    clip = tmp_path / "clip.flac"
    clip.write_bytes(b"clip")
    rows = [
        {
            "clip_id": str(index), "source_id": "source", "tier": "clean",
            "channel": "channel", "engine": "cobra", "profile_id": "profile",
            "model_revision": "model", "path": str(clip), "start": 0.0,
            "end": 1.0, "duration": 1.0,
        }
        for index in range(10)
    ]
    monkeypatch.setattr(
        "omni_curator.scribe.swservice.transcribe_file",
        lambda *_args, **_kwargs: {"error": "backend unavailable"},
    )
    output = _scribe_sample(rows, engines=["cobra"], limit=10, model="scribe-v2", language=None)
    assert len(output) == 3
    assert {row["error"] for row in output} == {"backend unavailable"}
