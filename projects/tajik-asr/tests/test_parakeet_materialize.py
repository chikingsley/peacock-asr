from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from omni_finetune_core.parquet import partition_dir, write_shard

from tajik_asr.parakeet.materialize import build_parser, materialize


def test_materialize_writes_deterministic_audio_manifest_and_summary(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset" / "version=0"
    output_root = tmp_path / "output"
    shard_dir = partition_dir(dataset_root, "youtube-demo", "test", "tgk_Cyrl")
    payloads = [b"fLaC-one", b"fLaC-two"]
    write_shard(
        ["як", "ду"],
        [np.frombuffer(payload, dtype=np.int8) for payload in payloads],
        [16_000, 48_000],
        shard_dir,
        0,
    )
    args = build_parser().parse_args(
        [
            "--dataset-root",
            str(dataset_root),
            "--output-root",
            str(output_root),
            "--max-duration",
            "2",
            "--dataset-revision",
            "revision-1",
        ]
    )

    rows, excluded, duration_seconds, manifest = materialize(args)

    assert (rows, excluded, duration_seconds) == (1, 1, 1.0)
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["text"] == "як"
    assert records[0]["dataset_revision"] == "revision-1"
    assert records[0]["audio_sha256"]
    assert Path(records[0]["audio_filepath"]).read_bytes() == payloads[0]
    summary = json.loads((output_root / "materialization.json").read_text())
    assert summary["rows"] == 1
    assert summary["excluded_over_max_duration"] == 1

    assert materialize(args)[:3] == (1, 1, 1.0)


def test_materialize_dry_run_does_not_create_output(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset" / "version=0"
    output_root = tmp_path / "output"
    shard_dir = partition_dir(dataset_root, "youtube-demo", "test", "tgk_Cyrl")
    write_shard(
        ["як"],
        [np.frombuffer(b"fLaC-one", dtype=np.int8)],
        [16_000],
        shard_dir,
        0,
    )
    args = build_parser().parse_args(
        [
            "--dataset-root",
            str(dataset_root),
            "--output-root",
            str(output_root),
            "--dry-run",
        ]
    )

    assert materialize(args)[:3] == (1, 0, 1.0)
    assert not output_root.exists()
