from __future__ import annotations

import json
from array import array
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from parakeet_finetune_core.materialize import build_parser, materialize
from parakeet_finetune_core.project import ParakeetProject


def _write_partition(root: Path, *, corpus: str, split: str, rows: list[dict]) -> None:
    partition = root / f"corpus={corpus}" / f"split={split}" / "language=eng_Latn"
    partition.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), partition / "part-00000.parquet")


def test_materialize_writes_deterministic_manifest_and_audio(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset" / "version=0"
    payload = b"fLaC-payload"
    _write_partition(
        dataset,
        corpus="meetings",
        split="train",
        rows=[
            {
                "text": "hello world",
                "audio_bytes": list(array("b", payload)),
                "audio_size": 16_000,
            },
            {
                "text": "too long",
                "audio_bytes": list(array("b", b"other")),
                "audio_size": 640_000,
            },
        ],
    )
    project = ParakeetProject(name="english", language="eng_Latn", root=tmp_path)
    args = build_parser(project).parse_args(
        [
            "--dataset-root",
            str(dataset),
            "--output-root",
            str(tmp_path / "materialized"),
            "--max-duration",
            "30",
        ]
    )

    stats, manifest = materialize(args)

    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert stats.rows == 1
    assert stats.excluded_duration == 1
    assert stats.rows_by_corpus == {"meetings": 1}
    assert records[0]["text"] == "hello world"
    assert records[0]["corpus"] == "meetings"
    assert Path(records[0]["audio_filepath"]).read_bytes() == payload
    assert materialize(args)[0].rows == 1


def test_materialize_dry_run_does_not_create_output(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset" / "version=0"
    _write_partition(
        dataset,
        corpus="read",
        split="dev",
        rows=[{"text": "reference", "audio_bytes": [1, 2], "audio_size": 32_000}],
    )
    project = ParakeetProject(name="english", language="eng_Latn", root=tmp_path)
    output = tmp_path / "materialized"
    args = build_parser(project).parse_args(
        [
            "--dataset-root",
            str(dataset),
            "--output-root",
            str(output),
            "--split",
            "dev",
            "--dry-run",
        ]
    )

    stats, manifest = materialize(args)

    assert stats.rows == 1
    assert manifest == output / "manifests" / "dev.jsonl"
    assert not output.exists()
