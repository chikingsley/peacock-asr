from __future__ import annotations

import gzip
import json
from pathlib import Path

from p004_training_from_scratch.canonical.data import (
    DurationBucketBatchSampler,
    ManifestCut,
    read_manifest_cuts,
)


def test_read_manifest_cuts_reads_all_when_limit_is_none(tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake")
    manifest_path = tmp_path / "cuts.jsonl.gz"
    payloads = [
        {
            "id": f"cut-{index}",
            "duration": 1.0 + index,
            "recording": {"sources": [{"source": str(audio_path)}]},
            "supervisions": [{"text": "AA BB"}],
        }
        for index in range(3)
    ]
    with gzip.open(manifest_path, "wt", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(f"{json.dumps(payload)}\n")

    cuts = read_manifest_cuts(
        manifest_path,
        limit=None,
        token_table={"<eps>": 0, "AA": 1, "BB": 2},
    )

    assert [cut.cut_id for cut in cuts] == ["cut-0", "cut-1", "cut-2"]


def test_read_manifest_cuts_treats_zero_limit_as_full_read(tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake")
    manifest_path = tmp_path / "cuts.jsonl.gz"
    payload = {
        "id": "cut-1",
        "duration": 1.25,
        "recording": {"sources": [{"source": str(audio_path)}]},
        "supervisions": [{"text": "AA BB"}],
    }
    with gzip.open(manifest_path, "wt", encoding="utf-8") as handle:
        handle.write(f"{json.dumps(payload)}\n")

    cuts = read_manifest_cuts(
        manifest_path,
        limit=0,
        token_table={"<eps>": 0, "AA": 1, "BB": 2},
    )

    assert len(cuts) == 1
    assert cuts[0].cut_id == "cut-1"


def test_duration_bucket_batch_sampler_emits_all_indices_once() -> None:
    cuts = [
        ManifestCut(
            cut_id=f"cut-{index}",
            audio_path=Path(f"/tmp/cut-{index}.wav"),
            phones=("AA",),
            duration_seconds=float(index + 1),
        )
        for index in range(10)
    ]
    sampler = DurationBucketBatchSampler(
        cuts=cuts,
        batch_size=3,
        shuffle=True,
        seed=42,
        bucket_size_multiplier=2,
    )

    batches = list(iter(sampler))
    flattened = [index for batch in batches for index in batch]

    assert sorted(flattened) == list(range(10))
    assert all(1 <= len(batch) <= 3 for batch in batches)
