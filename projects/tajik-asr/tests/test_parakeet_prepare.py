import hashlib
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tajik_asr.parakeet import prepare
from tajik_asr.parakeet.prepare import annotate, verify_snapshot


def test_ambiguous_source_match_excludes_every_heldout_candidate() -> None:
    record = {"corpus": "youtube-channel", "text": "салом", "duration": 1.0}
    index = {("youtube-channel", "салом", 16000): ["channel_train_0001", "channel_test_0002"]}
    counts = Counter()
    assert annotate(record, index, frozenset({"channel_test"}), "train", counts) is None
    assert counts["heldout_video"] == 1


def test_unique_match_carries_explicit_pseudo_label_provenance() -> None:
    record = {"corpus": "youtube-channel", "text": "салом", "duration": 1.0}
    index = {("youtube-channel", "салом", 16000): ["channel_train_0001"]}
    result = annotate(record, index, frozenset(), "train", Counter())
    assert result["source_video_ids"] == ["channel_train"]
    assert result["label_origin"] == "scribe_ensemble_llm_fusion"
    assert result["license"] == "unknown"


def test_unmatched_evaluation_row_requires_resolution() -> None:
    record = {"corpus": "fleurs", "text": "салом", "duration": 1.0}
    with pytest.raises(ValueError, match="requires a source identity"):
        annotate(record, {}, frozenset(), "test", Counter())


def test_cross_split_reference_identity_is_excluded_from_train() -> None:
    record = {"corpus": "fleurs", "text": "салом", "duration": 1.0}
    index = {("fleurs", "салом", 16000): ["train_id", "test_id"]}
    counts = Counter()
    assert (
        annotate(
            record,
            index,
            frozenset(),
            "train",
            counts,
            reserved_samples=frozenset({"test_id"}),
        )
        is None
    )
    assert counts["evaluation_identity"] == 1


@pytest.mark.parametrize("dry_run", [False, True])
def test_all_splits_reserve_candidate_videos_before_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, dry_run: bool
) -> None:
    monkeypatch.setattr(prepare, "ROOT", tmp_path)
    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests/heldout_test_videos.json").write_text('{"video_ids": []}')
    monkeypatch.setattr(prepare, "verify_snapshot", lambda *_: {"verified_shards": 3})
    index = {}
    for split, pairs in {
        "dev": [("dev", "channel_eval_0001")],
        "test": [("test", "channel_other_0001")],
        "train": [("overlap", "channel_eval_0002"), ("clean", "channel_clean_0001")],
    }.items():
        path = (
            tmp_path / "dataset" / "corpus=youtube-channel" / f"split={split}" / "language=tgk_Cyrl"
        )
        path.mkdir(parents=True)
        rows = []
        for text, sample_id in pairs:
            index[("youtube-channel", text, 16000)] = [sample_id]
            rows.append({"text": text, "audio_bytes": text.encode(), "audio_size": 16000})
        pq.write_table(pa.Table.from_pylist(rows), path / "part.parquet")
    monkeypatch.setattr(prepare, "source_index", lambda _: index)
    output = tmp_path / "output"
    args = [
        "--all-splits",
        "--dataset-root",
        str(tmp_path / "dataset"),
        "--output-root",
        str(output),
    ]
    assert prepare.main([*args, *(["--dry-run"] if dry_run else [])]) == 0
    if dry_run:
        assert not output.exists()
    else:
        report = json.loads((output / "source-audit.json").read_text())
        assert report["status"] == "complete"
        assert report["splits"]["train"]["rows"] == 1
        assert report["splits"]["train"]["excluded_provenance"] == 1
        row = json.loads((output / "manifests/train.jsonl").read_text())
        assert row["source_sample_ids"] == ["channel_clean_0001"]


def test_snapshot_requires_every_shard_and_its_content_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from huggingface_hub import HfApi

    item = SimpleNamespace(
        rfilename="part.parquet",
        size=4,
        lfs=SimpleNamespace(sha256=hashlib.sha256(b"data").hexdigest()),
    )
    monkeypatch.setattr(
        HfApi,
        "dataset_info",
        lambda *_args, **_kwargs: SimpleNamespace(sha="pinned", siblings=[item]),
    )
    with pytest.raises(ValueError, match="Dataset restore requires"):
        verify_snapshot(tmp_path, "repo", "pinned")
    (tmp_path / "part.parquet").write_bytes(b"oops")
    with pytest.raises(ValueError, match="content hash differs"):
        verify_snapshot(tmp_path, "repo", "pinned")
    (tmp_path / "part.parquet").write_bytes(b"data")
    assert verify_snapshot(tmp_path, "repo", "pinned")["verified_shards"] == 1
