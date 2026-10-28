from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from parakeet_finetune_core.splits import validate_splits

if TYPE_CHECKING:
    from pathlib import Path


def manifest(root: Path, name: str, **fields: object) -> Path:
    audio = root / f"{name}.flac"
    audio.touch()
    path = root / f"{name}.jsonl"
    path.write_text(json.dumps({"audio_filepath": str(audio), **fields}) + "\n")
    return path


def test_different_clips_from_reserved_video_fail(tmp_path):
    train = manifest(tmp_path, "train", source_video_ids=["channel_video"])
    evaluation = manifest(tmp_path, "test", source_video_ids=["channel_video"])
    with pytest.raises(ValueError, match="overlaps"):
        validate_splits([train], [evaluation])


def test_copied_audio_with_different_path_fails(tmp_path):
    train = manifest(tmp_path, "train", audio_sha256="same-audio")
    evaluation = manifest(tmp_path, "test", audio_sha256="same-audio")
    with pytest.raises(ValueError, match="overlaps"):
        validate_splits([train], [evaluation])


def test_historical_manifest_path_respects_frozen_videos(tmp_path):
    audio = tmp_path / "labeled" / "channel" / "video" / "cuts" / "seg_0000.flac"
    audio.parent.mkdir(parents=True)
    audio.touch()
    train = manifest(tmp_path, "train", audio_filepath=str(audio))
    with pytest.raises(ValueError, match="overlaps"):
        validate_splits([train], [], heldout_videos=frozenset({"channel_video"}))


def test_youtube_requires_video_provenance(tmp_path):
    train = manifest(tmp_path, "train", corpus="youtube-channel")
    with pytest.raises(ValueError, match="requires source_video_ids"):
        validate_splits([train], [], require_video_ids=True)


def test_independent_manifest_identities_pass(tmp_path):
    train = manifest(tmp_path, "train", source_video_ids=["train_video"])
    evaluation = manifest(tmp_path, "test", source_video_ids=["test_video"])
    assert validate_splits([train], [evaluation]) == {"train_rows": 1, "evaluation_rows": 1}


def test_missing_audio_fails_before_model_load(tmp_path):
    train = manifest(tmp_path, "train")
    (tmp_path / "train.flac").unlink()
    with pytest.raises(ValueError, match="unavailable"):
        validate_splits([train], [])


@pytest.mark.parametrize("identity", ["sample_id", "source_video_id"])
def test_direct_export_identities_match_materialized_manifests(tmp_path, identity):
    plural = "source_sample_ids" if identity == "sample_id" else "source_video_ids"
    train = manifest(tmp_path, "train", **{identity: "reserved"})
    evaluation = manifest(tmp_path, "evaluation", **{plural: ["reserved"]})
    with pytest.raises(ValueError, match="overlaps"):
        validate_splits([train], [evaluation])


def test_overlap_in_later_shard_is_checked(tmp_path):
    first = manifest(tmp_path, "first", source_video_ids=["clean"])
    second = manifest(tmp_path, "second", source_video_ids=["reserved"])
    evaluation = manifest(tmp_path, "evaluation", source_video_ids=["reserved"])
    shards = tmp_path / "shards"
    shards.mkdir()
    first.rename(shards / "shard_00000.jsonl")
    second.rename(shards / "shard_00001.jsonl")
    with pytest.raises(ValueError, match="overlaps"):
        validate_splits([shards], [evaluation])
