from pathlib import Path

import pytest

from parakeet_finetune_core.manifests import manifest_paths, manifest_pattern
from parakeet_finetune_core.tdt import train_ds


def test_shards_keep_one_shuffled_source_and_its_explicit_weight(tmp_path: Path):
    for index in range(3):
        (tmp_path / f"shard_{index:05d}.jsonl").write_text("{}\n")
    pattern = str(tmp_path / "shard_{00000..00002}.jsonl")
    assert manifest_pattern(tmp_path) == pattern
    config = train_ds([(tmp_path, 0.12)], 30, 120, 4, 0)
    assert config["input_cfg"] == [{"type": "nemo", "manifest_filepath": pattern, "weight": 0.12}]
    assert config["shuffle"] is True


def test_missing_middle_shard_fails_before_loading(tmp_path: Path):
    for index in (0, 2):
        (tmp_path / f"shard_{index:05d}.jsonl").touch()
    with pytest.raises(ValueError, match="contiguous"):
        manifest_paths(tmp_path)
