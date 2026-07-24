from __future__ import annotations

from omni_curator.ingest.huggingface import _derived_split


def test_derived_split_is_group_stable_and_train_only() -> None:
    first = _derived_split("train", group="meeting-a", validation_fraction=0.5, split_seed=17)
    assert first in {"train", "dev"}
    assert (
        _derived_split("train", group="meeting-a", validation_fraction=0.5, split_seed=17) == first
    )
    assert (
        _derived_split("test", group="meeting-a", validation_fraction=1.0, split_seed=17) == "test"
    )
    assert (
        _derived_split("train", group="meeting-a", validation_fraction=0.0, split_seed=17)
        == "train"
    )
    assert (
        _derived_split("train", group="meeting-a", validation_fraction=1.0, split_seed=17) == "dev"
    )
    assert (
        _derived_split("train.100", group="speaker-a", validation_fraction=0.0, split_seed=17)
        == "train"
    )
    assert (
        _derived_split("train.100", group="speaker-a", validation_fraction=1.0, split_seed=17)
        == "dev"
    )
