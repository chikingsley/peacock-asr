from __future__ import annotations

import pytest

from peacock_asr.train_profile import load_train_profile


def test_load_preflight_profile() -> None:
    profile = load_train_profile("preflight")
    assert profile.name == "preflight"
    assert profile.num_epochs == 1
    assert profile.push_to_hub is False
    assert profile.max_train_samples == 256
    assert profile.max_eval_samples == 64


def test_load_main_profile() -> None:
    profile = load_train_profile("main")
    assert profile.name == "main"
    assert profile.num_epochs == 3
    assert profile.push_to_hub is True
    assert profile.max_train_samples is None
    assert profile.max_eval_samples is None


def test_missing_profile_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_train_profile("does-not-exist")

