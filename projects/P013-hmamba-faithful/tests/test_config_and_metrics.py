from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml
from pydantic import ValidationError

from p012.config import HMambaConfig
from p012.trainer import mdd_detection_metrics


def test_config_matches_paper_appendix_defaults() -> None:
    config = yaml.safe_load(Path("conf/so762/HMamba.yaml").read_text())
    assert config["d_conv"] == 4
    assert config["feat_drop"] == 0.1
    assert "pool_mode" not in config
    assert "block_type" not in config


def test_vocab_size_matches_vocab_file() -> None:
    config = yaml.safe_load(Path("conf/so762/HMamba.yaml").read_text())
    vocab = json.loads(Path("local/so762/vocab_merge.json").read_text())
    assert config["vocab_size"] == len(vocab)


def test_mdd_detection_metrics() -> None:
    logits = torch.tensor(
        [
            [
                [0.0, 3.0, 0.0],  # predict 1, canonical 1 => correct
                [3.0, 0.0, 0.0],  # predict 0, canonical 1 => predicted mis
                [0.0, 0.0, 3.0],  # predict 2, canonical 2 => correct
                [0.0, 0.0, 3.0],  # predict 2, canonical 1 => predicted mis
            ]
        ]
    )
    canophns = torch.tensor([[1, 1, 2, 1]])
    realphns = torch.tensor([[1, 2, 2, 1]])
    mask = torch.tensor([[True, True, True, True]])

    precision, recall, f1 = mdd_detection_metrics(logits, canophns, realphns, mask)
    assert precision == 0.5
    assert recall == 1.0
    assert abs(f1 - (2 / 3)) < 1e-6


def test_config_rejects_unknown_fields() -> None:
    config = yaml.safe_load(Path("conf/so762/HMamba.yaml").read_text())
    config["pool_mode"] = "score-attn"
    with pytest.raises(ValidationError):
        HMambaConfig.model_validate(config)


def test_config_rejects_invalid_raw_dim() -> None:
    config = yaml.safe_load(Path("conf/so762/HMamba.yaml").read_text())
    config["raw_dim"] = 4
    with pytest.raises(ValidationError):
        HMambaConfig.model_validate(config)
