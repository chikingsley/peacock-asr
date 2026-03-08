"""Stable canonical trainer entrypoint for bounded production-like runs.

References
- PyTorch `torch.compile`: https://pytorch.org/docs/stable/generated/torch.compile.html
- Hugging Face Accelerate quicktour: https://huggingface.co/docs/accelerate/main/en/quicktour
- Conformer paper: https://arxiv.org/abs/2005.08100
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from p004_training_from_scratch.canonical.common import (
    CanonicalAttentionBackend,
    CanonicalModelType,
)
from p004_training_from_scratch.canonical.train_smoke import (
    DEFAULT_DEV_MANIFEST,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_TOKENS_PATH,
    DEFAULT_TRAIN_MANIFEST,
    LossComputeDType,
    run_canonical_train_smoke,
)


@dataclass(frozen=True, slots=True)
class CanonicalTrainConfig:
    train_manifest: Path = DEFAULT_TRAIN_MANIFEST
    dev_manifest: Path = DEFAULT_DEV_MANIFEST
    tokens_path: Path = DEFAULT_TOKENS_PATH
    train_limit: int = 24
    dev_limit: int = 8
    epochs: int = 3
    batch_size: int = 4
    model_type: CanonicalModelType = "conformer"
    attention_backend: CanonicalAttentionBackend = "mha"
    hidden_dim: int = 192
    encoder_layers: int = 3
    attention_heads: int = 4
    conv_kernel_size: int = 15
    dropout: float = 0.1
    learning_rate: float = 3e-4
    loss_compute_dtype: LossComputeDType = "model"
    seed: int = 42
    resume_from: Path | None = None
    enable_compile: bool = False
    num_workers: int = 4
    bucket_size_multiplier: int = 50
    pin_memory: bool = True
    allow_online_trackers: bool = True
    with_wandb: bool = True


FROZEN_STABLE_TRAIN_CONFIG = CanonicalTrainConfig()


def build_stable_train_config(**overrides: Any) -> CanonicalTrainConfig:
    return replace(FROZEN_STABLE_TRAIN_CONFIG, **overrides)


def run_canonical_train(
    *,
    output_dir: Path,
    config: CanonicalTrainConfig = FROZEN_STABLE_TRAIN_CONFIG,
) -> dict[str, Any]:
    return run_canonical_train_smoke(
        output_dir=output_dir,
        train_manifest=config.train_manifest,
        dev_manifest=config.dev_manifest,
        tokens_path=config.tokens_path,
        train_limit=config.train_limit,
        dev_limit=config.dev_limit,
        epochs=config.epochs,
        batch_size=config.batch_size,
        model_type=config.model_type,
        attention_backend=config.attention_backend,
        hidden_dim=config.hidden_dim,
        encoder_layers=config.encoder_layers,
        attention_heads=config.attention_heads,
        conv_kernel_size=config.conv_kernel_size,
        dropout=config.dropout,
        learning_rate=config.learning_rate,
        loss_compute_dtype=config.loss_compute_dtype,
        seed=config.seed,
        resume_from=config.resume_from,
        enable_compile=config.enable_compile,
        num_workers=config.num_workers,
        bucket_size_multiplier=config.bucket_size_multiplier,
        pin_memory=config.pin_memory,
        allow_online_trackers=config.allow_online_trackers,
        with_wandb=config.with_wandb,
    )


__all__ = [
    "DEFAULT_OUTPUT_ROOT",
    "FROZEN_STABLE_TRAIN_CONFIG",
    "CanonicalTrainConfig",
    "build_stable_train_config",
    "run_canonical_train",
]
