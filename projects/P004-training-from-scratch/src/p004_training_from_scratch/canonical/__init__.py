"""Canonical-lane runtime helpers for P004."""

from .preflight import (
    DEFAULT_MACHINE_OUTPUT,
    DEFAULT_OUTPUT,
    DEFAULT_SMOKE_AUDIO_LIST,
    run_canonical_preflight,
)
from .train_smoke import DEFAULT_OUTPUT_ROOT, run_canonical_train_smoke
from .trainer import (
    FROZEN_STABLE_TRAIN_CONFIG,
    CanonicalTrainConfig,
    build_stable_train_config,
    run_canonical_train,
)

__all__ = [
    "DEFAULT_MACHINE_OUTPUT",
    "DEFAULT_OUTPUT",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_SMOKE_AUDIO_LIST",
    "FROZEN_STABLE_TRAIN_CONFIG",
    "CanonicalTrainConfig",
    "build_stable_train_config",
    "run_canonical_preflight",
    "run_canonical_train",
    "run_canonical_train_smoke",
]
