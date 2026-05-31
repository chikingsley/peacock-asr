"""Run the in-housed wav2vec2-ASR recipe from a typed config.

Replaces each project's hand-rolled ``runpy`` wrapper: given a :class:`TrainingConfig`
(validated, typed) this writes the recipe YAML into the run dir and invokes the in-housed
recipe (``omni_finetune_core.recipe``) through fairseq2's recipe CLI. Resume semantics are
the recipe's: point ``output_dir`` at an existing worktree with ``common.no_sweep_dir``
set to continue it, otherwise a fresh ``ws_{world_size}.{hash}`` run is created.
"""

from __future__ import annotations

import os
import runpy
import sys
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from pathlib import Path

    from omni_finetune_core.config import TrainingConfig

RECIPE_MODULE = "omni_finetune_core.recipe"


def configure_environment(cache_root: Path) -> None:
    """Set HF/fairseq2 cache + CUDA env defaults under ``cache_root`` (idempotent)."""
    os.environ.setdefault("HF_HOME", str(cache_root / ".hf-cache"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_root / ".hf-cache/datasets"))
    os.environ.setdefault("FAIRSEQ2_CACHE_DIR", str(cache_root / ".fairseq2-cache/assets"))
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")


def run_recipe(
    config_file: Path, output_dir: Path, *, extra_args: list[str] | None = None
) -> None:
    """Invoke the recipe CLI (via runpy) with an existing config file + output dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sys.argv = [
        RECIPE_MODULE,
        str(output_dir),
        "--config-file",
        str(config_file),
        *(extra_args or []),
    ]
    runpy.run_module(RECIPE_MODULE, run_name="__main__")


def train(
    config: TrainingConfig, output_dir: Path, *, extra_args: list[str] | None = None
) -> Path:
    """Write ``config`` to ``output_dir/config.yaml`` and run the recipe. Returns the path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    config_file = output_dir / "config.yaml"
    config_file.write_text(
        yaml.safe_dump(config.to_recipe_dict(), sort_keys=False), encoding="utf-8"
    )
    run_recipe(config_file, output_dir, extra_args=extra_args)
    return config_file
