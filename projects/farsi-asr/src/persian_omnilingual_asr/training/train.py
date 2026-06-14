"""Launch an Omnilingual ASR fine-tune via the shared omni-finetune-core recipe.

Entry point ``persian-omni-train``. Mirrors ``tajik_omnilingual_asr.training.train``:
pick a preset (or pass ``--config-file``/``--output-dir`` directly), then hand off to
:func:`omni_finetune_core.train.run_recipe`. The vendored-recipe ``runpy``/``sys.path``
shim of the old ``finetune_omni`` path is gone — the recipe now lives in the package.

New runs land under ``<project>/runs/`` (distinct from the legacy ``finetune_omni/runs``).
Configs not covered by a preset (e.g. the scribe-v3/v4 continuations) are run via
``--config-file CONFIG --output-dir DIR``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from omni_finetune_core.train import configure_environment, run_recipe

ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = Path(__file__).resolve().parent / "configs"
RUNS_ROOT = ROOT / "runs"


@dataclass(frozen=True)
class TrainingPreset:
    config: Path
    output_dir: Path


PRESETS = {
    "fleurs-300m": TrainingPreset(
        config=CONFIG_DIR / "fleurs-fa-ir-ctc-300m-v2-finetune.yaml",
        output_dir=RUNS_ROOT / "fleurs-fa-ir-ft",
    ),
    "thomcles-continue": TrainingPreset(
        config=CONFIG_DIR / "thomcles-ctc-300m-v2-continue-from-fleurs.yaml",
        output_dir=RUNS_ROOT / "fleurs-fa-ir-thomcles-continue",
    ),
    "persian-clean-100h-300m": TrainingPreset(
        config=CONFIG_DIR / "persian-asr-clean-100h-filter-ctc-300m-v2.yaml",
        output_dir=RUNS_ROOT / "persian-clean-100h-filter",
    ),
    "persian-balanced-100h-300m": TrainingPreset(
        config=CONFIG_DIR / "persian-asr-balanced-100h-filter-ctc-300m-v2.yaml",
        output_dir=RUNS_ROOT / "persian-balanced-100h-filter",
    ),
    "persian-target-100h-300m": TrainingPreset(
        config=CONFIG_DIR / "persian-asr-target-100h-filter-ctc-300m-v2.yaml",
        output_dir=RUNS_ROOT / "persian-target-100h-filter",
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an Omnilingual ASR training recipe.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default=None)
    parser.add_argument("--config-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("recipe_args", nargs=argparse.REMAINDER)
    return parser


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.preset is not None:
        preset = PRESETS[args.preset]
        config_file = args.config_file or preset.config
        output_dir = args.output_dir or preset.output_dir
    else:
        if args.config_file is None or args.output_dir is None:
            raise SystemExit("--config-file and --output-dir are required without --preset")
        config_file = args.config_file
        output_dir = args.output_dir
    return config_file.resolve(), output_dir.resolve()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_file, output_dir = resolve_paths(args)

    configure_environment(ROOT)

    recipe_args = list(args.recipe_args)
    if recipe_args and recipe_args[0] == "--":
        recipe_args = recipe_args[1:]

    run_recipe(config_file, output_dir, extra_args=recipe_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
