from __future__ import annotations

import argparse
import os
import runpy
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OMNI_ROOT = ROOT / "vendor" / "omnilingual-asr"
OMNI_RECIPE_MODULE = "workflows.recipes.wav2vec2.asr"
OMNI_RECIPE_PATH = Path("workflows/recipes/wav2vec2/asr/__main__.py")


@dataclass(frozen=True)
class TrainingPreset:
    config: Path
    output_dir: Path


PRESETS = {
    "fleurs-300m": TrainingPreset(
        config=ROOT / "configs/omni/fleurs-fa-ir-ctc-300m-v2-finetune.yaml",
        output_dir=ROOT / "runs/omni-ctc-300m-fleurs-fa-ir-ft",
    ),
    "thomcles-continue": TrainingPreset(
        config=ROOT / "configs/omni/thomcles-ctc-300m-v2-continue-from-fleurs.yaml",
        output_dir=ROOT / "runs/omni-ctc-300m-fleurs-fa-ir-thomcles-continue",
    ),
    "persian-clean-100h-300m": TrainingPreset(
        config=ROOT / "configs/omni/persian-asr-clean-100h-filter-ctc-300m-v2.yaml",
        output_dir=ROOT / "runs/omni-ctc-300m-persian-clean-100h-filter",
    ),
    "persian-balanced-100h-300m": TrainingPreset(
        config=ROOT / "configs/omni/persian-asr-balanced-100h-filter-ctc-300m-v2.yaml",
        output_dir=ROOT / "runs/omni-ctc-300m-persian-balanced-100h-filter",
    ),
    "persian-target-100h-300m": TrainingPreset(
        config=ROOT / "configs/omni/persian-asr-target-100h-filter-ctc-300m-v2.yaml",
        output_dir=ROOT / "runs/omni-ctc-300m-persian-target-100h-filter",
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


def configure_environment() -> None:
    os.environ.setdefault("HF_HOME", str(ROOT / ".hf-cache"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(ROOT / ".hf-cache/datasets"))
    os.environ.setdefault("FAIRSEQ2_ASSET_DIR", str(ROOT / ".fairseq2-assets"))
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
    os.environ.setdefault("TORCH_SHOW_CPP_STACKTRACES", "1")


def configure_recipe_source() -> None:
    recipe = OMNI_ROOT / OMNI_RECIPE_PATH
    if not recipe.is_file():
        raise RuntimeError(f"Missing vendored Omnilingual ASR recipe: {recipe}")
    sys.path.insert(0, str(OMNI_ROOT / "src"))
    sys.path.insert(0, str(OMNI_ROOT))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_file, output_dir = resolve_paths(args)
    configure_environment()
    configure_recipe_source()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    os.chdir(OMNI_ROOT)

    recipe_args = list(args.recipe_args)
    if recipe_args and recipe_args[0] == "--":
        recipe_args = recipe_args[1:]
    sys.argv = [
        OMNI_RECIPE_MODULE,
        str(output_dir),
        "--config-file",
        str(config_file),
        *recipe_args,
    ]
    runpy.run_module(OMNI_RECIPE_MODULE, run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
