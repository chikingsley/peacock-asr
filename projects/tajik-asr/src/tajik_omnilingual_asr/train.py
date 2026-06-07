from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from omni_finetune_core.train import configure_environment, run_recipe

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(__file__).resolve().parent / "configs"


@dataclass(frozen=True)
class TrainingPreset:
    config: Path
    output_dir: Path


PRESETS = {
    "tajik-corpus-v0-300m": TrainingPreset(
        config=CONFIG_DIR / "tajik-asr-corpus-v0-ctc-300m-v2.yaml",
        output_dir=ROOT / "runs/omni-ctc-300m-tajik-asr-corpus-v0",
    ),
    # v1 = v0 real Tajik + transliterated-Persian FLEURS augmentation (aggressive mix).
    "tajik-corpus-v1-300m": TrainingPreset(
        config=CONFIG_DIR / "tajik-asr-corpus-v1-ctc-300m-v2.yaml",
        output_dir=ROOT / "runs/omni-ctc-300m-tajik-asr-corpus-v1",
    ),
    # v2 = the new curator pipeline's full export: FLEURS + 41 YouTube channels (~1,400 h),
    # script-aware Scribe verification, WER <= 0.35 + descriptor-junk filter.
    "tajik-corpus-v2-300m": TrainingPreset(
        config=CONFIG_DIR / "tajik-asr-corpus-v2-ctc-300m-v2.yaml",
        output_dir=ROOT / "runs/omni-ctc-300m-tajik-asr-corpus-v2",
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
