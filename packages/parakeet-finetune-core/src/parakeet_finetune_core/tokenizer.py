"""Shared CLI for training NeMo/SentencePiece ASR tokenizers."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from parakeet_finetune_core.project import ParakeetProject


def build_parser(project: ParakeetProject) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Train a {project.name} Parakeet ASR SentencePiece tokenizer."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--manifest", type=Path)
    input_group.add_argument("--data-file", type=Path)
    parser.add_argument("--output-root", type=Path, default=project.tokenizers)
    parser.add_argument("--name", default=None)
    parser.add_argument("--nemo-root", type=Path, default=project.default_nemo_root())
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--spe-type", choices=["bpe", "unigram", "char", "word"], default="bpe")
    parser.add_argument("--character-coverage", type=float, default=1.0)
    parser.add_argument("--sample-size", type=int, default=-1)
    parser.add_argument("--max-sentencepiece-length", type=int, default=-1)
    parser.add_argument("--byte-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--split-digits", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lower-case", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def tokenizer_data_root(project: ParakeetProject, args: argparse.Namespace) -> Path:
    name = args.name or project.tokenizer_name(args.spe_type, args.vocab_size)
    return args.output_root / name


def build_script_args(project: ParakeetProject, args: argparse.Namespace) -> tuple[Path, list[str]]:
    script = args.nemo_root / "scripts/tokenizers/process_asr_text_tokenizer.py"
    if not script.exists():
        raise FileNotFoundError(script)
    script_args = [
        "--data_root",
        str(tokenizer_data_root(project, args)),
        "--tokenizer",
        "spe",
        "--spe_type",
        args.spe_type,
        "--vocab_size",
        str(args.vocab_size),
        "--spe_character_coverage",
        str(args.character_coverage),
        "--spe_sample_size",
        str(args.sample_size),
        "--log",
    ]
    if args.max_sentencepiece_length > 0:
        script_args.extend(["--spe_max_sentencepiece_length", str(args.max_sentencepiece_length)])
    if args.manifest is not None:
        script_args.extend(["--manifest", str(args.manifest)])
    if args.data_file is not None:
        script_args.extend(["--data_file", str(args.data_file)])
    if args.byte_fallback:
        script_args.append("--spe_byte_fallback")
    if args.split_digits:
        script_args.append("--spe_split_digits")
    if not args.lower_case:
        script_args.append("--no_lower_case")
    return script, script_args


def build_command(project: ParakeetProject, args: argparse.Namespace) -> list[str]:
    script, script_args = build_script_args(project, args)
    return ["runpy", str(script), *script_args]


def run_script(script: Path, script_args: list[str]) -> None:
    old_argv = sys.argv
    try:
        sys.argv = [str(script), *script_args]
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = old_argv


def train_tokenizer_main(project: ParakeetProject, argv: list[str] | None = None) -> int:
    project.configure_environment()
    args = build_parser(project).parse_args(argv)
    script, script_args = build_script_args(project, args)
    command = ["runpy", str(script), *script_args]
    print(" ".join(command))
    if args.dry_run:
        return 0
    args.output_root.mkdir(parents=True, exist_ok=True)
    run_script(script, script_args)
    return 0
