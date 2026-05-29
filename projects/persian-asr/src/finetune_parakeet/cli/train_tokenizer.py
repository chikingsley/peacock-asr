from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from finetune_parakeet.paths import (
    DEFAULT_NEMO_ROOT,
    DEFAULT_TOKENIZER_ROOT,
    configure_external_caches,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a Persian ASR SentencePiece tokenizer with NeMo's tokenizer script."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--manifest", type=Path)
    input_group.add_argument("--data-file", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_TOKENIZER_ROOT)
    parser.add_argument("--name", default=None)
    parser.add_argument("--nemo-root", type=Path, default=DEFAULT_NEMO_ROOT)
    parser.add_argument("--python", default=sys.executable)
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


def tokenizer_data_root(args: argparse.Namespace) -> Path:
    name = args.name or f"fa_spe_{args.spe_type}_v{args.vocab_size}"
    return args.output_root / name


def build_command(args: argparse.Namespace) -> list[str]:
    script = args.nemo_root / "scripts/tokenizers/process_asr_text_tokenizer.py"
    if not script.exists():
        raise FileNotFoundError(script)
    command = [
        args.python,
        str(script),
        "--data_root",
        str(tokenizer_data_root(args)),
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
        command.extend(["--spe_max_sentencepiece_length", str(args.max_sentencepiece_length)])
    if args.manifest is not None:
        command.extend(["--manifest", str(args.manifest)])
    if args.data_file is not None:
        command.extend(["--data_file", str(args.data_file)])
    if args.byte_fallback:
        command.append("--spe_byte_fallback")
    if args.split_digits:
        command.append("--spe_split_digits")
    if not args.lower_case:
        command.append("--no_lower_case")
    return command


def main(argv: list[str] | None = None) -> int:
    configure_external_caches()
    args = build_parser().parse_args(argv)
    command = build_command(args)
    print(" ".join(command))
    if args.dry_run:
        return 0
    args.output_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
