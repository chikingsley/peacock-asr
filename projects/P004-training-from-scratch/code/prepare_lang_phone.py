#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///
# References:
# - icefall TIMIT `prepare_lang.py` entrypoint:
#   https://github.com/k2-fsa/icefall/blob/master/egs/timit/ASR/local/prepare_lang.py
# - icefall CTC utilities:
#   https://github.com/k2-fsa/icefall/tree/master/icefall/ctc
#
# Usage here:
# - Build a minimal `lang_phone` identity lexicon so the upstream reference lane
#   can consume the lab's ARPABET vocabulary without inventing a new tokenizer.
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_VOCAB = Path(
    "/home/simon/github/peacock-asr/projects/P003-compact-backbones/code/training/vocab.json"
)
DEFAULT_LANG_DIR = Path(
    "/home/simon/github/peacock-asr/projects/P004-training-from-scratch/experiments/data/lang_phone"
)
DEFAULT_ICEFALL_PREPARE_LANG = Path(
    "/home/simon/github/peacock-asr/projects/P004-training-from-scratch/third_party/"
    "icefall/egs/timit/ASR/local/prepare_lang.py"
)
DEFAULT_ICEFALL_ROOT = Path(
    "/home/simon/github/peacock-asr/projects/P004-training-from-scratch/third_party/"
    "icefall"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an icefall lang_phone directory from the peacock ARPABET vocab."
        )
    )
    parser.add_argument(
        "--vocab-json",
        type=Path,
        default=DEFAULT_VOCAB,
        help=f"Path to phoneme vocab JSON (default: {DEFAULT_VOCAB})",
    )
    parser.add_argument(
        "--lang-dir",
        type=Path,
        default=DEFAULT_LANG_DIR,
        help=f"Output lang_phone directory (default: {DEFAULT_LANG_DIR})",
    )
    parser.add_argument(
        "--icefall-prepare-lang",
        type=Path,
        default=DEFAULT_ICEFALL_PREPARE_LANG,
        help=(
            "Path to icefall prepare_lang.py "
            f"(default: {DEFAULT_ICEFALL_PREPARE_LANG})"
        ),
    )
    parser.add_argument(
        "--icefall-root",
        type=Path,
        default=DEFAULT_ICEFALL_ROOT,
        help=(
            "Path to icefall repo root for PYTHONPATH "
            f"(default: {DEFAULT_ICEFALL_ROOT})"
        ),
    )
    parser.add_argument(
        "--skip-prepare-lang",
        action="store_true",
        help=(
            "Only write lexicon.txt and phone_list.txt; do not invoke "
            "icefall prepare_lang.py"
        ),
    )
    return parser.parse_args()


def ordered_phones(vocab: dict[str, int]) -> list[str]:
    phone_items = [
        (token, idx)
        for token, idx in vocab.items()
        if not token.startswith("[") and not token.startswith("<")
    ]
    phone_items.sort(key=lambda item: item[1])
    return [token for token, _ in phone_items]


def write_identity_lexicon(lang_dir: Path, phones: list[str]) -> None:
    lexicon_path = lang_dir / "lexicon.txt"
    phone_list_path = lang_dir / "phone_list.txt"
    fallback_phone = phones[0]

    with lexicon_path.open("w", encoding="utf-8") as handle:
        for phone in phones:
            handle.write(f"{phone} {phone}\n")
        handle.write(f"<UNK> {fallback_phone}\n")

    with phone_list_path.open("w", encoding="utf-8") as handle:
        for phone in phones:
            handle.write(f"{phone}\n")


def _prepend_env_path(env: dict[str, str], key: str, path: Path) -> None:
    if not path.is_dir():
        return
    previous = env.get(key, "")
    env[key] = f"{path}:{previous}" if previous else str(path)


def build_child_env(icefall_root: Path) -> dict[str, str]:
    child_env = os.environ.copy()
    env_root = Path(sys.executable).parent.parent
    nvrtc_dir = (
        env_root
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
        / "nvidia"
        / "cuda_nvrtc"
        / "lib"
    )
    _prepend_env_path(child_env, "LD_LIBRARY_PATH", nvrtc_dir)
    _prepend_env_path(child_env, "PYTHONPATH", icefall_root)
    return child_env


def main() -> int:
    args = parse_args()

    if not args.vocab_json.is_file():
        msg = f"Vocab JSON not found: {args.vocab_json}"
        raise FileNotFoundError(msg)

    args.lang_dir.mkdir(parents=True, exist_ok=True)

    vocab = json.loads(args.vocab_json.read_text(encoding="utf-8"))
    phones = ordered_phones(vocab)
    if not phones:
        msg = "No phones found in vocab JSON after filtering special tokens"
        raise RuntimeError(msg)

    write_identity_lexicon(args.lang_dir, phones)
    logger.info("Wrote %d phones to %s", len(phones), args.lang_dir / "lexicon.txt")

    if args.skip_prepare_lang:
        logger.info("Skipped icefall prepare_lang.py")
        return 0

    if not args.icefall_prepare_lang.is_file():
        msg = f"icefall prepare_lang.py not found: {args.icefall_prepare_lang}"
        raise FileNotFoundError(msg)

    cmd = [
        sys.executable,
        str(args.icefall_prepare_lang),
        "--lang-dir",
        str(args.lang_dir),
    ]
    logger.info("Running: %s", " ".join(cmd))
    try:
        subprocess.run(  # noqa: S603
            cmd,
            check=True,
            env=build_child_env(args.icefall_root),
        )
    except subprocess.CalledProcessError:
        logger.exception(
            "icefall prepare_lang.py failed; lang scaffolding remains at %s",
            args.lang_dir,
        )
        raise

    logger.info("Prepared lang_phone at %s", args.lang_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
