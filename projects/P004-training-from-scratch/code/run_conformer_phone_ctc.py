#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///
# References:
# - icefall LibriSpeech Conformer CTC recipe:
#   https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/conformer_ctc
# - icefall recipe docs:
#   https://k2-fsa.github.io/icefall/recipes/Non-streaming-ASR/librispeech/conformer_ctc.html
#
# Usage here:
# - Keep the upstream reference lane close to stock icefall.
# - Only inject the local `lang_phone`, manifest, env, and exp-dir wiring.
from __future__ import annotations

import argparse
import logging
import os
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path("/home/simon/github/peacock-asr")
P004_ROOT = ROOT / "projects" / "P004-training-from-scratch"
P004_ENV = P004_ROOT / ".venv-icefall"
ICEFALL_ROOT = P004_ROOT / "third_party" / "icefall"
RECIPE_DIR = ICEFALL_ROOT / "egs" / "librispeech" / "ASR" / "conformer_ctc"
DEFAULT_MANIFEST_DIR = P004_ROOT / "experiments" / "data" / "manifests_phone_raw"
DEFAULT_LANG_DIR = P004_ROOT / "experiments" / "data" / "lang_phone"
DEFAULT_EXP_DIR = P004_ROOT / "experiments" / "checkpoints" / "conformer_ctc_phone"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the P004 phone-CTC Conformer recipe through the dedicated "
            "icefall env."
        )
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=DEFAULT_MANIFEST_DIR,
        help=f"Manifest directory (default: {DEFAULT_MANIFEST_DIR})",
    )
    parser.add_argument(
        "--lang-dir",
        type=Path,
        default=DEFAULT_LANG_DIR,
        help=f"lang_phone directory (default: {DEFAULT_LANG_DIR})",
    )
    parser.add_argument(
        "--exp-dir",
        type=Path,
        default=DEFAULT_EXP_DIR,
        help=f"Experiment directory (default: {DEFAULT_EXP_DIR})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved train command and exit without launching",
    )
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Additional train.py args passed through unchanged",
    )
    return parser.parse_args()


def _prepend_env_path(env: dict[str, str], key: str, path: Path) -> None:
    if not path.is_dir():
        return
    previous = env.get(key, "")
    env[key] = f"{path}:{previous}" if previous else str(path)


def build_env() -> dict[str, str]:
    env = os.environ.copy()
    nvrtc_dir = (
        P004_ENV
        / "lib"
        / "python3.11"
        / "site-packages"
        / "nvidia"
        / "cuda_nvrtc"
        / "lib"
    )
    _prepend_env_path(env, "LD_LIBRARY_PATH", nvrtc_dir)
    _prepend_env_path(env, "PYTHONPATH", ICEFALL_ROOT)
    return env


def main() -> int:
    args = parse_args()
    passthrough_args = list(args.args)
    while passthrough_args[:1] == ["--"]:
        passthrough_args = passthrough_args[1:]

    train_py = RECIPE_DIR / "train.py"
    env_python = P004_ENV / "bin" / "python"
    if not env_python.is_file():
        msg = f"P004 env python not found: {env_python}"
        raise FileNotFoundError(msg)
    if not train_py.is_file():
        msg = f"icefall conformer_ctc train.py not found: {train_py}"
        raise FileNotFoundError(msg)

    cmd = [
        str(env_python),
        str(train_py),
        "--world-size",
        "1",
        "--num-epochs",
        "20",
        "--exp-dir",
        str(args.exp_dir),
        "--lang-dir",
        str(args.lang_dir),
        "--manifest-dir",
        str(args.manifest_dir),
        "--full-libri",
        "0",
        "--att-rate",
        "0",
        "--num-decoder-layers",
        "0",
        "--on-the-fly-feats",
        "1",
        "--enable-musan",
        "0",
        "--max-duration",
        "20",
        "--num-workers",
        "4",
        *passthrough_args,
    ]

    logger.info("Running: %s", " ".join(cmd))
    if args.dry_run:
        return 0

    subprocess.run(  # noqa: S603
        cmd,
        check=True,
        cwd=RECIPE_DIR,
        env=build_env(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
