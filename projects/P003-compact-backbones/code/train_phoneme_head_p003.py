#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "torchaudio",
#     "transformers>=4.40",
#     "datasets",
#     "accelerate",
#     "jiwer",
#     "evaluate",
# ]
# ///
"""Project-local wrapper for the P003 phoneme-head trainer."""

from __future__ import annotations

import sys
from pathlib import Path

from _launcher_lib import emit, run_script


def main() -> int:
    repo_root = Path(__file__).resolve().parents[3]
    training_script = (
        repo_root
        / "projects"
        / "P003-compact-backbones"
        / "code"
        / "training"
        / "train_phoneme_head.py"
    )
    hf_home = repo_root / ".cache" / "models" / "huggingface"
    hub_cache = hf_home / "hub"

    hf_home.mkdir(parents=True, exist_ok=True)
    hub_cache.mkdir(parents=True, exist_ok=True)

    emit(
        "Launching trainer: "
        f"{training_script} {' '.join(sys.argv[1:])}".rstrip()
    )
    return run_script(
        training_script,
        "p003_train_phoneme_head",
        sys.argv[1:],
    )


if __name__ == "__main__":
    raise SystemExit(main())
