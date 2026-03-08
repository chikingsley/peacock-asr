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

import importlib.util
import sys
from pathlib import Path
from typing import Any


def _emit(message: str) -> None:
    sys.stdout.write(f"{message}\n")


def _load_module(script_path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        msg = f"Could not load module from {script_path}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _run_script(script_path: Path, module_name: str, argv: list[str]) -> int:
    previous_argv = sys.argv[:]
    try:
        sys.argv = [str(script_path), *argv]
        module = _load_module(script_path, module_name)
        module.main()
    finally:
        sys.argv = previous_argv
    return 0


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

    _emit(
        "Launching trainer: "
        f"{training_script} {' '.join(sys.argv[1:])}".rstrip()
    )
    return _run_script(
        training_script,
        "p003_train_phoneme_head",
        sys.argv[1:],
    )


if __name__ == "__main__":
    raise SystemExit(main())
