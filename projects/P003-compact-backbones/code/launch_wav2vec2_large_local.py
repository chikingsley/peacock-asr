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
"""Launch the canonical local wav2vec2-large phoneme-head training run."""

from __future__ import annotations

import importlib.util
import os
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


def _run_script(
    script_path: Path,
    module_name: str,
    argv: list[str],
    env: dict[str, str],
) -> int:
    previous_argv = sys.argv[:]
    previous_env = {key: os.environ.get(key) for key in env}
    try:
        os.environ.update(env)
        sys.argv = [str(script_path), *argv]
        module = _load_module(script_path, module_name)
        module.main()
    finally:
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        sys.argv = previous_argv
    return 0


def main() -> int:
    repo_root = Path(__file__).resolve().parents[3]
    training_wrapper = (
        repo_root
        / "projects"
        / "P003-compact-backbones"
        / "code"
        / "train_phoneme_head_p003.py"
    )
    hf_home = repo_root / ".cache" / "models" / "huggingface"
    hub_cache = hf_home / "hub"
    output_dir = (
        repo_root
        / "projects"
        / "P003-compact-backbones"
        / "experiments"
        / "checkpoints"
        / "wav2vec2-large-phoneme-en"
    )

    hf_home.mkdir(parents=True, exist_ok=True)
    hub_cache.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("HF_HOME", str(hf_home))
    env.setdefault("HF_HUB_CACHE", str(hub_cache))
    env.setdefault("TRANSFORMERS_CACHE", str(hub_cache))
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    argv = [
        "--model-name",
        "facebook/wav2vec2-large-960h-lv60-self",
        "--output-dir",
        str(output_dir),
        "--hub-repo",
        "Peacockery/wav2vec2-large-phoneme-en",
        "--batch-size",
        "1",
        "--gradient-accumulation",
        "32",
        "--learning-rate",
        "3e-5",
        "--num-epochs",
        "3",
        "--dataloader-workers",
        "4",
    ]

    argv.extend(sys.argv[1:])
    _emit(
        "Launching wav2vec2-large: "
        f"{training_wrapper} {' '.join(argv)}".rstrip()
    )
    return _run_script(
        training_wrapper,
        "p003_train_phoneme_head_wrapper_large",
        argv,
        env,
    )


if __name__ == "__main__":
    raise SystemExit(main())
