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
"""Launch the canonical local HuBERT-base phoneme-head training run."""

from __future__ import annotations

import importlib.util
import os
import subprocess
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


def _split_launcher_args(argv: list[str]) -> tuple[list[str], bool, str | None]:
    training_args: list[str] = []
    auto_score = True
    launch_after: str | None = None
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == "--no-score-after":
            auto_score = False
            index += 1
            continue
        if arg == "--score-after":
            auto_score = True
            index += 1
            continue
        if arg == "--launch-after":
            if index + 1 >= len(argv):
                raise SystemExit("--launch-after requires a value")
            launch_after = argv[index + 1]
            index += 2
            continue
        training_args.append(arg)
        index += 1
    return training_args, auto_score, launch_after


def _trigger_post_train_scoring(
    *,
    repo_root: Path,
    backend: str,
    eval_yaml: Path,
    label: str,
    launch_after: str | None,
) -> None:
    script_path = (
        repo_root
        / "projects"
        / "P003-compact-backbones"
        / "code"
        / "trigger_post_train_scoring.py"
    )
    cmd = [
            sys.executable,
            str(script_path),
            "--backend",
            backend,
            "--eval-yaml",
            str(eval_yaml),
            "--label",
            label,
            "--split",
            "both",
            "--device",
            "cuda",
        ]
    if launch_after:
        cmd.extend(["--launch-after", launch_after])
    subprocess.run(cmd, cwd=repo_root, check=True, text=True)  # noqa: S603


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
        / "hubert-base-phoneme-en"
    )

    hf_home.mkdir(parents=True, exist_ok=True)
    hub_cache.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("HF_HOME", str(hf_home))
    env.setdefault("HF_HUB_CACHE", str(hub_cache))
    env.setdefault("TRANSFORMERS_CACHE", str(hub_cache))

    argv = [
        "--model-name",
        "facebook/hubert-base-ls960",
        "--output-dir",
        str(output_dir),
        "--hub-repo",
        "Peacockery/hubert-base-phoneme-en",
        "--batch-size",
        "8",
        "--gradient-accumulation",
        "8",
        "--learning-rate",
        "3e-5",
        "--num-epochs",
        "3",
        "--dataloader-workers",
        "4",
    ]

    extra_args, auto_score, launch_after = _split_launcher_args(sys.argv[1:])
    argv.extend(extra_args)
    _emit(
        "Launching HuBERT-base: "
        f"{training_wrapper} {' '.join(argv)}".rstrip()
    )
    exit_code = _run_script(
        training_wrapper,
        "p003_train_phoneme_head_wrapper",
        argv,
        env,
    )
    if exit_code == 0 and auto_score:
        _emit("Training complete; triggering automatic prewarm and scoring sweep.")
        _trigger_post_train_scoring(
            repo_root=repo_root,
            backend="hf:Peacockery/hubert-base-phoneme-en",
            eval_yaml=(
                repo_root
                / "projects"
                / "P003-compact-backbones"
                / "experiments"
                / "sweeps"
                / "final"
                / "eval_hubert_base.yaml"
            ),
            label="hubert_base",
            launch_after=launch_after,
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
