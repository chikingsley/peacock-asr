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

import os
import sys
from pathlib import Path

from _launcher_lib import emit, run_script, split_launcher_args, trigger_post_train_scoring


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

    extra_args, auto_score, launch_after = split_launcher_args(sys.argv[1:])
    argv.extend(extra_args)
    emit(
        "Launching wav2vec2-large: "
        f"{training_wrapper} {' '.join(argv)}".rstrip()
    )
    exit_code = run_script(
        training_wrapper,
        "p003_train_phoneme_head_wrapper_large",
        argv,
        env,
    )
    if exit_code == 0 and auto_score:
        emit("Training complete; triggering automatic prewarm and scoring sweep.")
        trigger_post_train_scoring(
            repo_root=repo_root,
            backend="hf:Peacockery/wav2vec2-large-phoneme-en",
            eval_yaml=(
                repo_root
                / "projects"
                / "P003-compact-backbones"
                / "experiments"
                / "sweeps"
                / "final"
                / "eval_wav2vec2_large.yaml"
            ),
            label="wav2vec2_large",
            launch_after=launch_after,
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
