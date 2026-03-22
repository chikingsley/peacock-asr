#!/usr/bin/env python3
"""Launch OmniASR CTC 300M v2 phoneme fine-tuning in a dedicated Python 3.12 env."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _launcher_lib import split_launcher_args, trigger_post_train_scoring


def main(argv: list[str] | None = None) -> int:
    raw_args = sys.argv[1:] if argv is None else argv
    args, auto_score, launch_after = split_launcher_args(raw_args)
    repo_root = Path(__file__).resolve().parents[4]
    project_root = repo_root / "projects" / "P003-compact-backbones"
    omni_root = (
        repo_root
        / "projects"
        / "P004-training-from-scratch"
        / "third_party"
        / "omnilingual-asr"
    )
    launcher = (
        project_root
        / "code"
        / "omniasr"
        / "train_impl.py"
    )

    cmd = [
        "uv",
        "run",
        "--python",
        "3.12",
        "--with",
        "tbb>=2021.8",
        "--with-editable",
        str(omni_root),
        str(launcher),
        *args,
    ]
    completed = subprocess.run(cmd, cwd=repo_root, check=False)  # noqa: S603
    if completed.returncode == 0 and auto_score and "--check-only" not in args:
        trigger_post_train_scoring(
            repo_root=repo_root,
            backend=(
                "omni:/home/simon/github/peacock-asr/projects/P003-compact-backbones/"
                "experiments/checkpoints/omniasr-ctc-300m-v2-phoneme-en"
            ),
            eval_yaml=(
                "/home/simon/github/peacock-asr/projects/P003-compact-backbones/"
                "experiments/sweeps/final/eval_omniasr_ctc_300m_v2_phoneme.yaml"
            ),
            label="omniasr_ctc_300m_v2_phoneme",
            launch_after=launch_after,
        )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
