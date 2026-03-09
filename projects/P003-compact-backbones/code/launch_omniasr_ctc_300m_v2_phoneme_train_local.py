#!/usr/bin/env python3
"""Launch OmniASR CTC 300M v2 phoneme fine-tuning in a dedicated Python 3.12 env."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


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


def _trigger_post_train_scoring(*, repo_root: Path, launch_after: str | None) -> None:
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
            (
                "omni:/home/simon/github/peacock-asr/projects/P003-compact-backbones/"
                "experiments/checkpoints/omniasr-ctc-300m-v2-phoneme-en"
            ),
            "--eval-yaml",
            (
                "/home/simon/github/peacock-asr/projects/P003-compact-backbones/"
                "experiments/sweeps/final/eval_omniasr_ctc_300m_v2_phoneme.yaml"
            ),
            "--label",
            "omniasr_ctc_300m_v2_phoneme",
            "--split",
            "both",
            "--device",
            "cuda",
        ]
    if launch_after:
        cmd.extend(["--launch-after", launch_after])
    subprocess.run(cmd, cwd=repo_root, check=True, text=True)  # noqa: S603


def main(argv: list[str] | None = None) -> int:
    raw_args = sys.argv[1:] if argv is None else argv
    args, auto_score, launch_after = _split_launcher_args(raw_args)
    repo_root = Path(__file__).resolve().parents[3]
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
        / "launch_omniasr_ctc_300m_v2_phoneme_train_impl.py"
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
        _trigger_post_train_scoring(repo_root=repo_root, launch_after=launch_after)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
