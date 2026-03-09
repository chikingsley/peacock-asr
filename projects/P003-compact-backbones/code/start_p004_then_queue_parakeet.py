#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# ///
"""Start P004 conformer scoring, then queue Parakeet training after it finishes."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    repo_root = project_root.parents[1]

    trigger_script = project_root / "code" / "trigger_post_train_scoring.py"
    p004_backend = (
        "p004:/home/simon/github/peacock-asr/projects/P004-training-from-scratch/"
        "experiments/checkpoints/canonical_phone_ctc/"
        "canonical_local_conformer_full_trainclean100_e3_resume_20260307_a"
    )
    p004_eval_yaml = (
        project_root
        / "experiments"
        / "sweeps"
        / "final"
        / "eval_p004_conformer_full_trainclean100_e3.yaml"
    )

    trigger = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(trigger_script),
            "--backend",
            p004_backend,
            "--eval-yaml",
            str(p004_eval_yaml),
            "--label",
            "p004_conformer_full_trainclean100_e3",
            "--split",
            "both",
            "--device",
            "cuda",
            "--launch-after",
            (
                "uv run --project "
                "/home/simon/github/peacock-asr/projects/P003-compact-backbones "
                "python "
                "/home/simon/github/peacock-asr/projects/P003-compact-backbones/"
                "code/start_parakeet_then_queue_omni.py"
            ),
        ],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )
    lines = [line.strip() for line in trigger.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("Failed to capture P004 sweep ref from trigger output.")
    sweep_ref = lines[-1]
    sys.stdout.write(f"{sweep_ref}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
