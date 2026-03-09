#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# ///
"""Launch Parakeet training/scoring, then queue Omni training after scoring finishes."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    repo_root = project_root.parents[1]
    parakeet_launcher = project_root / "code" / "launch_parakeet_ctc_0_6b_local.py"

    subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(parakeet_launcher),
            "--launch-after",
            (
                "uv run --project "
                "/home/simon/github/peacock-asr/projects/P003-compact-backbones "
                "python "
                "/home/simon/github/peacock-asr/projects/P003-compact-backbones/"
                "code/launch_omniasr_ctc_300m_v2_phoneme_train_local.py"
            ),
        ],
        cwd=repo_root,
        check=True,
        text=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
