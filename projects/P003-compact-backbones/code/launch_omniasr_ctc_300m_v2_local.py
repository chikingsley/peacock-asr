#!/usr/bin/env python3
"""Launch the OmniASR CTC 300M v2 probe in a dedicated Python 3.12 env."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    repo_root = Path(__file__).resolve().parents[3]
    project_root = repo_root / "projects" / "P003-compact-backbones"
    omni_root = (
        repo_root
        / "projects"
        / "P004-training-from-scratch"
        / "third_party"
        / "omnilingual-asr"
    )
    probe = project_root / "code" / "launch_omniasr_ctc_300m_v2_probe.py"

    cmd = [
        "uv",
        "run",
        "--python",
        "3.12",
        "--with",
        "tbb>=2021.8",
        "--with-editable",
        str(omni_root),
        str(probe),
        *args,
    ]
    completed = subprocess.run(cmd, cwd=repo_root, check=False)  # noqa: S603
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
