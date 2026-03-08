#!/usr/bin/env python3
"""Launch the first real Citrinet P2-B train-clean-100 experiment."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
UTC_TZ = getattr(__import__("datetime"), "UTC", None) or timezone.__dict__["utc"]
SCRIPT_PATH = REPO_ROOT / (
    "projects/P003-compact-backbones/code/citrinet/scripts/train_citrinet_p2b.py"
)
DEFAULT_DATA_ROOT = REPO_ROOT / (
    "projects/P003-compact-backbones/experiments/citrinet/data/train_clean_100_full"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / (
    "projects/P003-compact-backbones/experiments/citrinet/checkpoints"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--precision", default="16-mixed")
    parser.add_argument("--wandb-project", default="peacock-asr-p003-citrinet")
    parser.add_argument("--wandb-entity", default="peacockery")
    parser.add_argument("--wandb-group", default="p003-citrinet-p2b")
    parser.add_argument("--wandb-mode", default="online")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_manifest = args.data_root / "manifests/train.jsonl"
    eval_manifest = args.data_root / "manifests/eval.jsonl"
    timestamp = datetime.now(UTC_TZ).strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"citrinet-256-p2b-train-clean-100-{timestamp}"
    output_dir = args.output_root / run_name

    cmd = [
        str(args.python_bin),
        str(SCRIPT_PATH),
        "--train-manifest",
        str(train_manifest),
        "--eval-manifest",
        str(eval_manifest),
        "--output-dir",
        str(output_dir),
        "--run-name",
        run_name,
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--lr",
        str(args.lr),
        "--max-epochs",
        str(args.max_epochs),
        "--max-steps",
        str(args.max_steps),
        "--precision",
        args.precision,
        "--wandb-project",
        args.wandb_project,
        "--wandb-entity",
        args.wandb_entity,
        "--wandb-group",
        args.wandb_group,
        "--wandb-mode",
        args.wandb_mode,
    ]

    sys.stdout.write(shlex.join(cmd) + "\n")
    if args.dry_run:
        return
    subprocess.run(cmd, check=True)  # noqa: S603


if __name__ == "__main__":
    main()
