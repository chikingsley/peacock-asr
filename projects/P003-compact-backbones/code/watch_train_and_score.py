#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# ///
"""Watch an in-flight training PID and trigger post-train scoring when it finishes."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _training_succeeded(output_dir: Path, log_path: Path) -> bool:
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    if "Done." in log_text:
        return True
    if "Traceback" in log_text:
        return False
    return (output_dir / "config.json").exists() and (
        output_dir / "preprocessor_config.json"
    ).exists()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--eval-yaml", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-log", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "test", "both"], default="both")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--poll-seconds", type=int, default=60)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    script_path = Path(__file__).resolve().with_name("trigger_post_train_scoring.py")
    project_root = Path(__file__).resolve().parents[1]
    logs_dir = project_root / "experiments" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    watch_log = logs_dir / f"watch_train_{args.label}.log"

    with watch_log.open("a", encoding="utf-8") as handle:
        handle.write(
            f"[{datetime.now().astimezone().isoformat()}] "
            f"watching pid={args.pid} label={args.label}\n"
        )
        handle.flush()
        while _process_exists(args.pid):
            time.sleep(max(5, args.poll_seconds))

        handle.write(
            f"[{datetime.now().astimezone().isoformat()}] pid={args.pid} exited\n"
        )
        handle.flush()

    if not _training_succeeded(args.output_dir, args.train_log):
        msg = (
            f"Training for {args.label} exited without a successful completion marker: "
            f"{args.train_log}"
        )
        raise RuntimeError(msg)

    cmd = [
        sys.executable,
        str(script_path),
        "--backend",
        args.backend,
        "--eval-yaml",
        str(args.eval_yaml.resolve()),
        "--label",
        args.label,
        "--split",
        args.split,
        "--device",
        args.device,
    ]
    subprocess.run(  # noqa: S603
        cmd,
        cwd=project_root.parents[1],
        check=True,
        text=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
