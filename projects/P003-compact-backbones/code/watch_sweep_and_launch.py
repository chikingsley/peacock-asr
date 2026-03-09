#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "wandb>=0.22.1",
# ]
# ///
"""Watch a W&B sweep and launch a follow-up command when it finishes cleanly."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from datetime import datetime
from pathlib import Path

import wandb


def _write_state(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["updated_at"] = datetime.now().astimezone().isoformat()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _sweep_finished_ok(sweep: wandb.apis.public.sweeps.Sweep) -> bool:
    states = {run.state for run in sweep.runs}
    if any(state in {"failed", "crashed", "killed", "preempted"} for state in states):
        return False
    return sweep.state.upper() in {"FINISHED", "CANCELED"} and not any(
        state in {"running", "queued", "pending"} for state in states
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-ref", required=True, help="entity/project/sweep_id")
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--launch",
        required=True,
        help="Shell command to run after the sweep finishes cleanly.",
    )
    parser.add_argument("--poll-seconds", type=int, default=60)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    logs_dir = project_root / "experiments" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    state_path = logs_dir / f"watch_sweep_{args.label}.json"
    log_path = logs_dir / f"watch_sweep_{args.label}.log"

    api = wandb.Api()
    launch_cmd = shlex.split(args.launch)
    state: dict[str, object] = {
        "label": args.label,
        "sweep_ref": args.sweep_ref,
        "launch_cmd": launch_cmd,
        "stage": "watching",
    }
    _write_state(state_path, state)

    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"[{datetime.now().astimezone().isoformat()}] watching {args.sweep_ref}\n"
        )
        handle.flush()
        while True:
            sweep = api.sweep(args.sweep_ref)
            run_states = {run.id: run.state for run in sweep.runs}
            state.update(
                {
                    "stage": "watching",
                    "sweep_state": sweep.state,
                    "run_states": run_states,
                }
            )
            _write_state(state_path, state)
            if _sweep_finished_ok(sweep):
                break
            if any(
                run_state in {"failed", "crashed", "killed", "preempted"}
                for run_state in run_states.values()
            ):
                state["stage"] = "blocked_failed_run"
                _write_state(state_path, state)
                handle.write(
                    "["
                    f"{datetime.now().astimezone().isoformat()}"
                    "] failed run detected; not launching follow-up\n"
                )
                handle.flush()
                return 1
            time.sleep(max(10, args.poll_seconds))

        state["stage"] = "launching"
        _write_state(state_path, state)
        handle.write(
            "["
            f"{datetime.now().astimezone().isoformat()}"
            "] launching follow-up: "
            f"{' '.join(launch_cmd)}\n"
        )
        handle.flush()
        subprocess.run(launch_cmd, cwd=project_root.parents[1], check=True, text=True)  # noqa: S603
        state["stage"] = "done"
        _write_state(state_path, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
