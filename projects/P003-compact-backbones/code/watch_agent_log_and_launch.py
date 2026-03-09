#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# ///
"""Watch a local W&B agent PID and launch a follow-up command from log state."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from datetime import datetime
from pathlib import Path


def _write_state(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["updated_at"] = datetime.now().astimezone().isoformat()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _analyze_log(log_path: Path) -> dict[str, object]:
    text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    return {
        "pcc_count": text.count("PCC:"),
        "has_traceback": "Traceback (most recent call last):" in text,
        "has_agent_cleanup": "Cleaning up finished run:" in text,
        "has_agent_exit": "Received exit command. Killing runs and quitting." in text,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument("--expected-results", type=int, required=True)
    parser.add_argument("--launch", required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    logs_dir = project_root / "experiments" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    state_path = logs_dir / f"watch_agent_{args.label}.json"
    log_path = logs_dir / f"watch_agent_{args.label}.log"

    launch_cmd = shlex.split(args.launch)
    state: dict[str, object] = {
        "label": args.label,
        "pid": args.pid,
        "log_path": str(args.log_path),
        "expected_results": args.expected_results,
        "launch_cmd": launch_cmd,
        "stage": "watching",
    }
    _write_state(state_path, state)

    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"[{datetime.now().astimezone().isoformat()}] watching pid {args.pid}\n"
        )
        handle.flush()
        while _pid_alive(args.pid):
            state.update(
                {
                    "stage": "watching",
                    "agent_alive": True,
                    "log_summary": _analyze_log(args.log_path),
                }
            )
            _write_state(state_path, state)
            time.sleep(max(10, args.poll_seconds))

        summary = _analyze_log(args.log_path)
        state.update(
            {
                "stage": "agent_exited",
                "agent_alive": False,
                "log_summary": summary,
            }
        )
        _write_state(state_path, state)

        pcc_count = (
            int(summary["pcc_count"])
            if isinstance(summary["pcc_count"], int)
            else 0
        )
        has_traceback = (
            bool(summary["has_traceback"])
            if isinstance(summary["has_traceback"], bool)
            else True
        )
        success = not has_traceback and pcc_count >= args.expected_results
        if not success:
            state["stage"] = "blocked_incomplete_log"
            _write_state(state_path, state)
            handle.write(
                "["
                f"{datetime.now().astimezone().isoformat()}"
                "] agent exited without enough successful results; "
                "not launching follow-up\n"
            )
            handle.flush()
            return 1

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
