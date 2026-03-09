#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "PyYAML>=6.0",
# ]
# ///
"""Prewarm k2 and launch the canonical eval sweep after a training run finishes."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def _run(
    cmd: list[str],
    *,
    cwd: Path,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    if cmd and cmd[0] == "uv":
        resolved_uv = shutil.which("uv")
        if resolved_uv is None:
            msg = "Could not resolve 'uv' on PATH."
            raise RuntimeError(msg)
        cmd = [resolved_uv, *cmd[1:]]
    return subprocess.run(  # noqa: S603
        cmd,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=capture_output,
    )


def _parse_yaml_scalar(path: Path, key: str) -> str:
    pattern = re.compile(rf"^{re.escape(key)}:\s*(\S+)\s*$")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line.strip())
        if match:
            return match.group(1).strip("\"'")
    msg = f"Could not find {key!r} in {path}"
    raise ValueError(msg)


def _parse_sweep_id(stdout: str, stderr: str) -> str:
    combined = f"{stdout}\n{stderr}"
    patterns = [
        re.compile(r"Created sweep with ID:\s*([a-z0-9]+)", re.IGNORECASE),
        re.compile(r"wandb agent\s+\S+/(\S+)/([a-z0-9]+)", re.IGNORECASE),
        re.compile(r"/sweeps/([a-z0-9]+)", re.IGNORECASE),
    ]
    for pattern in patterns:
        match = pattern.search(combined)
        if match:
            return match.group(match.lastindex or 1)
    msg = f"Could not parse sweep id from wandb output:\n{combined}"
    raise RuntimeError(msg)


def _write_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["updated_at"] = datetime.now().astimezone().isoformat()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _expected_results(eval_yaml: Path) -> int:
    data = yaml.safe_load(eval_yaml.read_text(encoding="utf-8"))
    parameters = data.get("parameters", {}) if isinstance(data, dict) else {}
    seed_cfg = parameters.get("seed", {}) if isinstance(parameters, dict) else {}
    if isinstance(seed_cfg, dict):
        if "values" in seed_cfg and isinstance(seed_cfg["values"], list):
            return len(seed_cfg["values"])
        if "value" in seed_cfg:
            return 1
    return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--eval-yaml", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--split", choices=["train", "test", "both"], default="both")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--launch-after",
        help="Shell command to launch after the scoring agent exits successfully.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    repo_root = Path(__file__).resolve().parents[3]
    logs_dir = project_root / "experiments" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    state_path = logs_dir / f"post_train_{args.label}.json"

    eval_yaml = args.eval_yaml.resolve()
    backend = args.backend

    state: dict[str, Any] = {
        "label": args.label,
        "backend": backend,
        "eval_yaml": str(eval_yaml),
        "project_root": str(project_root),
        "repo_root": str(repo_root),
        "stage": "prewarm",
    }
    _write_state(state_path, state)

    prewarm_cmd = [
        "uv",
        "run",
        "--project",
        str(project_root),
        "python",
        "-m",
        "p003_compact.cli",
        "prewarm-k2",
        "--backend",
        backend,
        "--split",
        args.split,
        "--device",
        args.device,
    ]
    _run(prewarm_cmd, cwd=repo_root)

    state["stage"] = "create_sweep"
    _write_state(state_path, state)

    sweep_cmd = [
        "uv",
        "run",
        "--project",
        str(project_root),
        "wandb",
        "sweep",
        str(eval_yaml),
    ]
    sweep_result = _run(sweep_cmd, cwd=repo_root, capture_output=True)
    sweep_id = _parse_sweep_id(sweep_result.stdout, sweep_result.stderr)
    entity = _parse_yaml_scalar(eval_yaml, "entity")
    project = _parse_yaml_scalar(eval_yaml, "project")
    sweep_ref = f"{entity}/{project}/{sweep_id}"

    agent_log = logs_dir / f"sweep_eval_{args.label}_{sweep_id}.log"
    resolved_uv = shutil.which("uv")
    if resolved_uv is None:
        msg = "Could not resolve 'uv' on PATH."
        raise RuntimeError(msg)
    with agent_log.open("ab") as handle:
        agent = subprocess.Popen(  # noqa: S603
            [
                resolved_uv,
                "run",
                "--project",
                str(project_root),
                "wandb",
                "agent",
                sweep_ref,
            ],
            cwd=repo_root,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=False,
        )

    state.update(
        {
            "stage": "agent_started",
            "sweep_id": sweep_id,
            "sweep_ref": sweep_ref,
            "agent_pid": agent.pid,
            "agent_log": str(agent_log),
        }
    )
    _write_state(state_path, state)

    if args.launch_after:
        watch_script = project_root / "code" / "watch_agent_log_and_launch.py"
        watch_log = logs_dir / f"watch_agent_{args.label}.log"
        expected_results = _expected_results(eval_yaml)
        with watch_log.open("ab") as handle:
            watcher = subprocess.Popen(  # noqa: S603
                [
                    resolved_uv,
                    "run",
                    "--project",
                    str(project_root),
                    "python",
                    str(watch_script),
                    "--pid",
                    str(agent.pid),
                    "--label",
                    args.label,
                    "--log-path",
                    str(agent_log),
                    "--expected-results",
                    str(expected_results),
                    "--launch",
                    args.launch_after,
                ],
                cwd=repo_root,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                text=False,
            )
        state.update(
            {
                "watcher_pid": watcher.pid,
                "watcher_log": str(watch_log),
                "launch_after": args.launch_after,
                "expected_results": expected_results,
            }
        )
        _write_state(state_path, state)
    sys.stdout.write(f"{sweep_ref}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
