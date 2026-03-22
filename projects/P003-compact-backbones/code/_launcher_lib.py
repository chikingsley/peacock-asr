"""Shared utilities for P003 launcher and orchestration scripts."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def emit(message: str) -> None:
    sys.stdout.write(f"{message}\n")


def load_module(script_path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        msg = f"Could not load module from {script_path}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def run_script(
    script_path: Path,
    module_name: str,
    argv: list[str],
    env: dict[str, str] | None = None,
) -> int:
    previous_argv = sys.argv[:]
    previous_env = {key: os.environ.get(key) for key in env} if env else {}
    try:
        if env:
            os.environ.update(env)
        sys.argv = [str(script_path), *argv]
        module = load_module(script_path, module_name)
        module.main()
    finally:
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        sys.argv = previous_argv
    return 0


def split_launcher_args(argv: list[str]) -> tuple[list[str], bool, str | None]:
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


def trigger_post_train_scoring(
    *,
    repo_root: Path,
    backend: str,
    eval_yaml: Path | str,
    label: str,
    launch_after: str | None,
) -> None:
    script_path = (
        repo_root
        / "projects"
        / "P003-compact-backbones"
        / "code"
        / "orchestration"
        / "trigger_post_train_scoring.py"
    )
    cmd = [
        sys.executable,
        str(script_path),
        "--backend",
        backend,
        "--eval-yaml",
        str(eval_yaml),
        "--label",
        label,
        "--split",
        "both",
        "--device",
        "cuda",
    ]
    if launch_after:
        cmd.extend(["--launch-after", launch_after])
    subprocess.run(cmd, cwd=repo_root, check=True, text=True)  # noqa: S603


def write_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["updated_at"] = datetime.now().astimezone().isoformat()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
