#!/usr/bin/env python3
"""Launch the wav2vec2-large eval sweep with safe k2 budgets and queue follow-ups."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _write_state(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["updated_at"] = datetime.now().astimezone().isoformat()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[3]
    project_root = repo_root / "projects" / "P003-compact-backbones"
    logs_dir = project_root / "experiments" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    sweep_yaml = (
        project_root / "experiments" / "sweeps" / "final" / "eval_wav2vec2_large.yaml"
    )
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("Could not resolve 'uv' on PATH.")
    sweep_create = subprocess.run(  # noqa: S603
        [uv, "run", "--project", str(project_root), "wandb", "sweep", str(sweep_yaml)],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )
    combined = f"{sweep_create.stdout}\n{sweep_create.stderr}"
    markers = ("Created sweep with ID:", "Creating sweep with ID:")
    sweep_id = None
    for line in combined.splitlines():
        for marker in markers:
            if marker in line:
                sweep_id = line.split(marker, 1)[1].strip()
                break
        if sweep_id is not None:
            break
    if sweep_id is None:
        raise RuntimeError(f"Could not parse sweep id from output:\n{combined}")
    sweep_ref = f"peacockery/peacock-asr-p003-compact-backbones/{sweep_id}"

    env = {
        "CTC_SCALAR_BATCH_UTTERANCES": "4",
        "CTC_SCALAR_BATCH_PHONE_POSITIONS": "48",
        "CTC_SCALAR_BATCH_CASE_FRAME_BUDGET": "24000",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }

    agent_log = logs_dir / f"sweep_eval_wav2vec2_large_{sweep_id}.log"
    with agent_log.open("ab") as handle:
        agent = subprocess.Popen(  # noqa: S603
            [
                uv,
                "run",
                "--project",
                str(project_root),
                "wandb",
                "agent",
                sweep_ref,
            ],
            cwd=repo_root,
            env={**os.environ, **env},
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=False,
        )

    state_path = logs_dir / "post_train_wav2vec2_large.json"
    _write_state(
        state_path,
        {
            "label": "wav2vec2_large",
            "backend": "hf:Peacockery/wav2vec2-large-phoneme-en",
            "eval_yaml": str(sweep_yaml),
            "project_root": str(project_root),
            "repo_root": str(repo_root),
            "stage": "agent_started",
            "sweep_id": sweep_id,
            "sweep_ref": sweep_ref,
            "agent_pid": agent.pid,
            "agent_log": str(agent_log),
        },
    )

    watcher_log = logs_dir / "watch_sweep_wav2vec2_large.log"
    with watcher_log.open("ab") as handle:
        watcher = subprocess.Popen(  # noqa: S603
            [
                uv,
                "run",
                "--project",
                str(project_root),
                "python",
                str(project_root / "code" / "watch_sweep_and_launch.py"),
                "--sweep-ref",
                sweep_ref,
                "--label",
                "wav2vec2_large",
                "--launch",
                (
                    "uv run --project "
                    "/home/simon/github/peacock-asr/projects/P003-compact-backbones "
                    "python "
                    "/home/simon/github/peacock-asr/projects/P003-compact-backbones/"
                    "code/start_p004_then_queue_parakeet.py"
                ),
                "--poll-seconds",
                "60",
            ],
            cwd=repo_root,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=False,
        )

    sys.stdout.write(json.dumps(
        {
            "sweep_ref": sweep_ref,
            "agent_pid": agent.pid,
            "agent_log": str(agent_log),
            "watcher_pid": watcher.pid,
            "watcher_log": str(watcher_log),
            "env": env,
        },
        indent=2,
    ) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
