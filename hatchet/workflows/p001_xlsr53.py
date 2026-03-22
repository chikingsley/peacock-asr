"""Concrete Hatchet workflow for the canonical P001 XLSR-53 baseline."""

from __future__ import annotations

import re
import shutil
import subprocess
from datetime import timedelta
from pathlib import Path
from typing import Any

from hatchet_sdk import Context
from pydantic import BaseModel

from hatchet.client import hatchet

REPO_ROOT = Path(__file__).resolve().parents[2]
P001_PROJECT_DIR = REPO_ROOT / "projects" / "P001-gop-baselines"
CANONICAL_SWEEP_YAML = (
    P001_PROJECT_DIR
    / "experiments"
    / "sweeps"
    / "final"
    / "phase1_xlsr_a3_gopt.yaml"
)


class P001XLSR53Input(BaseModel):
    label: str = "xlsr53-phase1-a3-gopt"
    project_id: str = "P001"
    backend: str = "xlsr-espeak"
    sweep_yaml: str = str(CANONICAL_SWEEP_YAML.relative_to(REPO_ROOT))
    wandb_entity: str = "peacockery"
    wandb_project: str = "peacock-asr-p001-gop-baselines"
    agent_count: int = 5
    split: str = "both"
    device: str = "cuda"
    skip_prewarm: bool = True


def default_p001_xlsr53_input() -> P001XLSR53Input:
    return P001XLSR53Input()


def parse_sweep_id(output: str) -> str:
    patterns = [
        re.compile(r"Created sweep with ID:\s*([a-z0-9]+)", re.IGNORECASE),
        re.compile(r"wandb agent\s+\S+/(\S+)/([a-z0-9]+)", re.IGNORECASE),
        re.compile(r"/sweeps/([a-z0-9]+)", re.IGNORECASE),
    ]
    for pattern in patterns:
        match = pattern.search(output)
        if match:
            return match.group(match.lastindex or 1)
    msg = f"Could not parse sweep id from wandb output:\n{output}"
    raise RuntimeError(msg)


def _resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _run_checked(
    cmd: list[str],
    *,
    cwd: Path,
    ctx: Context,
) -> subprocess.CompletedProcess[str]:
    rendered_cmd = " ".join(cmd)
    ctx.log(f"Running: {rendered_cmd}")
    if cmd and cmd[0] == "uv":
        resolved_uv = shutil.which("uv")
        if resolved_uv is None:
            msg = "Could not resolve 'uv' on PATH."
            raise RuntimeError(msg)
        cmd = [resolved_uv, *cmd[1:]]
    completed = subprocess.run(  # noqa: S603
        cmd,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    )
    if completed.stdout:
        ctx.log(completed.stdout.strip())
    if completed.stderr:
        ctx.log(completed.stderr.strip())
    return completed


p001_xlsr53_workflow = hatchet.workflow(
    name="p001-xlsr53-phase1-a3-gopt",
    input_validator=P001XLSR53Input,
)


@p001_xlsr53_workflow.task(
    execution_timeout=timedelta(hours=4),
)
def prewarm_xlsr53(
    input: P001XLSR53Input,
    ctx: Context,
) -> dict[str, Any]:
    if input.skip_prewarm:
        ctx.log("Skipping prewarm step for xlsr53 workflow.")
        return {"skipped": True}

    cmd = [
        "uv",
        "run",
        "--project",
        "projects/P001-gop-baselines",
        "python",
        "-m",
        "p001_gop.cli",
        "prewarm-k2",
        "--backend",
        input.backend,
        "--split",
        input.split,
        "--device",
        input.device,
    ]
    _run_checked(cmd, cwd=REPO_ROOT, ctx=ctx)
    return {
        "skipped": False,
        "backend": input.backend,
        "split": input.split,
        "device": input.device,
        "command": cmd,
    }


@p001_xlsr53_workflow.task(
    execution_timeout=timedelta(minutes=10),
)
def create_xlsr53_sweep(
    input: P001XLSR53Input,
    ctx: Context,
) -> dict[str, Any]:
    sweep_yaml = _resolve_repo_path(input.sweep_yaml)
    cmd = [
        "uv",
        "run",
        "--project",
        "projects/P001-gop-baselines",
        "wandb",
        "sweep",
        str(sweep_yaml.relative_to(REPO_ROOT)),
    ]
    completed = _run_checked(cmd, cwd=REPO_ROOT, ctx=ctx)
    output = "\n".join(
        part for part in [completed.stdout.strip(), completed.stderr.strip()] if part
    )
    sweep_id = parse_sweep_id(output)
    sweep_path = f"{input.wandb_entity}/{input.wandb_project}/{sweep_id}"
    return {
        "sweep_id": sweep_id,
        "sweep_path": sweep_path,
        "sweep_url": (
            f"https://wandb.ai/{input.wandb_entity}/{input.wandb_project}"
            f"/sweeps/{sweep_id}"
        ),
        "sweep_yaml": str(sweep_yaml.relative_to(REPO_ROOT)),
    }


@p001_xlsr53_workflow.task(
    execution_timeout=timedelta(days=3),
    parents=[create_xlsr53_sweep],
)
def run_xlsr53_agent(
    input: P001XLSR53Input,
    ctx: Context,
) -> dict[str, Any]:
    sweep = ctx.task_output(create_xlsr53_sweep)
    cmd = [
        "uv",
        "run",
        "--project",
        "projects/P001-gop-baselines",
        "wandb",
        "agent",
        "--count",
        str(input.agent_count),
        sweep["sweep_path"],
    ]
    _run_checked(cmd, cwd=REPO_ROOT, ctx=ctx)
    return {
        "status": "completed",
        "agent_count": input.agent_count,
        "sweep_id": sweep["sweep_id"],
        "sweep_path": sweep["sweep_path"],
        "sweep_url": sweep["sweep_url"],
        "label": input.label,
    }
