"""Hatchet workflow for the first P003 compact-backbones orchestration pass."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import perf_counter
from typing import Any

from hatchet_sdk import Context
from pydantic import BaseModel

from hatchet.client import hatchet

REPO_ROOT = Path(__file__).resolve().parents[2]
P003_EXPERIMENTS_DIR = REPO_ROOT / "projects" / "P003-compact-backbones" / "experiments"
CHECKPOINT_ROOT = P003_EXPERIMENTS_DIR / "checkpoints" / "hatchet"
RESULTS_ROOT = P003_EXPERIMENTS_DIR / "hatchet-results"
RUNNER_SCRIPT = REPO_ROOT / "hatchet" / "scripts" / "run_p001_eval.py"
JSON_RESULT_MARKER = "JSON_RESULT::"
WANDB_URL_PATTERN = re.compile(r"https://wandb\.ai/\S+/runs/[A-Za-z0-9]+")


class P003CompactBackbonesInput(BaseModel):
    label: str = "xlsr53-compare"
    project_id: str = "P003 Compact Backbones"
    backend: str = "xlsr-espeak"
    seed: int = 501
    device: str = "cuda"
    workers: int = 0
    split: str = "both"
    limit: int = 0
    prewarm_limit: int = 0
    no_cache: bool = True
    score_variant: str = "gop_sf"
    score_alpha: float = 0.5
    scalar_device: str = "cuda"
    wandb_entity: str = "peacockery"
    wandb_project: str = "peacock-asr-p001-gop-baselines"
    wandb_group: str = "p003-compact-backbones-xlsr53-compare"
    wandb_phase: str = "baseline"
    run_prefix: str = "XLSR-53"
    skip_prewarm: bool = True
    reuse_existing: bool = True


def default_p003_compact_backbones_input() -> P003CompactBackbonesInput:
    return P003CompactBackbonesInput()


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-")
    return slug or "run"


def _variant_roots(input: P003CompactBackbonesInput, variant: str) -> tuple[Path, Path]:
    label_slug = _slug(input.label)
    results_dir = RESULTS_ROOT / label_slug
    checkpoints_dir = CHECKPOINT_ROOT / label_slug / variant
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    return results_dir, checkpoints_dir


def _wandb_env(
    input: P003CompactBackbonesInput,
    *,
    variant: str,
    scalar_backend: str,
    scalar_device: str,
    checkpoints_dir: Path,
) -> dict[str, str]:
    run_prefix = input.run_prefix if variant == "python" else f"{input.run_prefix} k2"
    tags = [
        "p003",
        "compact-backbones",
        "xlsr53",
        "gopt",
        variant,
        scalar_backend,
    ]
    return {
        "PEACOCK_WANDB_ENTITY": input.wandb_entity,
        "PEACOCK_WANDB_PROJECT": input.wandb_project,
        "PEACOCK_WANDB_PROJECT_ID": input.project_id,
        "PEACOCK_WANDB_GROUP": input.wandb_group,
        "PEACOCK_WANDB_PHASE": input.wandb_phase,
        "PEACOCK_WANDB_JOB_ID": variant,
        "PEACOCK_WANDB_RUN_PREFIX": run_prefix,
        "PEACOCK_WANDB_JOB_TYPE": "eval",
        "PEACOCK_WANDB_TAGS": ",".join(tags),
        "PEACOCK_CHECKPOINTS_DIR": str(checkpoints_dir),
        "PEACOCK_CTC_SCALAR_BACKEND": scalar_backend,
        "PEACOCK_CTC_SCALAR_DEVICE": scalar_device,
    }


def _extract_json_result(output: str) -> dict[str, Any]:
    for line in reversed(output.splitlines()):
        if line.startswith(JSON_RESULT_MARKER):
            return json.loads(line.removeprefix(JSON_RESULT_MARKER).strip())
    msg = f"Could not locate JSON result marker in output:\n{output}"
    raise RuntimeError(msg)


def _extract_wandb_url(output: str) -> str | None:
    match = WANDB_URL_PATTERN.search(output)
    if match:
        return match.group(0)
    return None


def _load_eval_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = f"Expected eval payload dict in {path}, got {type(payload)!r}"
        raise RuntimeError(msg)
    required = {"elapsed_s", "eval_name", "metrics"}
    missing = sorted(required.difference(payload))
    if missing:
        msg = f"Eval payload {path} is missing keys: {missing}"
        raise RuntimeError(msg)
    return payload


def _run_logged(
    cmd: list[str],
    *,
    cwd: Path,
    ctx: Context,
    env: dict[str, str] | None = None,
) -> str:
    rendered_cmd = " ".join(cmd)
    ctx.log(f"Running: {rendered_cmd}")
    resolved_cmd = list(cmd)
    if resolved_cmd and resolved_cmd[0] == "uv":
        resolved_uv = shutil.which("uv")
        if resolved_uv is None:
            msg = "Could not resolve 'uv' on PATH."
            raise RuntimeError(msg)
        resolved_cmd[0] = resolved_uv
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    process = subprocess.Popen(  # noqa: S603
        resolved_cmd,
        cwd=cwd,
        env=merged_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output_lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        stripped = line.rstrip()
        output_lines.append(stripped)
        if stripped:
            ctx.log(stripped)
    return_code = process.wait()
    output = "\n".join(output_lines)
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, resolved_cmd, output=output)
    return output


def _run_eval_variant(
    input: P003CompactBackbonesInput,
    *,
    variant: str,
    scalar_backend: str,
    scalar_device: str,
    ctx: Context,
) -> dict[str, Any]:
    results_dir, checkpoints_dir = _variant_roots(input, variant)
    output_path = results_dir / f"{variant}.json"
    if input.reuse_existing and output_path.exists():
        ctx.log(f"Reusing existing {variant} result from {output_path}.")
        payload = _load_eval_payload(output_path)
        payload["variant"] = variant
        payload["scalar_backend"] = scalar_backend
        payload["scalar_device"] = scalar_device
        payload["job_id"] = variant
        payload["results_path"] = str(output_path)
        payload["checkpoints_root"] = str(checkpoints_dir)
        payload.setdefault("wandb_url", None)
        payload.setdefault("wall_time_s", payload["elapsed_s"])
        return payload
    env = _wandb_env(
        input,
        variant=variant,
        scalar_backend=scalar_backend,
        scalar_device=scalar_device,
        checkpoints_dir=checkpoints_dir,
    )
    cmd = [
        "uv",
        "run",
        "--project",
        "projects/P001-gop-baselines",
        "python",
        str(RUNNER_SCRIPT.relative_to(REPO_ROOT)),
        "--backend",
        input.backend,
        "--device",
        input.device,
        "--seed",
        str(input.seed),
        "--workers",
        str(input.workers),
        "--limit",
        str(input.limit),
        "--score-variant",
        input.score_variant,
        "--score-alpha",
        str(input.score_alpha),
        "--output",
        str(output_path.relative_to(REPO_ROOT)),
    ]
    if input.no_cache:
        cmd.append("--no-cache")
    start = perf_counter()
    try:
        output = _run_logged(cmd, cwd=REPO_ROOT, ctx=ctx, env=env)
    except subprocess.CalledProcessError as exc:
        if not output_path.exists():
            raise
        ctx.log(
            f"{variant} subprocess exited non-zero after writing {output_path}; "
            "reusing persisted result.",
        )
        output = exc.output or ""
    wall_time_s = perf_counter() - start
    if JSON_RESULT_MARKER in output:
        payload = _extract_json_result(output)
    else:
        payload = _load_eval_payload(output_path)
    payload["variant"] = variant
    payload["scalar_backend"] = scalar_backend
    payload["scalar_device"] = scalar_device
    payload["job_id"] = variant
    payload["results_path"] = str(output_path)
    payload["checkpoints_root"] = str(checkpoints_dir)
    payload["wandb_url"] = _extract_wandb_url(output)
    payload["wall_time_s"] = wall_time_s
    return payload


p003_compact_backbones_workflow = hatchet.workflow(
    name="p003-compact-backbones-xlsr53-compare",
    input_validator=P003CompactBackbonesInput,
)


@p003_compact_backbones_workflow.task(
    execution_timeout=timedelta(hours=8),
)
def run_xlsr53_python_baseline(
    input: P003CompactBackbonesInput,
    ctx: Context,
) -> dict[str, Any]:
    return _run_eval_variant(
        input,
        variant="python",
        scalar_backend="python",
        scalar_device="cpu",
        ctx=ctx,
    )


@p003_compact_backbones_workflow.task(
    execution_timeout=timedelta(hours=8),
    parents=[run_xlsr53_python_baseline],
)
def prewarm_xlsr53_k2(
    input: P003CompactBackbonesInput,
    ctx: Context,
) -> dict[str, Any]:
    if input.skip_prewarm:
        ctx.log("Skipping k2 prewarm before XLSR-53 comparison run.")
        return {
            "prewarm_skipped": True,
            "backend": input.backend,
            "split": input.split,
            "limit": input.prewarm_limit,
            "device": input.device,
            "scalar_device": input.scalar_device,
            "finished_at_utc": datetime.now(tz=UTC).isoformat(),
        }

    cmd = [
        "uv",
        "run",
        "--project",
        "projects/P001-gop-baselines",
        "peacock-asr",
        "prewarm-k2",
        "--backend",
        input.backend,
        "--split",
        input.split,
        "--limit",
        str(input.prewarm_limit),
        "--device",
        input.device,
    ]
    env = {
        "PEACOCK_CTC_SCALAR_BACKEND": "k2",
        "PEACOCK_CTC_SCALAR_DEVICE": input.scalar_device,
    }
    start = perf_counter()
    output = _run_logged(cmd, cwd=REPO_ROOT, ctx=ctx, env=env)
    wall_time_s = perf_counter() - start
    return {
        "prewarm_skipped": False,
        "backend": input.backend,
        "split": input.split,
        "limit": input.prewarm_limit,
        "device": input.device,
        "scalar_device": input.scalar_device,
        "wall_time_s": wall_time_s,
        "finished_at_utc": datetime.now(tz=UTC).isoformat(),
        "wandb_url": _extract_wandb_url(output),
    }


@p003_compact_backbones_workflow.task(
    execution_timeout=timedelta(hours=8),
    parents=[prewarm_xlsr53_k2],
)
def run_xlsr53_k2_baseline(
    input: P003CompactBackbonesInput,
    ctx: Context,
) -> dict[str, Any]:
    return _run_eval_variant(
        input,
        variant="k2",
        scalar_backend="k2",
        scalar_device=input.scalar_device,
        ctx=ctx,
    )


@p003_compact_backbones_workflow.task(
    execution_timeout=timedelta(minutes=5),
    parents=[run_xlsr53_python_baseline, run_xlsr53_k2_baseline],
)
def summarize_xlsr53_compare(
    input: P003CompactBackbonesInput,
    ctx: Context,
) -> dict[str, Any]:
    python_run = ctx.task_output(run_xlsr53_python_baseline)
    k2_run = ctx.task_output(run_xlsr53_k2_baseline)
    python_elapsed = float(python_run["elapsed_s"])
    k2_elapsed = float(k2_run["elapsed_s"])
    speedup = python_elapsed / k2_elapsed if k2_elapsed else None
    summary = {
        "label": input.label,
        "project_id": input.project_id,
        "backend": input.backend,
        "seed": input.seed,
        "python_elapsed_s": python_elapsed,
        "k2_elapsed_s": k2_elapsed,
        "k2_speedup_vs_python": speedup,
        "python_pcc": python_run["metrics"]["pcc"],
        "k2_pcc": k2_run["metrics"]["pcc"],
        "python_mse": python_run["metrics"]["mse"],
        "k2_mse": k2_run["metrics"]["mse"],
        "python_wandb_url": python_run.get("wandb_url"),
        "k2_wandb_url": k2_run.get("wandb_url"),
    }
    speedup_text = f"{speedup:.3f}x" if speedup is not None else "n/a"
    ctx.log(
        "Completed XLSR-53 compare: "
        f"python={python_elapsed:.2f}s, k2={k2_elapsed:.2f}s, "
        f"speedup={speedup_text}",
    )
    return summary
