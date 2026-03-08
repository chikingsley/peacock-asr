#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = []
# ///

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
UV_PROJECT = "projects/P001-gop-baselines"
FINAL_ROOT = PROJECT_ROOT / "projects/P001-gop-baselines/experiments/final"
SWEEPS_DIR = PROJECT_ROOT / "projects/P001-gop-baselines/experiments/sweeps/final"
AGENTS_DIR = FINAL_ROOT / "agents"
ALPHA_DIR = FINAL_ROOT / "alpha_sweeps"
BATCH_DIR = FINAL_ROOT / "batches"
CHECKPOINTS_DIR = FINAL_ROOT / "checkpoints"
MASTER_LOG = AGENTS_DIR / "launch_p001_paper_close_py.log"


@dataclass(frozen=True)
class SweepJob:
    name: str
    path: Path


PHASE1_SWEEPS = [
    SweepJob(
        "phase1_original_a1_scalar",
        SWEEPS_DIR / "phase1_original_a1_scalar.yaml",
    ),
    SweepJob(
        "phase1_original_a2_feats",
        SWEEPS_DIR / "phase1_original_a2_feats.yaml",
    ),
    SweepJob("phase1_original_a3_gopt", SWEEPS_DIR / "phase1_original_a3_gopt.yaml"),
    SweepJob("phase1_xlsr_a1_scalar", SWEEPS_DIR / "phase1_xlsr_a1_scalar.yaml"),
    SweepJob("phase1_xlsr_a2_feats", SWEEPS_DIR / "phase1_xlsr_a2_feats.yaml"),
    SweepJob("phase1_xlsr_a3_gopt", SWEEPS_DIR / "phase1_xlsr_a3_gopt.yaml"),
]

PHASE2_SWEEPS = [
    SweepJob("phase2_original_b1_gopsf", SWEEPS_DIR / "phase2_original_b1_gopsf.yaml"),
    SweepJob(
        "phase2_original_b2_logit_margin",
        SWEEPS_DIR / "phase2_original_b2_logit_margin.yaml",
    ),
    SweepJob(
        "phase2_original_b3_logit_combined_a025",
        SWEEPS_DIR / "phase2_original_b3_logit_combined_a025.yaml",
    ),
    SweepJob(
        "phase2_original_b4_logit_combined_a050",
        SWEEPS_DIR / "phase2_original_b4_logit_combined_a050.yaml",
    ),
    SweepJob(
        "phase2_original_b5_logit_combined_a075",
        SWEEPS_DIR / "phase2_original_b5_logit_combined_a075.yaml",
    ),
    SweepJob("phase2_xlsr_b1_gopsf", SWEEPS_DIR / "phase2_xlsr_b1_gopsf.yaml"),
    SweepJob(
        "phase2_xlsr_b2_logit_margin",
        SWEEPS_DIR / "phase2_xlsr_b2_logit_margin.yaml",
    ),
    SweepJob(
        "phase2_xlsr_b3_logit_combined_a025",
        SWEEPS_DIR / "phase2_xlsr_b3_logit_combined_a025.yaml",
    ),
    SweepJob(
        "phase2_xlsr_b4_logit_combined_a050",
        SWEEPS_DIR / "phase2_xlsr_b4_logit_combined_a050.yaml",
    ),
    SweepJob(
        "phase2_xlsr_b5_logit_combined_a075",
        SWEEPS_DIR / "phase2_xlsr_b5_logit_combined_a075.yaml",
    ),
]

ALL_SWEEPS = [*PHASE1_SWEEPS, *PHASE2_SWEEPS]


def utc_now() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(message: str) -> None:
    line = f"[{utc_now()}] {message}"
    sys.stdout.write(f"{line}\n")
    sys.stdout.flush()
    MASTER_LOG.parent.mkdir(parents=True, exist_ok=True)
    with MASTER_LOG.open("a", encoding="utf-8") as file:
        file.write(f"{line}\n")


def run_and_tee(
    command: list[str],
    *,
    log_path: Path,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(  # noqa: S603
        command,
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    captured: list[str] = []
    assert proc.stdout is not None
    with log_path.open("w", encoding="utf-8") as file:
        for line in proc.stdout:
            captured.append(line)
            sys.stdout.write(line)
            sys.stdout.flush()
            file.write(line)
        return_code = proc.wait()
    stdout = "".join(captured)
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command, output=stdout)
    return subprocess.CompletedProcess(command, return_code, stdout=stdout)


def extract_sweep_ref(stdout: str) -> str:
    match = re.findall(
        r"peacockery/peacock-asr-p001-gop-baselines/[A-Za-z0-9]+",
        stdout,
    )
    if not match:
        raise RuntimeError("Failed to parse W&B sweep reference from sweep output.")
    return match[-1]


def run_sweep(job: SweepJob) -> None:
    create_log = AGENTS_DIR / f"{job.name}_create.log"
    agent_log = AGENTS_DIR / f"{job.name}.log"

    log(f"Creating sweep {job.name}")
    created = run_and_tee(
        ["uv", "run", "--project", UV_PROJECT, "wandb", "sweep", str(job.path)],
        log_path=create_log,
    )
    sweep_ref = extract_sweep_ref(created.stdout)

    log(f"Running agent {sweep_ref}")
    run_and_tee(
        ["uv", "run", "--project", UV_PROJECT, "wandb", "agent", sweep_ref],
        log_path=agent_log,
    )
    log(f"Completed sweep {job.name}")


def alpha_env(*, backend_tag: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "HF_CHECKPOINT_UPLOAD": "false",
            "PEACOCK_WANDB_PROJECT": "peacock-asr-p001-gop-baselines",
            "PEACOCK_WANDB_ENTITY": "peacockery",
            "PEACOCK_WANDB_GROUP": (
                f"p001-paper-close-phase2b-{backend_tag}-alpha-sweep"
            ),
            "PEACOCK_WANDB_TRACK": "track05",
            "PEACOCK_WANDB_PROJECT_ID": "P001",
            "PEACOCK_WANDB_PHASE": "phase2b",
            "PEACOCK_WANDB_JOB_ID": "alpha-sweep",
            "PEACOCK_WANDB_RUN_PREFIX": "p001-paper-close",
            "PEACOCK_WANDB_JOB_TYPE": "analysis",
            "PEACOCK_WANDB_TAGS": f"p001,paper-close,phase2b,{backend_tag},alpha-sweep",
        },
    )
    return env


def run_alpha_sweep(*, backend: str, backend_tag: str) -> None:
    log(f"Running alpha sweep for {backend}")
    log_path = AGENTS_DIR / f"phase2b_{backend_tag}_alpha_sweep.log"
    run_and_tee(
        [
            "uv",
            "run",
            "--project",
            UV_PROJECT,
            "peacock-asr",
            "sweep-alpha",
            "--backend",
            backend,
            "--alpha-start",
            "0.0",
            "--alpha-stop",
            "1.0",
            "--alpha-step",
            "0.05",
            "--output-dir",
            str(ALPHA_DIR),
        ],
        log_path=log_path,
        env=alpha_env(backend_tag=backend_tag),
    )
    log(f"Completed alpha sweep for {backend}")


def ensure_dirs() -> None:
    for path in [AGENTS_DIR, ALPHA_DIR, BATCH_DIR, CHECKPOINTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def iter_sweeps(phase: str) -> list[SweepJob]:
    if phase == "phase1":
        return PHASE1_SWEEPS
    if phase == "phase2":
        return PHASE2_SWEEPS
    if phase == "all":
        return ALL_SWEEPS
    return []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch the P001 paper-close W&B campaign.",
    )
    parser.add_argument(
        "--phase",
        choices=["all", "phase1", "phase2", "phase2b"],
        default="all",
        help="Which part of the campaign to run.",
    )
    parser.add_argument(
        "--start-at",
        default=None,
        help="Optional sweep name to resume from within phase1/phase2/all.",
    )
    args = parser.parse_args()

    ensure_dirs()
    os.environ["HF_CHECKPOINT_UPLOAD"] = "false"
    os.environ["PEACOCK_WANDB_PROJECT"] = "peacock-asr-p001-gop-baselines"
    os.environ["PEACOCK_WANDB_ENTITY"] = "peacockery"

    log(f"Starting P001 paper-close campaign phase={args.phase}")

    if args.phase in {"all", "phase1", "phase2"}:
        sweeps = iter_sweeps(args.phase)
        if args.start_at:
            names = [job.name for job in sweeps]
            if args.start_at not in names:
                raise ValueError(f"--start-at must be one of: {', '.join(names)}")
            start_index = names.index(args.start_at)
            sweeps = sweeps[start_index:]
        for job in sweeps:
            run_sweep(job)

    if args.phase in {"all", "phase2b"}:
        run_alpha_sweep(backend="original", backend_tag="original")
        run_alpha_sweep(backend="xlsr-espeak", backend_tag="xlsr")

    log("P001 paper-close campaign complete")


if __name__ == "__main__":
    main()
