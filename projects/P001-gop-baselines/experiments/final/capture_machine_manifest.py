#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = []
# ///

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[4]
FINAL_ROOT = PROJECT_ROOT / "projects/P001-gop-baselines/experiments/final"
MANIFESTS_DIR = FINAL_ROOT / "manifests"
ALPHA_SWEEPS_DIR = FINAL_ROOT / "alpha_sweeps"
BATCHES_DIR = FINAL_ROOT / "batches"
CHECKPOINTS_DIR = FINAL_ROOT / "checkpoints"
NVIDIA_SMI_FIELDS = 7

PUBLIC_ENV_KEYS = [
    "CUDA_VISIBLE_DEVICES",
    "HF_HOME",
    "HF_DATASETS_CACHE",
    "TRANSFORMERS_CACHE",
    "WANDB_MODE",
    "PEACOCK_WANDB_PROJECT",
    "PEACOCK_WANDB_ENTITY",
    "PEACOCK_WANDB_GROUP",
    "PEACOCK_WANDB_PHASE",
    "PEACOCK_WANDB_JOB_ID",
    "PEACOCK_WANDB_RUN_PREFIX",
    "RUNPOD_POD_ID",
    "RUNPOD_CONTAINER_ID",
    "RUNPOD_TEMPLATE_ID",
    "RUNPOD_GPU_COUNT",
    "RUNPOD_PUBLIC_IP",
    "RUNPOD_TCP_PORT_22",
    "PEACOCK_RUNPOD_POD_ID",
    "PEACOCK_RUNPOD_CONTAINER_ID",
    "PEACOCK_RUNPOD_TEMPLATE_ID",
    "PEACOCK_RUNPOD_TEMPLATE_NAME",
    "PEACOCK_RUNPOD_GPU_TYPE",
    "PEACOCK_RUNPOD_GPU_COUNT",
    "PEACOCK_RUNPOD_HOURLY_COST",
    "PEACOCK_RUNPOD_VOLUME_GB",
    "PEACOCK_RUNPOD_IMAGE_NAME",
    "PEACOCK_RUNPOD_IMAGE_DIGEST",
    "PEACOCK_RUNPOD_MACHINE_ID",
]


@dataclass(frozen=True)
class ManifestPaths:
    manifests_dir: Path
    output_root: Path
    batches_dir: Path
    alpha_sweeps_dir: Path
    checkpoints_dir: Path


def utc_now() -> datetime:
    return datetime.now(tz=UTC)


def iso_now() -> str:
    return utc_now().isoformat()


def slug_now() -> str:
    return utc_now().strftime("%Y%m%dT%H%M%SZ")


def resolve_executable(name: str) -> str | None:
    return shutil.which(name)


def run_command(
    command: list[str],
    *,
    cwd: Path = PROJECT_ROOT,
) -> subprocess.CompletedProcess[str] | None:
    executable = resolve_executable(command[0])
    if executable is None:
        return None
    return subprocess.run(  # noqa: S603
        [executable, *command[1:]],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )


def run_stdout(command: list[str], *, cwd: Path = PROJECT_ROOT) -> str | None:
    result = run_command(command, cwd=cwd)
    if result is None or result.returncode != 0:
        return None
    text = result.stdout.strip()
    return text or None


def run_json(
    command: list[str],
    *,
    cwd: Path = PROJECT_ROOT,
) -> dict[str, Any] | list[Any] | None:
    raw = run_stdout(command, cwd=cwd)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cpu_model() -> str | None:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                _, _, value = line.partition(":")
                return value.strip() or None
    processor = platform.processor().strip()
    return processor or None


def total_memory_bytes() -> int | None:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None
    return int(page_size) * int(pages)


def lscpu_info() -> dict[str, Any]:
    raw = run_stdout(["lscpu", "--json"])
    if raw is None:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    rows = payload.get("lscpu", [])
    info: dict[str, Any] = {}
    for row in rows:
        field = row.get("field", "").strip().rstrip(":")
        data = row.get("data")
        if field:
            info[field] = data
    return info


def nvidia_gpu_info() -> list[dict[str, Any]]:
    raw = run_stdout(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu,driver_version,uuid",
            "--format=csv,noheader,nounits",
        ],
    )
    if raw is None:
        return []
    gpus: list[dict[str, Any]] = []
    for line in raw.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != NVIDIA_SMI_FIELDS:
            continue
        name, total, used, util, temp, driver, uuid = parts
        gpus.append(
            {
                "name": name,
                "memory_total_mb": int(total),
                "memory_used_mb": int(used),
                "utilization_gpu_pct": int(util),
                "temperature_c": int(temp),
                "driver_version": driver,
                "uuid": uuid,
            },
        )
    return gpus


def git_sha() -> str | None:
    return run_stdout(["git", "rev-parse", "HEAD"])


def selected_env() -> dict[str, str]:
    values: dict[str, str] = {}
    for key in PUBLIC_ENV_KEYS:
        value = os.environ.get(key)
        if value:
            values[key] = value
    return values


def resolve_path(path: Path | None, default: Path) -> Path:
    if path is None:
        return default
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def provider_name(args: argparse.Namespace) -> str:
    if args.provider != "auto":
        return args.provider
    if any(key.startswith("RUNPOD_") for key in os.environ) or Path("/runpod").exists():
        return "runpod"
    return "local"


def filtered_runpod_env() -> dict[str, str]:
    return {
        key: value
        for key, value in selected_env().items()
        if key.startswith(("RUNPOD_", "PEACOCK_RUNPOD_"))
    }


def resolve_runpod_pod_id(args: argparse.Namespace) -> str | None:
    if args.pod_id:
        return args.pod_id
    if os.environ.get("PEACOCK_RUNPOD_POD_ID"):
        return os.environ["PEACOCK_RUNPOD_POD_ID"]
    return os.environ.get("RUNPOD_POD_ID")


def env_fallback(*keys: str) -> str | None:
    for key in keys:
        value = os.environ.get(key)
        if value:
            return value
    return None


def runpod_control_plane(args: argparse.Namespace) -> dict[str, Any]:
    pod_id = resolve_runpod_pod_id(args)
    control: dict[str, Any] = {
        "pod_id": pod_id,
        "env": filtered_runpod_env(),
    }
    if pod_id is None:
        return control

    pod_get = run_json(
        [
            "runpodctl",
            "pod",
            "get",
            pod_id,
            "--include-machine",
            "--include-network-volume",
        ],
    )
    if isinstance(pod_get, dict):
        control["control_plane"] = {
            key: pod_get.get(key)
            for key in [
                "id",
                "name",
                "desiredStatus",
                "imageName",
                "templateId",
                "gpuCount",
                "gpuTypeId",
                "costPerHr",
                "volumeInGb",
                "machineId",
                "machine",
                "networkVolume",
            ]
            if key in pod_get
        }

    ssh_info = run_json(["runpodctl", "ssh", "info", pod_id, "--verbose"])
    if isinstance(ssh_info, dict):
        control["ssh_info"] = {
            key: ssh_info.get(key)
            for key in ["pod_id", "pod_name", "ip", "port", "command"]
            if key in ssh_info
        }

    overrides = {
        "container_id": (
            args.runpod_container_id
            or env_fallback("PEACOCK_RUNPOD_CONTAINER_ID", "RUNPOD_CONTAINER_ID")
        ),
        "template_id": (
            args.runpod_template_id
            or env_fallback("PEACOCK_RUNPOD_TEMPLATE_ID", "RUNPOD_TEMPLATE_ID")
        ),
        "template_name": (
            args.runpod_template_name
            or env_fallback("PEACOCK_RUNPOD_TEMPLATE_NAME")
        ),
        "image_name": (
            args.runpod_image_name
            or env_fallback("PEACOCK_RUNPOD_IMAGE_NAME")
        ),
        "image_digest": (
            args.runpod_image_digest
            or env_fallback("PEACOCK_RUNPOD_IMAGE_DIGEST")
        ),
        "gpu_type": (
            args.runpod_gpu_type
            or env_fallback("PEACOCK_RUNPOD_GPU_TYPE")
        ),
        "gpu_count": (
            args.runpod_gpu_count
            or env_fallback("PEACOCK_RUNPOD_GPU_COUNT", "RUNPOD_GPU_COUNT")
        ),
        "hourly_cost_usd": (
            args.runpod_hourly_cost
            or env_fallback("PEACOCK_RUNPOD_HOURLY_COST")
        ),
        "volume_gb": (
            args.runpod_volume_gb
            or env_fallback("PEACOCK_RUNPOD_VOLUME_GB")
        ),
        "machine_id": (
            args.runpod_machine_id
            or env_fallback("PEACOCK_RUNPOD_MACHINE_ID")
        ),
        "notes": args.runpod_note,
    }
    explicit_overrides = {
        key: value
        for key, value in overrides.items()
        if value is not None
    }
    if explicit_overrides:
        control["overrides"] = explicit_overrides

    return control


def project_probe() -> dict[str, Any]:
    uv_bin = resolve_executable("uv")
    if uv_bin is None:
        return {}
    code = """
import importlib.metadata as md
import json
import platform
import sys

import torch

from p001_gop.dataset import DATASET_REVISION
from p001_gop.settings import settings

packages = [
    "torch",
    "transformers",
    "wandb",
    "numpy",
    "scipy",
    "pydantic",
    "pydantic-settings",
    "huggingface-hub",
    "accelerate",
]
versions = {}
for package in packages:
    try:
        versions[package] = md.version(package)
    except md.PackageNotFoundError:
        versions[package] = None

payload = {
    "python_version": platform.python_version(),
    "python_executable": sys.executable,
    "dataset_revision": DATASET_REVISION,
    "settings": {
        "cache_dir": str(settings.cache_dir),
        "checkpoints_dir": str(settings.checkpoints_dir),
        "wandb_project": settings.wandb_project,
        "wandb_entity": settings.wandb_entity,
        "wandb_track": settings.wandb_track,
        "wandb_project_id": settings.wandb_project_id,
    },
    "versions": versions,
    "torch": {
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
    },
}
print(json.dumps(payload))
""".strip()
    result = subprocess.run(  # noqa: S603
        [uv_bin, "run", "python", "-c", code],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return {}
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {}


def build_host_manifest(*, args: argparse.Namespace) -> dict[str, Any]:
    uname = platform.uname()
    lscpu = lscpu_info()
    disk = shutil.disk_usage(PROJECT_ROOT)
    probe = project_probe()
    return {
        "captured_at": iso_now(),
        "source": {
            "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "repo_root": str(PROJECT_ROOT),
            "git_sha": git_sha(),
        },
        "provider": provider_name(args),
        "host": {
            "hostname": socket.gethostname(),
            "platform": uname.system,
            "platform_release": uname.release,
            "platform_version": uname.version,
            "machine": uname.machine,
            "processor": uname.processor or None,
        },
        "hardware": {
            "cpu_model": cpu_model(),
            "logical_cpu_count": os.cpu_count(),
            "socket_count": lscpu.get("Socket(s)"),
            "cores_per_socket": lscpu.get("Core(s) per socket"),
            "threads_per_core": lscpu.get("Thread(s) per core"),
            "memory_total_bytes": total_memory_bytes(),
            "disk_total_bytes": disk.total,
            "disk_used_bytes": disk.used,
            "disk_free_bytes": disk.free,
            "gpus": nvidia_gpu_info(),
        },
        "software": {
            "python_version": probe.get("python_version") or platform.python_version(),
            "python_executable": probe.get("python_executable") or sys.executable,
            "uv_version": run_stdout(["uv", "--version"]),
            "dependency_versions": probe.get("versions", {}),
            "torch": probe.get("torch", {}),
            "uv_lock_sha256": sha256_file(PROJECT_ROOT / "uv.lock"),
            "pyproject_sha256": sha256_file(PROJECT_ROOT / "pyproject.toml"),
        },
        "runpod": runpod_control_plane(args),
    }


def build_run_context(
    *,
    args: argparse.Namespace,
    host_manifest_path: Path,
    paths: ManifestPaths,
) -> dict[str, Any]:
    probe = project_probe()
    settings_info = probe.get("settings", {})
    return {
        "captured_at": iso_now(),
        "project": {
            "project_id": args.project_id,
            "project_slug": args.project_slug,
            "campaign_label": args.campaign,
            "phase": args.phase,
        },
        "paths": {
            "repo_root": str(PROJECT_ROOT),
            "final_root": str(paths.output_root),
            "manifests_dir": str(paths.manifests_dir),
            "cache_dir": settings_info.get("cache_dir"),
            "settings_checkpoints_dir": settings_info.get("checkpoints_dir"),
            "campaign_checkpoints_dir": str(paths.checkpoints_dir),
            "campaign_batches_dir": str(paths.batches_dir),
            "campaign_alpha_sweeps_dir": str(paths.alpha_sweeps_dir),
            "output_root": str(paths.output_root),
        },
        "wandb": {
            "entity": settings_info.get("wandb_entity"),
            "project": settings_info.get("wandb_project"),
            "track": settings_info.get("wandb_track"),
            "group_prefix": args.wandb_group_prefix,
        },
        "dataset": {
            "name": args.dataset_name,
            "revision": probe.get("dataset_revision"),
        },
        "launcher": {
            "entrypoint": args.launcher,
            "notes": args.note,
        },
        "environment": {
            "selected_env": selected_env(),
        },
        "remote_execution": {
            "provider": provider_name(args),
            "runpod": runpod_control_plane(args),
        },
        "host_manifest": {
            "path": display_path(host_manifest_path),
            "sha256": sha256_file(host_manifest_path),
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture project-local machine and run-context manifests.",
    )
    parser.add_argument(
        "--project-id",
        default="P001",
        help="Project identifier stored in the run-context manifest.",
    )
    parser.add_argument(
        "--project-slug",
        default="gop-baselines",
        help="Project slug stored in the run-context manifest.",
    )
    parser.add_argument(
        "--campaign",
        default="paper-close",
        help="Campaign label to store in the run-context manifest.",
    )
    parser.add_argument(
        "--phase",
        default="all",
        help="Phase label for the run-context manifest.",
    )
    parser.add_argument(
        "--dataset-name",
        default="SpeechOcean762",
        help="Dataset name stored in the run-context manifest.",
    )
    parser.add_argument(
        "--wandb-group-prefix",
        default="p001-paper-close",
        help="Canonical W&B group prefix for this campaign.",
    )
    parser.add_argument(
        "--launcher",
        default="projects/P001-gop-baselines/experiments/final/launch_p001_paper_close.py",
        help="Canonical launcher entrypoint for this campaign.",
    )
    parser.add_argument(
        "--note",
        default="Captured during the live P001 paper-close rerun campaign.",
        help="Optional note stored in the run-context manifest.",
    )
    parser.add_argument(
        "--provider",
        choices=["auto", "local", "runpod"],
        default="auto",
        help="Override provider detection when capturing manifests.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Project-local output root for this campaign.",
    )
    parser.add_argument(
        "--manifests-dir",
        type=Path,
        default=None,
        help="Directory where manifest JSON files should be written.",
    )
    parser.add_argument(
        "--batches-dir",
        type=Path,
        default=None,
        help="Canonical batch artifact directory for this campaign.",
    )
    parser.add_argument(
        "--alpha-sweeps-dir",
        type=Path,
        default=None,
        help="Canonical alpha-sweep artifact directory for this campaign.",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=None,
        help="Canonical checkpoint artifact directory for this campaign.",
    )
    parser.add_argument(
        "--pod-id",
        "--runpod-pod-id",
        default=None,
        help="Optional RunPod pod ID for control-plane metadata capture via runpodctl.",
    )
    parser.add_argument(
        "--runpod-container-id",
        default=None,
        help="Optional RunPod container ID override stored in the RunPod block.",
    )
    parser.add_argument(
        "--runpod-template-id",
        default=None,
        help="Optional template ID override stored in the RunPod manifest block.",
    )
    parser.add_argument(
        "--runpod-template-name",
        default=None,
        help="Optional template name override stored in the RunPod manifest block.",
    )
    parser.add_argument(
        "--runpod-image-name",
        default=None,
        help="Optional image name override stored in the RunPod manifest block.",
    )
    parser.add_argument(
        "--runpod-image-digest",
        default=None,
        help="Optional image digest override stored in the RunPod manifest block.",
    )
    parser.add_argument(
        "--runpod-gpu-type",
        default=None,
        help="Optional GPU type override stored in the RunPod manifest block.",
    )
    parser.add_argument(
        "--runpod-gpu-count",
        type=int,
        default=None,
        help="Optional GPU count override stored in the RunPod manifest block.",
    )
    parser.add_argument(
        "--runpod-hourly-cost",
        type=float,
        default=None,
        help="Optional hourly USD cost override stored in the RunPod manifest block.",
    )
    parser.add_argument(
        "--runpod-volume-gb",
        type=int,
        default=None,
        help="Optional volume size override stored in the RunPod manifest block.",
    )
    parser.add_argument(
        "--runpod-machine-id",
        default=None,
        help="Optional machine ID override stored in the RunPod manifest block.",
    )
    parser.add_argument(
        "--runpod-note",
        default=None,
        help="Optional freeform note stored in the RunPod manifest block.",
    )
    return parser.parse_args()


def resolve_paths_from_args(args: argparse.Namespace) -> ManifestPaths:
    output_root = resolve_path(args.output_root, FINAL_ROOT)
    manifests_dir = resolve_path(args.manifests_dir, output_root / "manifests")
    batches_dir = resolve_path(args.batches_dir, output_root / "batches")
    alpha_sweeps_dir = resolve_path(
        args.alpha_sweeps_dir, output_root / "alpha_sweeps",
    )
    checkpoints_dir = resolve_path(
        args.checkpoints_dir, output_root / "checkpoints",
    )
    return ManifestPaths(
        manifests_dir=manifests_dir,
        output_root=output_root,
        batches_dir=batches_dir,
        alpha_sweeps_dir=alpha_sweeps_dir,
        checkpoints_dir=checkpoints_dir,
    )


def main() -> None:
    args = parse_args()
    paths = resolve_paths_from_args(args)
    paths.manifests_dir.mkdir(parents=True, exist_ok=True)
    stamp = slug_now()

    host_path = paths.manifests_dir / f"{stamp}_{args.campaign}_host_manifest.json"
    host_manifest = build_host_manifest(args=args)
    write_json(host_path, host_manifest)

    context_path = paths.manifests_dir / f"{stamp}_{args.campaign}_run_context.json"
    run_context = build_run_context(
        args=args,
        host_manifest_path=host_path,
        paths=paths,
    )
    write_json(context_path, run_context)

    write_json(paths.manifests_dir / "latest_host_manifest.json", host_manifest)
    write_json(paths.manifests_dir / "latest_run_context.json", run_context)

    print(display_path(host_path))  # noqa: T201
    print(display_path(context_path))  # noqa: T201


if __name__ == "__main__":
    main()
