from __future__ import annotations

import importlib.metadata as metadata
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from p004_training_from_scratch.settings import PROJECT_ROOT, ProjectSettings

DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "validation" / "machine_manifest.json"
PACKAGE_NAMES = (
    "torch",
    "torchcodec",
    "wandb",
    "python-dotenv",
    "vastai-sdk",
    "lhotse",
    "k2",
)


def capture_machine_manifest(*, output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    settings = ProjectSettings.from_env()
    payload = {
        "captured_at": datetime.now(tz=UTC).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "git_sha": _run_stdout(["git", "rev-parse", "HEAD"]),
        "host": _host_info(),
        "hardware": _hardware_info(),
        "software": _software_info(),
        "tracking": {
            "wandb_enabled": settings.wandb_enabled,
            "wandb_entity": settings.wandb_entity,
            "wandb_project": settings.wandb_project,
            "wandb_mode": settings.wandb_mode,
        },
        "public_env": settings.public_env(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        f"{json.dumps(payload, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    return payload


def _host_info() -> dict[str, Any]:
    uname = platform.uname()
    return {
        "hostname": socket.gethostname(),
        "platform": uname.system,
        "platform_release": uname.release,
        "platform_version": uname.version,
        "machine": uname.machine,
        "processor": uname.processor or None,
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
    }


def _hardware_info() -> dict[str, Any]:
    disk = shutil.disk_usage(PROJECT_ROOT)
    return {
        "logical_cpu_count": os.cpu_count(),
        "memory_total_bytes": _total_memory_bytes(),
        "disk_total_bytes": disk.total,
        "disk_used_bytes": disk.used,
        "disk_free_bytes": disk.free,
        "gpus": _nvidia_gpu_info(),
    }


def _software_info() -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for package_name in PACKAGE_NAMES:
        try:
            versions[package_name] = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            versions[package_name] = None

    return {
        "uv_version": _run_stdout(["uv", "--version"]),
        "dependency_versions": versions,
        "pyproject_sha256": _sha256_file(PROJECT_ROOT / "pyproject.toml"),
        "uv_lock_sha256": _sha256_file(PROJECT_ROOT / "uv.lock"),
    }


def _nvidia_gpu_info() -> list[dict[str, Any]]:
    raw = _run_stdout(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,utilization.gpu,driver_version,uuid",
            "--format=csv,noheader,nounits",
        ]
    )
    if raw is None:
        return []

    gpus: list[dict[str, Any]] = []
    for line in raw.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            continue
        name, total, used, util, driver, uuid = parts
        gpus.append(
            {
                "name": name,
                "memory_total_mb": int(total),
                "memory_used_mb": int(used),
                "utilization_gpu_pct": int(util),
                "driver_version": driver,
                "uuid": uuid,
            }
        )
    return gpus


def _run_stdout(command: list[str]) -> str | None:
    executable = shutil.which(command[0])
    if executable is None:
        return None
    completed = subprocess.run(
        [executable, *command[1:]],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    stdout = completed.stdout.strip()
    return stdout or None


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _total_memory_bytes() -> int | None:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None
    return int(page_size) * int(pages)


__all__ = [
    "DEFAULT_OUTPUT",
    "capture_machine_manifest",
]
