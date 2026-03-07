#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///
# References:
# - icefall LibriSpeech Conformer CTC recipe:
#   https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/conformer_ctc
# - PyTorch 2.9 release blog:
#   https://pytorch.org/blog/pytorch-2-9/
# - PyTorch FlexAttention + FlashAttention-4 on Hopper/Blackwell:
#   https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/
#
# Usage here:
# - Validate the reference lane without starting a long train.
# - Record the current state of the environment, manifests, lang files, and GPU.
from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

P004_ROOT = Path("/home/simon/github/peacock-asr/projects/P004-training-from-scratch")
P004_ENV = P004_ROOT / ".venv-icefall"
ENV_PYTHON = P004_ENV / "bin" / "python"
ICEFALL_ROOT = P004_ROOT / "third_party" / "icefall"
RECIPE_DIR = ICEFALL_ROOT / "egs" / "librispeech" / "ASR" / "conformer_ctc"
LANG_DIR = P004_ROOT / "experiments" / "data" / "lang_phone"
MANIFEST_DIR = P004_ROOT / "experiments" / "data" / "manifests_phone_raw"
BUILD_SUMMARY = MANIFEST_DIR / "build_summary.json"
DEFAULT_OUTPUT = (
    P004_ROOT / "experiments" / "validation" / "reference_setup.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the P004 reference lane and write a JSON summary."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Where to write the validation JSON (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def _prepend_env_path(env: dict[str, str], key: str, path: Path) -> None:
    if not path.is_dir():
        return
    previous = env.get(key, "")
    env[key] = f"{path}:{previous}" if previous else str(path)


def build_runtime_env() -> dict[str, str]:
    env = os.environ.copy()
    nvrtc_dir = (
        P004_ENV
        / "lib"
        / "python3.11"
        / "site-packages"
        / "nvidia"
        / "cuda_nvrtc"
        / "lib"
    )
    _prepend_env_path(env, "LD_LIBRARY_PATH", nvrtc_dir)
    _prepend_env_path(env, "PYTHONPATH", ICEFALL_ROOT)
    return env


def run_command(
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    completed = subprocess.run(  # noqa: S603
        cmd,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    return {
        "cmd": cmd,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def file_info(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "is_dir": path.is_dir(),
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
    }


def count_lines(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def inspect_manifest(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        first = json.loads(next(handle))
    supervision = first.get("supervisions", [])
    first_supervision = supervision[0] if supervision else {}
    recording = first.get("recording", {})
    return {
        "path": str(path),
        "first_cut_id": first.get("id"),
        "duration": first.get("duration"),
        "num_supervisions": len(supervision),
        "first_phone_text": first_supervision.get("text"),
        "sampling_rate": recording.get("sampling_rate"),
        "num_samples": recording.get("num_samples"),
    }


def import_probe(*, prepared_env: bool) -> dict[str, Any]:
    probe_code = """
import json
import lhotse
import torch
import k2
print(json.dumps({
  "python": {
    "executable": __import__("sys").executable,
    "version": __import__("sys").version.split()[0],
  },
  "torch": {
    "version": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_version": torch.version.cuda,
  },
  "k2": {"version": getattr(k2, "__version__", "unknown")},
  "lhotse": {"version": getattr(lhotse, "__version__", "unknown")},
}))
""".strip()
    result = run_command(
        [str(ENV_PYTHON), "-c", probe_code],
        env=build_runtime_env() if prepared_env else None,
    )
    parsed: dict[str, Any] = {
        "prepared_env": prepared_env,
        "returncode": result["returncode"],
        "stderr": result["stderr"],
    }
    if result["returncode"] == 0:
        parsed["details"] = json.loads(result["stdout"])
    else:
        parsed["details"] = None
    return parsed


def gpu_info() -> dict[str, Any] | None:
    if shutil.which("nvidia-smi") is None:
        return None

    summary = run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version,utilization.gpu,"
            "utilization.memory",
            "--format=csv,noheader",
        ]
    )
    processes = run_command(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )

    active_processes = []
    for line in processes["stdout"].splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("No running"):
            continue
        parts = [part.strip() for part in stripped.split(",")]
        if len(parts) != 3:
            continue
        pid, process_name, used_memory = parts
        ps_result = run_command(
            ["ps", "-p", pid, "-o", "pid=,ppid=,etime=,cmd="]
        )
        active_processes.append(
            {
                "pid": int(pid),
                "process_name": process_name,
                "used_gpu_memory_mib": int(used_memory),
                "ps": ps_result["stdout"],
            }
        )

    return {
        "summary": summary["stdout"],
        "active_processes": active_processes,
    }


def compute_status(report: dict[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    checks: list[str] = []

    required_paths = report["paths"]
    missing = [
        name
        for name, info in required_paths.items()
        if name != "output" and not info["exists"]
    ]
    if missing:
        issues.append(f"Missing required paths: {', '.join(missing)}")
    else:
        checks.append("All required reference-lane paths exist.")

    prepared_probe = report["import_probes"]["prepared_env"]
    if prepared_probe["returncode"] != 0:
        issues.append("Prepared-env import probe failed.")
    else:
        checks.append("Prepared-env import probe succeeded.")

    raw_probe = report["import_probes"]["raw_env"]
    if raw_probe["returncode"] != 0 and "libnvrtc.so.12" in raw_probe["stderr"]:
        issues.append(
            "Raw .venv-icefall imports need LD_LIBRARY_PATH to locate libnvrtc.so.12."
        )

    build_summary = report["build_summary"]
    if build_summary is None:
        issues.append("Missing build_summary.json for the phone manifests.")
    else:
        checks.append("Manifest build summary is present.")

    gpu = report["gpu"]
    if gpu and gpu["active_processes"]:
        issues.append("Target GPU is already occupied by another process.")

    overall = "pass" if not issues else "partial"
    return {
        "overall": overall,
        "checks": checks,
        "issues": issues,
    }


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    build_summary = None
    if BUILD_SUMMARY.is_file():
        build_summary = json.loads(BUILD_SUMMARY.read_text(encoding="utf-8"))

    manifests = {}
    for manifest in sorted(MANIFEST_DIR.glob("librispeech_cuts_*.jsonl.gz")):
        manifests[manifest.name] = inspect_manifest(manifest)

    lang_files = {
        "tokens": file_info(LANG_DIR / "tokens.txt"),
        "lexicon": file_info(LANG_DIR / "lexicon.txt"),
        "phone_list": file_info(LANG_DIR / "phone_list.txt"),
        "words": file_info(LANG_DIR / "words.txt"),
    }
    lang_counts = {
        "tokens": count_lines(LANG_DIR / "tokens.txt"),
        "lexicon": count_lines(LANG_DIR / "lexicon.txt"),
        "phone_list": count_lines(LANG_DIR / "phone_list.txt"),
        "words": count_lines(LANG_DIR / "words.txt"),
    }

    report: dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "paths": {
            "output": file_info(args.output),
            "p004_root": file_info(P004_ROOT),
            "env_python": file_info(ENV_PYTHON),
            "icefall_root": file_info(ICEFALL_ROOT),
            "recipe_dir": file_info(RECIPE_DIR),
            "manifest_dir": file_info(MANIFEST_DIR),
            "lang_dir": file_info(LANG_DIR),
        },
        "build_summary": build_summary,
        "manifests": manifests,
        "lang_files": lang_files,
        "lang_counts": lang_counts,
        "import_probes": {
            "raw_env": import_probe(prepared_env=False),
            "prepared_env": import_probe(prepared_env=True),
        },
        "gpu": gpu_info(),
    }
    report["status"] = compute_status(report)

    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["paths"]["output"] = file_info(args.output)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {args.output}")
    print(f"overall={report['status']['overall']}")
    for check in report["status"]["checks"]:
        print(f"check: {check}")
    for issue in report["status"]["issues"]:
        print(f"issue: {issue}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
