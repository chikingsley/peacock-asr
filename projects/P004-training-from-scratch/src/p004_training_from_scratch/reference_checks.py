from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

P004_ROOT = Path("/home/simon/github/peacock-asr/projects/P004-training-from-scratch")
ICEFALL_ENV = P004_ROOT / ".venv-icefall"
ENV_PYTHON = ICEFALL_ENV / "bin" / "python"
ICEFALL_ROOT = P004_ROOT / "third_party" / "icefall"
RECIPE_DIR = ICEFALL_ROOT / "egs" / "librispeech" / "ASR" / "conformer_ctc"
VALIDATOR = P004_ROOT / "code" / "validate_reference_setup.py"
DEFAULT_OUTPUT = P004_ROOT / "experiments" / "validation" / "reference_checks.json"


@dataclass(frozen=True)
class CheckSpec:
    name: str
    description: str
    kind: Literal["script", "pytest"]
    targets: tuple[str, ...]
    include_recipe_dir: bool = False


@dataclass(frozen=True)
class CheckResult:
    name: str
    description: str
    kind: str
    ok: bool
    returncode: int
    duration_s: float
    command: tuple[str, ...]
    stdout: str
    stderr: str
    details: dict[str, Any] | None = None


def _prepend_env_path(env: dict[str, str], key: str, path: Path) -> None:
    if not path.is_dir():
        return
    previous = env.get(key, "")
    env[key] = f"{path}:{previous}" if previous else str(path)


def build_reference_env(*, include_recipe_dir: bool = False) -> dict[str, str]:
    env = os.environ.copy()
    nvrtc_dir = (
        ICEFALL_ENV
        / "lib"
        / "python3.11"
        / "site-packages"
        / "nvidia"
        / "cuda_nvrtc"
        / "lib"
    )
    _prepend_env_path(env, "LD_LIBRARY_PATH", nvrtc_dir)
    _prepend_env_path(env, "PYTHONPATH", ICEFALL_ROOT)
    if include_recipe_dir:
        _prepend_env_path(env, "PYTHONPATH", RECIPE_DIR)
    return env


def build_check_specs(*, validation_output: Path) -> tuple[CheckSpec, ...]:
    return (
        CheckSpec(
            name="reference_setup",
            description="Run the saved reference-lane workspace validator.",
            kind="script",
            targets=(str(VALIDATOR), "--output", str(validation_output)),
        ),
        CheckSpec(
            name="icefall_checkpoint",
            description="Run upstream icefall checkpoint I/O tests.",
            kind="pytest",
            targets=("third_party/icefall/test/test_checkpoint.py",),
        ),
        CheckSpec(
            name="icefall_graph_compiler",
            description="Run upstream icefall CTC graph compiler tests.",
            kind="pytest",
            targets=("third_party/icefall/test/test_graph_compiler.py",),
        ),
        CheckSpec(
            name="icefall_conformer_components",
            description="Run recipe-local Conformer component tests.",
            kind="pytest",
            targets=(
                "third_party/icefall/egs/librispeech/ASR/conformer_ctc/test_subsampling.py",
                "third_party/icefall/egs/librispeech/ASR/conformer_ctc/test_label_smoothing.py",
            ),
            include_recipe_dir=True,
        ),
    )


def _build_command(spec: CheckSpec) -> tuple[str, ...]:
    if spec.kind == "script":
        return (str(ENV_PYTHON), *spec.targets)
    return (str(ENV_PYTHON), "-m", "pytest", "-q", *spec.targets)


def _load_validator_status(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    status = payload.get("status")
    if not isinstance(status, dict):
        return None
    return status


def run_check(spec: CheckSpec) -> CheckResult:
    command = _build_command(spec)
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        cwd=P004_ROOT,
        env=build_reference_env(include_recipe_dir=spec.include_recipe_dir),
    )
    duration_s = time.perf_counter() - started
    details: dict[str, Any] | None = None
    ok = completed.returncode == 0
    if spec.name == "reference_setup" and spec.kind == "script":
        details = _load_validator_status(Path(spec.targets[-1]))
        ok = ok and details is not None and details.get("overall") == "pass"
    return CheckResult(
        name=spec.name,
        description=spec.description,
        kind=spec.kind,
        ok=ok,
        returncode=completed.returncode,
        duration_s=duration_s,
        command=command,
        stdout=completed.stdout.strip(),
        stderr=completed.stderr.strip(),
        details=details,
    )


def run_reference_checks(*, output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    validation_output = output.parent / f"{output.stem}_reference_setup{output.suffix}"
    specs = build_check_specs(validation_output=validation_output)
    results = [run_check(spec) for spec in specs]
    payload = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "output_path": str(output),
        "validation_output_path": str(validation_output),
        "overall_ok": all(result.ok for result in results),
        "checks": [asdict(result) for result in results],
    }
    output.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")
    return payload


__all__ = [
    "DEFAULT_OUTPUT",
    "CheckResult",
    "CheckSpec",
    "_build_command",
    "_load_validator_status",
    "build_check_specs",
    "build_reference_env",
    "run_check",
    "run_reference_checks",
]
