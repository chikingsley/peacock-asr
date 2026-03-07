from __future__ import annotations

import json
from pathlib import Path

from p004_training_from_scratch.reference_checks import (
    ENV_PYTHON,
    ICEFALL_ROOT,
    RECIPE_DIR,
    VALIDATOR,
    CheckSpec,
    _build_command,
    _load_validator_status,
    build_check_specs,
    build_reference_env,
)


def test_build_reference_env_includes_icefall_root() -> None:
    env = build_reference_env()
    pythonpath = env["PYTHONPATH"].split(":")
    assert str(ICEFALL_ROOT) in pythonpath
    assert str(RECIPE_DIR) not in pythonpath


def test_build_reference_env_can_include_recipe_dir() -> None:
    env = build_reference_env(include_recipe_dir=True)
    pythonpath = env["PYTHONPATH"].split(":")
    assert str(ICEFALL_ROOT) in pythonpath
    assert str(RECIPE_DIR) in pythonpath


def test_build_check_specs_shapes_validator_and_recipe_tests(tmp_path: Path) -> None:
    output = tmp_path / "reference_setup.json"
    specs = build_check_specs(validation_output=output)
    assert specs[0].targets == (str(VALIDATOR), "--output", str(output))
    assert specs[-1].include_recipe_dir is True


def test_build_command_for_script_check() -> None:
    spec = CheckSpec(
        name="reference_setup",
        description="validator",
        kind="script",
        targets=("code/validate_reference_setup.py",),
    )
    assert _build_command(spec) == (str(ENV_PYTHON), "code/validate_reference_setup.py")


def test_build_command_for_pytest_check() -> None:
    spec = CheckSpec(
        name="icefall_checkpoint",
        description="checkpoint test",
        kind="pytest",
        targets=("third_party/icefall/test/test_checkpoint.py",),
    )
    assert _build_command(spec) == (
        str(ENV_PYTHON),
        "-m",
        "pytest",
        "-q",
        "third_party/icefall/test/test_checkpoint.py",
    )


def test_load_validator_status_reads_status_mapping(tmp_path: Path) -> None:
    payload = {
        "status": {
            "overall": "partial",
            "checks": ["ok"],
            "issues": ["gpu busy"],
        }
    }
    path = tmp_path / "validator.json"
    path.write_text(f"{json.dumps(payload)}\n", encoding="utf-8")
    assert _load_validator_status(path) == payload["status"]
