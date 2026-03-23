"""Tests for settings.py."""

from __future__ import annotations

from pathlib import Path


def test_settings_uses_project_local_env_file() -> None:
    from p010.settings import PROJECT_ROOT, Settings

    env_file = Settings.model_config["env_file"]
    assert isinstance(env_file, Path)
    assert env_file == PROJECT_ROOT / ".env"
    assert env_file.is_absolute()


def test_settings_env_prefix_is_p010() -> None:
    from p010.settings import Settings

    assert Settings.model_config["env_prefix"] == "P010_"
