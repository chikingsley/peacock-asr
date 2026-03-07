from __future__ import annotations

import pytest

from p004_training_from_scratch.settings import (
    DEFAULT_WANDB_ENTITY,
    DEFAULT_WANDB_MODE,
    DEFAULT_WANDB_PROJECT,
    ProjectSettings,
)


@pytest.fixture(autouse=True)
def clean_relevant_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "VAST_API_KEY",
        "WANDB_API_KEY",
        "WANDB_ENTITY",
        "WANDB_PROJECT",
        "WANDB_MODE",
        "HF_TOKEN",
        "HF_HOME",
    ):
        monkeypatch.delenv(key, raising=False)


def test_project_settings_defaults() -> None:
    settings = ProjectSettings.from_env(load_dotenv_file=False)
    assert settings.vast_api_key is None
    assert settings.wandb_api_key is None
    assert settings.wandb_entity == DEFAULT_WANDB_ENTITY
    assert settings.wandb_project == DEFAULT_WANDB_PROJECT
    assert settings.wandb_mode == DEFAULT_WANDB_MODE
    assert settings.wandb_enabled is True


def test_project_settings_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WANDB_ENTITY", "custom-entity")
    monkeypatch.setenv("WANDB_PROJECT", "custom-project")
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("HF_HOME", "/tmp/hf")
    settings = ProjectSettings.from_env(load_dotenv_file=False)
    assert settings.wandb_entity == "custom-entity"
    assert settings.wandb_project == "custom-project"
    assert settings.wandb_mode == "disabled"
    assert settings.wandb_enabled is False
    assert settings.public_env() == {
        "WANDB_ENTITY": "custom-entity",
        "WANDB_PROJECT": "custom-project",
        "WANDB_MODE": "disabled",
        "HF_HOME": "/tmp/hf",
    }
