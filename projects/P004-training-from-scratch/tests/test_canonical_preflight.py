from __future__ import annotations

from pathlib import Path

import pytest

from p004_training_from_scratch.canonical.preflight import (
    read_smoke_audio_paths,
    resolve_preflight_wandb_mode,
)
from p004_training_from_scratch.settings import ProjectSettings


def test_resolve_preflight_wandb_mode_respects_disabled(tmp_path: Path) -> None:
    settings = ProjectSettings(
        vast_api_key=None,
        wandb_api_key=None,
        wandb_entity="peacockery",
        wandb_project="p004",
        wandb_mode="disabled",
        hf_token=None,
        hf_home=None,
    )

    mode, reason = resolve_preflight_wandb_mode(
        settings,
        allow_online=True,
        netrc_path=tmp_path / ".netrc",
    )

    assert mode == "disabled"
    assert "disabled" in reason


def test_resolve_preflight_wandb_mode_forces_offline_without_auth(
    tmp_path: Path,
) -> None:
    settings = ProjectSettings(
        vast_api_key=None,
        wandb_api_key=None,
        wandb_entity="peacockery",
        wandb_project="p004",
        wandb_mode="online",
        hf_token=None,
        hf_home=None,
    )

    mode, reason = resolve_preflight_wandb_mode(
        settings,
        allow_online=True,
        netrc_path=tmp_path / ".netrc",
    )

    assert mode == "offline"
    assert "offline" in reason


def test_resolve_preflight_wandb_mode_allows_online_with_credentials(
    tmp_path: Path,
) -> None:
    settings = ProjectSettings(
        vast_api_key=None,
        wandb_api_key="token",
        wandb_entity="peacockery",
        wandb_project="p004",
        wandb_mode="online",
        hf_token=None,
        hf_home=None,
    )

    mode, reason = resolve_preflight_wandb_mode(
        settings,
        allow_online=True,
        netrc_path=tmp_path / ".netrc",
    )

    assert mode == "online"
    assert "credentials" in reason


def test_read_smoke_audio_paths_returns_first_existing_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_one = tmp_path / "one.wav"
    audio_two = tmp_path / "two.wav"
    audio_one.write_bytes(b"one")
    audio_two.write_bytes(b"two")
    audio_list = tmp_path / "audio_files.txt"
    audio_list.write_text(
        "\n".join(
            [
                audio_one.relative_to(tmp_path).as_posix(),
                audio_two.relative_to(tmp_path).as_posix(),
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "p004_training_from_scratch.canonical.preflight.PROJECT_ROOT",
        tmp_path,
    )

    paths = read_smoke_audio_paths(audio_list, limit=2)

    assert paths == [audio_one, audio_two]


def test_read_smoke_audio_paths_rejects_missing_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_list = tmp_path / "audio_files.txt"
    audio_list.write_text("missing.wav\n", encoding="utf-8")
    monkeypatch.setattr(
        "p004_training_from_scratch.canonical.preflight.PROJECT_ROOT",
        tmp_path,
    )

    with pytest.raises(FileNotFoundError):
        read_smoke_audio_paths(audio_list, limit=1)
