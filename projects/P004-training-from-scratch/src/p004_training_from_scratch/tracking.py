from __future__ import annotations

from pathlib import Path
from typing import Literal

from p004_training_from_scratch.settings import ProjectSettings

WandbMode = Literal["online", "offline", "disabled", "shared"]


def resolve_local_wandb_mode(
    settings: ProjectSettings,
    *,
    allow_online: bool,
    netrc_path: Path | None = None,
    offline_reason: str = "forcing offline mode for local canonical task",
) -> tuple[WandbMode, str]:
    if settings.wandb_mode == "disabled":
        return ("disabled", "disabled by project settings")
    if settings.wandb_mode == "offline":
        return ("offline", "offline mode requested by project settings")

    credentials_present = bool(settings.wandb_api_key)
    if netrc_path is None:
        netrc_path = Path.home() / ".netrc"
    if netrc_path.is_file():
        credentials_present = True

    if allow_online and credentials_present:
        return ("online", "online mode allowed and W&B credentials were found")

    return ("offline", offline_reason)

__all__ = ["WandbMode", "resolve_local_wandb_mode"]
