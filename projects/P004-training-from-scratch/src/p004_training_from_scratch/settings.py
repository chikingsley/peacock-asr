from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(
    os.environ.get("P004_PROJECT_ROOT", str(DEFAULT_PROJECT_ROOT))
).expanduser()
DEFAULT_WANDB_ENTITY = "peacockery"
DEFAULT_WANDB_PROJECT = "peacock-asr-p004-training-from-scratch"
DEFAULT_WANDB_MODE = "online"


@dataclass(frozen=True, slots=True)
class ProjectSettings:
    vast_api_key: str | None
    wandb_api_key: str | None
    wandb_entity: str
    wandb_project: str
    wandb_mode: str
    hf_token: str | None
    hf_home: str | None
    runpod_api_key: str | None = None

    @classmethod
    def from_env(
        cls,
        *,
        load_dotenv_file: bool = True,
        dotenv_path: Path | None = None,
    ) -> ProjectSettings:
        if load_dotenv_file:
            load_dotenv(dotenv_path or (PROJECT_ROOT / ".env"))
        return cls(
            runpod_api_key=_optional_env("RUNPOD_API_KEY"),
            vast_api_key=_optional_env("VAST_API_KEY"),
            wandb_api_key=_optional_env("WANDB_API_KEY"),
            wandb_entity=os.environ.get("WANDB_ENTITY", DEFAULT_WANDB_ENTITY),
            wandb_project=os.environ.get("WANDB_PROJECT", DEFAULT_WANDB_PROJECT),
            wandb_mode=os.environ.get("WANDB_MODE", DEFAULT_WANDB_MODE),
            hf_token=_optional_env("HF_TOKEN"),
            hf_home=_optional_env("HF_HOME"),
        )

    @property
    def wandb_enabled(self) -> bool:
        return self.wandb_mode != "disabled"

    def public_env(self) -> dict[str, str]:
        payload = {
            "WANDB_ENTITY": self.wandb_entity,
            "WANDB_PROJECT": self.wandb_project,
            "WANDB_MODE": self.wandb_mode,
        }
        if self.hf_home:
            payload["HF_HOME"] = self.hf_home
        return payload


def _optional_env(key: str) -> str | None:
    value = os.environ.get(key)
    if value is None:
        return None
    value = value.strip()
    return value or None


__all__ = [
    "DEFAULT_PROJECT_ROOT",
    "DEFAULT_WANDB_ENTITY",
    "DEFAULT_WANDB_MODE",
    "DEFAULT_WANDB_PROJECT",
    "PROJECT_ROOT",
    "ProjectSettings",
]
