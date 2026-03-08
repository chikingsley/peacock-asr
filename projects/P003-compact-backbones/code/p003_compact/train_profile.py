"""Typed training profile loader for fixed phoneme-head runs."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class TrainProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    output_dir: str
    num_epochs: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    gradient_accumulation: int = Field(ge=1)
    learning_rate: float = Field(gt=0.0)
    train_splits: list[str] = Field(min_length=1)
    eval_split: str
    max_train_samples: int | None = Field(default=None, ge=1)
    max_eval_samples: int | None = Field(default=None, ge=1)
    push_to_hub: bool
    dataloader_workers: int = Field(default=4, ge=0)

    @field_validator("name", "output_dir", "eval_split")
    @classmethod
    def _validate_non_empty(cls, value: str) -> str:
        stripped = value.strip()
        if stripped == "":
            msg = "Profile string fields must not be empty."
            raise ValueError(msg)
        return stripped

    @field_validator("train_splits")
    @classmethod
    def _validate_train_splits(cls, value: list[str]) -> list[str]:
        cleaned = [split.strip() for split in value if split.strip() != ""]
        if len(cleaned) == 0:
            msg = "train_splits must contain at least one non-empty split."
            raise ValueError(msg)
        return cleaned


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def profile_path(name: str) -> Path:
    profile_name = name.strip()
    if profile_name == "":
        msg = "Profile name must not be empty."
        raise ValueError(msg)
    return _repo_root() / "code" / "training" / "profiles" / f"{profile_name}.yaml"


def load_train_profile(name: str) -> TrainProfile:
    path = profile_path(name)
    if not path.exists():
        msg = (
            f"Training profile '{name}' not found at {path}. "
            "Expected one of: preflight, main."
        )
        raise FileNotFoundError(msg)

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    try:
        profile = TrainProfile.model_validate(raw)
    except ValidationError as e:
        msg = f"Invalid training profile '{path}':\n{e}"
        raise ValueError(msg) from e
    return profile
