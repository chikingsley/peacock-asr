from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
    import torch


def _find_project_root() -> Path | None:
    current = Path(__file__).resolve().parent
    for _ in range(10):
        if (current / "pyproject.toml").exists():
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None


def _default_cache_dir() -> Path:
    project_root = _find_project_root()
    if project_root is not None:
        return project_root / ".cache"
    return Path.home() / ".cache" / "peacock-asr"


def _default_env_file() -> Path | None:
    project_root = _find_project_root()
    if project_root is None:
        return None
    return project_root / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_default_env_file(),
        extra="ignore",
    )

    cache_dir: Path = Field(default_factory=_default_cache_dir)
    ctc_gop_model_path: Path | None = None
    ctc_gop_processor_path: Path | None = None
    ctc_feature_backend: str = Field(
        default="batched",
        validation_alias=AliasChoices(
            "PEACOCK_CTC_FEATURE_BACKEND",
            "CTC_FEATURE_BACKEND",
        ),
    )  # "batched" (GPU) or "loop" (serial)
    ctc_scalar_backend: str = Field(
        default="k2",
        validation_alias=AliasChoices(
            "PEACOCK_CTC_SCALAR_BACKEND",
            "CTC_SCALAR_BACKEND",
        ),
    )  # "python" or "k2"
    ctc_scalar_device: str = Field(
        default="auto",
        validation_alias=AliasChoices(
            "PEACOCK_CTC_SCALAR_DEVICE",
            "CTC_SCALAR_DEVICE",
        ),
    )  # "cpu", "cuda", or "auto"
    ctc_scalar_batch_utterances: int = Field(
        default=8,
        validation_alias=AliasChoices(
            "PEACOCK_CTC_SCALAR_BATCH_UTTERANCES",
            "CTC_SCALAR_BATCH_UTTERANCES",
        ),
    )
    ctc_scalar_batch_phone_positions: int = Field(
        default=32,
        validation_alias=AliasChoices(
            "PEACOCK_CTC_SCALAR_BATCH_PHONE_POSITIONS",
            "CTC_SCALAR_BATCH_PHONE_POSITIONS",
        ),
    )
    ctc_scalar_batch_case_frame_budget: int = Field(
        default=5000,
        validation_alias=AliasChoices(
            "PEACOCK_CTC_SCALAR_BATCH_CASE_FRAME_BUDGET",
            "CTC_SCALAR_BATCH_CASE_FRAME_BUDGET",
        ),
    )
    ctc_posterior_batch_size: int = Field(
        default=8,
        validation_alias=AliasChoices(
            "PEACOCK_CTC_POSTERIOR_BATCH_SIZE",
            "CTC_POSTERIOR_BATCH_SIZE",
        ),
    )
    ctc_posterior_transport_dtype: str = Field(
        default="float32",
        validation_alias=AliasChoices(
            "PEACOCK_CTC_POSTERIOR_TRANSPORT_DTYPE",
            "CTC_POSTERIOR_TRANSPORT_DTYPE",
        ),
    )
    num_workers: int = 1
    device: str = Field(
        default="auto",
        validation_alias=AliasChoices("PEACOCK_DEVICE", "DEVICE"),
    )
    hf_checkpoint_repo: str | None = None
    hf_checkpoint_upload: bool = False
    hf_checkpoint_repo_subdir: str = "runs"
    hf_train_repo: str = "Peacockery/w2v-bert-phoneme-en"
    hf_token: str | None = None
    checkpoints_dir_override: Path | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "PEACOCK_CHECKPOINTS_DIR",
            "PEACOCK_ASR_CHECKPOINTS_DIR",
        ),
    )
    wandb_mode: str | None = Field(
        default=None,
        validation_alias="WANDB_MODE",
    )
    wandb_sweep_id: str | None = Field(
        default=None,
        validation_alias="WANDB_SWEEP_ID",
    )
    wandb_project: str = Field(
        default="peacock-asr-p003-compact-backbones",
        validation_alias=AliasChoices("PEACOCK_WANDB_PROJECT", "WANDB_PROJECT"),
    )
    wandb_entity: str = Field(
        default="peacockery",
        validation_alias=AliasChoices("PEACOCK_WANDB_ENTITY", "WANDB_ENTITY"),
    )
    wandb_group: str | None = Field(
        default=None,
        validation_alias=AliasChoices("PEACOCK_WANDB_GROUP", "WANDB_RUN_GROUP"),
    )
    wandb_job_type: str = Field(
        default="eval",
        validation_alias="PEACOCK_WANDB_JOB_TYPE",
    )
    wandb_run_prefix: str = Field(
        default="",
        validation_alias="PEACOCK_WANDB_RUN_PREFIX",
    )
    wandb_tags: str = Field(
        default="",
        validation_alias="PEACOCK_WANDB_TAGS",
    )
    wandb_track: str = Field(
        default="track10",
        validation_alias="PEACOCK_WANDB_TRACK",
    )
    wandb_project_id: str = Field(
        default="P003",
        validation_alias="PEACOCK_WANDB_PROJECT_ID",
    )
    wandb_phase: str | None = Field(
        default=None,
        validation_alias="PEACOCK_WANDB_PHASE",
    )
    wandb_job_id: str | None = Field(
        default=None,
        validation_alias="PEACOCK_WANDB_JOB_ID",
    )

    @property
    def torch_device(self) -> torch.device:
        """Resolve 'auto' to the best available device."""
        import torch  # noqa: PLC0415

        if self.device == "auto":
            return torch.device(
                "cuda" if torch.cuda.is_available() else "cpu",
            )
        return torch.device(self.device)

    @property
    def scalar_torch_device(self) -> torch.device:
        """Resolve scalar GOP backend device."""
        import torch  # noqa: PLC0415

        if self.ctc_scalar_device == "auto":
            return torch.device(
                "cuda" if torch.cuda.is_available() else "cpu",
            )
        return torch.device(self.ctc_scalar_device)

    @property
    def models_dir(self) -> Path:
        d = self.cache_dir / "models"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def data_dir(self) -> Path:
        d = self.cache_dir / "data"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def features_dir(self) -> Path:
        d = self.cache_dir / "features"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def checkpoints_dir(self) -> Path:
        d = self.checkpoints_dir_override or (self.cache_dir / "checkpoints")
        d.mkdir(parents=True, exist_ok=True)
        return d


settings = Settings()
