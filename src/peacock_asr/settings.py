from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
    import torch


def _default_cache_dir() -> Path:
    current = Path(__file__).resolve().parent
    for _ in range(10):
        if (current / "pyproject.toml").exists():
            return current / ".cache"
        parent = current.parent
        if parent == current:
            break
        current = parent
    return Path.home() / ".cache" / "peacock-asr"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PEACOCK_ASR_",
        env_file=".env",
        extra="ignore",
    )

    cache_dir: Path = Field(default_factory=_default_cache_dir)
    ctc_gop_model_path: Path | None = None
    ctc_gop_processor_path: Path | None = None
    ctc_feature_backend: str = "batched"  # "batched" (GPU) or "loop" (serial)
    num_workers: int = 1
    device: str = "auto"
    hf_checkpoint_repo: str | None = None
    hf_checkpoint_upload: bool = False
    hf_checkpoint_repo_subdir: str = "runs"
    hf_train_repo: str = Field(
        default="Peacockery/w2v-bert-phoneme-en",
        validation_alias=AliasChoices(
            "PEACOCK_ASR_HF_TRAIN_REPO",
            "HF_TRAIN_REPO",
        ),
    )
    hf_token: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "HF_TOKEN",
            "HUGGINGFACE_HUB_TOKEN",
            "PEACOCK_ASR_HF_TOKEN",
        ),
    )
    mlflow_tracking_uri: str = Field(
        default="https://mlflow.peacockery.studio",
        validation_alias=AliasChoices(
            "MLFLOW_TRACKING_URI",
            "PEACOCK_ASR_MLFLOW_TRACKING_URI",
        ),
    )
    mlflow_experiment_name: str = Field(
        default="peacock-asr",
        validation_alias=AliasChoices(
            "MLFLOW_EXPERIMENT_NAME",
            "PEACOCK_ASR_MLFLOW_EXPERIMENT_NAME",
        ),
    )
    mlflow_enable_system_metrics_logging: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING",
            "PEACOCK_ASR_MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING",
        ),
    )
    mlflow_system_metrics_sampling_interval: int = Field(
        default=10,
        ge=1,
        validation_alias=AliasChoices(
            "MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL",
            "PEACOCK_ASR_MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL",
        ),
    )
    mlflow_system_metrics_samples_before_logging: int = Field(
        default=1,
        ge=1,
        validation_alias=AliasChoices(
            "MLFLOW_SYSTEM_METRICS_SAMPLES_BEFORE_LOGGING",
            "PEACOCK_ASR_MLFLOW_SYSTEM_METRICS_SAMPLES_BEFORE_LOGGING",
        ),
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
        d = self.cache_dir / "checkpoints"
        d.mkdir(parents=True, exist_ok=True)
        return d


settings = Settings()
