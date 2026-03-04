from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import Field
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
    hf_train_repo: str = "Peacockery/w2v-bert-phoneme-en"
    hf_token: str | None = None

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
