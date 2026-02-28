from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_prefix": "GOPT_BENCH_", "env_file": ".env"}

    cache_dir: Path = Path.home() / ".cache" / "gopt-bench"
    ctc_gop_model_path: Path | None = None
    ctc_gop_processor_path: Path | None = None
    num_workers: int = 1
    device: str = "auto"

    @property
    def torch_device(self) -> object:
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


settings = Settings()
