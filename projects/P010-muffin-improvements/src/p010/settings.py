"""Global experiment settings via pydantic-settings."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Data
    speechocean762_path: Path = Field(description="Root of SpeechOcean762 dataset")
    librispeech_path: Path | None = Field(default=None)

    # Tracking
    traceo_url: str = Field(default="http://localhost:4000")
    traceo_api_key: str = Field(default="")
    run_name: str = Field(default="unnamed")
    project_name: str = Field(default="p010-muffin")

    # Training
    device: str = Field(default="cuda")
    seed: int = Field(default=42)
    num_seeds: int = Field(default=3, description="Min seeds per config per lab methodology")

    # Primary metric
    primary_metric: str = Field(default="pcc", description="PCC — matches all published SpeechOcean762 work")
