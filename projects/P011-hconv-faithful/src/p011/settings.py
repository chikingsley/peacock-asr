"""Global experiment settings via pydantic-settings.

All hyperparameters match train_hierCB.sh exactly unless noted.
Override via environment variables or a .env file.
"""

import os
from enum import StrEnum
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from p011.ssl_features import SSL_MODEL_KEYS, SSLModelKey, parse_ssl_model_keys, ssl_feature_dim

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _default_features_dir() -> Path:
    """Resolve the default features directory from env or the project .env file."""
    for env_key in ("P011_FEATURES_DIR", "P010_FEATURES_DIR"):
        env_value = os.getenv(env_key)
        if env_value:
            return Path(env_value).expanduser()

    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            for env_key in ("P011_FEATURES_DIR", "P010_FEATURES_DIR"):
                if line.startswith(f"{env_key}="):
                    return Path(line.split("=", 1)[1].strip()).expanduser()

    raise RuntimeError(
        "features_dir was not provided and neither P011_FEATURES_DIR nor P010_FEATURES_DIR is set"
    )


class SSLInterfaceMode(StrEnum):
    """How SSL features are fed into the pronunciation model."""

    NONE = "none"
    HCONV = "hconv"
    CHCONV = "chconv"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_prefix="P011_",
        extra="ignore",
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    features_dir: Path = Field(
        default_factory=_default_features_dir,
        description="Root directory containing seq_data_librispeech_v4/ (HF cache or local)",
    )
    frame_store_dir: Path | None = Field(
        default=None,
        description="Optional override for ssl_frame_store_v1; defaults to <features_dir>/ssl_frame_store_v1",
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    embed_dim: int = Field(default=24, description="Transformer embedding dimension")
    p_depth: int = Field(default=3, description="Number of phone-level BlockCNN layers")
    w_depth: int = Field(default=2, description="Number of word-level BlockCNN layers")
    u_depth: int = Field(default=1, description="Number of utterance-level BlockCNN layers")
    num_heads: int = Field(default=1, description="Attention heads in BlockCNN")
    ssl_drop: float = Field(default=0.2, description="Dropout applied to concatenated SSL features")
    ssl_interface: SSLInterfaceMode = Field(
        default=SSLInterfaceMode.NONE,
        description="How SSL features are fused before entering the pronunciation model",
    )
    ssl_models: tuple[SSLModelKey, ...] = Field(
        default=SSL_MODEL_KEYS,
        description="Subset of SSL feature streams to load, kept in the repo's canonical concat order",
    )
    ssl_output_dim: int | None = Field(
        default=None,
        description="Projected SSL width after HConv/CHConv; defaults to the selected raw SSL width",
    )

    # ── Training ──────────────────────────────────────────────────────────────
    lr: float = Field(default=1e-3)
    batch_size: int = Field(default=25)
    grad_accum_steps: int = Field(default=1, ge=1, description="Micro-batches to accumulate before optimizer step")
    n_epochs: int = Field(default=100)
    noise: float = Field(default=0.0, description="Scale of input noise augmentation on GOP features")
    seed: int = Field(default=22, description="First seed in the paper's seed list (22,33,44,55,66)")
    device: str = Field(default="cuda")

    # ── Loss weights ──────────────────────────────────────────────────────────
    loss_w_phn: float = Field(default=3.0, description="MuFFIN §V.B: L_p=3, L_w=1, L_u=1")
    loss_w_word: float = Field(default=1.0)
    loss_w_utt: float = Field(default=1.0)

    # ── ConPCO ────────────────────────────────────────────────────────────────
    use_conpco: bool = Field(default=False)
    loss_w_pco: float = Field(default=1.0)
    loss_w_clap: float = Field(default=1.0)
    pco_ld: float = Field(default=0.5, description="Diversity term weight λ_d")
    pco_lt: float = Field(default=0.1, description="Tightness term weight λ_t")
    pco_mg: float = Field(default=1.0, description="Ordinal margin")
    clap_t2a: float = Field(default=0.5, description="Text-to-audio CLAP loss weight")

    # ── MDD (Phase 1c — MuFFIN full) ──────────────────────────────────────────
    use_mdd: bool = Field(default=False)
    loss_w_mdd: float = Field(default=1.0, description="Weight for MDD detection (BCE) loss")
    loss_w_diag: float = Field(default=1.0, description="Weight for MDD diagnosis (CE) loss")

    # ── PhnVar (MuFFIN §IV cont) ────────────────────────────────────────────
    use_phnvar: bool = Field(default=False, description="Enable phoneme-specific logit perturbation")
    phnvar_sigma: float = Field(default=4.0, description="PhnVar noise std (clamped [-1,1]; BLV default)")
    phnvar_alpha: float = Field(default=1.0, description="QF weight in PhnVar (paper: 1.0)")
    phnvar_beta: float = Field(default=1.0, description="DF weight in PhnVar (paper: 1.0)")

    # ── Pretraining (MuFFIN §V.B, ref [41]) ─────────────────────────────────
    pretrain_epochs: int = Field(default=100, description="Pretraining epochs (HierTFR default)")

    # ── W&B ───────────────────────────────────────────────────────────────────
    wandb_project: str = Field(default="p011-hconv-faithful")
    wandb_entity: str | None = Field(default=None)

    @field_validator("ssl_models", mode="before")
    @classmethod
    def _parse_ssl_models(cls, value: object) -> tuple[SSLModelKey, ...]:
        if value is None or value == "":
            return SSL_MODEL_KEYS
        if isinstance(value, (list, tuple, set)):
            return parse_ssl_model_keys([str(item) for item in value])
        if isinstance(value, str):
            return parse_ssl_model_keys(value)
        raise TypeError(f"Unsupported ssl_models value: {value!r}")

    @property
    def selected_ssl_dim(self) -> int:
        """Concatenated SSL width for the configured subset."""
        return ssl_feature_dim(self.ssl_models)

    @property
    def resolved_frame_store_dir(self) -> Path:
        """Frame-store root used by the paper-faithful HConv path."""
        return self.frame_store_dir or (self.features_dir / "ssl_frame_store_v1")

    @property
    def resolved_ssl_output_dim(self) -> int:
        """Interface output width after optional projection."""
        return self.ssl_output_dim or self.selected_ssl_dim
