"""Global experiment settings via pydantic-settings.

All hyperparameters match train_hierCB.sh exactly unless noted.
Override via environment variables or a .env file.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_prefix="P010_",
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    features_dir: Path = Field(
        description="Root directory containing seq_data_librispeech_v4/ (HF cache or local)"
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    embed_dim: int = Field(default=24, description="Transformer embedding dimension")
    p_depth: int = Field(default=3, description="Number of phone-level BlockCNN layers")
    w_depth: int = Field(default=2, description="Number of word-level BlockCNN layers")
    u_depth: int = Field(default=1, description="Number of utterance-level BlockCNN layers")
    num_heads: int = Field(default=1, description="Attention heads in BlockCNN")
    ssl_drop: float = Field(default=0.2, description="Dropout applied to concatenated SSL features")

    # ── Training ──────────────────────────────────────────────────────────────
    lr: float = Field(default=1e-3)
    batch_size: int = Field(default=25)
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
    phnvar_sigma: float = Field(default=1.0, description="Gaussian noise std for PhnVar (not specified in paper)")
    phnvar_alpha: float = Field(default=1.0, description="QF weight in PhnVar (paper: 1.0)")
    phnvar_beta: float = Field(default=1.0, description="DF weight in PhnVar (paper: 1.0)")

    # ── Pretraining (MuFFIN §V.B, ref [41]) ─────────────────────────────────
    pretrain_epochs: int = Field(default=100, description="Pretraining epochs (HierTFR default)")

    # ── W&B ───────────────────────────────────────────────────────────────────
    wandb_project: str = Field(default="p010-muffin")
    wandb_entity: str | None = Field(default=None)
