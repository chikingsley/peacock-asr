"""Project-level configuration shared by Parakeet tokenizer, CTC, and TDT CLIs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, kw_only=True)
class ParakeetProject:
    """Language-specific defaults for shared Parakeet fine-tuning commands."""

    name: str
    language: str
    root: Path
    data_root: Path | None = None
    nemo_root: Path | None = None
    model_root: Path | None = None
    tokenizer_root: Path | None = None
    runs_root: Path | None = None
    default_ctc_model: Path | None = None
    default_tdt_model: str | Path | None = None
    default_hybrid_model: Path | None = None
    default_train_manifest: Path | None = None
    default_validation_manifest: Path | None = None
    default_tokenizer_dir: Path | None = None
    default_tdt_checkpoint: Path | None = None
    default_ctc_checkpoint: Path | None = None
    # promoted final models for eval (stable .nemo), kept separate from the training base models
    default_eval_ctc_model: str | Path | None = None
    default_eval_tdt_model: str | Path | None = None
    default_eval_kind: str = "ctc"
    default_eval_normalizer: str | None = None
    default_eval_normalizer_language: str | None = None
    default_tokenizer_name: str | None = None
    default_ctc_run_name: str | None = None
    default_tdt_run_name: str | None = None
    env_prefix: str | None = None

    @property
    def data(self) -> Path:
        return self.data_root or self.root / "data"

    @property
    def hf_home(self) -> Path:
        return self.data / "hf-cache"

    @property
    def models(self) -> Path:
        return self.model_root or self.root / "models" / "parakeet"

    @property
    def tokenizers(self) -> Path:
        return self.tokenizer_root or self.root / "tokenizers" / "parakeet"

    @property
    def runs(self) -> Path:
        return self.runs_root or self.root / "runs" / "parakeet"

    @property
    def env_key(self) -> str:
        return (self.env_prefix or self.name).upper().replace("-", "_")

    def default_nemo_root(self) -> Path:
        """Resolve the NeMo recipe checkout used for tokenizer/recipe wrapper commands."""
        env_specific = os.environ.get(f"{self.env_key}_NEMO_ROOT")
        if env_specific:
            return Path(env_specific).expanduser()
        env_generic = os.environ.get("PARAKEET_NEMO_ROOT")
        if env_generic:
            return Path(env_generic).expanduser()
        if self.nemo_root is not None:
            return self.nemo_root.expanduser()
        packaged = Path(__file__).resolve().parent / "nemo_recipes"
        if packaged.exists():
            return packaged
        msg = (
            "No NeMo recipe root resolved. Set `nemo_root` on the ParakeetProject "
            "(e.g. base_models/parakeet/nemo_recipes), or set "
            f"{self.env_key}_NEMO_ROOT / PARAKEET_NEMO_ROOT. The cross-project "
            "farsi-asr/src/finetune_parakeet fallback has been removed."
        )
        raise RuntimeError(msg)

    def configure_environment(self) -> None:
        """Set cache defaults under the language project root."""
        os.environ.setdefault("HF_HOME", str(self.hf_home))
        os.environ.setdefault("HF_DATASETS_CACHE", str(self.hf_home / "datasets"))
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

    def tokenizer_name(self, spe_type: str, vocab_size: int) -> str:
        if self.default_tokenizer_name is not None:
            return self.default_tokenizer_name
        return f"{self.language.lower()}_spe_{spe_type}_v{vocab_size}"
