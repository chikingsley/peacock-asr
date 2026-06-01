"""Typed (Pydantic v2) training configuration for the Omni CTC fine-tune recipe.

The language projects currently hand-write YAML that fairseq2 parses untyped. These
models give the same structure a validated, IDE-discoverable shape: build a
``TrainingConfig`` in Python (or load a parsed dict), get validation + sensible defaults,
and emit the recipe-shaped dict via :meth:`TrainingConfig.to_recipe_dict`.

Defaults mirror the omni 300M CTC recipe as run in the Persian/Tajik projects.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Dtype = Literal["torch.bfloat16", "torch.float16", "torch.float32"]


class _Base(BaseModel):
    # extra="forbid" catches typos in config keys; the `model` field needs the protected
    # namespace disabled so it doesn't collide with Pydantic's `model_*` methods.
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class AsrTaskConfig(_Base):
    min_audio_len: int = 16_000
    max_audio_len: int = 960_000
    max_num_elements: int = 2_000_000
    batch_shuffle_window: int = 1
    normalize_audio: bool = True
    example_shuffle_window: int = 1

    @model_validator(mode="after")
    def _element_budget_fits_one_clip(self) -> AsrTaskConfig:
        # The omni batcher rounds max_num_elements DOWN to a multiple of max_audio_len; if the
        # budget is smaller than one clip it rounds to 0 and the run dies with the cryptic
        # "max_seq_len must be <= max_num_elements (0)". Catch it here instead.
        if self.max_num_elements < self.max_audio_len:
            msg = (
                f"max_num_elements ({self.max_num_elements}) < max_audio_len "
                f"({self.max_audio_len}): the batcher rounds the element budget down to a "
                "multiple of max_audio_len, so this rounds to 0. Raise max_num_elements to at "
                "least max_audio_len (ideally a few multiples)."
            )
            raise ValueError(msg)
        return self


class FragmentLoading(_Base):
    cache: bool = True


class MixtureParquetStorageConfig(_Base):
    dataset_summary_path: str
    beta_corpus: float = 0.5
    beta_language: float = 0.5
    fragment_loading: FragmentLoading = Field(default_factory=FragmentLoading)


class DatasetConfig(_Base):
    name: str
    train_split: str = "train"
    valid_split: str = "dev"
    storage_mode: Literal["MIXTURE_PARQUET"] = "MIXTURE_PARQUET"
    task_mode: Literal["ASR"] = "ASR"
    mixture_parquet_storage_config: MixtureParquetStorageConfig
    asr_task_config: AsrTaskConfig = Field(default_factory=AsrTaskConfig)


class ModelConfig(_Base):
    name: str
    # The dtype the model's parameters load in. Left unset (fairseq2 default float32) for the 300M
    # static-mixed-precision path; set to "torch.bfloat16" for the 1B PURE-bf16 path, where
    # mixed_precision.mode="off" runs everything in this dtype (no fp32 master copy — see presets).
    dtype: Dtype | None = None


class TokenizerConfig(_Base):
    name: str


class OptimizerInner(_Base):
    lr: float = 1e-5


class OptimizerConfig(_Base):
    config: OptimizerInner = Field(default_factory=OptimizerInner)


class LrSchedulerInner(_Base):
    stage_ratio: list[float] = Field(default_factory=lambda: [0.1, 0.4, 0.5])
    start_lr_scale: float = 0.01
    final_lr_scale: float = 0.05


class LrSchedulerConfig(_Base):
    name: str = "tri_stage"
    config: LrSchedulerInner = Field(default_factory=LrSchedulerInner)


class MixedPrecision(_Base):
    # "static" (fairseq2 default): fwd/bwd in `dtype`, optimizer step in full fp32 (keeps an fp32
    # master copy — safe, but costs memory). "off": whole run in `model.dtype` (pure bf16, no fp32
    # copy — needed to fit the 1B). "auto": torch.amp autocast. Left unset = fairseq2's "static".
    mode: Literal["off", "static", "auto"] | None = None
    dtype: Dtype = "torch.bfloat16"


class GradAccumulation(_Base):
    num_batches: int = 1


class ActivationCheckpointing(_Base):
    mode: Literal["layerwise"] = "layerwise"
    every_nth_layer: int = 1


class TrainerConfig(_Base):
    freeze_encoder_for_n_steps: int = 0
    mixed_precision: MixedPrecision = Field(default_factory=MixedPrecision)
    grad_accumulation: GradAccumulation = Field(default_factory=GradAccumulation)
    activation_checkpointing: ActivationCheckpointing = Field(
        default_factory=ActivationCheckpointing
    )
    # Gradient-norm clip. Unset for the 300M; set to 1.0 for the 1B pure-bf16 path, where the
    # missing fp32 master copy makes AdamW numerically riskier and clipping guards against spikes.
    max_grad_norm: float | None = None
    gc_every_n_steps: int = 100


class RegimeConfig(_Base):
    num_steps: int = Field(gt=0)
    score_metric: Literal["wer", "loss"] | None = None
    validate_every_n_steps: int = 1_000
    validate_after_n_steps: int = 0
    checkpoint_every_n_steps: int = 1_000
    checkpoint_after_n_steps: int = 0
    keep_best_n_checkpoints: int | None = None
    keep_last_n_checkpoints: int | None = None
    save_model_only: str | None = None
    publish_metrics_every_n_steps: int = 1
    publish_metrics_after_n_steps: int = 0


class CommonConfig(_Base):
    # no_sweep_dir=true + an explicit run dir resumes an existing worktree instead of
    # spawning a fresh ws_{world_size}.{hash} subdir (the sha1 covers the whole config,
    # so any change otherwise cold-starts a new run).
    no_sweep_dir: bool = False
    sweep_format: str = "ws_{world_size}.{hash}"


class TrainingConfig(_Base):
    model: ModelConfig
    dataset: DatasetConfig
    tokenizer: TokenizerConfig
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    lr_scheduler: LrSchedulerConfig | None = None
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    regime: RegimeConfig
    common: CommonConfig = Field(default_factory=CommonConfig)

    def to_recipe_dict(self) -> dict[str, object]:
        """Recipe-shaped dict (drop unset optionals) for YAML emission / fairseq2."""
        return self.model_dump(exclude_none=True)
