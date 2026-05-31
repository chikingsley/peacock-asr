"""Two canonical fine-tune regimes as type-checked ``TrainingConfig`` builders.

These encode what actually worked on a ~12 GB GPU across the Persian/Tajik omni 300M CTC
fine-tunes, so a person or agent doesn't re-derive (or re-break) the setup each time:

- :func:`gpu_max_finetune` — a fresh fine-tune that maxes the local GPU (bf16 + grad-accum +
  layerwise activation checkpointing). The everyday default.
- :func:`warm_restart` — a second-wind run from a best checkpoint's weights with a fresh optimizer
  and a lower peak LR. Use only after a plain run plateaus (the gain is usually small — the lever
  is more/better data).

See ``TRAINING.md`` for the reasoning, the step/epoch math, and the footguns.
"""

from __future__ import annotations

from omni_finetune_core.config import (
    ActivationCheckpointing,
    AsrTaskConfig,
    CommonConfig,
    DatasetConfig,
    GradAccumulation,
    LrSchedulerConfig,
    LrSchedulerInner,
    MixedPrecision,
    MixtureParquetStorageConfig,
    ModelConfig,
    OptimizerConfig,
    OptimizerInner,
    RegimeConfig,
    TokenizerConfig,
    TrainerConfig,
    TrainingConfig,
)

#: Anchor for the step/epoch estimate: tajik v0 ran ~125 steps/epoch on 5.8 h of audio at
#: ``max_num_elements=2_000_000`` with grad-accum 2 (4000 steps ≈ 32 epochs).
_STEPS_PER_EPOCH_ANCHOR = 125.0
_ANCHOR_HOURS = 5.8
_ANCHOR_MAX_ELEMENTS = 2_000_000
_ANCHOR_GRAD_ACCUM = 2


def _gpu12_trainer() -> TrainerConfig:
    """The trainer block that fits omni 300M CTC on ~12 GB: bf16 + grad-accum + layerwise ckpt."""
    return TrainerConfig(
        mixed_precision=MixedPrecision(dtype="torch.bfloat16"),
        grad_accumulation=GradAccumulation(num_batches=2),
        activation_checkpointing=ActivationCheckpointing(mode="layerwise", every_nth_layer=1),
    )


def _dataset(name: str, summary_path: str, *, max_num_elements: int) -> DatasetConfig:
    return DatasetConfig(
        name=name,
        mixture_parquet_storage_config=MixtureParquetStorageConfig(
            dataset_summary_path=summary_path
        ),
        asr_task_config=AsrTaskConfig(max_num_elements=max_num_elements),
    )


def _best_wer_regime(num_steps: int, *, validate_every: int) -> RegimeConfig:
    """Validate/checkpoint on the same cadence and keep the best-WER checkpoints (ship best)."""
    return RegimeConfig(
        num_steps=num_steps,
        score_metric="wer",
        validate_every_n_steps=validate_every,
        validate_after_n_steps=validate_every - 1,
        checkpoint_every_n_steps=validate_every,
        checkpoint_after_n_steps=validate_every - 1,
        keep_best_n_checkpoints=3,
        keep_last_n_checkpoints=1,
        save_model_only="all_but_last",
    )


def recommend_num_steps(
    hours: float,
    *,
    target_epochs: int = 30,
    max_num_elements: int = _ANCHOR_MAX_ELEMENTS,
    grad_accum: int = _ANCHOR_GRAD_ACCUM,
) -> int:
    """Rough step budget for ``target_epochs`` (20-30 is the small-data sweet spot).

    Calibrated on a single anchor (tajik v0) and scaled linearly, so treat it as a starting
    point: read the real steps/epoch off the first epoch's logs and adjust. Always early-stop on
    the dev-WER plateau rather than trusting this number.
    """
    steps_per_epoch = (
        _STEPS_PER_EPOCH_ANCHOR
        * (hours / _ANCHOR_HOURS)
        * (_ANCHOR_MAX_ELEMENTS / max_num_elements)
        * (_ANCHOR_GRAD_ACCUM / grad_accum)
    )
    return max(1, round(steps_per_epoch * target_epochs))


def gpu_max_finetune(
    *,
    model: str,
    dataset: str,
    tokenizer: str,
    dataset_summary_path: str,
    num_steps: int,
    lr: float = 1e-5,
    max_num_elements: int = _ANCHOR_MAX_ELEMENTS,
    validate_every: int = 200,
) -> TrainingConfig:
    """Regime A — fresh fine-tune that maxes a ~12 GB GPU. ``lr_scheduler`` left to the recipe
    default (tri-stage: warmup 10% / hold 40% / decay 50%, peaking at ``lr``).
    """
    return TrainingConfig(
        model=ModelConfig(name=model),
        dataset=_dataset(dataset, dataset_summary_path, max_num_elements=max_num_elements),
        tokenizer=TokenizerConfig(name=tokenizer),
        optimizer=OptimizerConfig(config=OptimizerInner(lr=lr)),
        trainer=_gpu12_trainer(),
        regime=_best_wer_regime(num_steps, validate_every=validate_every),
    )


def warm_restart(
    *,
    checkpoint_card: str,
    dataset: str,
    tokenizer: str,
    dataset_summary_path: str,
    num_steps: int,
    peak_lr: float = 2e-6,
    max_num_elements: int = _ANCHOR_MAX_ELEMENTS,
    validate_every: int = 500,
    no_sweep_dir: bool = True,
) -> TrainingConfig:
    """Regime B — warm-restart from a best-checkpoint card.

    ``checkpoint_card`` must point at the prior best checkpoint's weights (NOT the base card):
    loading weights via the model card gives a FRESH optimizer + schedule, unlike a plain resume
    (which restores the decayed tri-stage LR and only polishes). ``peak_lr`` defaults to 2e-6
    (~20% of a from-scratch 1e-5). ``no_sweep_dir=True`` + an explicit ``--output-dir`` at train
    time writes into the dir you choose instead of cold-starting a fresh ``ws_{hash}`` worktree.
    """
    return TrainingConfig(
        model=ModelConfig(name=checkpoint_card),
        dataset=_dataset(dataset, dataset_summary_path, max_num_elements=max_num_elements),
        tokenizer=TokenizerConfig(name=tokenizer),
        optimizer=OptimizerConfig(config=OptimizerInner(lr=peak_lr)),
        lr_scheduler=LrSchedulerConfig(config=LrSchedulerInner()),
        trainer=_gpu12_trainer(),
        regime=_best_wer_regime(num_steps, validate_every=validate_every),
        common=CommonConfig(no_sweep_dir=no_sweep_dir),
    )
