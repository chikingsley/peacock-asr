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
    FragmentLoading,
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


def _gpu12_trainer_1b() -> TrainerConfig:
    """The trainer block that fits omni 1B CTC on ~12 GB: PURE bf16 + grad clip + grad-accum 4.

    ``mixed_precision.mode="off"`` runs everything — including the AdamW step — in bf16, dropping
    the fp32 optimizer master copy the default "static" mode keeps (~8 GB vs the ~16 GB the safe
    fp32-optimizer path needs for the 1B). Pure-bf16 AdamW is numerically riskier, so
    ``max_grad_norm=1.0`` clips spikes and grad-accum 4 restores effective batch size lost to the
    small per-step element budget. Validated on the Persian 1B run (peaks ~9.1 GiB at 960k elems).
    """
    return TrainerConfig(
        mixed_precision=MixedPrecision(mode="off", dtype="torch.bfloat16"),
        grad_accumulation=GradAccumulation(num_batches=4),
        activation_checkpointing=ActivationCheckpointing(mode="layerwise", every_nth_layer=1),
        max_grad_norm=1.0,
    )


def _dataset(
    name: str,
    summary_path: str,
    *,
    max_num_elements: int,
    max_audio_len: int = 960_000,
    fragment_cache_dir: str | None = None,
) -> DatasetConfig:
    return DatasetConfig(
        name=name,
        mixture_parquet_storage_config=MixtureParquetStorageConfig(
            dataset_summary_path=summary_path,
            fragment_loading=FragmentLoading(cache_dir=fragment_cache_dir),
        ),
        asr_task_config=AsrTaskConfig(
            max_num_elements=max_num_elements, max_audio_len=max_audio_len
        ),
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
    fragment_cache_dir: str | None = None,
) -> TrainingConfig:
    """Regime A — fresh fine-tune that maxes a ~12 GB GPU. ``lr_scheduler`` left to the recipe
    default (tri-stage: warmup 10% / hold 40% / decay 50%, peaking at ``lr``).

    Set ``fragment_cache_dir`` to real disk for large corpora — the fairseq2 default caches
    Arrow fragments under /tmp, and a tmpfs overflow kills the run mid-training.
    """
    return TrainingConfig(
        model=ModelConfig(name=model),
        dataset=_dataset(
            dataset, dataset_summary_path,
            max_num_elements=max_num_elements, fragment_cache_dir=fragment_cache_dir,
        ),
        tokenizer=TokenizerConfig(name=tokenizer),
        optimizer=OptimizerConfig(config=OptimizerInner(lr=lr)),
        trainer=_gpu12_trainer(),
        regime=_best_wer_regime(num_steps, validate_every=validate_every),
    )


def gpu_max_finetune_1b(
    *,
    model: str,
    dataset: str,
    tokenizer: str,
    dataset_summary_path: str,
    num_steps: int,
    lr: float = 1e-5,
    max_num_elements: int = 960_000,
    max_audio_len: int = 480_000,
    validate_every: int = 1_000,
    fragment_cache_dir: str | None = None,
) -> TrainingConfig:
    """Regime A for the **1B** model — pure-bf16 fit on a ~12 GB GPU.

    Same shape as :func:`gpu_max_finetune` but for the 1B: ``model.dtype`` is bf16 and the trainer
    runs pure bf16 (no fp32 optimizer copy — see :func:`_gpu12_trainer_1b`). Clips are capped at
    ``max_audio_len`` (480k samples = 30 s) to bound activation memory; the omni inference pipeline
    caps at 40 s anyway, and ``max_num_elements`` defaults to 960k (~two 30 s clips per batch).
    Watch the first ~1-2k steps for loss spikes / NaNs and drop ``lr`` if it destabilizes.
    """
    return TrainingConfig(
        model=ModelConfig(name=model, dtype="torch.bfloat16"),
        dataset=_dataset(
            dataset, dataset_summary_path,
            max_num_elements=max_num_elements, max_audio_len=max_audio_len,
            fragment_cache_dir=fragment_cache_dir,
        ),
        tokenizer=TokenizerConfig(name=tokenizer),
        optimizer=OptimizerConfig(config=OptimizerInner(lr=lr)),
        trainer=_gpu12_trainer_1b(),
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
    fragment_cache_dir: str | None = None,
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
        dataset=_dataset(
            dataset, dataset_summary_path,
            max_num_elements=max_num_elements, fragment_cache_dir=fragment_cache_dir,
        ),
        tokenizer=TokenizerConfig(name=tokenizer),
        optimizer=OptimizerConfig(config=OptimizerInner(lr=peak_lr)),
        lr_scheduler=LrSchedulerConfig(config=LrSchedulerInner()),
        trainer=_gpu12_trainer(),
        regime=_best_wer_regime(num_steps, validate_every=validate_every),
        common=CommonConfig(no_sweep_dir=no_sweep_dir),
    )
