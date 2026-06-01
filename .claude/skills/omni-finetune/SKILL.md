---
name: omni-finetune
description: Use when fine-tuning an OmniASR CTC model (300M or 1B) on an exported omni-parquet dataset with omni-finetune-core. Covers the presets (gpu_max_finetune, gpu_max_finetune_1b, warm_restart), step budgeting, and the GPU footguns. Run after curate-verify-export has produced a datasets/vN.
---

# Fine-tune OmniASR CTC (300M / 1B)

Train on an exported `data/datasets/vN/` (omni-parquet from `curate-verify-export`). Reference:
`packages/omni-finetune-core/TRAINING.md`. Presets are **type-checked builders** in
`omni_finetune_core.presets`; you build a `TrainingConfig` in Python, then `train()` writes the
recipe YAML and runs the in-housed fairseq2 recipe.

There is **no `omni-finetune` console script** — launch from a small project script. The only CLI
the package ships is `omni-tune` (batch-element-budget tuning sweep). Both the **300M** and **1B**
regimes fit the ~12 GB GPU — the 300M via the default "static" mixed precision (fp32 optimizer
copy), the 1B via **pure bf16** (`gpu_max_finetune_1b`, see below).

## Launch (project venv)

```python
from pathlib import Path
from omni_finetune_core.presets import gpu_max_finetune, recommend_num_steps
from omni_finetune_core.train import configure_environment, train

cache_root = Path("data")
configure_environment(cache_root)   # HF/fairseq2 caches + CUDA env defaults (idempotent)

cfg = gpu_max_finetune(
    model="omni_ctc_300m_v2_base",
    dataset="<dataset_asset_name>",          # the parquet dataset asset card
    tokenizer="omni_asr_tokenizer_written_v2_local",
    dataset_summary_path=".../language_distribution_0.tsv",
    num_steps=recommend_num_steps(hours=5.8, target_epochs=30),  # ~3750 @ 5.8h
)
train(cfg, output_dir=Path("runs/v1"))       # writes config.yaml, runs the recipe
```

## Regimes

- **`gpu_max_finetune`** (everyday default): fresh fine-tune from the base card. `lr 1e-5`,
  recipe-default tri-stage schedule (warmup 10% → hold 40% → decay 50%), best-WER checkpointing.
- **`gpu_max_finetune_1b`** (the 1B model): same shape, but **pure bf16** — `model.dtype` bf16 +
  `mixed_precision.mode="off"` (no fp32 optimizer copy, ~8 GB vs ~16 GB), with `max_grad_norm 1.0`
  + grad-accum 4 for stability and clips capped at 30 s. `model="omniASR_CTC_1B_v2"` (upstream base,
  auto-downloaded). Watch the first ~1–2k steps for loss spikes / NaNs; drop `lr` if it destabilizes.
- **`warm_restart`** (only after a dev-WER plateau): loads weights from a **best-checkpoint card**
  (not base) → fresh optimizer + lower peak (`peak_lr 2e-6`, explicit tri-stage). Real but tiny
  payoff (Persian: dev 11.27→11.15). The lever is almost always **more/better data**, not a rewarm.

```python
from omni_finetune_core.presets import warm_restart
cfg = warm_restart(
    checkpoint_card="omni_ctc_300m_v2_scribe_v4_20260530_best",   # the BEST ckpt, not base
    dataset="scribe_v4", tokenizer="omniASR_tokenizer_written_v2",
    dataset_summary_path=".../language_distribution_0.tsv",
    num_steps=10_000, peak_lr=2e-6,
)
```

## Step / epoch budget

- Target **~20–30 epochs** for small-data fine-tunes. `recommend_num_steps(hours=, target_epochs=)`
  anchors on ~125 steps/epoch @ 5.8 h / `max_num_elements 2.0M` / grad-accum 2.
- **Early-stop on the dev-WER plateau; ship the best, not the last** checkpoint (`score_metric=wer`,
  `keep_best_n=3`). UER (char error) bottoms out early; word-level error is what plateaus.

## GPU-max block (baked into every preset, fits 300M on 12 GB)

`bfloat16`, grad-accum 2, layerwise activation checkpointing (every layer — the big saver),
`min_audio_len 16_000` (1 s) / `max_audio_len 960_000` (60 s). Peak ~83% / 9.45 GiB on Tajik.
If you OOM, the first lever is `max_num_elements`, not these.

## Footguns (each has bitten us)

1. **Sweep-hash cold-start.** fairseq2's run dir is `ws_{world_size}.{sha1(config)}` — any config
   change spawns a fresh worktree and cold-starts. To continue an existing one set
   `common.no_sweep_dir=True` and pass an explicit `output_dir`.
2. **Plain resume runs at the decayed LR** (~5e-7 floor — only polishes). For a real second wind use
   `warm_restart` (weights via card = fresh optimizer), not a resume.
3. **Batcher rounds `max_num_elements` down** to a multiple of `max_audio_len`; pick them so it
   doesn't round to 0 (`max_seq_len must be <= max_num_elements (0)`).
4. **Sample rate must be 16 kHz.** The pipeline decodes at native rate with no resample; mixed rates
   overshoot the element budget and OOM. Curate to 16 kHz before parquet (the curator does this).
