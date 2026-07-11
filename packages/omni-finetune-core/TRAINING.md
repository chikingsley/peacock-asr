# Fine-tuning omni CTC (300M + 1B) — what works

Two regimes that have actually run on our ~12 GB GPU, captured so we (or an agent) don't re-derive or re-break them. Both are type-checked builders in `omni_finetune_core.presets`; build one in Python, emit YAML via `TrainingConfig.to_recipe_dict`, run with `train.py`. Knobs that depend on the dataset (steps, element budget, paths) are arguments; the parts that "just have to be right" are baked in.

## The shared GPU-max trainer block

What fits the 300M CTC model on 12 GB — used in **every** run:

- `mixed_precision: torch.bfloat16`
- `grad_accumulation.num_batches: 2`
- `activation_checkpointing: layerwise, every_nth_layer 1` ← the big memory saver
- `min_audio_len 16_000` (1 s), `max_audio_len 960_000` (60 s)

Peak GPU sat ~83% (9.45 GiB) on Tajik with this. If you OOM, the first lever is `max_num_elements` (the batch element budget), not these.

## Regime A — `gpu_max_finetune` (the everyday default)

Fresh fine-tune from the base card. `lr 1e-5`, the recipe-default tri-stage schedule (warmup 10% → hold 40% → decay 50%), best-WER checkpointing.

```python
from omni_finetune_core.presets import gpu_max_finetune, recommend_num_steps

cfg = gpu_max_finetune(
    model="omni_ctc_300m_v2_base",
    dataset="tajik_asr_corpus", tokenizer="omni_asr_tokenizer_written_v2_local",
    dataset_summary_path=".../language_distribution_0.tsv",
    num_steps=recommend_num_steps(hours=5.8, target_epochs=30),  # ≈3750
)
```

## The 1B model — `gpu_max_finetune_1b` (pure bf16)

The 1B only fits on 12 GB in **pure bf16**: `mixed_precision.mode="off"` + `model.dtype="torch.bfloat16"` means there is **no fp32 master copy**, so weights + AdamW states stay in bf16 (~8 GB) instead of the ~16 GB the safe fp32-optimizer default ("static") needs. The trade-off is numerical: pure-bf16 AdamW is spikier, so the preset adds `max_grad_norm 1.0` (clip) + `grad_accumulation 4`, and caps clips at `max_audio_len 480_000` (30 s) to bound activation memory. Validated on the Persian 1B run — peaks ~9.1 GiB / 80% at `max_num_elements 960_000`. Watch the first ~1–2k steps for loss spikes / NaNs and drop `lr` if it destabilizes.

```python
from omni_finetune_core.presets import gpu_max_finetune_1b

cfg = gpu_max_finetune_1b(
    model="omniASR_CTC_1B_v2",  # upstream base card (auto-downloaded); no local card needed
    dataset="georgian_asr_corpus", tokenizer="omniASR_tokenizer_written_v2",
    dataset_summary_path=".../language_distribution_0.tsv",
    num_steps=34_000,
)
```

The difference from the 300M in one line: **300M** keeps an fp32 optimizer copy ("static" mixed precision); **1B** can't afford one, so it runs everything in bf16 and clips gradients to stay stable.

## Regime B — `warm_restart` (only after a plateau)

Second wind from a **best-checkpoint** card: loading weights via the card gives a **fresh optimizer + schedule** at a lower peak (`2e-6`, ~20% of from-scratch), with an **explicit** tri-stage so the LR is what you intend. Use it when a plain run has plateaued and you want to test if there's more. Be honest about the payoff: on scribe-v4 it moved dev-WER 11.27 → 11.15 and test 17.50 → 17.44 — real but tiny. **The lever is almost always more/better data, not another rewarm.**

```python
cfg = warm_restart(
    checkpoint_card="omni_ctc_300m_v2_scribe_v4_20260530_best",  # the BEST ckpt, not base
    dataset="scribe_v4", tokenizer="omniASR_tokenizer_written_v2",
    dataset_summary_path=".../language_distribution_0.tsv",
    num_steps=10_000, peak_lr=2e-6,
)
# train with an explicit --output-dir (no_sweep_dir=True is already set so it lands there).
```

## How many steps / epochs

- **Target ~20–30 epochs** for these small-data fine-tunes (matches HF Wav2Vec2/XLSR practice).
- **~125 steps/epoch** on 5.8 h at `max_num_elements 2.0M` / grad-accum 2 (the anchor in `recommend_num_steps`); it scales ~linearly with hours and inversely with the element budget and grad-accum. It's a *starting point* — read the real steps/epoch off the first epoch's logs.
- **Always early-stop on the dev-WER plateau** and **ship the best, not the last** checkpoint (`score_metric=wer`, `keep_best_n=3`). Char error (UER) maxes out early and stays low; the remaining error is word-level and is what plateaus.

## Footguns (each has bitten us)

1. **Sweep-hash cold-start.** fairseq2's run dir is `ws_{world_size}.{sha1(whole config)}`, so *any* config change spawns a fresh worktree and **cold-starts**. To continue/keep an existing worktree, set `common.no_sweep_dir=True` and pass an explicit `--output-dir`.
1. **Plain resume runs at the decayed LR.** Resuming restores the tri-stage LR at its ~5e-7 floor — it only polishes. For a real second wind use `warm_restart` (weights via card = fresh optimizer), not a resume.
1. **Batcher rounds the element budget down.** `max_num_elements` is rounded down to a multiple of `max_audio_len`; pick them so it doesn't round to 0 (`max_seq_len must be <= max_num_elements (0)`).
1. **Sample rate must be 16 kHz.** The ASR pipeline decodes at the file's native rate with no resample; mixed rates overshoot the element budget and OOM (this was the Tajik OOM). Curate to 16 kHz before parquet.

## What these produced

| run               | regime                | result                                        |
| ----------------- | --------------------- | --------------------------------------------- |
| Tajik v0 (5.8 h)  | A, 4000 steps         | best dev-WER **17.1%** @ step 1800, UER ~1.7% |
| Persian scribe-v4 | A then B (rewarm 10k) | dev 11.27 → **11.15**, test 17.50 → **17.44** |
