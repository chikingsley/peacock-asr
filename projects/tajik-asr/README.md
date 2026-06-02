# Tajik ASR

Fine-tuning OmniASR CTC for Tajik (`tgk_Cyrl`). A thin per-language project — config (`sources.py`)
+ wiring (`curate.py`, `train.py`, `eval.py`, `assets.py`) over the shared packages
[`omni-curator`](../../packages/omni-curator) (data) and
[`omni-finetune-core`](../../packages/omni-finetune-core) (training).

## Pipeline

```
curate download → label → merge → (ingest) → verify → export → train → eval
```

- `tajik-curate download [--tier clean]` — pull YouTube channel audio (`sources.YOUTUBE_CHANNELS`)
  to `data/create/<slug>/` (16 kHz FLAC, resumable, cookie-authenticated).
- `tajik-curate label [--channel X]` — VAD-segment + Scribe-ensemble label each video into a
  per-channel store `data/channels/<slug>/store.sqlite` (parallel-safe, incremental).
- `tajik-curate merge` — fold the per-channel stores into the master `data/curator.sqlite`.
- `tajik-curate ingest fleurs|commonvoice` — existing-labeled datasets into the store.
- `tajik-curate verify` — Scribe-score every clip (WER/CER persisted for the export gate).
- `tajik-curate export vN [--max-wer ...]` — `Selection` over the pool → omni-parquet under
  `data/datasets/vN` (normalize → language gate → quality filter → tokenizer-coverage gate).
- `tajik-train` / `tajik-eval` — fine-tune / score via `omni-finetune-core` presets + `assets.py` cards.

## Data layout (`data/`, gitignored)

```
data/
  create/<slug>/    downloaded channel audio
  channels/<slug>/  per-channel labeled stores
  labeled/<slug>/   per-video VAD clip cuts (label workspace)
  canonical_audio/  resampled ingest clips
  raw/              transient dataset downloads (HF cache, Common Voice)
  curator.sqlite    master pool (merge target)
  datasets/vN/      exported omni-parquet ablations
```

## Sources (`sources.py`)

Wired now: **FLEURS** `tg_tj`, and **43 vetted YouTube channels** (19 clean — news / audiobook /
narration / lessons; 24 noisy — talk / podcast / vlog).

Planned (see `TODO.md`, not yet in `sources.py`): **Common Voice 25** `tg` and HF datasets
(`muhtasham/tajik-asr-augmented-test`, …) via `omni_curator.ingest.huggingface.load_hf_audio`.

## Legacy

The original combined dataset (`dataset_prep/artifacts/tajik_asr_combined_v0`, gitignored) and its
trained checkpoint are kept as provenance: 1,884 / 263 / 440 train/dev/test from `fleurs_tg_tj`
(1,815) + `common_voice_25_tg` (572) + `muhtasham_tajik_asr_augmented_test` (200), Scribe-v2 corpus
WER train 11.7% / dev 13.1% / test 14.7%. The `data/datasets` pipeline supersedes it.
