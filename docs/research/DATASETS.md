# Dataset Pipeline Guide

How data flows from raw audio to GPU training tensors, and which steps are needed for which models.

## The Three Layers

Every HuggingFace dataset goes through these layers:

```text
Hub (Parquet)  →  Local Cache (Arrow)  →  Training (PyTorch tensors)
   storage          first-load cache         every batch
   compressed       memory-mapped            on GPU
```

### 1. Hub Storage (Parquet)

Parquet is a compressed columnar format optimized for storage and transfer. When you `push_to_hub()` or `upload_folder()`, data ends up as `.parquet` files on HuggingFace servers.

This is what you see at `https://huggingface.co/datasets/Org/Name`.

### 2. Local Cache (Arrow)

The HuggingFace `datasets` library uses Apache Arrow as its native format. Arrow files are memory-mapped, meaning the OS can load just the rows you need without reading the whole file.

**On first load**, `load_dataset()` downloads the parquet files and converts them to Arrow format. This is cached at `HF_HOME/hub/datasets--Org--Name/`. On subsequent loads, it skips straight to the cached Arrow files.

The Arrow conversion is I/O-bound (not CPU-heavy). For our 283K-example dataset (~155GB parquet), it takes ~45 minutes on the first load. After that, loading is near-instant.

### 3. Training Tensors

During training, the DataLoader reads from the Arrow cache and converts to PyTorch tensors via the data collator. This happens every batch.

## What We Store in Each Dataset

### Raw Audio Dataset: `gilkeyio/librispeech-alignments`

Contains the original audio and phoneme-level alignments:

| Column | Type | Size |
|--------|------|------|
| `audio` | Audio (wav bytes) | ~50KB avg |
| `phonemes` | list[dict] | phone, start, end times |
| `words` | list[dict] | word-level alignments |
| `transcript` | string | full text |

Total: ~160GB across all splits.

### Preprocessed Feature Dataset: `Peacockery/librispeech-phoneme-features`

Pre-extracted mel spectrogram features for w2v-bert-2.0:

| Column | Type | Size |
|--------|------|------|
| `input_features` | list[list[float32]] | [n_frames, 160], ~480KB avg |
| `labels` | list[int64] | phoneme label IDs |
| `input_length` | int64 | number of frames |
| `phone_count` | int64 | number of phones |

Total: ~155GB across 4 splits (283,888 examples).

## Which Models Need What

### w2v-bert-2.0 (SeamlessM4T family)

**Feature extractor**: `SeamlessM4TFeatureExtractor` — computes mel spectrograms.
**Cost**: ~27ms per sample on CPU. This is the expensive step.
**Input key**: `input_features` (float32 mel spectrogram)

**With preprocessed dataset**: Features are pre-computed. Training loads them directly from Arrow cache. No CPU feature extraction needed. Faster on local SSD; may be slower on NFS (see "Storage Medium Matters" below).

**Without preprocessed dataset**: Uses `set_transform()` to extract features on-the-fly in DataLoader workers. Each worker decodes audio + computes mel every epoch.

### wav2vec2-base, HuBERT-base (wav2vec2 family)

**Feature extractor**: `Wav2Vec2FeatureExtractor` — just normalizes raw audio (subtract mean, divide by std).
**Cost**: ~0.3ms per sample. Trivial.
**Input key**: `input_values` (float32 normalized waveform)

**Preprocessing is NOT needed.** The "feature extraction" is a simple normalization that's 91x faster than mel extraction. The overhead of storing/loading preprocessed data would exceed the savings.

These models consume raw audio directly via `set_transform()`.

### Summary Table

| Model | Extractor | Cost/sample | Needs preprocessing? | Input key |
|-------|-----------|-------------|---------------------|-----------|
| w2v-bert-2.0 | SeamlessM4T (mel) | 27ms | Yes, big speedup | `input_features` |
| wav2vec2-base | Wav2Vec2 (normalize) | 0.3ms | No, not worth it | `input_values` |
| HuBERT-base | Wav2Vec2 (normalize) | 0.3ms | No, not worth it | `input_values` |
| Whisper | WhisperFeatureExtractor (mel) | ~25ms | Yes, would help | `input_features` |

## Storage Medium Matters: NFS vs Local SSD

The choice between preprocessed data and on-the-fly `set_transform()` depends heavily on where your data lives.

### The tradeoff

| Method | CPU cost/sample | I/O per sample | Best when |
|--------|----------------|----------------|-----------|
| `set_transform()` (raw audio) | 27ms (mel) or 0.3ms (normalize) | ~50KB (compressed audio) | NFS / network storage, or cheap CPU |
| Preprocessed Arrow | 0ms | ~480KB (float32 features) | Local SSD, fast disk I/O |

### What we measured (RunPod, NFS-backed volume)

| Method | Step time | Notes |
|--------|-----------|-------|
| `set_transform()` on-the-fly mel | **12.5s/step** | 8 DataLoader workers compute features in parallel |
| Preprocessed Arrow from NFS | **13.5s/step** | Reads 10x more data per sample over network |

Preprocessed data was **8% slower** on NFS because reading 480KB of float32 features per sample over the network costs more than reading 50KB of compressed audio and computing the mel on CPU.

### When to use which

- **NFS / network storage**: Use `set_transform()` — smaller I/O footprint wins
- **Local SSD**: Use preprocessed — zero CPU overhead wins
- **Cheap feature extraction** (wav2vec2 normalize, 0.3ms): Always use `set_transform()` — not worth storing
- **Expensive feature extraction** (mel, 27ms) + local SSD: Preprocess once, train many times

### What the HF docs say

The [HF datasets docs](https://huggingface.co/docs/datasets/process) recommend `set_transform()` for operations run every epoch (augmentations) and `.map()` for one-time preprocessing. They also [note](https://discuss.huggingface.co/t/custom-20gb-arrow-dataset-very-slow-to-train/146611) that Arrow memory-mapping on NFS can be slower than local disk, and suggest using a non-NFS intermediate directory when possible. The [Arrow architecture page](https://huggingface.co/docs/datasets/en/about_arrow) explains that memory-mapped access is fast for local disk but depends on I/O bandwidth — which is exactly where NFS falls short for large feature tensors.

## Training Script Flags

`projects/P003-compact-backbones/code/training/train_phoneme_head.py` handles both paths:

```bash
# With preprocessed data (w2v-bert-2.0)
uv run --project projects/P003-compact-backbones python projects/P003-compact-backbones/code/training/train_phoneme_head.py \
  --preprocessed-dataset Peacockery/librispeech-phoneme-features \
  --model-name facebook/w2v-bert-2.0

# With raw audio (wav2vec2-base) — no preprocessing needed
uv run --project projects/P003-compact-backbones python projects/P003-compact-backbones/code/training/train_phoneme_head.py \
  --model-name facebook/wav2vec2-base

# With raw audio (w2v-bert-2.0) — works but slower
uv run --project projects/P003-compact-backbones python projects/P003-compact-backbones/code/training/train_phoneme_head.py \
  --model-name facebook/w2v-bert-2.0
```

When `--preprocessed-dataset` is set:

- Loads pre-extracted features directly
- Skips `set_transform()` entirely
- Skips filtering (already done during preprocessing)
- Just loads → collate → train

When not set:

- Loads raw audio from `gilkeyio/librispeech-alignments`
- Applies `set_transform()` per batch to decode audio + extract features
- Filters invalid examples on first load

## Creating New Preprocessed Datasets

If you need to preprocess for a new model family (e.g., Whisper), the general approach:

1. Load the raw audio dataset
2. Apply the model's feature extractor to every example
3. Save the extracted features + labels as a new dataset
4. Push to Hub

See `projects/P003-compact-backbones/code/training/preprocess_features.py` for the w2v-bert-2.0 implementation and `docs/lessons/preprocessing-at-scale.md` for lessons learned about doing this at scale on RunPod.

Key gotchas:

- RunPod CPU pods have cgroup memory limits (~64GB) that `free` doesn't show
- NFS page cache counts toward cgroup — use `posix_fadvise(DONTNEED)` to evict
- HF datasets `.map()` with `num_proc>1` + `keep_in_memory=True` = fork deadlock
- Best approach for large datasets: bypass HF datasets, use pyarrow + huggingface_hub directly

## Cache Management

Arrow caches can be large. Locations:

```bash
# HF datasets cache (Arrow files)
$HF_HOME/hub/datasets--Org--Name/

# Default HF_HOME
~/.cache/huggingface/

# On RunPod, set to NFS to avoid filling container disk
HF_HOME=/runpod/hf_cache
```

To clear a dataset's cache:

```bash
rm -rf $HF_HOME/hub/datasets--Org--Name/
```

Next load will re-download parquet and reconvert to Arrow.
