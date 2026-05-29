# Persian ASR

Local project for Persian ASR dataset curation, Omnilingual CTC benchmarking and
fine-tuning, and Parakeet experiments.

## Setup

```bash
cd /home/simon/github/peacock-asr/projects/persian-asr
uv sync
```

The active Python entry points are defined in `pyproject.toml`:

- `persian-benchmark-asr`
- `persian-benchmark-omni`
- `persian-benchmark-sherpa-onnx`
- `persian-build-candidate-manifests`
- `persian-dataset-export-nemo-manifest`
- `persian-dataset-ingest`
- `persian-dataset-run-nemo-curator`
- `persian-export-nemo-manifest`
- `persian-finetune-parakeet`
- `persian-ingest-corpus`
- `persian-prepare-omni-curated`
- `persian-prepare-omni-fleurs`
- `persian-prepare-omni-thomcles`
- `persian-repair-neyshekar`
- `persian-run-nemo-curator`
- `persian-score-omni-manifest`
- `persian-train-omni`
- `persian-train-tokenizer`

## Project Layout

```text
src/persian_asr_dataset/           # dataset ledger, source ingest, NeMo Curator export/scoring
src/persian_omnilingual_asr/       # Omnilingual benchmarks, data prep, scoring, training
src/persian_parakeet_asr/          # Parakeet tokenizer and fine-tune launch helpers
vendor/omnilingual-asr/            # pinned Facebook/Meta Omnilingual ASR checkout
vendor/nemo/                       # pinned NVIDIA NeMo checkout for ASR scripts
vendor/mobius/                     # pinned FluidInference/Mobius checkout for CoreML tooling
```

The dataset-prep commands remain available because they are the reproducible recipe
for rebuilding the Omnilingual parquet mirrors under `data/raw`. They are rebuild
tools; normal curation uses the SQLite ledger and scorer commands.

## Current Prepared Data

- `data/raw/fleurs_fa_ir_omni`: Omnilingual parquet prepared from `google/fleurs`.
- `data/raw/thomcles_persian_omni`: Omnilingual parquet prepared from
  `Thomcles/Persian-Farsi-Speech`.
- `data/raw/mozilla_data_collective`: Common Voice 25 Mozilla Data Collective archive
  and extracted files.
- `data/raw/neyshekar_v3_asr_aligned`: repaired Neyshekar source archive and metadata.
- `data/raw/asr_farsi_youtube_pourmand1376`: ASR Farsi YouTube source metadata.
- `data/raw/worldspeech_fa_ir`: WorldSpeech Persian source metadata.
- `data/curation/persian_corpus.sqlite`: master curation ledger.
- `data/selection/candidate-manifests`: selected candidate manifest sets.
- `data/training/omnilingual`: Omnilingual training datasets.
- `tokenizers/corpora`: tokenizer text corpora.
- `tokenizers/parakeet`: trained Parakeet tokenizer artifacts.

## Main Workflow

1. Preserve raw source metadata in a SQLite curation ledger.
2. Score/filter ledger rows with NeMo Curator and
   `nvidia/stt_fa_fastconformer_hybrid_large`.
3. Export accepted rows into Omnilingual parquet for training.
4. Benchmark each training run into its own folder under `benchmarks/`.
5. Use Omni CTC scoring as an agreement/error-analysis signal after the NeMo pass.

Training uses the Python entry point:

```bash
uv run persian-train-omni --preset fleurs-300m
uv run persian-train-omni --preset thomcles-continue
```

Corpus ingestion writes source rows to the curation ledger:

```bash
uv run persian-ingest-corpus --source common_voice_25_0
uv run persian-ingest-corpus --source fleurs_omni
uv run persian-ingest-corpus --source thomcles_omni
```

NeMo manifest export materializes audio files and JSONL for Curator:

```bash
uv run persian-export-nemo-manifest --run-id nemo-fa-preview --limit 100
```

NeMo Curator scoring uses the FastConformer CTC head by default:

```bash
uv run persian-run-nemo-curator \
  --manifest /home/simon/github/peacock-asr/projects/persian-asr/data/curation/nemo_runs/nemo-fa-preview/manifest/manifest.jsonl
```

## Benchmarks

Each benchmark run should write directly under:

```text
benchmarks/<run-id>/
```

with:

- `summary.md`
- `samples.jsonl`
- `run.log` when available

The current FLEURS test summaries are already in this shape.

## Notes

- `HF_HOME` should default to this project's `.hf-cache` for local commands.
- Final and best Omnilingual checkpoints referenced by `.fairseq2-assets` live under
  `models/omnilingual/checkpoints/`. `runs/` is for logs, metrics, and resumable
  training state while a run is active.
