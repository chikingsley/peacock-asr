# Persian ASR

Local project for Persian ASR dataset curation, Omnilingual CTC benchmarking and
fine-tuning, and Parakeet experiments.

## Setup

```bash
cd /home/simon/github/peacock-asr/projects/farsi-asr
uv sync
```

The active Python entry points are defined in `pyproject.toml`:

- `farsi-curate`
- `farsi-omni-train`
- `farsi-omni-eval`
- `farsi-omni-eval-lm`
- `farsi-parakeet-train-tokenizer`
- `farsi-parakeet-train-ctc`
- `farsi-parakeet-train-tdt`
- `farsi-parakeet-train-nemo`
- `farsi-parakeet-eval`

## Project Layout

```text
src/farsi_asr/          # project config and thin CLIs
src/farsi_asr/omni/     # Omnilingual training/eval wrappers
src/farsi_asr/parakeet/ # Parakeet tokenizer/train/eval wrappers
```

The shared curation and training logic lives in `packages/omni-curator`,
`packages/omni-finetune-core`, and `packages/parakeet-finetune-core`.

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

1. Preserve source metadata through the shared curator source registry and stores.
2. Score/filter rows with the project ASR gates and shared curator tooling.
3. Export accepted rows into Omnilingual parquet for training.
4. Benchmark each training run into its own folder under `benchmarks/`.
5. Use Omni CTC scoring as an agreement/error-analysis signal.

Training uses the Python entry point:

```bash
uv run farsi-omni-train --help
```

Corpus curation is routed through the shared curator CLI:

```bash
uv run farsi-curate --help
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
