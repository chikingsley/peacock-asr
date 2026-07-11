# Persian ASR

Local project for Persian ASR dataset curation, Omnilingual CTC benchmarking and fine-tuning, and Parakeet experiments.

This directory has two active Markdown documents: this operational README and `EXPERIMENTS.md`, the append-only result ledger. Retired plans, run notes, source lists, and decisions remain available through Git history rather than living beside the current workflow.

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

Shared curation quality audits use `omni-quality` from `packages/omni-curator`.

## Project Layout

```text
src/farsi_asr/          # project config and thin CLIs
src/farsi_asr/omni/     # Omnilingual training/eval wrappers
src/farsi_asr/parakeet/ # Parakeet tokenizer/train/eval wrappers
```

The shared curation and training logic lives in `packages/omni-curator`, `packages/omni-finetune-core`, and `packages/parakeet-finetune-core`.

## Current Data

- `data` is a machine-local symlink to `/mnt/tiny-2t/peacock-asr/farsi-asr/data`.
- `data/parakeet/manifests/gate2_full_train.jsonl` is the current 74,752-row, approximately 173-hour matched-audio training surface used for the Parakeet recipe ablations.
- `Peacockery/farsi-asr-corpus-v4` is the published 985-hour, seven-corpus training export. Its machine-labeled portions already passed the Scribe-v4 agreement filter.
- `Peacockery/farsi-asr-wer35-fastconformer` is the older 416,056-row, 500.51-hour FastConformer-WER≤35 export. The Hub artifact contains its kept rows, not the 92,462 rejected candidates.
- `Peacockery/omni-ctc-300m-farsi` and `Peacockery/parakeet-ctc-109m-farsi` are the published CTC models; the latter is pinned under `base_models/parakeet/parakeet-ctc-109m-farsi/model.nemo` and is compatible with NeMo Forced Aligner.
- `src/farsi_asr/sources.py` is the executable corpus and YouTube-channel registry.

## Main Workflow

1. Preserve source metadata through the shared curator source registry and stores.
1. Record additive quality signals before choosing or changing any filter threshold.
1. Score/filter rows with the project ASR gates and shared curator tooling.
1. Export accepted rows into Omnilingual parquet for training.
1. Benchmark each training run into its own folder under `benchmarks/`.
1. Use Omni CTC scoring as an agreement/error-analysis signal.

Training uses the Python entry point:

```bash
uv run farsi-omni-train --help
```

Corpus curation is routed through the shared curator CLI:

```bash
uv run farsi-curate --help
```

## Data Quality

`omni-quality` provides the two new bounded audit lanes. `edge` records NeMo-SDP-style beginning/end ASR disagreement plus full WER/CER from a fixed draft recognizer. `nfa-prepare` records and applies the project language normalizer, while `nfa-run` and `nfa-summarize` run the version-matched NeMo Forced Aligner with a Persian CTC model and record word coverage, leading/trailing margins, aligned-span coverage, and exact model/tool hashes. These signals never delete rows by themselves.

For a new corpus or materially different source domain:

1. Freeze a deterministic bounded sample with `omni-quality sample`.
1. Produce draft hypotheses with one fixed model and record its exact path/hash.
1. Run `omni-quality edge` without thresholds, then `omni-quality nfa-prepare`, CTC forced alignment, and `omni-quality nfa-summarize`.
1. Manually audit stratified clean/middle/tail bins before selecting thresholds.
1. Compare a size-matched random training subset with a size-matched cleaned subset using the same training recipe, step count, and held-out benchmarks.

The current V4 audit lane uses `C1Tech/whisper-base-fa` at its measured batch-1 setting because it has the best accessible greedy WER on the shared Persian benchmark. This remains an independent audit/comparator lane: its hypotheses never become training labels, while Scribe remains the accepted high-quality teacher/verifier lane. Pass `--tokenizer-model` and `--rejected-output` to `nfa-prepare` so NeMo token-case incompatibilities are retained with explicit status before alignment. CTC alignment validates whether the supplied text can be placed monotonically over the audio and exposes suspicious clip margins; it does not prove that every word is true.

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
- Final and best Omnilingual checkpoints referenced by `.fairseq2-assets` live under `models/omnilingual/checkpoints/`. `runs/` is for logs, metrics, and resumable training state while a run is active.
- Persian normalization follows the pinned NVIDIA surface: ZWNJ becomes a space, punctuation is normalized consistently on references and hypotheses, and a recipe change requires a fresh data export plus benchmark.
- Production YouTube VAD is Silero 6.2.1 ONNX on CPU, `conservative-v1`, threshold 0.5. The same shared postprocessor and per-clip provenance apply to every VAD adapter.
