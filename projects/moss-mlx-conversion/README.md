# MOSS MLX Conversion

This project tracks a full MLX conversion plan for
[`OpenMOSS-Team/MOSS-Transcribe-preview-2B`](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-preview-2B).

## Documents

Use the docs this way:

- [docs/PLAN.md](docs/PLAN.md): durable architecture and upstream-reference
  plan. Update when the strategy changes.
- [docs/PROGRESS.md](docs/PROGRESS.md): current state, verified commands,
  latest measurements, and next steps. Update during active work.
- [CHANGELOG.md](CHANGELOG.md): terse history of completed milestones.

Start with [docs/PLAN.md](docs/PLAN.md). The plan is written to stay close to the
existing `mlx-audio` and FluidInference `mobius` conversion patterns while
calling out where MOSS differs from the already-ported Qwen3/Cohere-style
speech models.

Current execution state is tracked in [docs/PROGRESS.md](docs/PROGRESS.md).

Current short version:

- Linux/PyTorch reference and processor parity are complete for the pinned MOSS
  snapshot.
- BF16 MLX-layout weights have been written under
  `artifacts/mlx/MOSS-Transcribe-preview-2B-bf16/`.
- `moss-mlx-smoke` passed on Apple Silicon for the LibriSpeech fixture. It
  loaded the converted weights, built audio prompt embeddings, matched the first
  5 generated token IDs, and matched the PyTorch reference transcript exactly.
- The Mac working copy is organized at
  `/Users/simonpeacocks/GitHub/moss-mlx-conversion`.
- `moss-streaming-eval` streams LibriSpeech rows and audio assets from
  Hugging Face without writing audio files, then reports WER/CER with `jiwer`.
  The first 20 `openslr/librispeech_asr` clean-test rows ran at 1.58% WER and
  0.65 RTF on Apple Silicon.

## Layout

```text
src/moss_mlx_conversion/
  reference/    PyTorch/HF reference capture and processor parity
  conversion/   safetensor inspection and BF16 MLX conversion
  runtime/      MLX smoke transcription and streamed eval
  model/        MLX audio encoder, adapter, and MOSS wrapper
  config.py     shared MOSS/Qwen/audio config parsing
  processor.py  local MOSS processor implementation
```
