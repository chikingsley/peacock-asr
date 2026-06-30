# MOSS MLX Conversion

This project tracks a full MLX conversion plan for
[`OpenMOSS-Team/MOSS-Transcribe-preview-2B`](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-preview-2B).

## Documents

Use the docs this way:

- [docs/PLAN.md](docs/PLAN.md): durable architecture and upstream-reference
  plan. Update when the strategy changes.
- [docs/PROGRESS.md](docs/PROGRESS.md): current state, verified commands,
  latest measurements, and next steps. Update during active work.
- [docs/COREML_MOBIUS.md](docs/COREML_MOBIUS.md): private CoreML/Mobius track,
  component split, static shapes, and validation gates.
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
- The Mac working copy at `/Users/simonpeacocks/GitHub/moss-mlx-conversion`
  is the active Apple Silicon/CoreML workbench; retained outputs are copied
  back under ignored local `artifacts/coreml/`.
- `moss-streaming-eval` streams LibriSpeech rows and audio assets from
  Hugging Face without writing audio files, then reports WER/CER with `jiwer`.
  The paired 100-row clean-test baseline ran at 1.80% WER / 1.61 RTFx on MLX
  BF16 Apple Silicon, versus 2.01% WER / 19.03 RTFx on PyTorch BF16 with the
  RTX 5070.
- Gated real-weight tests now cover MLX weight load, fixture transcription, and
  a one-row streamed eval when explicitly enabled on Apple Silicon.
- Local backend shape now exposes `MossTranscribeBackend.generate(...)` and an
  `STTOutput` contract for later `mlx-audio` integration work.
- Quantized private candidates were tested. The strongest current candidate is
  `text-decoder-4bit-g64`: 2.81 GB weights and 2.48 RTFx on the first 20
  LibriSpeech clean-test rows, with no BF16 WER regression on that slice. BF16,
  `text-decoder-4bit-g64`, and `all-4bit-g64` weights are retained locally;
  weaker 8-bit candidates keep reports/manifests only.
- Full LibriSpeech benchmarking was stopped after Parakeet v3 completed and
  MOSS partials confirmed the architecture is not FluidAudio-speed in MLX.
  Current strategic read: use MOSS as an open-weights teacher/reference unless
  a separate CoreML/ANE decoder experiment is explicitly scoped.
- `moss-coreml-plan` now writes a private Mobius-style CoreML conversion
  contract under `artifacts/coreml/`, including component boundaries, fixed
  prefill/cache shapes, and parity gates. This is planning/export groundwork,
  not a CoreML model yet.
- Private CoreML fixture exports now cover token embedding, audio
  encoder+adapter, full decoder prefill, and a full one-token cache-external
  decoder step. All exported packages validated against PyTorch fixture
  tensors on `home-mac` and compiled with `xcrun coremlcompiler`.
- No public upload/branch/PR action has been taken.

## Layout

```text
src/moss_mlx_conversion/
  backend/      Local STTOutput/backend shape and serial serving adapter
  reference/    PyTorch/HF reference capture, paired eval, and processor parity
  conversion/   safetensor inspection, BF16 conversion, quantize/package CLIs
  runtime/      MLX smoke transcription, quantized loading, and streamed eval
  model/        MLX audio encoder, adapter, and MOSS wrapper
  coreml/       private CoreML/Mobius planning and export contract tools
  config.py     shared MOSS/Qwen/audio config parsing
  processor.py  local MOSS processor implementation
```

```text
coreml/
  README.md     private CoreML workbench notes for future export scripts
```
