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
  prefill/cache shapes, and parity gates. This is the private CoreML workbench,
  not a public FluidAudio branch.
- Private CoreML fixture exports now cover token embedding, audio
  encoder+adapter, full decoder prefill, fixed append-cache decoder step,
  padded 768-slot cache-external decoder step, and a fused stateful decoder
  with CoreML State API KV buffers. All exported packages validated against
  PyTorch fixture tensors on `home-mac` and compiled with
  `xcrun coremlcompiler`.
- The fused stateful decoder is the current Mobius-style single-model decoder:
  one CoreML model handles prompt prefill and the next decode call with the
  same internal state, ranking fixture tokens `4197` then `1059`. It is proven
  on short prompts, but a longer non-fixture row exposed a stateful decode
  stability bug.
- `run_stateful_fixture_pipeline.py` now wires the exported token embedding,
  audio encoder+adapter, audio-mask merge, Qwen3 RoPE/masks, and stateful
  decoder in one CoreML process on `home-mac`. The component-merged fixture
  ranks `4197` then `1059`; total measured time for the one fixture was
  21.32s, with 20.61s in decoder prefill and 0.226s in the first decode step.
- A private Swift/CoreML fixture runner now loads the compiled `.mlmodelc`
  bundles with `MLModel`, reuses `MLState`, and greedy-decodes the fixture.
  It also loads the Qwen ByteLevel tokenizer JSON and decodes generated IDs.
  The first 5 generated IDs and text match exactly; the 52-token run has a
  comma-only drift after token 10 and normalized WER/CER `0.0`.
- The Swift runner now supports a compact prompt fixture and constructs the
  fixed English MOSS prompt locally from prefix/suffix token constants plus the
  audio placeholder count. The compact path reports `prompt_source=compact`,
  matches the 5-token fixture exactly, and has the same normalized-WER-zero
  52-token punctuation drift.
- The Swift runner now also has a fixture-level `--audio` path. It reads the
  LibriSpeech WAV, computes MOSS/Whisper `[128, 1484]` log-mel features in
  Swift, reports mel max/mean diff `0.003906` / `0.000515` against the saved
  fixture tensor, matches the 5-token output exactly, and keeps normalized
  WER/CER `0.0` on the 52-token run.
- The first production-shaped CoreML audio package now accepts padded
  30-second mel input `[128, 3000]` with real seqlens and masked invalid audio
  tokens. The Swift `--audio --audio-max-frames 3000` path matches the fixture
  52-token generated IDs/text exactly under `--compute-units cpu-gpu`; default
  `.all` dispatch currently fails this audio package on ANE.
- The Swift/CoreML runner now has reference-text scoring and EOS stop. A
  non-fixture LibriSpeech clean-test row (`6930-75918-0001`, 14.23s) ran
  through the padded audio path, generated 47 tokens, stopped on `151645`, and
  matched the reference after normalization with WER/CER `0.0`.
- `moss-swift-coreml-eval` now runs a repeatable Swift/CoreML batch eval over
  streamed Hugging Face rows by materializing short WAV/reference pairs and
  calling the Swift runner. The first two-row clean-test batch scored WER/CER
  `0.0` with 1.43 RTFx on summed Swift model time.
- The 20-row Swift/CoreML gate exposed the stateful decoder roadblock: rows
  0-2 completed with WER/CER `0.0`, but row 3 has prompt length 313 and the
  first stateful decode step produced no finite logits. A new explicit-cache
  decoder path bypasses that failure: fixed-length prefill-cache packages for
  prompt lengths 195 and 313 plus the padded step decoder scored WER/CER
  `0.0` on rows 1 and 3. This is still a correctness bridge, not a final
  FluidAudio backend, because prefill is fixed-length and decode moves full
  padded KV arrays per token.
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
