# FluidAudio Integration Handoff

This project is still private. Do not push branches, open PRs, or publish
artifacts from this handoff without an explicit request.

## Current Evidence

- Private Swift/CoreML runtime exists in `swift/MossCoreMLFixture`.
- The shared 512-token explicit-cache path completed the first 20 LibriSpeech
  clean-test rows with WER `0.0158` and CER `0.00418`.
- Process-per-row eval:
  `artifacts/evals/librispeech-test-clean-swift-coreml-external-cache-512-20/summary.json`
  recorded 164.49s audio, 216.29s summed Swift model time, 0.76 RTFx, and
  1382.60s wall.
- Persistent Swift batch eval:
  `artifacts/evals/librispeech-test-clean-swift-coreml-external-cache-512-batch-20/summary.json`
  recorded the same WER/CER, 132.95s summed Swift model time, 1.24 RTFx, and
  691.58s wall.
- The persistent batch proves the required FluidAudio runtime property: compiled
  CoreML models must be loaded once and reused across utterances.
- Fixture-free runtime manifest eval:
  `artifacts/evals/librispeech-test-clean-swift-coreml-runtime-manifest-cache-512-batch-20/summary.json`
  used `runtime/moss_runtime_manifest.json`, completed 20/20 rows, and produced
  identical row IDs, normalized hypotheses, prompt lengths, generated-token
  counts, WER, and CER versus the prior persistent batch.
- Private FluidAudio scaffold patch:
  `patches/fluid-audio-moss-private-scaffold.patch` adds
  `Sources/FluidAudio/ASR/MOSS` plus `fluidaudiocli moss-transcribe` and
  `moss-benchmark`.
  It was applied uncommitted to `/Users/simonpeacocks/GitHub/FluidAudio`,
  built with `swift build -c release`, and smoke-tested on row
  `6930-75918-0001`.
- FluidAudio CLI smoke, same row and CoreML artifacts:
  `fluidaudiocli moss-transcribe ... --cpu-gpu --repeat 2` produced the same
  transcript on both runs, generated 47 tokens, stopped on EOS, and used prompt
  length 195 / 185 audio tokens. Cold model load was 78.95s; repeated
  transcribe processing times were 52.50s and 41.07s for 14.23s audio.
- FluidAudio 20-row benchmark:
  `artifacts/evals/fluid-audio-moss-benchmark-20/summary.json` completed 20/20
  rows with WER `0.0158`, CER `0.00418`, 164.49s audio, 710.41s full manager
  processing, and 0.23 RTFx. This matches the prior WER/CER exactly and proves
  the FluidAudio code shape, while confirming the speed blocker.
- 512-cache FluidAudio benchmark:
  `artifacts/evals/fluid-audio-moss-benchmark-cache512-20/summary.json`
  completed the same 20/20 rows with WER `0.0158`, CER `0.00418`, 164.49s
  audio, 237.57s full manager processing, 215.89s model timing, 21.68s host
  overhead, and 0.69 RTFx. The improvement comes from pairing the 512-token
  prefill package with a new 512-slot padded decoder step, avoiding the
  512-to-768 prefill K/V padding copy for this short-row gate.
- Cache-preset FluidAudio smoke:
  `artifacts/evals/fluid-audio-moss-benchmark-cache-preset-short512-offset1-limit1/summary.json`
  completed row `6930-75918-0001` through `--cache-preset short-512` with
  WER/CER `0.0`, 14.23s audio, 45.87s processing, 44.02s model timing, and
  0.31 RTFx. The paired overflow probe with `--max-tokens 400` failed before
  decoder execution with `requires cache length 712, but cache length is 512`.
- Bundle-manifest FluidAudio smoke:
  `runtime/moss_bundle_manifest.json` now carries package paths and cache
  presets. The private Mac CLI run under
  `artifacts/evals/fluid-audio-moss-benchmark-bundle-manifest-short512-offset1-limit1/summary.json`
  used `--bundle-manifest ... --cache-preset short-512` with no manual package,
  tokenizer, or runtime-manifest flags and completed row `6930-75918-0001`
  with WER/CER `0.0`, 14.23s audio, 19.41s processing, 18.03s model timing,
  and 0.73 RTFx.
- Local active bundle smoke:
  `scripts/build_fluid_audio_bundle.sh` created
  `/Users/simonpeacocks/GitHub/moss-mlx-conversion/bundles/moss-fluid-audio-coreml-active`
  with package-local manifest paths. The bundle contains `compiled`,
  `compiled_audio_30s`, `compiled_prefill_cache_512`,
  `compiled_step_padded`, `compiled_step_padded_512`,
  `moss_tokenizer.json`, `moss_runtime_manifest.json`, and
  `moss_bundle_manifest.json`. A private FluidAudio run with only
  `--model-dir <bundle> --cache-preset short-512` completed the same row with
  WER/CER `0.0`, 14.23s audio, 18.43s processing, 16.98s model timing, and
  0.77 RTFx.

## Relevant FluidAudio Shape

Reference checkout inspected and then modified privately on `home-mac`:

- Path: `/Users/simonpeacocks/GitHub/FluidAudio`
- Commit: `a95ec26 Validate downloaded model artifacts before caching (#740) (#741)`
- Private state: uncommitted MOSS scaffold only; no branch, commit, push, PR, or
  model publication.
- Closest runtime precedent:
  `Sources/FluidAudio/ASR/Cohere/CoherePipeline.swift`
- Closest model-name precedent:
  `Sources/FluidAudio/ModelNames.swift`, `ModelNames.CohereTranscribe`
- Closest docs:
  `Documentation/ASR/Cohere.md`, `Documentation/Models.md`

Cohere is the closest precedent because it is also an encoder-decoder ASR path
with an external KV-cache decoder. Parakeet is not close architecturally; it is
useful mainly for benchmark expectations and package/download style.

## Required FluidAudio Files

A real FluidAudio integration should be a new backend, not a small model-name
addition:

- `Sources/FluidAudio/ASR/MOSS/MossAsrConfig.swift`
- `Sources/FluidAudio/ASR/MOSS/MossModels.swift`
- `Sources/FluidAudio/ASR/MOSS/MossPipeline.swift`
- `Sources/FluidAudio/ASR/MOSS/MossTokenizer.swift`
- `Sources/FluidAudio/ASR/MOSS/MossMelFrontend.swift` or reuse shared Swift
  Whisper log-mel code if it is promoted to `Shared`.
- `Sources/FluidAudioCLI/Commands/ASR/MOSS/MossTranscribeCommand.swift`
- `Sources/FluidAudioCLI/Commands/ASR/MOSS/MossBenchmarkCommand.swift`
- `Documentation/ASR/MOSS.md`
- `Documentation/Models.md` row
- `Sources/FluidAudio/ModelNames.swift` entries for the model repo and required
  files.

## Required Model Bundle

The current private bundle needs these files:

- `moss_token_embedding.mlmodelc`
- `moss_audio_encoder_adapter_30s_padded.mlmodelc`
- `moss_decoder_prefill_cache_512.mlmodelc`
- `moss_decoder_step_padded_fixture.mlmodelc` for the 768-cache fallback.
- `moss_decoder_step_padded_512.mlmodelc` for short prompts that fit a
  512-token total prompt+decode window.
- `moss_decoder_prefill_cache_768.mlmodelc` exists experimentally and
  Torch-validates, but it is not currently usable under `cpu-gpu` in
  FluidAudio because CoreML/MPSGraph crashes before the first row.
- `moss_tokenizer.json`
- `runtime/moss_runtime_manifest.json`, or the same fields embedded in a model
  bundle manifest: prompt prefix/suffix token IDs, placeholder ID, hidden size,
  head dim, and RoPE theta. The CLI still passes EOS token, audio-frame limit,
  cache length, and prefill bucket length as runtime options.
- `runtime/moss_bundle_manifest.json`, currently passed with
  `--bundle-manifest`, maps the compiled package paths, tokenizer path,
  runtime manifest path, default cache preset, and named cache presets. Relative
  paths are resolved against `--model-dir`.

The private CLI currently exposes these cache shortcuts:

- `--cache-preset short-512`: 512-token padded prefill plus 512-cache decoder
  step. This is the fastest validated short-row path.
- `--cache-preset compat-768`: 512-token padded prefill plus 768-cache decoder
  step. This keeps the older compatibility path.
- `--cache-preset matched-768`: 768-token padded prefill plus 768-cache decoder
  step. This remains experimental because the current `cpu-gpu` CoreML runtime
  crashes in MPSGraph before row output.

The old `moss_swift_fixture_compact.json` remains acceptable for regression
fixtures but is no longer the production-shaped runtime config contract.

## Why This Is Not Just Model Registration

MOSS needs runtime logic that FluidAudio does not already have as a generic ASR
path:

- Whisper-compatible 128-bin mel frontend with 30-second static padding.
- Qwen chat-style prompt construction with audio placeholder replacement.
- Token embedding, audio embedding, and host-side merged-embedding assembly.
- Padded prefill with `last_token_mask`.
- External-cache decode step that passes and updates full padded key/value
  tensors. The private scaffold defaults to the 768-cache package, but the
  512-cache package is faster when the prompt plus generated tokens fit.
- Qwen ByteLevel tokenizer decode and special-token skipping.

Adding `ModelNames.MOSS` without a manager would only download files; it would
not make them runnable.

## Runtime Caveats

- The CoreML State API fused decoder remains unstable at row 3 prompt length
  313. The production path currently uses explicit cache tensors.
- The explicit-cache path is correct but memory-bandwidth heavy because every
  generated token copies full padded KV arrays through CoreML.
- The private FluidAudio scaffold proves the code can live inside FluidAudio
  and reuse loaded models across calls. The initial 768-cache benchmark was
  only 0.23x RTFx by full manager processing time. The 512-cache benchmark
  raises that gate to 0.69x RTFx, but this is still much slower than
  FluidAudio's fast CTC/TDT-style ASR paths and still pays explicit-cache KV
  movement each generated token.
- A 512-token prefill bucket covers the first 20 clean-test rows. Full
  LibriSpeech needs prompt/decode-length buckets or a larger padded prefill and
  step package.
- A matched 768-token prefill package was exported and compiled, but the
  private FluidAudio `cpu-gpu` probe crashed in MPSGraph with
  `shape.count = 0 != strides.count = 2`; CPU-only did not produce a row before
  the probe was stopped. Treat matched 768 as blocked pending CoreML runtime
  debugging.
- The audio path is single-window only. Long audio needs FluidAudio-style
  chunking and stitching before it is a general user-facing ASR backend.
- Use `cpu-gpu` for the current padded audio package. Default `.all` routed to
  ANE on `home-mac` and failed for this package.

## Benchmark Position

The current persistent MOSS path is useful as a quality/reference backend, not
a Parakeet-speed backend:

- MOSS private persistent batch, first 20 clean rows: WER `1.58%`, RTFx `1.24`.
- MOSS private FluidAudio scaffold, first 20 clean rows with 512 cache: WER
  `1.58%`, RTFx `0.69`.
- FluidAudio Cohere docs report full LibriSpeech test-clean around WER `1.77%`,
  total-audio/compute RTFx `1.72`.
- Local FluidAudio Parakeet v3 full test-clean result recorded in this project:
  WER `2.63%`, RTFx `39.61`.

MOSS quality is competitive on this small window, but the autoregressive Qwen
decoder makes the runtime fundamentally different from Parakeet.

## Next Concrete Step

The first private FluidAudio scaffold exists as a patch and an uncommitted Mac
checkout change. Keep it private and unpushed.

1. Add automatic prompt/decode bucket selection beyond the current manual
   `short-512` path.
2. Profile and reduce explicit-cache KV movement; this is the current
   production-speed blocker.
3. Add long-audio chunking and stitching beyond the single 30-second window.
4. Add Hugging Face download/model-name metadata only after a private model
   bundle exists.
