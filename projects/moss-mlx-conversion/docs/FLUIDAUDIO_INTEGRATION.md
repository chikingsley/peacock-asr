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

## Relevant FluidAudio Shape

Read-only reference checkout inspected on `home-mac`:

- Path: `/Users/simonpeacocks/GitHub/FluidAudio`
- Commit: `a95ec26 Validate downloaded model artifacts before caching (#740) (#741)`
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
- `Sources/FluidAudioCLI/Commands/ASR/MossTranscribeCommand.swift`
- `Sources/FluidAudioCLI/Commands/ASR/MossBenchmark.swift`
- `Documentation/ASR/MOSS.md`
- `Documentation/Models.md` row
- `Sources/FluidAudio/ModelNames.swift` entries for the model repo and required
  files.

## Required Model Bundle

The current private bundle needs these files:

- `moss_token_embedding.mlmodelc`
- `moss_audio_encoder_adapter_30s_padded.mlmodelc`
- `moss_decoder_prefill_cache_512.mlmodelc`
- `moss_decoder_step_padded_fixture.mlmodelc`
- `moss_tokenizer.json`
- A small config/manifest file replacing the current compact fixture JSON:
  prompt prefix/suffix token IDs, placeholder ID, hidden size, head dim,
  RoPE theta, EOS token, audio-token stride, max audio frames, cache length,
  prefill bucket length.

The current `moss_swift_fixture_compact.json` is acceptable for the private
fixture runner but should not be the public runtime config contract.

## Why This Is Not Just Model Registration

MOSS needs runtime logic that FluidAudio does not already have as a generic ASR
path:

- Whisper-compatible 128-bin mel frontend with 30-second static padding.
- Qwen chat-style prompt construction with audio placeholder replacement.
- Token embedding, audio embedding, and host-side merged-embedding assembly.
- Padded prefill with `last_token_mask`.
- External-cache decode step that passes and updates full
  `[28, 1, 8, 768, 128]` key/value tensors.
- Qwen ByteLevel tokenizer decode and special-token skipping.

Adding `ModelNames.MOSS` without a manager would only download files; it would
not make them runnable.

## Runtime Caveats

- The CoreML State API fused decoder remains unstable at row 3 prompt length
  313. The production path currently uses explicit cache tensors.
- The explicit-cache path is correct but memory-bandwidth heavy because every
  generated token copies full padded KV arrays through CoreML.
- A 512-token prefill bucket covers the first 20 clean-test rows. Full
  LibriSpeech needs either prompt-length buckets or a larger padded prefill.
- The audio path is single-window only. Long audio needs FluidAudio-style
  chunking and stitching before it is a general user-facing ASR backend.
- Use `cpu-gpu` for the current padded audio package. Default `.all` routed to
  ANE on `home-mac` and failed for this package.

## Benchmark Position

The current persistent MOSS path is useful as a quality/reference backend, not
a Parakeet-speed backend:

- MOSS private persistent batch, first 20 clean rows: WER `1.58%`, RTFx `1.24`.
- FluidAudio Cohere docs report full LibriSpeech test-clean around WER `1.77%`,
  total-audio/compute RTFx `1.72`.
- Local FluidAudio Parakeet v3 full test-clean result recorded in this project:
  WER `2.63%`, RTFx `39.61`.

MOSS quality is competitive on this small window, but the autoregressive Qwen
decoder makes the runtime fundamentally different from Parakeet.

## Next Concrete Step

If we decide to actually modify FluidAudio, start with a private local branch in
`/Users/simonpeacocks/GitHub/FluidAudio` and keep it unpushed:

1. Add `MossAsrConfig`, `MossModels`, and `MossPipeline` by lifting the proven
   Swift runtime out of `MossCoreMLFixture`.
2. Replace fixture JSON dependency with a small runtime manifest.
3. Add manual model loading first; only add Hugging Face download metadata after
   the local model directory path works.
4. Add a CLI `moss-transcribe` command that accepts `--model-dir` and one WAV.
5. Add a `moss-benchmark` command for LibriSpeech rows, reusing the same
   normalization/WER behavior as the private harness.
6. Re-run the 20-row gate through FluidAudio CLI and require matching WER/CER
   and a single-process wall profile.
