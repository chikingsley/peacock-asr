# MOSS CoreML Workbench

Private CoreML/Mobius-style workbench for MOSS-Transcribe. This directory is for
conversion scripts and local notes only; it is not a public FluidInference fork.

Generate the current component contract:

```bash
uv run --project projects/moss-mlx-conversion --locked moss-coreml-plan \
  --output projects/moss-mlx-conversion/artifacts/coreml/moss-coreml-plan.json
```

The generated JSON is the source of truth for the first CoreML export pass:

- component names and inputs/outputs
- static prompt, audio, and cache shapes
- MOSS-specific deltas from Qwen3-ASR and Cohere
- parity gates before benchmark work

The current export scripts are fixture-first probes. They intentionally validate
one known LibriSpeech fixture before generalizing shapes or building a runtime
loop.

## Component Probes

The first probe is the token embedding component. It proves CoreMLTools,
`.mlpackage` writing, and CoreML prediction on a MOSS tensor before touching the
decoder.

Extract a smaller embedding-only safetensors file from the local BF16 MLX
artifact:

```bash
uv run --project projects/moss-mlx-conversion \
  projects/moss-mlx-conversion/coreml/export_token_embedding.py \
  --weights projects/moss-mlx-conversion/artifacts/mlx/MOSS-Transcribe-preview-2B-bf16/weights.safetensors \
  --extract-only
```

Export and validate on macOS:

```bash
uv run --project projects/moss-mlx-conversion/coreml \
  projects/moss-mlx-conversion/coreml/export_token_embedding.py \
  --weights projects/moss-mlx-conversion/artifacts/coreml/moss-token-embedding-fp16.safetensors \
  --validate-predict \
  --overwrite
```

The later probes use the saved PyTorch BF16 fixture tensors:

```bash
uv run --project projects/moss-mlx-conversion/coreml --locked \
  projects/moss-mlx-conversion/coreml/export_audio_encoder_adapter.py \
  --trace-dtype fp32 \
  --compute-precision float16 \
  --wrapper static-fixture \
  --validate-predict \
  --overwrite
```

The production-shaped audio encoder probe pads the mel input to the 30-second
contract `[128, 3000]`, keeps the real `audio_data_seqlens`, masks invalid
audio-token positions inside the encoder attention, and returns the fixed
maximum `[390, 2048]` audio-embedding tensor:

```bash
uv run --project projects/moss-mlx-conversion/coreml --locked \
  projects/moss-mlx-conversion/coreml/export_audio_encoder_adapter.py \
  --wrapper static-padded \
  --frames 3000 \
  --trace-dtype fp32 \
  --compute-precision float16 \
  --validate-predict \
  --overwrite \
  --package-name moss_audio_encoder_adapter_30s_padded.mlpackage
```

```bash
uv run --project projects/moss-mlx-conversion/coreml --locked \
  projects/moss-mlx-conversion/coreml/export_decoder_prefill.py \
  --num-layers 28 \
  --trace-dtype fp32 \
  --compute-precision float16 \
  --validate-predict \
  --overwrite
```

```bash
uv run --project projects/moss-mlx-conversion/coreml --locked \
  projects/moss-mlx-conversion/coreml/export_decoder_step.py \
  --num-layers 28 \
  --trace-dtype fp32 \
  --compute-precision float16 \
  --validate-predict \
  --overwrite
```

The production-shaped cache-external step uses the same exporter with a fixed
768-token padded cache:

```bash
uv run --project projects/moss-mlx-conversion/coreml --locked \
  projects/moss-mlx-conversion/coreml/export_decoder_step.py \
  --cache-mode padded \
  --cache-len 768 \
  --num-layers 28 \
  --trace-dtype fp32 \
  --compute-precision float16 \
  --validate-predict \
  --overwrite \
  --package-name moss_decoder_step_padded_fixture.mlpackage
```

The Mobius-style stateful fused decoder uses CoreML State API buffers and
requires macOS 15+:

```bash
uv run --project projects/moss-mlx-conversion/coreml --locked \
  projects/moss-mlx-conversion/coreml/export_decoder_stateful.py \
  --cache-len 768 \
  --num-layers 28 \
  --trace-dtype fp32 \
  --compute-precision float16 \
  --validate-predict \
  --overwrite \
  --package-name moss_decoder_stateful_fused.mlpackage
```

The explicit-cache prefill exporter returns logits plus KV tensors. It supports
exact prompt-length packages and a padded package with `last_token_mask`, so
one compiled model can cover multiple prompt lengths inside the same bucket:

```bash
uv run --project projects/moss-mlx-conversion/coreml --locked \
  projects/moss-mlx-conversion/coreml/export_decoder_prefill_cache.py \
  --seq-len 512 \
  --mode padded \
  --validation-prompt-len 313 \
  --trace-dtype fp32 \
  --compute-precision float16 \
  --overwrite \
  --package-name moss_decoder_prefill_cache_512.mlpackage
```

Run the integrated CoreML fixture pipeline after all three runtime packages
exist:

```bash
uv run --project projects/moss-mlx-conversion/coreml --locked \
  projects/moss-mlx-conversion/coreml/run_stateful_fixture_pipeline.py \
  --packages-dir projects/moss-mlx-conversion/coreml/build \
  --merged-source coreml-components \
  --output projects/moss-mlx-conversion/coreml/build/moss_coreml_stateful_fixture_pipeline.json
```

The isolation mode feeds the decoder the saved reference merged embeddings
while still measuring the token/audio component boundary:

```bash
uv run --project projects/moss-mlx-conversion/coreml --locked \
  projects/moss-mlx-conversion/coreml/run_stateful_fixture_pipeline.py \
  --packages-dir projects/moss-mlx-conversion/coreml/build \
  --merged-source reference \
  --output projects/moss-mlx-conversion/coreml/build/moss_coreml_stateful_fixture_pipeline_reference_merged.json
```

Export the Swift-readable fixture JSON:

```bash
uv run --project projects/moss-mlx-conversion/coreml --locked \
  projects/moss-mlx-conversion/coreml/export_swift_fixture.py \
  --output projects/moss-mlx-conversion/artifacts/coreml/moss_swift_fixture.json
```

Export a compact fixture that omits the full prompt arrays and keeps only the
MOSS template prefix/suffix plus audio placeholder count:

```bash
uv run --project projects/moss-mlx-conversion/coreml --locked \
  projects/moss-mlx-conversion/coreml/export_swift_fixture.py \
  --compact-only \
  --output projects/moss-mlx-conversion/artifacts/coreml/moss_swift_fixture_compact.json
```

Stage the tokenizer next to the CoreML fixture artifacts:

```bash
cp projects/moss-mlx-conversion/artifacts/mlx/MOSS-Transcribe-preview-2B-bf16/tokenizer.json \
  projects/moss-mlx-conversion/artifacts/coreml/moss_tokenizer.json
```

Compile on macOS with:

```bash
xcrun coremlcompiler compile <package.mlpackage> <output-dir>
```

The retained decoder artifacts now cover four stages:

- `moss_decoder_step_fixture`: fixed append-cache fixture transition
  `past_len=203 -> 204`.
- `moss_decoder_step_padded_fixture`: fixed 768-slot external-cache contract
  with host-provided update mask, attention mask, and RoPE tensors.
- `moss_decoder_step_padded_512`: matched 512-slot external-cache contract for
  short prompts/decodes that fit the shared 512-token prefill bucket.
- `moss_decoder_prefill_cache_<seq-len>`: exact or padded prefill that returns
  explicit KV tensors for the padded step decoder. The validated 512-token
  padded package uses `last_token_mask` to select the real prompt end. The
  768-token padded package also Torch-validates, but currently crashes in the
  private FluidAudio `cpu-gpu` runtime before row output.
- `moss_decoder_stateful_fused`: one fused decoder package with final norm,
  tied LM head projection, and 56 CoreML State API KV tensors. It validates
  `prefill -> one decode step` with a single CoreML state object, but it still
  needs a Swift runtime loop before it is a FluidAudio backend.

Integrated fixture result on `home-mac`:

- Component path: CoreML token embeddings + CoreML audio embeddings merged by
  `audio_input_mask`, then stateful decoder. Top-1 tokens match `4197` then
  `1059`.
- Reference-merged isolation path: exact saved merged embeddings into the same
  stateful decoder. Top-1 tokens also match `4197` then `1059`.
- Component audio embeddings differ from the saved BF16 reference by max/mean
  `0.002686` / `0.000354`; the full merged prompt differs by max/mean
  `0.002686` / `0.000337`.
- The runner's raw `prefill_logits_vs_reference` compares against the saved HF
  reference logits, not the exporter manifest's custom static-decoder parity
  target. Treat the runner as a runtime-contract proof first.

## Swift Fixture Runner

The private Swift package lives at `swift/MossCoreMLFixture`. It is a fixture
runner, not a FluidAudio source tree patch.

Build on macOS:

```bash
swift build --package-path swift/MossCoreMLFixture -c release
```

Run the 5-token greedy fixture against compiled `.mlmodelc` bundles:

```bash
swift run --package-path swift/MossCoreMLFixture -c release moss-coreml-fixture \
  --packages-dir coreml/build \
  --fixture artifacts/coreml/moss_swift_fixture.json \
  --output coreml/build/moss_swift_coreml_fixture_5tok.json
```

Run the saved 52-token fixture:

```bash
swift run --package-path swift/MossCoreMLFixture -c release moss-coreml-fixture \
  --packages-dir coreml/build \
  --fixture artifacts/coreml/moss_swift_fixture.json \
  --max-new-tokens 52 \
  --output coreml/build/moss_swift_coreml_fixture_52tok.json
```

Run the compact prompt-builder fixture:

```bash
swift run --package-path swift/MossCoreMLFixture -c release moss-coreml-fixture \
  --packages-dir coreml/build \
  --fixture artifacts/coreml/moss_swift_fixture_compact.json \
  --max-new-tokens 52 \
  --output coreml/build/moss_swift_coreml_fixture_compact_52tok.json
```

Run the fixture WAV through the Swift Whisper log-mel frontend, then the same
CoreML path:

```bash
swift run --package-path swift/MossCoreMLFixture -c release moss-coreml-fixture \
  --packages-dir coreml/build \
  --fixture artifacts/coreml/moss_swift_fixture_compact.json \
  --audio artifacts/cache/fixtures/librosa-libri1-16k.wav \
  --compare-fixture-audio \
  --max-new-tokens 52 \
  --output coreml/build/moss_swift_coreml_audio_frontend_52tok.json
```

Run the same WAV through the 30-second padded audio package:

```bash
xcrun coremlcompiler compile \
  coreml/build/moss_audio_encoder_adapter_30s_padded.mlpackage \
  coreml/build/compiled_audio_30s

swift run --package-path swift/MossCoreMLFixture -c release moss-coreml-fixture \
  --packages-dir coreml/build \
  --fixture artifacts/coreml/moss_swift_fixture_compact.json \
  --audio artifacts/cache/fixtures/librosa-libri1-16k.wav \
  --audio-max-frames 3000 \
  --compare-fixture-audio \
  --audio-package compiled_audio_30s/moss_audio_encoder_adapter_30s_padded.mlmodelc \
  --compute-units cpu-gpu \
  --max-new-tokens 52 \
  --output coreml/build/moss_swift_coreml_audio_30s_padded_cpu_gpu_52tok.json
```

Run a non-fixture LibriSpeech clean-test row through the same padded path with
reference-text scoring and EOS stop:

```bash
swift run --package-path swift/MossCoreMLFixture -c release moss-coreml-fixture \
  --packages-dir coreml/build \
  --fixture artifacts/coreml/moss_swift_fixture_compact.json \
  --audio artifacts/coreml/nonfixture_librispeech_clean_row1/audio.wav \
  --audio-max-frames 3000 \
  --audio-package compiled_audio_30s/moss_audio_encoder_adapter_30s_padded.mlmodelc \
  --compute-units cpu-gpu \
  --max-new-tokens 160 \
  --reference-text-file artifacts/coreml/nonfixture_librispeech_clean_row1/reference.txt \
  --output coreml/build/moss_swift_coreml_audio_30s_padded_cpu_gpu_librispeech_row1.json
```

Run a repeatable small batch from Hugging Face rows:

```bash
uv run --extra mac --locked moss-swift-coreml-eval \
  --offset 1 \
  --limit 2 \
  --page-size 2 \
  --max-new-tokens 160 \
  --output-dir artifacts/evals/librispeech-test-clean-swift-coreml-2
```

Run the row-3 explicit-cache smoke that bypasses the stateful long-prompt
decode failure:

```bash
swift run --package-path swift/MossCoreMLFixture -c release moss-coreml-fixture \
  --packages-dir coreml/build \
  --fixture artifacts/coreml/moss_swift_fixture_compact.json \
  --audio artifacts/evals/librispeech-test-clean-swift-coreml-20/audio/000003-6930-75918-0003.wav \
  --audio-max-frames 3000 \
  --audio-package compiled_audio_30s/moss_audio_encoder_adapter_30s_padded.mlmodelc \
  --prefill-cache-package compiled_prefill_cache_512/moss_decoder_prefill_cache_512.mlmodelc \
  --prefill-cache-seq-len 512 \
  --step-package compiled_step_padded/moss_decoder_step_padded_fixture.mlmodelc \
  --cache-len 768 \
  --compute-units cpu-gpu \
  --max-new-tokens 160 \
  --reference-text-file artifacts/evals/librispeech-test-clean-swift-coreml-20/reference/000003-6930-75918-0003.txt \
  --output coreml/build/moss_swift_coreml_external_cache_cpu_gpu_librispeech_row3.json
```

Swift result on `home-mac`:

- The 5-token greedy run exactly matched `[4197, 1059, 4158, 6177, 323]` and
  decoded to `with her white paint and`.
- The 52-token run matched the first 10 IDs, then inserted a comma token after
  `smokestack`; Swift decoded raw WER/CER are `0.0278` / `0.00442`, and
  normalized WER/CER are both `0.0`.
- The tokenizer-enabled 52-token run measured 24.95s total: 16.88s decoder
  prefill, 7.63s decoder decode calls, 0.27s decode token embeddings, 0.13s
  audio encoder+adapter.
- The compact fixture path reports `prompt_source=compact`, proving Swift is
  building the fixed MOSS prompt from `[151644, 872, 198, 151669]`,
  193 audio placeholder tokens, and `[151670, 151645, 198, 151644, 77091,
  198]` instead of consuming serialized `input_ids` / `audio_input_mask`.
  Compact 5-token output is exact; compact 52-token output has the same
  comma-only normalized-WER-zero drift.
- The Swift audio frontend path reports `prompt_source=compact_audio`, computes
  `[128, 1484]` Whisper log-mel features from the source WAV, and matches the
  saved PyTorch/Whisper fixture mel with max/mean abs diff `0.003906` /
  `0.000515`. The 5-token output is exact. The 52-token output keeps
  normalized WER/CER `0.0`, with raw WER/CER `0.0556` / `0.00885` from
  punctuation-only drift.
- The 30-second padded audio package validates against the fixture prefix with
  CoreML-vs-BF16 max/mean diff `0.003738` / `0.000462`. Through Swift with
  `--compute-units cpu-gpu`, the 5-token and 52-token runs match the expected
  generated IDs/text exactly. The 52-token run measured 7.07s total: 0.14s
  audio frontend, 1.30s audio encoder+adapter, 0.75s decoder prefill, and
  4.70s decoder decode calls.
- The Swift runner now supports reference-text scoring and EOS stop. The first
  non-fixture LibriSpeech clean-test row (`6930-75918-0001`, 14.23s) produced
  47 generated tokens, stopped on token `151645`, and matched the reference
  after normalization with WER/CER `0.0`. Total measured time was 8.25s.
- The first `moss-swift-coreml-eval` batch on two clean-test rows completed
  with WER/CER `0.0`, total audio 19.25s, summed Swift model time 13.43s, and
  RTFx 1.43. Wall time was 42.99s because the harness still launches a Swift
  process per row.
- The attempted 20-row batch completed rows 0-2 with WER/CER `0.0`
  (`22.76s` audio, `14.32s` summed Swift model time), then failed on row 3.
  Row 3 prefill succeeds with prompt length 313 and first token `from`, but
  the first stateful decode step returns no finite logits under both
  `cpu-gpu` and `cpu-only`. This is the current CoreML decoder stability
  boundary.
- The explicit-cache path bypasses the row-3 stateful failure. Row 1
  (`prompt_len=195`) and row 3 (`prompt_len=313`) both score normalized
  WER/CER `0.0` with the shared `compiled_prefill_cache_512` plus
  `compiled_step_padded`. Row 3 generated 77 tokens, stopped on EOS, and
  measured 13.80s model time for 23.32s audio (RTFx 1.69). This is a stronger
  correctness bridge, not the final FluidAudio backend: decode still moves
  full padded KV arrays through CoreML each token, and longer prompts need a
  larger or bucketed prefill package.
- The same shared 512-token explicit-cache path completed the first 20
  LibriSpeech clean-test rows with WER `0.0158`, CER `0.00418`, 164.49s audio,
  216.29s summed Swift model time, and 0.76 RTFx. Nonzero normalized WER rows
  were row 4 (`opened for them`), row 15 possessive normalization, row 17
  `Ralph` vs `Raoul`, and row 19 `moon beams` vs `moonbeams`. Wall time was
  1382.60s because the current batch harness launches a new Swift process for
  every row.
- `moss-swift-coreml-eval --swift-batch` now writes a JSONL manifest and calls
  the Swift runner once, keeping compiled CoreML models loaded across rows.
  The persistent 20-row run preserved WER `0.0158` / CER `0.00418`, reduced
  summed Swift model time to 132.95s, improved model-time RTFx to 1.24, and
  cut wall time to 691.58s. This removes process-per-row startup as the main
  issue; the remaining runtime cost is per-token decoder work and full padded
  KV tensor movement.
- The Swift runner now accepts `--runtime-manifest
  runtime/moss_runtime_manifest.json` for production-shaped prompt/model
  constants. The compact fixture JSON remains useful for regression tests, but
  it is no longer the runtime config contract. The 20-row manifest batch
  matched the prior persistent batch row-for-row on normalized hypotheses,
  prompt lengths, generated-token counts, WER, and CER; it measured RTFx 0.75
  and wall time 813.11s, so use it as a correctness/package gate rather than a
  performance comparison.
- `patches/fluid-audio-moss-private-scaffold.patch` ports the same
  external-cache path into a private FluidAudio `ASR/MOSS` scaffold with
  manual model-dir loading plus `fluidaudiocli moss-transcribe` and
  `moss-benchmark`. It builds on `home-mac`; the FluidAudio 20-row gate matches
  WER `0.0158` / CER `0.00418`, but measures only 0.23 RTFx by full manager
  processing time.
- The matched 512-cache decoder-step package can be selected with
  `--step-package compiled_step_padded_512/moss_decoder_step_padded_512.mlmodelc`
  and `--cache-len 512`, or through the private FluidAudio CLI shortcut
  `--cache-preset short-512`. On the same 20 rows it preserves WER/CER and
  improves full FluidAudio manager RTFx to 0.69 by avoiding the 512-to-768
  prefill cache padding copy. The shortcut also has an early cache-capacity
  guard, proven by an over-budget row-3 probe. It is a bucketed short-row path,
  not the general runtime.
- The tracked `runtime/moss_bundle_manifest.json` now lets the private
  FluidAudio CLI resolve package paths, tokenizer path, runtime manifest path,
  and cache presets through `--bundle-manifest`. A one-row `short-512` smoke
  with no manual package/tokenizer/runtime flags completed with WER/CER `0.0`
  and 0.73 RTFx.
- `scripts/build_fluid_audio_bundle.sh` builds the active local bundle under
  `bundles/moss-fluid-audio-coreml-active` with package-local manifest paths.
  On `home-mac`, that bundle runs through private FluidAudio as a plain
  `--model-dir` and completed the same row with WER/CER `0.0` and 0.77 RTFx.
- The matched 768-token prefill package can compile and Torch-validates with
  zero diff at prompt length 313, but it is not an active runtime path:
  `cpu-gpu` FluidAudio execution crashes in MPSGraph before the first row, and
  a CPU-only one-row probe did not produce output before being stopped.
- The same padded audio package failed under default `.all` compute-unit
  dispatch with an ANE inference error. Use `--compute-units cpu-gpu` for this
  package until compute placement is profiled more carefully.
- Still missing for a real FluidAudio backend: model store/download layout, a
  FluidAudio-style `ASR/MOSS` manager, a non-fixture benchmark harness, and
  long-audio chunking beyond the single 30-second static window.
