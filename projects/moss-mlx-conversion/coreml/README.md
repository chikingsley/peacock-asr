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

The retained decoder artifacts now cover three stages:

- `moss_decoder_step_fixture`: fixed append-cache fixture transition
  `past_len=203 -> 204`.
- `moss_decoder_step_padded_fixture`: fixed 768-slot external-cache contract
  with host-provided update mask, attention mask, and RoPE tensors.
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
- The same padded audio package failed under default `.all` compute-unit
  dispatch with an ANE inference error. Use `--compute-units cpu-gpu` for this
  package until compute placement is profiled more carefully.
- Still missing for a real FluidAudio backend: model store/download layout, a
  FluidAudio-style `ASR/MOSS` manager, a non-fixture benchmark harness, and
  long-audio chunking beyond the single 30-second static window.
