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
