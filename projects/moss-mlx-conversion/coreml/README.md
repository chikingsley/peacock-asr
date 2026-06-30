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

Actual `.mlpackage` export scripts should land here once the PyTorch wrapper
signatures are ready and the run is moved to macOS/CoreML.

## First Component Probe

The first real CoreML probe is the token embedding component. It proves
CoreMLTools, `.mlpackage` writing, and CoreML prediction on a MOSS tensor before
touching the decoder.

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
