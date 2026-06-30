# MOSS MLX Conversion Progress

Last updated: 2026-06-30

## Verified Linux/GPU Reference State

- Model snapshot pinned:
  `c98175cb20e48bd9be4e95f6c85f2af18899f780`
- `mlx-audio` reference clone:
  `cc4ddedaec649d739c9177bd47b9cbd9be674680`
- `mlx-lm` reference clone:
  `2ed22318cd6a2fcc5c2e0caa1e1fb0ddeb7cafd5`
- Metadata/code/tokenizer cache:
  `artifacts/cache/huggingface/models--OpenMOSS-Team--MOSS-Transcribe-preview-2B/snapshots/c98175cb20e48bd9be4e95f6c85f2af18899f780`
- Fixture:
  `artifacts/cache/fixtures/librosa-libri1-16k.wav`
- Processor parity:
  `artifacts/reference/processor-parity-pinned/processor_parity.json`
- PyTorch BF16 reference:
  `artifacts/reference/libri1-pytorch-bf16/reference_report.json`
- Reference tensors:
  `artifacts/reference/libri1-pytorch-bf16/reference_tensors.npz`
- Weight mapping report:
  `artifacts/mlx/MOSS-Transcribe-preview-2B-bf16/mapping-report.json`
- Converted BF16 MLX artifact:
  `artifacts/mlx/MOSS-Transcribe-preview-2B-bf16/weights.safetensors`
- Conversion report:
  `artifacts/mlx/MOSS-Transcribe-preview-2B-bf16/conversion-report.json`
- Apple Silicon smoke report:
  `artifacts/mlx-smoke/libri1-smoke-report.json`
- Apple Silicon streamed LibriSpeech eval:
  `artifacts/evals/librispeech-test-clean-streaming-20/summary.json`

## Document Policy

- `README.md` is the orientation entry point.
- `docs/PLAN.md` is durable architecture/planning context and should only be
  changed when the strategy changes.
- `docs/PROGRESS.md` is the live status file for commands, current results,
  measurements, and next steps.
- `CHANGELOG.md` is the terse historical record of completed milestones.

## Current Results

Processor parity against upstream remote code passes for:

- `input_ids`
- `attention_mask`
- `audio_input_mask`
- `audio_data_seqlens`
- `audio_data`

PyTorch BF16 reference generation on the LibriSpeech fixture produced:

```text
with her white paint and her scarlet smokestack the inverashiel, one of the two small steamers that during the summer months plied up and down the loch and incidentally carried on communication between inverashiel and crianan,
```

Reference tensor highlights:

- Prompt length: 203 tokens
- Audio placeholder count: 193
- Mel shape: `[128, 1484]`
- Audio hidden shape: `[193, 2048]`
- Adapter output shape: `[193, 2048]`
- Merged embedding shape: `[1, 203, 2048]`
- First generated token ID: `4197`

Weight inspection:

- Source tensors: 838
- Source dtype: 838 BF16 tensors
- Source parameters: 2,418,833,792
- Source bytes: 4,837,667,584
- Audio model tensors: 525
- Audio adapter tensors: 3
- Language model tensors: 310
- Skipped source tensors: 0
- Mapping destination tensors: 839, including the optional generated tied
  `lm_head.weight` candidate.
- Actual BF16 conversion saved tensors: 838. It does not duplicate
  `lm_head.weight` by default because `tie_word_embeddings=true` and the MLX
  model uses `model.embed_tokens.as_linear(...)`.
- Actual BF16 conversion skipped source tensors: 0
- Converted artifact size: 4.6G
- Checked converted tensor shapes:
  - `audio_model.conv2d1.weight`: `[480, 3, 3, 1]` BF16
  - `audio_model.conv2d2.weight`: `[480, 3, 3, 480]` BF16
  - `audio_model.conv2d3.weight`: `[480, 3, 3, 480]` BF16
  - `audio_model.conv_out.weight`: `[1280, 7680]` BF16
  - `audio_adapter.gate_proj.weight`: `[8192, 2048]` BF16
  - `model.embed_tokens.weight`: `[151936, 2048]` BF16
  - `model.layers.0.self_attn.q_proj.weight`: `[2048, 2048]` BF16

Local gates now passing:

- `uv run --locked ruff check src tests`
- `uv run --locked ty check src tests`
- `uv run --locked pytest -q`
- `uv run --locked moss-convert --help`
- `uv run --locked moss-mlx-smoke --help`
- Forbidden-command scan returns no matches.

Linux negative MLX smoke result:

- `moss-mlx-smoke` exits at the intended boundary with:
  `ModuleNotFoundError: MLX runtime is not available in this environment.`
  This proves the CLI wiring works up to the Apple Silicon requirement.

Apple Silicon MLX smoke result:

- Host: `home-mac`
- Platform: macOS `26.5.1`, `Darwin arm64`
- Mac project path: `/Users/simonpeacocks/GitHub/moss-mlx-conversion`
- Remote `uv`: `uv 0.11.24`
- `moss-mlx-smoke` loaded the BF16 converted artifact and generated the
  LibriSpeech fixture end to end.
- Prompt length: 203
- Audio placeholder count: 193
- Generated token count: 51
- First 5 generated token IDs: `[4197, 1059, 4158, 6177, 323]`
- First 5 token match vs PyTorch reference: true
- Transcript match vs PyTorch reference: true
- Total smoke elapsed time: 27.13 seconds
- Generation elapsed time: 7.83 seconds

MLX transcript:

```text
with her white paint and her scarlet smokestack the inverashiel, one of the two small steamers that during the summer months plied up and down the loch and incidentally carried on communication between inverashiel and crianan,
```

Mac gates passing under `--extra mac`:

- `uv run --extra mac --locked ruff check src tests`
- `uv run --extra mac --locked ty check src tests`
- `uv run --extra mac --locked -m pytest -q`

## Streaming LibriSpeech Eval

`moss-streaming-eval` uses the Hugging Face Dataset Viewer rows API for
metadata and signed audio asset URLs, then decodes streamed audio bytes in
memory with `soundfile`. It does not materialize per-utterance audio files.

Default target:

- Dataset: `openslr/librispeech_asr`
- Config: `clean`
- Split: `test`
- Offset: 0
- Limit: 20
- Metrics: `jiwer` WER/CER after lowercase and punctuation normalization

Command used on the Mac:

```bash
uv run --extra mac --locked moss-streaming-eval \
  --limit 20 \
  --page-size 20 \
  --output-dir artifacts/evals/librispeech-test-clean-streaming-20
```

Result:

- Completed rows: 20
- Corpus WER: 0.01580135440180587
- Mean sample WER: 0.01585844651952462
- CER: 0.004177109440267335
- Total streamed audio: 164.49 seconds
- Per-sample processing elapsed: 106.97 seconds
- Wall elapsed: 110.76 seconds
- RTF: 0.6503141830406948
- Speed multiple: 1.5377182692283724

Artifacts:

- Summary:
  `artifacts/evals/librispeech-test-clean-streaming-20/summary.json`
- Per-row predictions:
  `artifacts/evals/librispeech-test-clean-streaming-20/predictions.jsonl`

## Reproduction Commands

```bash
uv run --project projects/moss-mlx-conversion --locked moss-processor-parity \
  --revision c98175cb20e48bd9be4e95f6c85f2af18899f780 \
  --local-files-only \
  --dump-dir projects/moss-mlx-conversion/artifacts/reference/processor-parity-pinned
```

```bash
uv run --project projects/moss-mlx-conversion --locked moss-reference \
  --revision c98175cb20e48bd9be4e95f6c85f2af18899f780 \
  --dump-dir projects/moss-mlx-conversion/artifacts/reference/libri1-pytorch-bf16 \
  --max-new-tokens 128 \
  --save-large-tensors
```

```bash
uv run --project projects/moss-mlx-conversion --locked moss-inspect-weights \
  --revision c98175cb20e48bd9be4e95f6c85f2af18899f780 \
  --local-files-only \
  --output projects/moss-mlx-conversion/artifacts/mlx/MOSS-Transcribe-preview-2B-bf16/mapping-report.json
```

```bash
uv run --project projects/moss-mlx-conversion --locked moss-convert \
  --revision c98175cb20e48bd9be4e95f6c85f2af18899f780 \
  --local-files-only \
  --output-dir projects/moss-mlx-conversion/artifacts/mlx/MOSS-Transcribe-preview-2B-bf16
```

Apple Silicon runtime smoke command used:

```bash
uv run --project projects/moss-mlx-conversion --extra mac --locked moss-mlx-smoke \
  --model-dir projects/moss-mlx-conversion/artifacts/mlx/MOSS-Transcribe-preview-2B-bf16 \
  --reference-report projects/moss-mlx-conversion/artifacts/reference/libri1-pytorch-bf16/reference_report.json \
  --report projects/moss-mlx-conversion/artifacts/mlx-smoke/libri1-smoke-report.json \
  --max-new-tokens 128
```

## Next Chunk

The next full-stride chunk is broader validation and upstream shaping:

1. Expand the streaming eval beyond the first 20 rows and add optional PyTorch
   side-by-side reference scoring for the same row IDs.
2. Move the local backend shape toward an `mlx-audio` package under
   `mlx_audio/stt/models/moss_transcribe/`.
3. Add gated real-weight tests that skip unless the converted artifact exists.
4. Publish or stage a BF16 MLX repo only after the 20-file validation table is
   acceptable.
5. Start 8-bit text-decoder quantization only after BF16 quality is measured.

The minimum full conversion is now proven: PyTorch reference, processor parity,
BF16 MLX weights, strict MLX load, and one end-to-end Apple Silicon transcript
matching the PyTorch reference.
