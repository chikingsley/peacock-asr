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
- Paired 100-row MLX/PyTorch eval:
  `artifacts/evals/librispeech-test-clean-mlx-100/summary.json`
  `artifacts/evals/librispeech-test-clean-pytorch-100/summary.json`
  `artifacts/evals/librispeech-test-clean-mlx-vs-pytorch-100/summary.json`
- Quantized MLX artifacts:
  `artifacts/mlx/MOSS-Transcribe-preview-2B-text-decoder-8bit-g64/`
  `artifacts/mlx/MOSS-Transcribe-preview-2B-text-decoder-4bit-g64/`
  `artifacts/mlx/MOSS-Transcribe-preview-2B-all-8bit-g64/`
  `artifacts/mlx/MOSS-Transcribe-preview-2B-all-4bit-g64/`
  Complete local `weights.safetensors` files are retained for BF16, the best
  `text-decoder-4bit-g64` candidate, and the smallest `all-4bit-g64`
  candidate. The 8-bit candidate directories retain configs, reports,
  manifests, and eval summaries, but their multi-GB weight files were not kept
  after cleanup because they were weaker/noisier candidates and are
  reproducible from BF16 on Apple Silicon if needed.
- Private package manifests:
  `artifacts/packages/MOSS-Transcribe-preview-2B-bf16-manifest.json`
  `artifacts/packages/MOSS-Transcribe-preview-2B-text-decoder-8bit-g64-manifest.json`
  `artifacts/packages/MOSS-Transcribe-preview-2B-text-decoder-4bit-g64-manifest.json`
  `artifacts/packages/MOSS-Transcribe-preview-2B-all-8bit-g64-manifest.json`
  `artifacts/packages/MOSS-Transcribe-preview-2B-all-4bit-g64-manifest.json`

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
- `uv run --locked moss-pytorch-streaming-eval --help`
- `uv run --locked moss-compare-evals --help`
- `uv run --locked moss-transcribe --help`
- `uv run --locked moss-quantize --help`
- `uv run --locked moss-package-manifest --help`
- `uv run --locked moss-coreml-plan --help`
- Forbidden-command scan returns no matches.

Linux negative MLX smoke result:

- `moss-mlx-smoke` exits at the intended boundary with:
  `ModuleNotFoundError: MLX runtime is not available in this environment.`
  This proves the CLI wiring works up to the Apple Silicon requirement.

Apple Silicon MLX smoke result:

- Host: `home-mac`
- Platform: macOS `26.5.1`, `Darwin arm64`
- Mac project path used during validation:
  `/Users/simonpeacocks/GitHub/moss-mlx-conversion`
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
- `MOSS_MLX_RUN_REAL_WEIGHTS=1 uv run --extra mac --locked -m pytest -q tests/test_real_weights.py -m real_weights`
  - Result: 2 passed, 1 skipped. Covers real converted-weight load and
    LibriSpeech fixture transcription.
- `MOSS_MLX_RUN_REAL_WEIGHTS=1 MOSS_MLX_RUN_STREAMING=1 uv run --extra mac --locked -m pytest -q tests/test_real_weights.py::test_real_streamed_one_row_eval`
  - Result: 1 passed. Covers real streamed HF row/audio fetch plus MLX
    transcription.

## Backend Shape

The local runtime now has an `mlx-audio`-style shape without depending on
`mlx-audio` internals:

- `MossTranscribeBackend.from_pretrained(model_dir)`
- `MossTranscribeBackend.generate(audio, language="English")`
- `STTOutput(text, segments, language, total_time, prompt_tokens,
  generation_tokens, timings, raw)`
- `MossSerialAdapter` for a serial serving/broker path.

CLI smoke:

```bash
uv run --extra mac --locked moss-transcribe \
  --model-dir artifacts/mlx/MOSS-Transcribe-preview-2B-bf16 \
  --audio artifacts/cache/fixtures/librosa-libri1-16k.wav
```

This is intentionally still a local package shape, not an upstream
`mlx-audio` branch or PR.

## Streaming LibriSpeech Eval

`moss-streaming-eval` uses the Hugging Face Dataset Viewer rows API for
metadata and signed audio asset URLs, then decodes streamed audio bytes in
memory with `soundfile`. It does not materialize per-utterance audio files.
`moss-pytorch-streaming-eval` uses the same row/audio/normalization path against
the upstream PyTorch model, so MLX and PyTorch can be compared on identical row
IDs.

Default target:

- Dataset: `openslr/librispeech_asr`
- Config: `clean`
- Split: `test`
- Offset: 0
- Limit: 20
- Metrics: `jiwer` WER/CER after lowercase and punctuation normalization
- Speed reporting: RTFx is primary (`audio_duration / elapsed`, bigger is
  better); RTF is also recorded for compatibility.

Original 20-row command used on the Mac:

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
- RTFx: 1.5377182692283724

Artifacts:

- Summary:
  `artifacts/evals/librispeech-test-clean-streaming-20/summary.json`
- Per-row predictions:
  `artifacts/evals/librispeech-test-clean-streaming-20/predictions.jsonl`

## Paired 100-Row Baseline

Commands used:

```bash
uv run --extra mac --locked moss-streaming-eval \
  --limit 100 \
  --page-size 20 \
  --output-dir artifacts/evals/librispeech-test-clean-mlx-100
```

```bash
uv run --locked moss-pytorch-streaming-eval \
  --revision c98175cb20e48bd9be4e95f6c85f2af18899f780 \
  --local-files-only \
  --limit 100 \
  --page-size 20 \
  --output-dir artifacts/evals/librispeech-test-clean-pytorch-100
```

```bash
uv run --locked moss-compare-evals \
  --left artifacts/evals/librispeech-test-clean-mlx-100/predictions.jsonl \
  --right artifacts/evals/librispeech-test-clean-pytorch-100/predictions.jsonl \
  --left-name mlx-bf16 \
  --right-name pytorch-bf16 \
  --output-dir artifacts/evals/librispeech-test-clean-mlx-vs-pytorch-100
```

| Backend | Rows | WER | CER | Audio sec | Sample sec | Wall sec | RTFx | RTF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MLX BF16, Apple Silicon | 100 | 0.017970401691331923 | 0.005121510343442459 | 670.565 | 416.083 | 423.465 | 1.6116 | 0.6205 |
| PyTorch BF16, RTX 5070 | 100 | 0.0200845665961945 | 0.00612572805784294 | 670.565 | 35.233 | 51.615 | 19.0323 | 0.0525 |

MLX/PyTorch comparison:

- Compared rows: 100
- Exact hypothesis matches: 85
- Normalized hypothesis matches: 97
- First 5 generated IDs match: 92
- Equal per-row WER: 97
- MLX lower per-row WER: 3
- PyTorch lower per-row WER: 0

Conclusion: the BF16 MLX path does not show a quality regression on this
100-row subset. The few normalized text differences favor MLX by WER, mostly in
spelling/wording edge cases such as `sixteenth` vs `sixteen, one`.

Profiling from the 100-row MLX run:

- Total transcription time: 388.04 seconds inside `transcribe_waveform`
- Generation time: 332.10 seconds
- Audio feature/encoder/adapter time: 52.56 seconds
- Processor time: 1.79 seconds
- Embedding merge time: 1.54 seconds

Speed conclusion: generation is the main Apple Silicon bottleneck. The audio
tower is the secondary bottleneck. Processor and embedding merge are not worth
optimizing first.

## Speed Probe

An experimental `--generation-mode fast-greedy` path was added to skip the
generic MLX-LM per-token log-probability calculation. It preserved 20-row
quality but did not beat the default MLX-LM generation path on the same rows:

| Generation mode | Rows | WER | CER | RTFx | Generation sec |
| --- | ---: | ---: | ---: | ---: | ---: |
| `fast-greedy` | 20 | 0.01580135440180587 | 0.004177109440267335 | 1.8325 | 74.86 |
| `mlx-lm` | 20 | 0.01580135440180587 | 0.004177109440267335 | 1.8611 | 74.38 |

Default remains `mlx-lm`. The next speed levers are batching/serving shape,
prompt/cache reuse where applicable, MLX-Audio backend integration, and later
text-decoder quantization. Quantization should remain behind BF16 validation.

## Quantization Results

Quantization follows the MLX-LM/MLX-Audio prior-art contract:

- Apply `nn.quantize` / `mlx_lm.utils.quantize_model` after loading BF16
  weights.
- Persist `quantization` and `quantization_config` in `config.json`.
- Rebuild quantized module structure before `load_weights` by checking
  `*.scales` tensors in the saved artifact.
- Use scoped predicates so the text decoder, audio tower, adapter, or all
  quantizable modules can be tested independently.

Commands used on the Mac:

```bash
uv run --extra mac --locked moss-quantize \
  --bits 4 \
  --group-size 64 \
  --scope text-decoder \
  --output-dir artifacts/mlx/MOSS-Transcribe-preview-2B-text-decoder-4bit-g64 \
  --overwrite
```

All four quantized artifacts loaded and produced fixture transcripts through
`moss-mlx-smoke`. The text-decoder 4-bit and all-module 4-bit smokes preserved
the first 5 generated IDs; exact transcript match differs only by punctuation
or small wording on the fixture.

| Artifact | Scope | Bits | Weight bytes | Package bytes |
| --- | --- | ---: | ---: | ---: |
| BF16 | none | 16 | 4,837,667,584 | 4,854,176,440 |
| text-decoder-8bit-g64 | text decoder | 8 | 3,516,602,520 | 3,533,059,256 |
| text-decoder-4bit-g64 | text decoder | 4 | 2,811,958,960 | 2,828,415,698 |
| all-8bit-g64 | all quantizable modules | 8 | 2,574,711,976 | 2,591,214,120 |
| all-4bit-g64 | all quantizable modules | 4 | 1,367,701,071 | 1,384,203,216 |

20-row clean-test benchmark, first LibriSpeech rows:

| Backend | Rows | WER | CER | RTFx | Sample sec | Generation sec | Audio feature sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 `mlx-lm` | 20 | 0.01580135440180587 | 0.004177109440267335 | 1.8611 | 88.38 | 74.38 | 10.83 |
| text-decoder-8bit-g64 | 20 | 0.01580135440180587 | 0.004177109440267335 | 1.2069 | 136.29 | 111.54 | 21.90 |
| text-decoder-4bit-g64 | 20 | 0.013544018058690745 | 0.0029239766081871343 | 2.4793 | 66.34 | 50.78 | 12.53 |
| all-8bit-g64 | 20 | 0.01580135440180587 | 0.004177109440267335 | 2.3850 | 68.97 | 58.09 | 7.64 |
| all-4bit-g64 | 20 | 0.01580135440180587 | 0.004177109440267335 | 1.8047 | 91.14 | 74.27 | 13.89 |

The text-decoder 8-bit run is noisy: the first run before the backend label
fix measured 1.9949 RTFx, while the labeled rerun measured 1.2069 RTFx after a
long sequence of Mac jobs. Do not treat that variant's speed as final without
a fresh repeated benchmark.

BF16-vs-quant comparison on the same 20 rows:

| Variant | Exact matches | Normalized matches | First 5 ID matches | Equal WER | BF16 lower WER | Quant lower WER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| text-decoder-4bit-g64 | 13/20 | 19/20 | 18/20 | 19/20 | 0 | 1 |
| all-8bit-g64 | 19/20 | 20/20 | 20/20 | 20/20 | 0 | 0 |
| all-4bit-g64 | 11/20 | 20/20 | 18/20 | 20/20 | 0 | 0 |

Current quantization read:

- Best speed/quality candidate: `text-decoder-4bit-g64`.
- Best broad quantization candidate: `all-8bit-g64`; it reduces audio feature
  time and package size while matching BF16 WER on this slice.
- Not recommended yet: `all-4bit-g64`; it is smallest, but not faster than the
  better candidates and changes exact punctuation/wording more often.
- Next validation gate for any quantized artifact is a 100-row or full
  clean-test pass, preferably repeated after a cool Mac runtime window because
  the 20-row speed numbers show thermal/runtime variance.

## Private Package Manifests

`moss-package-manifest` writes local metadata only. It hashes files, embeds the
config, includes conversion/quantization reports when present, links eval
summaries, and records `"public_actions": "none"`.

Generated manifests:

- `artifacts/packages/MOSS-Transcribe-preview-2B-bf16-manifest.json`
- `artifacts/packages/MOSS-Transcribe-preview-2B-text-decoder-8bit-g64-manifest.json`
- `artifacts/packages/MOSS-Transcribe-preview-2B-text-decoder-4bit-g64-manifest.json`
- `artifacts/packages/MOSS-Transcribe-preview-2B-all-8bit-g64-manifest.json`
- `artifacts/packages/MOSS-Transcribe-preview-2B-all-4bit-g64-manifest.json`

No public branch, PR, push, or Hugging Face upload has been done.

## Mac Working Copy

The Mac working copy at `/Users/simonpeacocks/GitHub/moss-mlx-conversion` is
the active Apple Silicon/CoreML workbench. Retained artifacts are copied back
to this Linux project under ignored `artifacts/coreml/`. The separate
FluidAudio checkout at `/Users/simonpeacocks/GitHub/FluidAudio` was left
untouched.

Retained local weight files:

| Artifact | Local weight bytes |
| --- | ---: |
| BF16 | 4,837,764,136 |
| text-decoder-4bit-g64 | 2,811,958,960 |
| all-4bit-g64 | 1,367,701,071 |

The stopped benchmark logs, Parakeet full result, BF16 partial, and text4
concurrent partial are present under `artifacts/logs/` and `artifacts/evals/`.

## Stopped Full Benchmark Probe

The full LibriSpeech `test-clean` benchmark matrix was started on the Mac and
then intentionally stopped on 2026-06-30 after the architecture/speed read
showed MOSS is better treated as a teacher/reference model than as a
FluidAudio-speed serving backend.

Completed baseline:

| Backend | Rows | WER | CER | Overall RTFx | Total audio sec | Processing sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FluidAudio Parakeet TDT v3 | 2620 | 0.026338755621196294 | 0.010256972670352282 | 39.612813665714469 | 19452.480625 | 491.065361 |

Stopped MOSS partials:

| Backend | Completed rows | WER | CER | RTFx | Wall sec | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| MLX BF16 | 200 / 2620 | 0.01569995638901003 | 0.00492282019190655 | 1.133674799085734 | 1384.935804 | Slowed by concurrent text4 probe after row ~150. |
| text-decoder-4bit-g64 | 25 / 2620 | 0.01904761904761905 | 0.0056120659417748155 | 0.8532106051131099 | 230.813908 | Concurrent probe only; not a clean standalone speed number. |

Artifacts:

- `artifacts/evals/fluid-parakeet-v3-librispeech-test-clean-full/results.json`
- `artifacts/evals/librispeech-test-clean-mlx-bf16-full/partial-summary.json`
- `artifacts/evals/librispeech-test-clean-mlx-text-decoder-4bit-g64-full-concurrent/partial-summary.json`

Architectural read:

- MOSS is a speech-conditioned decoder-only Qwen stack, not a CTC/TDT
  acoustic-decoder model.
- The local timing profile shows generation dominates: on the 100-row BF16
  run, generation took 332.10 seconds versus 52.56 seconds for audio
  features/encoder/adapter.
- Quantization can reduce cost, but it does not change the basic
  token-by-token LLM decoding path.
- Practical role: use MOSS as an open-weights teacher/reference for quality,
  distillation, or data generation. Do not expect FluidAudio/Parakeet-class
  throughput without a much deeper architecture change or CoreML/ANE-specific
  decoder effort.

## FluidAudio/CoreML Notes

Current MOSS artifact type:

- The artifacts in `artifacts/mlx/` are MLX-layout safetensors packages.
- They are not CoreML models and are not native FluidAudio backends.
- The local `MossTranscribeBackend` mirrors an `mlx-audio`-style Python STT
  contract, not a Swift/CoreML/ANE runtime.

What FluidAudio does for supported models:

- Parakeet TDT uses a purpose-built ASR architecture: audio encoder plus a
  small TDT predictor/joint path that can skip audio frames by predicting token
  durations.
- Cohere Transcribe is encoder-decoder with a 48-layer Conformer encoder and
  an 8-layer transformer decoder. FluidAudio's documented CoreML port uses an
  INT8 encoder and a static-shape ANE-resident decoder/cache path so decode can
  stay on the Neural Engine.
- SenseVoice/Paraformer-style models are non-autoregressive or CTC-like enough
  that much of the transcript path can run in parallel, then use host-side
  decoding/detokenization.

How MOSS differs:

- MOSS injects audio embeddings into a Qwen3-1.7B-style decoder prompt and then
  generates transcript text autoregressively.
- A FluidAudio/CoreML port would likely need multiple `.mlpackage` pieces:
  audio frontend/encoder/adapter, decoder prefill, cache-external decoder step,
  and LM-head/tied embedding handling.
- The likely hard part is not conversion alone; it is designing static shapes,
  KV-cache IO, attention masks, and compute-unit placement so the Qwen decoder
  actually dispatches efficiently.
- A serious CoreML experiment should be scoped as a separate track, starting
  with a single short fixture and parity checkpoints before attempting
  benchmark work.

## Private CoreML/Mobius Workbench

This track is now scoped privately under the local MOSS conversion project:

- Workbench notes:
  `coreml/README.md`
- Detailed design:
  `docs/COREML_MOBIUS.md`
- Package module:
  `src/moss_mlx_conversion/coreml/`
- CLI:
  `moss-coreml-plan`

The first concrete artifact was a Mobius-style conversion contract. It can be
regenerated with:

```bash
uv run --project projects/moss-mlx-conversion --locked moss-coreml-plan \
  --output projects/moss-mlx-conversion/artifacts/coreml/moss-coreml-plan.json
```

Generated artifacts:

- `artifacts/coreml/moss-coreml-plan.json`
- `artifacts/coreml/moss-token-embedding-fp16.safetensors`
- `artifacts/coreml/moss-token-embedding-fp16.json`
- `artifacts/coreml/moss-audio-encoder-adapter-bf16.safetensors`
- `artifacts/coreml/moss-audio-encoder-adapter-bf16.json`
- `artifacts/coreml/moss-qwen3-decoder-bf16.safetensors`
- `artifacts/coreml/moss-qwen3-decoder-bf16.json`
- `artifacts/coreml/moss_token_embedding.mlpackage/`
- `artifacts/coreml/moss_token_embedding.mlmodelc/`
- `artifacts/coreml/moss_token_embedding.json`
- `artifacts/coreml/moss_audio_encoder_adapter_fixture.mlpackage/`
- `artifacts/coreml/moss_audio_encoder_adapter_fixture.mlmodelc/`
- `artifacts/coreml/moss_audio_encoder_adapter_fixture.json`
- `artifacts/coreml/moss_decoder_prefill_fixture.mlpackage/`
- `artifacts/coreml/moss_decoder_prefill_fixture.mlmodelc/`
- `artifacts/coreml/moss_decoder_prefill_fixture.json`
- `artifacts/coreml/moss_decoder_step_fixture.mlpackage/`
- `artifacts/coreml/moss_decoder_step_fixture.mlmodelc/`
- `artifacts/coreml/moss_decoder_step_fixture.json`
- `artifacts/coreml/moss_decoder_step_padded_fixture.mlpackage/`
- `artifacts/coreml/moss_decoder_step_padded_fixture.mlmodelc/`
- `artifacts/coreml/moss_decoder_step_padded_fixture.json`
- `artifacts/coreml/moss_decoder_step_padded_1layer_fixture.mlpackage/`
- `artifacts/coreml/moss_decoder_step_padded_1layer_fixture.mlmodelc/`
- `artifacts/coreml/moss_decoder_step_padded_1layer_fixture.json`
- `artifacts/coreml/moss_decoder_stateful_fused.mlpackage/`
- `artifacts/coreml/moss_decoder_stateful_fused.mlmodelc/`
- `artifacts/coreml/moss_decoder_stateful_fused.json`
- `artifacts/coreml/moss_decoder_stateful_fused_1layer.mlpackage/`
- `artifacts/coreml/moss_decoder_stateful_fused_1layer.mlmodelc/`
- `artifacts/coreml/moss_decoder_stateful_fused_1layer.json`
- `artifacts/coreml/moss_coreml_stateful_fixture_pipeline.json`
- `artifacts/coreml/moss_coreml_stateful_fixture_pipeline_reference_merged.json`
- `artifacts/coreml/moss_swift_fixture.json`
- `artifacts/coreml/moss_swift_fixture_compact.json`
- `artifacts/coreml/moss_swift_coreml_fixture_5tok.json`
- `artifacts/coreml/moss_swift_coreml_fixture_52tok.json`
- `artifacts/coreml/moss_swift_coreml_fixture_60tok.json`
- `artifacts/coreml/moss_tokenizer.json`
- `artifacts/coreml/moss_swift_coreml_fixture_5tok_tokenizer.json`
- `artifacts/coreml/moss_swift_coreml_fixture_52tok_tokenizer.json`
- `artifacts/coreml/moss_swift_coreml_fixture_compact_5tok.json`
- `artifacts/coreml/moss_swift_coreml_fixture_compact_52tok.json`

Current default contract:

| Item | Value |
| --- | ---: |
| Max audio seconds | 30 |
| Max mel frames | 3000 |
| Max MOSS audio tokens | 390 |
| Fixed prompt overhead tokens | 10 |
| Fixed prefill sequence length | 512 |
| Decode budget | 256 |
| Padded KV cache length | 768 |
| Per-layer KV cache shape | `[1, 8, 768, 128]` |
| Total FP16 KV cache | 84.0 MiB |

Planned pieces:

- Host mel frontend.
- `moss_audio_encoder_adapter.mlpackage`.
- `moss_token_embedding.mlpackage`.
- `moss_decoder_prefill.mlpackage`.
- `moss_decoder_step_cache_external.mlpackage`.
- `moss_decoder_stateful_fused.mlpackage`.

Mobius relation:

- Use Qwen3-ASR CoreML work for component split, prefill/step separation,
  RoPE precision checks, and cache padding warnings.
- Use Cohere Transcribe CoreML work for the cache-external decoder direction.
- Do not treat Mobius as a generic MOSS converter; MOSS still needs dedicated
  wrappers for its audio encoder, audio-mask embedding injection, and Qwen3
  decoder cache contract.

Fixture component probes completed on `home-mac`:

| Component | Fixture input | CoreML validation |
| --- | --- | --- |
| `moss_token_embedding` | token IDs `[1, 512]` | max/mean diff vs PyTorch `0.0` / `0.0` |
| `moss_audio_encoder_adapter_fixture` | mel `[128, 1484]` | output `[193, 2048]`; max/mean diff vs PyTorch `0.002675` / `0.000354` |
| `moss_decoder_prefill_fixture` | merged embeds `[1, 203, 2048]` | top-1 token `4197`; max/mean diff vs PyTorch `0.048508` / `0.017621` |
| `moss_decoder_step_fixture` | token `4197`, KV `[28, 1, 8, 203, 128]` | top-1 token `1059`; max/mean diff vs PyTorch `0.040039` / `0.015691` |
| `moss_decoder_step_padded_fixture` | token `4197`, padded KV `[28, 1, 8, 768, 128]` | top-1 token `1059`; padded Torch path matches append-cache Torch exactly on valid logits/cache slices; CoreML vs Torch logits max/mean diff `0.040039` / `0.015691` |
| `moss_decoder_stateful_fused` | prefill `[1, 203, 2048]`, then token `4197` with same CoreML state | prefill top-1 `4197`; step top-1 `1059`; 56 CoreML state tensors `[1, 8, 768, 128]`; CoreML vs static step logits max/mean diff `0.038696` / `0.015730` |
| `run_stateful_fixture_pipeline` component path | CoreML token IDs + CoreML mel/audio + host merge + stateful decoder | merged prompt max/mean diff vs saved BF16 reference `0.002686` / `0.000337`; prefill top-1 `4197`; step top-1 `1059`; total fixture time `21.32s`, with `20.61s` decoder prefill and `0.226s` first decode step |
| `run_stateful_fixture_pipeline` reference-merged isolation | saved merged embeds + stateful decoder | decoder input diff vs saved reference `0.0`; prefill top-1 `4197`; step top-1 `1059`; total fixture time `22.15s`, with `21.45s` decoder prefill and `0.143s` first decode step |
| Swift `moss-coreml-fixture` 5-token greedy | JSON fixture mel/token IDs + compiled `.mlmodelc` bundles + `MLState` | generated IDs exactly match `[4197, 1059, 4158, 6177, 323]`; total fixture time `18.17s`, with `16.96s` decoder prefill and `0.809s` decoder decode calls |
| Swift `moss-coreml-fixture` 52-token greedy | same Swift/CoreML path | first 10 IDs match, then CoreML inserts comma token `11` after `smokestack`; decoded normalized WER/CER are `0.0`; total fixture time `23.53s`, with `16.80s` decoder prefill and `6.33s` decoder decode calls |
| Swift tokenizer-enabled 5-token greedy | same Swift/CoreML path plus Qwen ByteLevel tokenizer JSON | generated IDs and decoded text exactly match; text `with her white paint and`; raw/normalized WER/CER `0.0` |
| Swift tokenizer-enabled 52-token greedy | same Swift/CoreML path plus Qwen ByteLevel tokenizer JSON | generated text inserts only the comma after `smokestack`; raw WER/CER `0.0278` / `0.00442`; normalized WER/CER `0.0`; total fixture time `24.95s`, with `16.88s` decoder prefill and `7.63s` decoder decode calls |
| Swift compact prompt 5-token greedy | compact fixture without serialized `input_ids` / `audio_input_mask` | `prompt_source=compact`; generated IDs and decoded text exactly match; total fixture time `16.17s`, with `15.19s` decoder prefill and `0.47s` decoder decode calls |
| Swift compact prompt 52-token greedy | same compact prompt-builder path | `prompt_source=compact`; first 10 IDs match; generated text inserts only the comma after `smokestack`; raw WER/CER `0.0278` / `0.00442`; normalized WER/CER `0.0`; total fixture time `22.74s`, with `16.32s` decoder prefill and `5.96s` decoder decode calls |

Retained full-component package sizes:

| Component | `.mlpackage` | `.mlmodelc` |
| --- | ---: | ---: |
| `moss_token_embedding` | 594M | 594M |
| `moss_audio_encoder_adapter_fixture` | 1.4G | 1.4G |
| `moss_decoder_prefill_fixture` | 3.3G | 3.3G |
| `moss_decoder_step_fixture` | 3.3G | 3.3G |
| `moss_decoder_step_padded_fixture` | 3.3G | 3.3G |
| `moss_decoder_stateful_fused` | 3.3G | 3.3G |

Notes:

- CoreMLTools warned that Torch 2.12.1 is newer than its tested Torch version.
  The conversions, CoreML predictions, and compile checks still passed.
- The audio encoder and decoder exporters use fixed LibriSpeech fixture shapes.
- The fixed append-cache decoder step proves the original `past_len=203` to
  `204` fixture transition.
- The padded decoder step proves the planned 768-token external-cache window
  with host-provided update mask, attention mask, and RoPE tensors.
- The stateful fused decoder proves the Mobius-style CoreML State API path for
  the fixture: one CoreML state object survives prefill and the first decode
  step. It requires macOS 15+ and is still not a Swift/FluidAudio runtime.
- The integrated Python/CoreML fixture runner proves the runtime contract
  across token embedding, audio encoder+adapter, audio-mask insertion, Qwen3
  RoPE/masks, and stateful decoder state reuse. It is still fixture-shaped and
  uses Python/CoreMLTools, not Swift.
- The runner's raw `prefill_logits_vs_reference` field compares against the
  saved HF reference logits. The stateful exporter manifest's smaller parity
  diffs compare against the local custom static decoder path. Do not mix those
  two numeric gates; use the runner primarily for component-wiring and token
  rank validation.
- The Swift fixture runner proves the same component/state contract through
  Swift `MLModel` and `MLState`, using compiled `.mlmodelc` bundles. It now
  builds the fixed English MOSS prompt from compact template fields:
  `[151644, 872, 198, 151669] + audio_placeholder_count * 0 + [151670,
  151645, 198, 151644, 77091, 198]`. It still consumes fixture mel data and
  does not implement a Swift mel frontend, model download/store, or a
  FluidAudio `ASR/MOSS` manager.
- The Swift fixture runner now has a Qwen ByteLevel tokenizer decode bridge for
  generated token IDs, including skipped special tokens such as `<|im_end|>`.
  It is a decoder/detokenizer path plus fixed prompt builder; arbitrary prompt
  tokenizer encoding and time-marker variants remain future runtime work.
- The Swift 52-token greedy drift is punctuation-only on the decoded fixture:
  expected text has `smokestack the inverashiel`; Swift emitted
  `smokestack, the inverashiel`. After lowercase/punctuation normalization,
  WER and CER are both `0.0`.
- The packages and compiled models are retained locally under ignored
  `artifacts/coreml/`.
- A working Mac copy exists at
  `/Users/simonpeacocks/GitHub/moss-mlx-conversion` with the CoreML uv
  environment and generated component artifacts. Local `artifacts/coreml/`
  remains the canonical copy for retained outputs.
- A reference-only FluidAudio clone at `/tmp/FluidAudio` was inspected at
  `a95ec26`; current `main` does not contain a merged Qwen3-ASR Swift manager.
  A MOSS backend would need a new `ASR/MOSS` manager/model store rather than a
  small model-name addition.

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

The private conversion now has two completed tracks:

1. MLX reference/runtime track: PyTorch reference, processor parity, BF16 MLX
   weights, strict MLX load, Apple Silicon transcript parity, gated
   real-weight tests, backend shape, quantized candidates, and private local
   manifests.
2. CoreML/Mobius fixture track: token embedding, audio encoder+adapter,
   prefill, append-cache step, padded external-cache step, fused stateful
   decoder, integrated Python/CoreML fixture runner, and private Swift
   `MLState` greedy fixture runner all validate and are retained locally.

The next real work is a Swift/CoreML runtime decision:

1. If MOSS remains a teacher/reference, use the MLX/PyTorch artifacts to build
   batch teacher transcription and quality gates.
2. If pursuing FluidAudio-level runtime, the remaining missing pieces are
   real Swift mel frontend, model store/download layout, an `ASR/MOSS` manager
   API around the proven Swift fixture core, and optional general prompt
   tokenizer/template support beyond the fixed English no-time-marker path.
3. Run a single real audio file through that Swift runtime and require the
   first 5 generated tokens plus normalized transcript parity before any WER
   benchmark.
4. Then run the existing 20-row clean-test eval and profile compute placement.
   Only after that should quantized CoreML or artifact publication be scoped.
5. Keep all work private. Public branch, PR, push, and Hugging Face upload
   remain out of scope until explicitly requested.
