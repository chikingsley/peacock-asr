# MOSS-Transcribe Preview 2B MLX Conversion Plan

## Goal

Build a real MLX conversion and runtime path for
[`OpenMOSS-Team/MOSS-Transcribe-preview-2B`](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-preview-2B),
then shape it so it can be upstreamed as an `mlx-audio` backend and later used
as the basis for a FluidInference `mobius` / FluidAudio CoreML port.

The execution style should be full-stride: implement a complete reference
pipeline, complete MLX model, complete weight converter, and complete generation
loop for a short audio fixture. If the full step fails, use parity checkpoints
to isolate the break and then continue from there. Do not limit the first pass
to exploratory fragments that cannot produce a transcript.

## Source Snapshot

Primary model:
[`OpenMOSS-Team/MOSS-Transcribe-preview-2B`](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-preview-2B)

Important upstream files:

- [`config.json`](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-preview-2B/raw/main/config.json)
- [`modeling_Moss.py`](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-preview-2B/raw/main/modeling_Moss.py)
- [`processing_Moss.py`](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-preview-2B/raw/main/processing_Moss.py)
- [`chat_template_default.py`](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-preview-2B/raw/main/chat_template_default.py)
- [`model.safetensors.index.json`](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-preview-2B/raw/main/model.safetensors.index.json)

Observed repository contents:

- `modeling_Moss.py`
- `processing_Moss.py`
- `chat_template_default.py`
- Qwen tokenizer files: `tokenizer.json`, `vocab.json`, `merges.txt`,
  `added_tokens.json`, `special_tokens_map.json`
- Single BF16 safetensors shard plus index

Model facts from the model card and config:

- Task: English ASR.
- License: Apache-2.0.
- Approximate size: 2.4B parameters, BF16 safetensors.
- Text backbone: Qwen3-1.7B style decoder.
- Audio encoder: Qwen3-Omni-MoE audio encoder.
- Adapter: gated MLP from audio hidden states to language-model embeddings.
- Audio frontend: 16 kHz, 128-bin Whisper log-mel, `n_fft=400`,
  `hop_length=160`.
- Text config: 28 Qwen3 decoder layers, hidden size 2048, 16 attention heads,
  8 KV heads, head dim 128, vocab size 151936, `rope_theta=1000000`.
- Audio config: 32 encoder layers, `d_model=1280`, 20 attention heads,
  FFN dim 5120, output dim 2048, 128 mel bins.
- Adapter hidden size: 8192.

Important trap: the top-level `hidden_size` in `config.json` is not the text
decoder hidden size to use for implementation. The actual language model
parameters live under `language_config`. Use `language_config.hidden_size=2048`
for the decoder and adapter output.

## Existing Work To Stay Close To

FluidInference `mobius`:
[`FluidInference/mobius`](https://github.com/FluidInference/mobius)

- `mobius` organizes conversion work as `models/{class}/{name}/{destination}`.
  For MOSS, the matching path would be
  `models/stt/moss-transcribe-preview-2b/mlx` for MLX-oriented work and later
  `models/stt/moss-transcribe-preview-2b/coreml` for CoreML.
- The repo uses separate `pyproject.toml` files per model/destination and `uv`
  for dependency isolation.

Closest FluidInference conversion references:

- [`models/stt/qwen3-asr-0.6b/coreml`](https://github.com/FluidInference/mobius/tree/main/models/stt/qwen3-asr-0.6b/coreml)
- [`models/stt/qwen3-asr-0.6b/coreml/convert-qwen3-asr.py`](https://github.com/FluidInference/mobius/blob/main/models/stt/qwen3-asr-0.6b/coreml/convert-qwen3-asr.py)
- [`models/stt/qwen3-asr-0.6b/coreml/QWEN3_ASR_COREML.md`](https://github.com/FluidInference/mobius/blob/main/models/stt/qwen3-asr-0.6b/coreml/QWEN3_ASR_COREML.md)
- [`models/stt/cohere-transcribe-03-2026/coreml`](https://github.com/FluidInference/mobius/tree/main/models/stt/cohere-transcribe-03-2026/coreml)
- [`mobius` PR #18: Qwen3-ASR stateful CoreML decoder conversion](https://github.com/FluidInference/mobius/pull/18)
- [`mobius` PR #70: ANE optimization candidates and playbook](https://github.com/FluidInference/mobius/pull/70)

Relevant FluidAudio integration work:

- [`FluidAudio` main repo](https://github.com/FluidInference/FluidAudio)
- [`PR #487: Cohere Transcribe INT8 encoder + FP16 cache-external decoder`](https://github.com/FluidInference/FluidAudio/pull/487)
- [`PR #537: Cohere static-shape decoder v2`](https://github.com/FluidInference/FluidAudio/pull/537)
- [`PR #676: remove experimental Qwen3-ASR backend`](https://github.com/FluidInference/FluidAudio/pull/676)
- [`PR #744: native Swift mel frontend for streaming ASR`](https://github.com/FluidInference/FluidAudio/pull/744)
- [`PR #709: Canary-1B-v2 CoreML AED engine`](https://github.com/FluidInference/FluidAudio/pull/709)

What to reuse conceptually from those:

- Treat audio encoder, adapter/projection, token embedding, decoder prefill, and
  one-token decode as separable components.
- Keep host-managed KV cache as the default mental model for eventual CoreML.
- Add parity probes before optimizing.
- Prefer fixed/static shapes when targeting ANE; PR #537 documents a real case
  where dynamic `attention_mask` shapes kept the decoder off ANE.
- Watch precision in decoder and LM head. The Qwen3-ASR CoreML notes document
  real overflow and cache-length failure modes.
- Do not assume a Qwen3-ASR Swift backend currently exists in FluidAudio main.
  PR #676 removed the experimental backend, so MOSS should be treated as a new
  integration with a fresh manager/API surface when it reaches FluidAudio.
- Native audio feature extraction matters on-device. PR #744 is a concrete
  example where moving mel extraction out of a flexible CoreML preprocessor
  fixed an iPadOS cold-start failure and made failures loud instead of silent.
- Encoder-decoder work such as Canary is not the MOSS architecture, but it is a
  useful example of how Fluid stages CoreML models, precision variants, CLI
  benchmarks, and follow-up cache-external decoder work.

Closest MLX references:

- [`mlx-audio`](https://github.com/Blaizzy/mlx-audio)
- [`mlx_audio/stt/models/qwen3_asr`](https://github.com/Blaizzy/mlx-audio/tree/main/mlx_audio/stt/models/qwen3_asr)
- [`mlx_audio/stt/models/qwen2_audio`](https://github.com/Blaizzy/mlx-audio/tree/main/mlx_audio/stt/models/qwen2_audio)
- [`mlx_audio/stt/models/cohere_asr`](https://github.com/Blaizzy/mlx-audio/tree/main/mlx_audio/stt/models/cohere_asr)
- [`mlx-audio` PR #811: Higgs Audio 3 STT support](https://github.com/Blaizzy/mlx-audio/pull/811)
- [`mlx-audio` PR #777: STT eval harness](https://github.com/Blaizzy/mlx-audio/pull/777)
- [`mlx-audio` PR #740: Mega-ASR STT model](https://github.com/Blaizzy/mlx-audio/pull/740)
- [`mlx-audio` PR #774: Nemotron cache-aware streaming](https://github.com/Blaizzy/mlx-audio/pull/774)
- [`mlx-audio` PR #783: Qwen3-ASR segment batching](https://github.com/Blaizzy/mlx-audio/pull/783)
- [`mlx-audio` PR #806: STT hot-path profiling optimizations](https://github.com/Blaizzy/mlx-audio/pull/806)

What to reuse from `mlx-audio`:

- Backend shape: model package under `mlx_audio/stt/models/<backend_name>/`.
- `STTOutput` return contract.
- `mlx_lm.generate.generate_step` for autoregressive generation with
  precomputed `input_embeddings`.
- Qwen3 ASR pattern: preprocess audio, encode audio, build prompt, replace
  audio-token embeddings, then call the Qwen decoder through MLX generation.
- Qwen2-Audio pattern: audio feature extraction, audio tower, projector, then
  embedding splicing before text generation.
- Higgs Audio 3 pattern from PR #811: Qwen3-1.7B text backbone, Whisper-style
  audio frontend, projector into text embeddings, sanitized checkpoint key
  mapping, MLX-native log-mel extraction, small shape tests, and a gated
  real-weight transcription test. This is the closest MLX implementation
  precedent for MOSS even though the exact audio tower differs.
- STT eval harness from PR #777: reuse the existing WER/RTF reporting shape for
  the MOSS validation report instead of inventing a separate evaluator.
- Mega-ASR PR #740: use its router/LoRA work only as a testing and model-card
  precedent. It is relevant for noisy-ASR evaluation and quantization reporting,
  not for the first MOSS architecture.
- Segment batching PR #783 and STT hot-path PR #806 are performance follow-ups,
  not first-pass requirements. Track them after BF16 transcript parity works.

Reference priority for MOSS:

1. Implement the first MLX pass closest to `mlx-audio` PR #811 plus the current
   `qwen3_asr` and `qwen2_audio` backends.
2. Validate with the `mlx-audio` PR #777 eval harness shape.
3. Use `mobius` PR #18 when designing the later CoreML component split and
   cache strategy.
4. Use FluidAudio PR #676 as the current-integration warning: MOSS will need a
   fresh Swift backend rather than a small patch to an existing Qwen3-ASR path.
5. Use `mobius` PR #70 and FluidAudio PR #744 for ANE and on-device frontend
   decisions after the MLX backend is correct.

## How MOSS Differs From Existing Qwen3/Cohere Ports

1. MOSS is not the same as native `Qwen3-ASR`.
   `Qwen3-ASR` has native Transformers classes and an existing MLX backend.
   MOSS uses custom remote code: `MossConfig`, `MossModel`,
   `MossForCausalLM`, and `MossProcessor`.

2. MOSS uses `Qwen3OmniMoeAudioEncoder`.
   Existing `mlx-audio` Qwen3-ASR code has an audio encoder implementation, but
   MOSS points to the Transformers `qwen3_omni_moe` audio encoder with
   `d_model=1280`, 32 layers, and 20 heads. Do not assume the existing
   `qwen3_asr` audio tower can be reused unchanged.

3. MOSS injection is mask-based, not special-token-ID-only.
   `MossProcessor` builds `input_ids` and an `audio_input_mask`. The model
   embeds all input IDs, runs the audio encoder and gated MLP, then uses
   `masked_scatter_` into positions where `audio_input_mask` is true.

4. The audio placeholder token ID is `0`.
   The audio placeholder is not a normal Qwen special token. The mask is the
   source of truth for where audio embeddings go.

5. The default prompt is template-driven.
   `processing_Moss.py` loads `chat_template_default.py`; the fallback legacy
   format exists but should not be the default parity target. Use the template
   path used in the model card.

6. Time markers are optional and disabled for baseline parity.
   `enable_time_marker=False` in the model-card inference snippet. Keep that
   setting for the first conversion and only add time-marker support after
   baseline parity.

7. MOSS is English-only.
   Unlike `Qwen3-ASR` and Cohere Transcribe, this does not need language prompt
   plumbing for the first backend. That keeps the first end-to-end target
   narrower.

8. Cohere is encoder-decoder with cross-attention; MOSS is audio-embedding
   injection into a decoder-only Qwen3 stack.
   Cohere's cache-external decoder plan is still useful for eventual CoreML
   mechanics, but the MLX implementation path should look closer to Qwen3-ASR
   and Qwen2-Audio than Cohere.

## Project Layout

Current local shape:

```text
projects/moss-mlx-conversion/
  README.md
  docs/
    PLAN.md
    PROGRESS.md
  pyproject.toml
  src/moss_mlx_conversion/
    __init__.py
    config.py
    dump.py
    mlx_compat.py
    paths.py
    processor.py
    conversion/
      __init__.py
      convert.py
      weights.py
    reference/
      __init__.py
      download.py
      hf.py
      processor_parity.py
      reference.py
    runtime/
      __init__.py
      audio.py
      streaming_eval.py
      transcribe.py
    model/
      __init__.py
      moss.py
      audio_encoder.py
      adapter.py
  tests/
    test_import.py
  artifacts/
    # ignored generated caches, weights, reports, and eval outputs
```

The first implementation ended up using a role-based package split rather than
the original flat-file sketch:

```text
reference/   upstream PyTorch/HF snapshot, processor parity, tensor dumps
conversion/  safetensor inspection and BF16 MLX weight conversion
runtime/     MLX smoke transcription and streamed HF evaluation
model/       MLX model modules: audio encoder, adapter, MOSS wrapper
docs/        plan and live progress/results
```

Historical sketch from the first planning pass:

```text
projects/moss-mlx-conversion/
  README.md
  PLAN.md
  pyproject.toml
  src/moss_mlx_conversion/
    config.py
    convert.py
    reference.py
    weights.py
    processor.py
    model/
      moss.py
      audio_encoder.py
      qwen3_text.py
      adapter.py
    validation/
      parity.py
      transcribe.py
  tests/
    test_processor_parity.py
    test_weight_mapping.py
    test_component_parity.py
  scripts/
    download_reference.py
    run_short_fixture.py
  artifacts/
```

Use `uv` for all Python commands. Start `pyproject.toml` from the visible local
template at `/home/simon/github/python-project-template/pyproject.toml` if this
becomes an executable package.

Keep large model artifacts, generated MLX weights, datasets, and run outputs out
of Git. Store them under ignored repo-local directories or the machine's normal
Peacock artifact locations once this moves beyond planning.

## Execution Plan

### Phase 1: Lock The Reference

Deliverable: a reference runner that can produce one known MOSS transcript and
dump component tensors.

Steps:

1. Create the local Python project under `projects/moss-mlx-conversion`.
2. Add normal dependencies for the conversion runtime:
   `torch`, `transformers`, `huggingface-hub`, `safetensors`, `soundfile`,
   `librosa`, `numpy`, `mlx`, `mlx-lm`, and `mlx-audio` if importing shared
   output types locally.
3. Write `scripts/download_reference.py` using `huggingface_hub.snapshot_download`
   with explicit allow patterns.
4. Write `reference.py` that mirrors the model-card inference flow exactly:
   load `MossForCausalLM` with `trust_remote_code=True`, load tokenizer, load
   `MossProcessor`, load `chat_template_default.py`, run greedy generation.
5. Use a short public fixture first, ideally the same LibriSpeech Mr. Quilter
   sample used by Qwen examples or another local 16 kHz WAV fixture.
6. Dump these parity tensors:
   - raw waveform stats
   - mel features from `MossProcessor`
   - `input_ids`, `attention_mask`, `audio_input_mask`, `audio_data_seqlens`
   - `audio_model(...).last_hidden_state`
   - `audio_adapter(...)`
   - final merged `inputs_embeds`
   - first prefill logits
   - first 5 greedy decode tokens and logits

Commands should look like:

```bash
uv run --project projects/moss-mlx-conversion moss-reference \
  --model-id OpenMOSS-Team/MOSS-Transcribe-preview-2B \
  --audio /path/to/fixture.wav \
  --dump-dir artifacts/reference/mr-quilter
```

Gate:

- The reference runner prints a plausible transcript.
- Tensor dumps are saved with shapes, dtypes, max/mean stats, and small checksums.

### Phase 2: Port Processor First, With Exact Parity

Deliverable: an MLX-side processor that reproduces MOSS prompt/token/mask/mel
layout before any model code is trusted.

Steps:

1. Reimplement `MelConfig` and `MossProcessor` behavior locally.
2. Keep `enable_time_marker=False`.
3. Load and interpret `chat_template_default.py` or vendor a JSON-equivalent
   static representation after verifying it is stable.
4. Match the exact feature extractor parameters:
   `feature_size=128`, `sampling_rate=16000`, `n_fft=400`, `hop_length=160`.
5. Match `_get_feat_extract_output_lengths` from `processing_Moss.py`.
6. Preserve both outputs:
   `input_ids` and `audio_input_mask`.

Gate:

- For the same waveform, local processor output equals upstream processor output
  for `input_ids`, `audio_input_mask`, and `audio_data_seqlens`.
- Mel parity is within tolerance. If the local NumPy/MLX mel path differs,
  keep the upstream Transformers feature extractor in the reference path and
  debug before moving to model parity.

### Phase 3: Port The MLX Model As A Full Vertical Slice

Deliverable: `MossMLXModel.generate(audio)` produces a transcript, even before
performance or quantization are tuned.

Implementation pieces:

1. `config.py`
   - Parse `MossConfig`.
   - Store `language_config`, `audio_config`, adapter hidden size, and special
     token IDs.
   - Ignore misleading top-level Qwen defaults that are not used by
     `MossModel`.

2. Text decoder
   - First choice: use `mlx_lm`'s Qwen3 implementation if its config matches
     MOSS `language_config`.
   - Fallback: adapt `mlx-audio`'s `qwen3_asr.TextModel` implementation.
   - Required behavior: accept `input_embeddings` for prefill and use KV cache
     for autoregressive decode.

3. Audio encoder
   - Port `Qwen3OmniMoeAudioEncoder` semantics from Transformers, not from
     memory.
   - Start full-size and BF16/FP32 compatible; do not make a tiny surrogate.
   - Preserve convolution/downsampling/windowing behavior and output shape.
   - If the upstream encoder contains MoE-specific branches, implement them
     directly rather than dropping them.

4. Adapter
   - Implement `MossGatedMLP` exactly:
     `down_proj(silu(gate_proj(x)) * up_proj(x))`.

5. Embedding merge
   - Embed `input_ids` through Qwen3 token embeddings.
   - Flatten/mask-replace positions where `audio_input_mask` is true.
   - Do not rely on placeholder token ID alone.

6. Generation
   - Build `input_embeddings` for the full prompt.
   - Use `mlx_lm.generate.generate_step` with `input_embeddings`.
   - Stop on `processor.end_token_id` / Qwen EOS.
   - Decode generated IDs through the original tokenizer.

Gate:

- End-to-end MLX greedy output matches PyTorch for a short fixture or diverges
  only after a known token step captured by parity logs.
- Component parity identifies where any divergence begins.

### Phase 4: Weight Conversion

Deliverable: `convert.py` maps upstream safetensors into an MLX checkpoint that
loads without ad hoc manual edits.

Steps:

1. Read `model.safetensors.index.json` and source safetensors with
   `safetensors`.
2. Build explicit weight maps for:
   - `model.audio_model.*`
   - `model.audio_adapter.gate_proj/up_proj/down_proj.*`
   - `model.language_model.*`
   - tied `lm_head.weight` / `embed_tokens.weight`
3. Write a mapping report:
   - source tensors consumed
   - destination tensors written
   - skipped tensors and why
   - shape mismatches
4. Save BF16 MLX weights first.
5. Add optional quantized variants only after BF16 parity:
   - likely 8-bit text decoder
   - keep audio tower unquantized until quality is known

Example command:

```bash
uv run --project projects/moss-mlx-conversion moss-convert \
  --model-id OpenMOSS-Team/MOSS-Transcribe-preview-2B \
  --output-dir artifacts/mlx/MOSS-Transcribe-preview-2B-bf16 \
  --dtype bf16 \
  --write-report artifacts/mlx/MOSS-Transcribe-preview-2B-bf16/mapping.json
```

Gate:

- `missing_source_tensors == 0`.
- `unmapped_destination_tensors == 0`, except explicitly generated buffers.
- Model loads and runs the short fixture.

### Phase 5: Parity And Quality Validation

Deliverable: a small but real validation report that can be pasted into an
upstream PR.

Test sets:

- One tiny smoke fixture for fast iteration.
- 20 LibriSpeech `test-clean` utterances for WER sanity.
- 100 LibriSpeech `test-clean` utterances once the pipeline is stable.
- A small noisy/accented English sample set if this becomes useful for Peacock.

Metrics:

- Exact transcript for smoke fixture.
- Token-by-token divergence position vs PyTorch.
- WER/CER on short LibriSpeech subset.
- RTFx on Apple Silicon.
- Prompt tokens, generation tokens, prompt tokens/sec, generation tokens/sec.
- Peak memory if easy to measure.

Required parity checkpoints:

- Processor parity.
- Audio encoder output max/mean difference.
- Adapter output max/mean difference.
- Prefill logits top-k overlap.
- First 5 generated token IDs.
- Full transcript.

Gate:

- BF16 MLX should be close enough to PyTorch that transcript quality is not
  obviously degraded on the smoke and 20-file sets.
- Quantized variants must be compared against BF16 before publishing.

### Phase 6: Package For `mlx-audio`

Deliverable: a contribution-ready backend and model card draft.

Backend name:

- `moss_transcribe`

Likely `mlx-audio` paths:

```text
mlx_audio/stt/models/moss_transcribe/
  README.md
  __init__.py
  config.py
  moss_transcribe.py
  audio.py
  tokenizer.py
```

API target:

```python
from mlx_audio.stt import load

model = load("mlx-community/MOSS-Transcribe-preview-2B-8bit")
result = model.generate("speech.wav")
print(result.text)
```

CLI target:

```bash
uv run mlx_audio.stt.generate \
  --model mlx-community/MOSS-Transcribe-preview-2B-8bit \
  --audio speech.wav \
  --format txt
```

Publishing target:

- BF16 repo first if size is acceptable.
- 8-bit repo after BF16 parity is stable.
- Preserve Apache-2.0 attribution.
- Model card should clearly say this is English ASR and a format conversion of
  the OpenMOSS release.

### Phase 7: CoreML / FluidAudio Follow-On

This is not the first implementation target, but the MLX plan should leave a
clean bridge to it.

Recommended `mobius` path:

```text
models/stt/moss-transcribe-preview-2b/coreml/
```

CoreML component split:

- `moss_audio_encoder_adapter.mlpackage`
- `moss_embedding.mlpackage` if FluidAudio wants explicit token embedding
- `moss_decoder_prefill.mlpackage`
- `moss_decoder_cache_external.mlpackage`
- optional static-shape v2 decoder if ANE dispatch needs it

Things to copy from Fluid's existing work:

- Qwen3-ASR conversion split between audio encoder, embedding, LM head,
  decoder prefill, and decode stack.
- Cohere host-managed KV cache design.
- Cohere v2 static decoder lesson: fixed mask shape can be necessary for ANE.
- `coreml-cli` profiling after conversion, not only successful compilation.

CoreML-specific risk list:

- Qwen3 RoPE layout must match the model implementation.
- Decoder precision may need FP32 even when weights are compact.
- Dynamic cache lengths can produce runtime or accuracy problems.
- ANE dispatch has to be verified; `.all` can silently fall back.
- MOSS audio encoder is larger/different than Qwen3-ASR 0.6B, so do not assume
  the same performance envelope.

## First Real Work Chunk

The first coding chunk should be large enough to produce a transcript or a
specific parity failure:

1. Create `pyproject.toml` and package skeleton.
2. Implement `moss-reference` and `moss-processor-parity`.
3. Download only metadata/code/tokenizer first, then the model weights when
   ready for the full run.
4. Run one short fixture through upstream PyTorch.
5. Dump parity tensors.
6. Implement local processor parity.
7. Start MLX model with text decoder and adapter wiring.

Done means either:

- a PyTorch reference transcript plus processor parity report exists, or
- the run fails at a concrete dependency/model-load issue with exact stderr and
  the next command needed.

## Open Questions To Resolve By Running Code

- Does current `mlx_lm.models.qwen3` accept MOSS `language_config` without a
  local copy?
- How close is `Qwen3OmniMoeAudioEncoder` to the existing `mlx-audio`
  `qwen3_asr.AudioEncoder` implementation?
- Does MOSS require any `transformers` behavior hidden in
  `Qwen3OmniMoeAudioEncoder` beyond standard attention, convolution, and
  downsampling?
- Is the model-card `chat_template_default.py` stable enough to vendor as data,
  or should the backend parse/load it?
- What is the first-token logit tolerance between PyTorch BF16 and MLX BF16 on
  this machine?
- Does 8-bit text decoder quantization preserve transcript quality?

## Success Criteria

Minimum success:

- Local reference runner produces an upstream MOSS transcript.
- MLX backend loads converted BF16 weights.
- One short audio fixture transcribes end to end.
- The plan has parity dumps that identify any remaining divergence.

Useful success:

- 20-file LibriSpeech subset is within a small WER delta of PyTorch reference.
- `mlx-audio` style backend can be run through Python and CLI.
- Conversion report has no unexplained missing or unused weights.

Upstream-ready success:

- BF16 and 8-bit HF repos are reproducible from `convert.py`.
- `mlx-audio` PR includes backend, docs, smoke tests, and validation table.
- Follow-on `mobius` issue or branch has a clear CoreML component split.
