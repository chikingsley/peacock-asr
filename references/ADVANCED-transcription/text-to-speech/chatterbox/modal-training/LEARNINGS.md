# Chatterbox TTS Fine-Tuning: Learnings

## Training Results (test run: 5 samples, 3 epochs)

| Metric | Epoch 1 | Epoch 2 | Epoch 3 |
|--------|---------|---------|---------|
| Train loss | 3.12 | 3.05 | 2.94 |
| Eval loss | 12.94 | 12.71 | 12.47 |

- 7.8M trainable parameters out of 540M total (1.46% with LoRA rank 32)
- Both standard TTS and voice-cloned audio samples generated successfully
- Train and eval loss both decreasing — LoRA is learning

## Architecture Findings

### Chatterbox is NOT standard transformers
- Custom `chatterbox.tts.ChatterboxTTS` model — not an `nn.Module` at the top level
- Cannot call `cb_model.parameters()` — must access `cb_model.t3` directly
- Three submodels: T3 (text→speech tokens), S3Gen (vocoder), VoiceEncoder (speaker identity)
- Only T3 is trained; S3Gen and VoiceEncoder stay frozen

### T3 Model Internals
- T3 wraps a `LlamaModel` (attribute `t3.tfmr`) for the standard model, GPT-2 for turbo
- Forward signature: `T3.forward(*, t3_cond: T3Cond, text_tokens, text_token_lens, speech_tokens, speech_token_lens, training=False)`
- `T3Cond` is a dataclass: `{speaker_emb, clap_emb, cond_prompt_speech_tokens, cond_prompt_speech_emb, emotion_adv}`
- `emotion_adv` defaults to `0.5` as a float, but T3 internally calls `.view()` on it — must pass as `torch.Tensor` with correct batch shape
- Must call `t3.prepare_conditioning(t3_cond)` before `t3.forward()` — it runs cond through encoder
- Output has `.text_logits` (shape: [B, seq_len, vocab_size]) and `.speech_logits`

### Key Config Values (Standard model)
| Config | Value |
|--------|-------|
| text_tokens_dict_size | 704 |
| speech_tokens_dict_size | 8194 |
| speaker_embed_size | 256 |
| start_text_token / stop_text_token | 255 / 0 |
| start_speech_token / stop_speech_token | 6561 / 6562 |
| max_text_tokens | 2048 |
| max_speech_tokens | 4096 |
| backbone | Llama_520M (532M params) |

### VoiceEncoder Expects Mel Spectrograms
- LSTM-based, expects `(batch, time, 40)` — 40 mel bands at 16kHz
- NOT raw audio waveforms — passing raw audio gives `Expected 40, got 709440`
- Mel extraction params: `n_fft=400, hop_length=160, win_length=400` at 16kHz sample rate
- Best approach: use `cb_model.prepare_conditionals(audio_path)` then read from `cb_model.conds`
- Fallback: manually compute mel spectrograms with `torchaudio.transforms.MelSpectrogram`

## Problems Encountered & Solutions

### 1. Modal secrets not in dev environment
**Problem**: `wandb-secret` Modal secret didn't exist in the dev environment.
**Solution**: Pass WandB API key from local `.env` as a function argument instead of relying on Modal secrets.

### 2. torchcodec / ffmpeg incompatibility
**Problem**: `datasets` tried to use `torchcodec` for audio loading, which had ffmpeg version conflicts.
**Solution**: Pin `datasets==3.3.2` which uses `soundfile` backend instead.

### 3. T3.loss() cross-entropy shape mismatch
**Problem**: `Expected target size [4, 704], got [4, 256]`. T3 outputs logits as `[B, seq_len, vocab]` but `F.cross_entropy` needs `[B, vocab, seq_len]`.
**Solution**: Transpose logits before passing to `F.cross_entropy`. Use manual loss computation instead of `T3.loss()`.

### 4. Dynamic vs fixed padding
**Problem**: T3.loss asserts `text_tokens.size(1) == text_token_lens.max()`. Fixed padding to 256 broke this invariant.
**Solution**: Dynamic padding — pad to max length in each batch, not a fixed constant.

### 5. HF Trainer's prediction_step incompatibility
**Problem**: Default `prediction_step` passes the batch as kwargs to `model()`, but T3 doesn't accept `speaker_emb` etc. as keyword args.
**Solution**: Override `prediction_step` in custom ChatterboxTrainer to call our `compute_loss` instead.

### 6. Turbo model loading fails
**Problem**: `ChatterboxTTS.from_pretrained(device=device, model_id="ResembleAI/chatterbox-turbo")` raises TypeError. Falls back to loading the standard (Llama) model.
**Status**: Unresolved. The turbo model variant loading needs investigation of the `from_pretrained` API.

## Tips for Future Work

1. **Always inspect model signatures first**: `inspect.signature(t3.forward)` and `inspect.getsource(t3.loss)` saved hours vs. guessing
2. **Chatterbox preprocessing is the bottleneck**: Speaker embedding extraction + S3Gen tokenization takes significant time. Cache to volume.
3. **Small datasets still work**: Even 5 samples showed decreasing loss. For production quality, recommend 1hr+ of audio data.
4. **Voice cloning is free**: Zero-shot, just pass `audio_prompt_path` at inference. No training needed for cloning ability.
5. **The `transformers` SDPA bug**: Set `TRANSFORMERS_ATTN_IMPLEMENTATION=eager` in the environment to avoid issues with voice references in some transformers versions.
