# Kyutai STT Fine-tuning: How It Works

This document explains the key concepts behind fine-tuning Kyutai speech-to-text models.

## 1. Model Architecture: Streaming Decoder with Audio Windows

Kyutai STT is a **streaming decoder-only model** designed for real-time transcription. Unlike Whisper (which sees all audio before generating text), Kyutai processes audio incrementally with **no lookahead**.

At each step t, the model sees:
- **Text history**: All previously generated text tokens (`[y₀, y₁, ..., y_{t-1}]`)
- **Audio history**: Audio frames up to the current window (NOT future audio)

```
step t inputs                              model                        output
─────────────────────────────────────   ┌──────────────────────┐       ───────
  text_history = [y₀ … y_{t-1}]      ──▶│                      │──▶     y_t
  audio_history = [a₀ … a_{window}]  ──▶│   STT Decoder f(·)   │
                                        └──────────────────────┘
```

**Key architectural features:**
- **KV caching**: Past computations are cached for efficient autoregressive generation
- **Windowed audio encoding**: Audio is encoded in windows as generation progresses
- **Causal attention**: Each position only attends to previous positions (no future lookahead)

**Input tensor shape**: `[batch_size, seq_len, 33]` where:
- Channel 0: Text tokens (one per frame)
- Channels 1-32: Audio codebook tokens from the neural codec

### Kyutai vs Whisper: Key Difference

| Aspect | Whisper | Kyutai |
|--------|---------|--------|
| Audio encoder | **Bidirectional** - sees ALL audio at once | **Causal** - sees audio up to current window only |
| Architecture | Encoder-decoder | Decoder-only |
| Use case | Batch transcription | **Streaming** / real-time |
| Fine-tuning data needs | Less data - encoder provides full context | **More data** - must learn from partial context |

This architectural difference means Kyutai may require more training data to fine-tune effectively, as the model must learn to predict text without seeing future audio.

### Training vs Inference

- **model.generate()**: Starts with `<bos>`, encodes audio in windows, generates text autoregressively with KV cache
- **model.forward()**: Single pass over the full sequence - used for training with teacher forcing (causal attention still applies)

For training, we prepare the full text sequence aligned to audio frames in advance, then run one forward pass.

## 2. Text Tokenization: UNK and PAD Tokens

Text tokens are timestamped - each frame (~80ms) gets exactly one token.

**Special tokens**:
- `<pad>` (ID=3): Represents silence or time when no text is being produced
- `<unk>` (ID=0): Marks word boundaries - appears one frame before each word starts
- `<bos>` (ID varies): Beginning of sequence, always at position 0

**Example timeline** (speech from ~0.5s to 2s):
```
<bos><pad><pad><pad><unk>hello<pad><unk>world<pad><pad>
```

The `<unk>` token serves as a "word boundary marker" - the model learns that `<unk>` precedes actual word tokens.

### STT Delay

A delay of ~0.5 seconds is added so the model has audio context before producing text. Additionally, text tokens are placed at the **end** of each word's audio (not the start), giving the model the full word's audio before predicting it:

```
Audio:  [  hello being spoken  ][  world being spoken  ]
Text:   <bos><pad><pad><pad><pad><unk>hello<pad><unk>world<pad>
                                     ↑
                          Token placed at word END + delay
```

## 3. Label Setup: Weighted PAD Token Loss

Labels are what we compare model outputs to during training. The key insight is:

**PAD tokens are downweighted in the loss function** (following the official moshi-finetune approach).

Since PAD tokens dominate the sequence (representing silence between words), we apply a reduced weight (default 0.25) to PAD token predictions. This prevents the model from learning to always predict PAD while still providing some learning signal for silence transitions.

```python
# From data_collator.py
labels = text_input_tokens.clone()
labels[0] = -100           # Never predict the initial BOS
labels[-1] = -100          # No target for last position

# Create loss weights: PAD tokens get reduced weight, others get full weight
loss_weights = torch.ones(seq_len, dtype=torch.float32)
is_pad = (labels == pad_token_id)
loss_weights[is_pad] = pad_weight  # e.g., 0.25
loss_weights[0] = 0.0   # BOS position - no loss
loss_weights[-1] = 0.0  # Last position - no loss
```

**Example** (with pad_weight=0.25):
```
Input:    <bos>  <pad>  <pad>  <unk>  hello  <pad>  <pad>
Labels:   -100   <pad>  <pad>  <unk>  hello  <pad>  <pad>
Weights:  0.0    0.25   0.25   1.0    1.0    0.25   0.25
```

The weighted loss is computed in the trainer by multiplying per-token cross-entropy loss by these weights.

## 4. Audio Tokenization

Audio is converted to discrete tokens using Kyutai's neural codec (Mimi, similar to EnCodec):

1. **Resample** to 24kHz
2. **Encode** with `model.codec_model.encode()` → produces 32 codebook streams
3. **Frame rate**: ~12.5 fps (80ms per frame, from `frame_size=1920` samples at 24kHz)

**Each frame = 32 audio tokens** (one from each codebook). The codec uses Residual Vector Quantization (RVQ):
- Codebook 1: coarse audio structure (most important)
- Codebooks 2-32: progressively finer residual details

So a 10-second clip → ~125 frames → 125 × 32 = 4000 audio tokens total.

```python
# From train.py
audio_inputs = processor(audio_24k, sampling_rate=24000)
codec_output = model.codec_model.encode(input_values, padding_mask)
audio_codes = codec_output.audio_codes.transpose(1, 2)  # [batch, seq_len, 32]
```

Each frame has 32 codebook tokens representing different aspects of the audio at that time.

### Batching Audio Encoding (Optional Optimization)

The current implementation encodes audio **sequentially** (one sample at a time) during preprocessing. However, both the processor and codec model support **batched encoding** with proper padding:

```python
# Current: Sequential (in process_sample loop)
audio_inputs = processor(audio_24k, sampling_rate=24000)  # Single sample
codec_output = model.codec_model.encode(input_values, padding_mask)

# Alternative: Batched
batch_audios = [sample['audio_24k'] for sample in samples]
batch_inputs = processor(batch_audios, sampling_rate=24000, padding=True)
# Returns:
#   input_values: [batch_size, max_audio_samples] - padded waveforms
#   padding_mask: [batch_size, max_audio_samples] - 1=real, 0=padding

codec_output = model.codec_model.encode(input_values, padding_mask)
# Returns audio_codes: [batch_size, 32, max_seq_len] - with padding

# Important: Strip padding after encoding to restore natural lengths
for i, sample in enumerate(samples):
    real_frames = calculate_real_frames(padding_mask[i])
    sample['audio_codes'] = codec_output.audio_codes[i, :, :real_frames]
```

**Why batching isn't currently used:**

| Factor | Impact |
|--------|--------|
| Forced alignment (MMS_FA) | Sequential, CPU-only - dominates preprocessing time |
| Codec encoding | Already fast on GPU (~100ms per sample) |
| Speedup potential | ~10-20% of total preprocessing time |
| Results caching | Preprocessing is one-time (pickled) |

**Key point**: If you batch the codec step, you must **strip the padding afterward** before saving. Otherwise, text scheduling would create sequences matching the padded length, conflating "silence PAD" (meaningful) with "batch PAD" (artifact).

The processor correctly returns `padding_mask` for variable-length audio, and the codec respects this mask during encoding. The architecture supports batching - it's just not worth the added complexity given the forced alignment bottleneck.

## 5. Collating Text and Audio for Training

The data collator combines text and audio tokens into the 33-channel input:

```python
# From data_collator.py

# 1. Create text tensor filled with PAD tokens
text_input_tokens = torch.full((seq_len,), pad_token_id)

# 2. Place scheduled tokens at their pre-computed positions
#    (token_info['start'] already contains word_end + stt_delay from schedule_tokens_discrete)
for token_info in token_schedule:
    frame_pos = int(token_info['start'] / frame_hop_s)
    text_input_tokens[frame_pos] = token_info['token_id']

# 3. Force BOS at position 0
text_input_tokens[0] = bos_token_id

# 4. Concatenate with audio codes
text_tokens = text_input_tokens.unsqueeze(-1)  # [seq_len, 1]
input_ids = torch.cat([text_tokens, audio_codes], dim=-1)  # [seq_len, 33]
```

### Token Schedule Generation

The `schedule_tokens_discrete()` function maps words to frame positions:

1. Get word-level timestamps from alignment
2. Add STT delay (~0.5s) to each word's end time
3. Place `<unk>` one frame before word start
4. Place word tokens starting at the delayed end time
5. Fill gaps with `<pad>`

**Spillover handling**: If a word's tokens would overlap the next word, either:
- `"truncate"`: Cut off tokens that don't fit
- `"shift"`: Push subsequent words later (preserves all tokens)

## 6. Text-Audio Alignment

To create time-aligned training data, we need word-level timestamps. Two approaches:

### Option A: Whisper Timestamped

Use Whisper's transcription with word-level timestamps:

```python
import whisper_timestamped as whisper
result = whisper.transcribe(model, audio, language="en")
# result["segments"][i]["words"] contains word timestamps
```

**Pros**: Simple, single model
**Cons**: Harder to get correct timestamps if transcript needs correction

### Option B: Forced Alignment (Recommended)

Use a dedicated alignment model (e.g., TorchAudio's MMS_FA) with your **ground truth text**:

```python
from torchaudio.pipelines import MMS_FA
bundle = MMS_FA
model = bundle.get_model()
tokenizer = bundle.get_tokenizer()
aligner = bundle.get_aligner()

# Align ground truth text to audio
emission, _ = model(waveform)  # Character probabilities per 20ms frame
tokens = tokenizer(normalized_words)  # Character-level tokenization
token_spans = aligner(emission[0], tokens)  # CTC alignment
```

**How MMS_FA works**:
1. Neural network outputs per-frame character probabilities ("emissions") at 16kHz/20ms frames
2. Text is split into characters
3. CTC aligner finds optimal path matching characters to frames via dynamic programming

**Parallelization**: Runs on CPU (batch_size=1). The neural forward pass could batch on GPU, but the CTC alignment step is inherently sequential per sample. Since alignment is a one-time preprocessing step (not repeated each epoch), CPU is sufficient.

**Pros**:
- Uses your exact ground truth text
- Can apply corrections to text and re-align
- Better for fine-tuning on specific vocabulary/terminology

**Cons**: Requires text normalization (MMS_FA only handles lowercase letters)

### Text Normalization for Alignment

MMS_FA needs normalized text, but we want to preserve original casing/punctuation:

```
Original:       "Phi-2 is fine-tuning!"
Normalized:     ["phi", "is", "fine", "tuning"]
Mapping:        [[0], [1], [2, 3]]

After alignment, reverse-map timestamps:
- "phi" timestamps → "Phi-2"
- "fine" + "tuning" timestamps → "fine-tuning!"
```

### Applying Corrections

If your dataset has transcription errors, forced alignment lets you:
1. Create a corrections file mapping wrong→right words
2. Apply corrections to both text and timestamps
3. Re-run alignment with corrected text

```json
// word_corrections.json
{
    "llama": "LLaMA",
    "gpt four": "GPT-4"
}
```

## Summary: The Training Pipeline

1. **Load audio** → resample to 24kHz
2. **Get timestamps** via forced alignment with ground truth text
3. **Encode audio** → 32 codebook streams via neural codec
4. **Schedule tokens** → map words to frames with UNK markers and STT delay
5. **Create labels** → apply weighted loss for PAD tokens, ignore BOS and last position
6. **Concatenate** → [text, audio] into [seq_len, 33] tensor
7. **Train** → forward pass, compute weighted loss, backprop

## 7. Platform Compatibility

### Mac/Apple Silicon (MPS)

**Important**: The 2.6B model (`kyutai/stt-2.6b-en-trfs`) does not work correctly with the transformers library on Mac (CPU or MPS). It generates empty outputs (only PAD tokens).

| Model | Mac (MPS/CPU) | CUDA |
|-------|---------------|------|
| `kyutai/stt-1b-en_fr-trfs` | ✅ Works | ✅ Works |
| `kyutai/stt-2.6b-en-trfs` | ❌ Empty output | ✅ Should work |

**Recommendations**:
- Use the 1B model for local development on Mac
- Use CUDA (cloud GPU) for fine-tuning the 2.6B model
- For inference-only on Mac, Kyutai recommends the MLX version (`kyutai/stt-2.6b-en-mlx`)

### HuggingFace Download Issues

If you encounter download errors with `xethub.hf.co`, the scripts automatically disable XET transfers by setting `HF_HUB_DISABLE_XET=1` before imports. This falls back to standard HTTP downloads.

## 8. Recommended Hyperparameters

Based on experiments with the `Trelis/llm-lingo` dataset (7 train, 6 validation samples):

| Parameter | Recommended Value | Notes |
|-----------|-------------------|-------|
| `pad_weight` | 0.25 | Weight for PAD token loss |
| `learning_rate` | 1e-4 | Higher than typical LLM fine-tuning |
| `num_epochs` | 5 | Prevents overfitting on small datasets |
| `lora_rank` | 32 | Good balance of capacity vs efficiency |
| `lora_alpha` | 32 | Equal to rank for stable training |

**Results**: These settings achieved +9.5% WER improvement over the base model on validation data.

**Hyperparameter search summary**:
| Pad Weight | LR | Epochs | WER Improvement |
|------------|-----|--------|-----------------|
| 0.5 | 2e-6 | 12 | -2.5% (worse) |
| 0.5 | 3e-5 | 6 | +5.2% |
| 0.25 | 5e-5 | 3 | +2.2% |
| 0.25 | 1e-4 | 3 | +7.4% |
| **0.25** | **1e-4** | **5** | **+9.5%** ✓ |

## 9. Converting Fine-tuned Models for Rust/Candle Server

The Kyutai Rust server (`stt-rs`) uses a different model format than HuggingFace Transformers. After fine-tuning with the transformers library, you need to convert the weights to Candle format to use them with the production Rust server.

### Model Format Differences

| Aspect | HuggingFace Transformers | Kyutai Candle |
|--------|-------------------------|---------------|
| Framework | PyTorch/Transformers | Rust Candle |
| Embedding | Combined `embed_tokens` (73570) | Split `text_emb` (8001) + `emb.{0-31}` (2049 each) |
| Q/K/V projections | Separate `q_proj`, `k_proj`, `v_proj` | Combined `in_proj_weight` |
| Layer norms | `input_layernorm.weight` [2048] | `norm1.alpha` [1, 1, 2048] |
| MLP | `mlp.fc1`, `mlp.fc2` | `gating.linear_in`, `gating.linear_out` |
| Audio codec | Included in model | Loaded separately |

### Conversion Script

Use `convert_transformers_to_candle.py` to convert fine-tuned models:

```bash
# Convert a fine-tuned model
uv run python convert_transformers_to_candle.py \
    --input ./kyutai-finetuned \
    --output ./kyutai-finetuned-candle/model.safetensors

# Or convert directly from HuggingFace (for testing)
uv run python convert_transformers_to_candle.py \
    --input kyutai/stt-1b-en_fr-trfs \
    --output ./converted/model.safetensors
```

The converter:
1. Splits the combined embedding into text and audio embeddings
2. Applies inverse rotary permutation to Q/K projections
3. Combines separate Q/K/V into `in_proj_weight`
4. Reshapes layer norms from [hidden] to [1, 1, hidden]
5. Renames layer components to match Candle naming
6. Copies `extra_heads` (VAD) from original Candle model
7. Skips `codec_model` weights (loaded separately by Candle server)

### Validation

Use `tests/test_conversion.py` to validate the conversion:

```bash
# Compare converted weights against original Candle model
uv run python tests/test_conversion.py \
    --original kyutai/stt-1b-en_fr-candle \
    --converted ./converted/model.safetensors

# Expected output:
# ✓ All keys and shapes match!
# ✓ All weight values match within tolerance!
# ✓ Conversion validation PASSED
```

### Using with Rust Server

After conversion, update the Rust server config to use your converted weights:

```toml
# config-stt-finetuned.toml
[modules.asr]
type = "BatchedAsr"
lm_model_file = "/path/to/kyutai-finetuned-candle/model.safetensors"
text_tokenizer_file = "hf://kyutai/stt-1b-en_fr-candle/tokenizer_en_fr_audio_8000.model"
audio_tokenizer_file = "hf://kyutai/stt-1b-en_fr-candle/mimi-pytorch-e351c8d8@125.safetensors"
```

Note: The tokenizer and audio codec (Mimi) are NOT fine-tuned, so you can continue using the original files from `kyutai/stt-*-candle`.
