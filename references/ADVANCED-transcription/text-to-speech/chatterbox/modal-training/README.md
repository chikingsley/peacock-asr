# Chatterbox TTS Fine-Tuning on Modal

Fine-tune Chatterbox TTS (by Resemble AI) T3 model with LoRA on Modal GPUs.

Only the T3 model (text → speech tokens) is trained. S3Gen (vocoder) and VoiceEncoder are frozen. Voice cloning is zero-shot — built into Chatterbox inference, no training needed.

## Supported Models

| Model | Size | Backbone | Use Case |
|-------|------|----------|----------|
| `ResembleAI/chatterbox-turbo` (default) | 350M | GPT-2 | Fast, English-focused |
| `ResembleAI/chatterbox` | 500M | Llama | Multilingual |

## Prerequisites

1. [Modal](https://modal.com/) account with GPU access
2. HuggingFace token (for datasets and model push)
3. WandB API key (optional, for experiment tracking)

```bash
# Setup
cp .env.example .env
# Edit .env with your HF_TOKEN and optionally WANDB_API_KEY
```

## Usage

```bash
cd text-to-speech/chatterbox/modal-training

# Quick test (dev environment, 5 samples, 3 epochs)
uv run modal run --env=dev chatterbox_modal_train.py --max-samples 5 --epochs 3

# Full training
uv run modal run chatterbox_modal_train.py

# Custom parameters
uv run modal run chatterbox_modal_train.py \
    --model-name ResembleAI/chatterbox \
    --language fr \
    --epochs 30 \
    --learning-rate 5e-6
```

## CLI Arguments

| Arg | Default | Description |
|-----|---------|-------------|
| `--dataset` | `Trelis/ronan_tts_medium_clean` | HF dataset (needs `text` + `audio` columns) |
| `--model-name` | `ResembleAI/chatterbox-turbo` | Base model |
| `--epochs` | `50` | Training epochs |
| `--batch-size` | `4` | Batch size |
| `--learning-rate` | `1e-5` | Learning rate |
| `--lora-rank` | `32` | LoRA rank |
| `--lora-alpha` | `64` | LoRA alpha |
| `--hub-repo` | `Trelis/chatterbox-ft` | HF Hub push target |
| `--language` | `en` | Language ID (for multilingual model) |
| `--max-samples` | `None` | Limit samples (for testing) |
| `--download` | `True` | Download samples locally after training |

## What Happens

1. **Preprocessing**: Extracts speaker embeddings (VoiceEncoder) + acoustic tokens (S3Gen) + text tokens from each audio sample. Cached to Modal volume.
2. **LoRA Training**: Applies LoRA to T3 attention layers, trains with cross-entropy loss on speech token prediction. Speaker embeddings zeroed 20% of the time for generalization.
3. **Merge & Save**: Merges LoRA weights into T3, saves to volume.
4. **Sample Generation**: Generates 5 standard TTS samples + 5 voice-cloned samples from validation data.
5. **Download**: WAV files downloaded to `./output/chatterbox-ft/samples/`.

## Expected Output

- `standard_*.wav` — generated from text only (model's default voice)
- `cloned_*.wav` — generated with voice cloning from a reference audio
- Eval loss should decrease across epochs (visible in WandB or logs)
