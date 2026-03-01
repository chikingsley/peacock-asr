# Orpheus TTS Modal Training

Fine-tune Orpheus TTS models with LoRA on Modal's H100 GPUs.

## Setup

1. Install uv if not already installed:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Install dependencies:
```bash
cd modal-training
uv sync
```

3. Set up Modal:
```bash
uv run modal setup
```

4. Create a `.env` file with your HuggingFace token:
```bash
HF_TOKEN=hf_your_token_here
```

## Usage

### Basic Training

Run training with default parameters:
```bash
uv run modal run orpheus_modal_train.py
```

### Custom Parameters

```bash
# Use the pretrained base model
uv run modal run orpheus_modal_train.py --model-name canopylabs/orpheus-tts-0.1-pretrained

# Use the fine-tuned model (default)
uv run modal run orpheus_modal_train.py --model-name unsloth/orpheus-3b-0.1-ft

# Custom dataset and epochs
uv run modal run orpheus_modal_train.py \
    --dataset Trelis/my-custom-dataset \
    --epochs 5 \
    --learning-rate 1e-4

# Push to HuggingFace Hub (private)
uv run modal run orpheus_modal_train.py \
    --hub-repo username/orpheus-ft-custom

# Quick test with limited samples
uv run modal run orpheus_modal_train.py --epochs 1 --max-samples 10
```

### All Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset` | `Trelis/ronan_tts_medium_clean` | HuggingFace dataset with text + audio columns |
| `--model-name` | `unsloth/orpheus-3b-0.1-ft` | Base model to fine-tune |
| `--epochs` | `3` | Number of training epochs |
| `--batch-size` | `1` | Training batch size |
| `--learning-rate` | `2e-4` | Learning rate for LoRA fine-tuning |
| `--lora-rank` | `32` | LoRA rank |
| `--lora-alpha` | `64` | LoRA alpha |
| `--hub-repo` | `None` | HuggingFace Hub repo to push model (private) |
| `--max-samples` | `None` | Limit number of training samples |
| `--download/--no-download` | `True` | Download audio samples after training |

## Dataset Format

The script supports two dataset formats:

### 1. Raw Audio Dataset (text + audio columns)
The script will tokenize audio on-the-fly using SNAC:
```python
{
    "text": "Hello, this is a test.",
    "audio": {"array": [...], "sampling_rate": 24000},
    "source": "speaker_name"  # optional
}
```

### 2. Pre-tokenized Dataset (input_ids + labels + attention_mask)
If your dataset already has `input_ids`, it will be used directly:
```python
{
    "input_ids": [128259, ...],
    "labels": [128259, ...],
    "attention_mask": [1, 1, ...]
}
```

## Model Options

### unsloth/orpheus-3b-0.1-ft (Default)
- Pre-fine-tuned version with optimizations
- Faster inference, good baseline quality

### canopylabs/orpheus-tts-0.1-pretrained
- Original pretrained model from Canopy Labs
- Use for training from scratch

## Outputs

After training:
- **Model**: Saved to Modal volume at `/data/output/orpheus-ft/`
- **Audio samples**: Generated from validation split at `/data/output/samples/`
- **HuggingFace Hub**: Pushed to specified repo (if `--hub-repo` provided)

Audio samples (from validation set) are automatically downloaded to `./output/samples/` locally for quality inspection.

## Volume Management

Training data and models are cached on a Modal volume (`orpheus-training-data`):
- `/data/hf_cache/` - HuggingFace model cache
- `/data/output/` - Training outputs

To clear the volume:
```bash
uv run modal volume delete orpheus-training-data
```

## LoRA Configuration

The script uses the same LoRA configuration as `train-lora.py`:
- Rank: 32
- Alpha: 64
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, down_proj, up_proj
- Modules to save: lm_head, embed_tokens
- RSLoRA: enabled

## Verification

1. Run a quick test:
```bash
uv run modal run orpheus_modal_train.py --epochs 1 --max-samples 5
```

2. Check generated audio samples in `./output/samples/`

3. Listen to verify voice quality matches expectations

4. If using `--hub-repo`, check your model on HuggingFace Hub
