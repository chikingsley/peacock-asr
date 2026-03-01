# Voxtral Modal Training

Fine-tune Voxtral-Mini-3B for transcription using LoRA on Modal GPUs.

## Prerequisites

- Modal account and CLI (`uv pip install modal && modal setup`)
- `.env` file with `HF_TOKEN` (copy from `.env.example`)
- Modal secret `wandb-secret` with `WANDB_API_KEY` (optional)

## Usage

```bash
# Quick test (3 samples, 1 epoch)
uv run modal run --env=dev voxtral_modal_train.py --max-samples 3 --epochs 1

# Full training
uv run modal run voxtral_modal_train.py --epochs 3

# Push to HuggingFace Hub
uv run modal run voxtral_modal_train.py --hub-repo your-org/voxtral-ft
```

## CLI Args

| Arg | Default | Description |
|-----|---------|-------------|
| `--dataset` | `Trelis/llm-lingo` | HuggingFace dataset |
| `--model-name` | `mistralai/Voxtral-Mini-3B-2507` | Base model |
| `--epochs` | `3` | Training epochs |
| `--batch-size` | `1` | Batch size |
| `--learning-rate` | `5e-5` | LoRA learning rate |
| `--lora-rank` | `32` | LoRA rank |
| `--lora-alpha` | `32` | LoRA alpha |
| `--hub-repo` | `Trelis/voxtral-ft` | HF Hub push target |
| `--max-samples` | `None` | Limit samples (for testing) |
| `--download` | `False` | Download files after training |

## LoRA Config

- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- RSLoRA enabled
- Audio tower frozen during training

## Output

- Baseline WER (before training) and fine-tuned WER (after training) printed
- Merged model saved to Modal volume at `/data/output/voxtral-ft`
- Optional: pushed to HuggingFace Hub

## Results (Trelis/llm-lingo)

- Baseline WER: 30.6% → Fine-tuned WER: 14.6% (16.0pp improvement)
