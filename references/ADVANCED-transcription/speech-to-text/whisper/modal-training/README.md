# Whisper Modal Training

Fine-tune Whisper-large-v3-turbo for transcription using LoRA on Modal GPUs.
Measures WER before and after training for comparison with Voxtral.

## Prerequisites

- Modal account and CLI (`uv pip install modal && modal setup`)
- `.env` file with `HF_TOKEN` (copy from `.env.example`)
- Modal secret `wandb-secret` with `WANDB_API_KEY` (optional)

## Usage

```bash
# Quick test (3 samples, 1 epoch)
uv run modal run --env=dev whisper_modal_train.py --max-samples 3 --epochs 1

# Full training
uv run modal run whisper_modal_train.py --epochs 2

# Push to HuggingFace Hub
uv run modal run whisper_modal_train.py --hub-repo your-org/whisper-ft
```

## CLI Args

| Arg | Default | Description |
|-----|---------|-------------|
| `--dataset` | `Trelis/llm-lingo` | HuggingFace dataset |
| `--model-name` | `openai/whisper-large-v3-turbo` | Base model |
| `--epochs` | `2` | Training epochs |
| `--batch-size` | `1` | Batch size |
| `--learning-rate` | `1e-4` | LoRA learning rate |
| `--lora-rank` | `32` | LoRA rank |
| `--lora-alpha` | `32` | LoRA alpha |
| `--hub-repo` | `Trelis/whisper-ft` | HF Hub push target |
| `--max-samples` | `None` | Limit samples (for testing) |
| `--download` | `False` | Download files after training |

## LoRA Config

- Target modules: `q_proj`, `v_proj`, `k_proj`, `out_proj`, `fc1`, `fc2`
- RSLoRA enabled
- Uses Seq2SeqTrainer with `predict_with_generate=True`

## Output

- Baseline WER (before training) and fine-tuned WER (after training) printed
- Merged model saved to Modal volume at `/data/output/whisper-ft`
- Optional: pushed to HuggingFace Hub

## Results (Trelis/llm-lingo)

- Baseline WER: 37.0% -> Fine-tuned WER: 15.1% (21.9pp improvement)
