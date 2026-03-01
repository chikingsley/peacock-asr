# Kyutai STT Fine-tuning

## Quick Start

```bash
# Install dependencies
uv sync

# Edit config.yaml to set your dataset and training options

# Run fine-tuning
uv run train.py --config config.yaml
```

## Evaluation

```bash
# Evaluate base model on validation split
uv run eval.py --config config.yaml --split validation

# Evaluate fine-tuned model
uv run eval.py --config config.yaml --model-path ./kyutai-finetuned --split validation

# Compare base vs fine-tuned (shows WER improvement)
uv run eval.py --config config.yaml --split validation --compare
```

## Configuration

Edit `config.yaml`:

- `model.model_id`: Base model (`kyutai/stt-1b-en_fr-trfs` or `kyutai/stt-2.6b-en_fr-trfs`)
- `dataset.name`: HuggingFace dataset with `audio` and `text` columns
- `lora.use_lora`: `true` for LoRA, `false` for full fine-tuning

## How It Works

See [KYUTAI_FINETUNING_EXPLAINED.md](KYUTAI_FINETUNING_EXPLAINED.md) for details on the model architecture, tokenization, and training data preparation.
