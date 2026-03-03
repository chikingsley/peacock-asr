# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "torchaudio",
#     "transformers>=4.40",
#     "datasets",
#     "accelerate",
#     "jiwer",
#     "evaluate",
#     "mlflow",
# ]
# ///
"""Train a w2v-BERT 2.0 CTC phoneme head on LibriSpeech alignments.

Usage (RunPod A100 80GB):
    uv run train_phoneme_head.py \
        --hub-repo your-username/w2v-bert-phoneme-en

Usage (quick test, no hub push):
    uv run train_phoneme_head.py --no-push \
        --max-train-samples 500 --max-eval-samples 100

Based on: https://huggingface.co/blog/fine-tune-w2v2-bert
Adapted from character/word CTC to ARPABET phoneme CTC using
gilkeyio/librispeech-alignments (960h, pre-labeled by MFA).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import evaluate
import numpy as np
import torch  # noqa: TC002 - used at runtime in DataCollatorCTCWithPadding
from datasets import Audio, concatenate_datasets, load_dataset
from transformers import (
    SeamlessM4TFeatureExtractor,
    Trainer,
    TrainingArguments,
    Wav2Vec2BertForCTC,
    Wav2Vec2BertProcessor,
    Wav2Vec2CTCTokenizer,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stress stripping: AY1 -> AY, IH0 -> IH, ER2 -> ER
# ---------------------------------------------------------------------------
STRESS_RE = re.compile(r"[012]$")


def strip_stress(phone: str) -> str:
    return STRESS_RE.sub("", phone)


# ---------------------------------------------------------------------------
# Data collator (copied from HF blog, handles variable-length padding)
# ---------------------------------------------------------------------------
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2BertProcessor
    padding: bool | str = True

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        input_features = [
            {"input_features": f["input_features"]} for f in features
        ]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.pad(
            input_features, padding=self.padding, return_tensors="pt"
        )
        labels_batch = self.processor.pad(
            labels=label_features, padding=self.padding, return_tensors="pt"
        )
        # Replace padding with -100 so CTC loss ignores it
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--vocab-json",
        type=Path,
        default=Path(__file__).parent / "vocab.json",
        help="Path to vocab.json (default: training/vocab.json)",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="w2v-bert-phoneme-en",
        help="Local output directory for checkpoints",
    )
    p.add_argument(
        "--no-push",
        action="store_true",
        help="Disable pushing checkpoints to HuggingFace Hub",
    )
    p.add_argument(
        "--hub-repo",
        type=str,
        default="w2v-bert-phoneme-en",
        help="HuggingFace Hub repo name",
    )
    p.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Per-device train batch size (default: 16)",
    )
    p.add_argument(
        "--gradient-accumulation",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4)",
    )
    p.add_argument(
        "--learning-rate",
        type=float,
        default=3e-5,
        help="Peak learning rate (default: 3e-5)",
    )
    p.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Limit training samples (for testing)",
    )
    p.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help="Limit eval samples (for testing)",
    )
    p.add_argument(
        "--train-splits",
        nargs="+",
        default=["train_clean_100", "train_clean_360", "train_other_500"],
        help="Dataset splits to use for training",
    )
    p.add_argument(
        "--eval-split",
        type=str,
        default="dev_clean",
        help="Dataset split to use for evaluation",
    )
    return p.parse_args()


def main() -> None:  # noqa: PLR0915
    args = parse_args()

    # --- Load vocab ---
    if not args.vocab_json.exists():
        logger.error("vocab.json not found at %s", args.vocab_json)
        sys.exit(1)

    with args.vocab_json.open() as f:
        vocab = json.load(f)
    logger.info("Loaded vocab with %d tokens", len(vocab))

    # --- Create tokenizer and processor ---
    # Save vocab.json to a temp dir for Wav2Vec2CTCTokenizer
    tokenizer_dir = Path(args.output_dir) / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    with (tokenizer_dir / "vocab.json").open("w") as f:
        json.dump(vocab, f)

    tokenizer = Wav2Vec2CTCTokenizer(
        str(tokenizer_dir / "vocab.json"),
        unk_token="[UNK]",  # noqa: S106
        pad_token="[PAD]",  # noqa: S106
        word_delimiter_token=None,  # No word boundaries for phone sequences
    )

    feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(
        "facebook/w2v-bert-2.0"
    )
    processor = Wav2Vec2BertProcessor(
        feature_extractor=feature_extractor, tokenizer=tokenizer
    )

    # --- Load dataset ---
    logger.info("Loading librispeech-alignments dataset...")

    train_splits = []
    for split_name in args.train_splits:
        logger.info("  Loading split: %s", split_name)
        ds = load_dataset(
            "gilkeyio/librispeech-alignments", split=split_name
        )
        train_splits.append(ds)
    train_ds = concatenate_datasets(train_splits)

    eval_ds = load_dataset(
        "gilkeyio/librispeech-alignments", split=args.eval_split
    )

    if args.max_train_samples:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
    if args.max_eval_samples:
        eval_ds = eval_ds.select(range(min(args.max_eval_samples, len(eval_ds))))

    logger.info("Train: %d examples, Eval: %d examples", len(train_ds), len(eval_ds))

    # --- Resample audio to 16kHz ---
    train_ds = train_ds.cast_column("audio", Audio(sampling_rate=16_000))
    eval_ds = eval_ds.cast_column("audio", Audio(sampling_rate=16_000))

    # --- Preprocessing: extract phone sequences and audio features ---
    def prepare_dataset(batch):
        audio = batch["audio"]

        # Extract audio features
        batch["input_features"] = processor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        batch["input_length"] = len(batch["input_features"])

        # Extract phone sequence from phoneme alignments
        # Strip stress markers: AY1 -> AY, IH0 -> IH
        phones = [strip_stress(p["phoneme"]) for p in batch["phonemes"]]

        # Filter to phones in our vocab (skip silence markers, etc.)
        phones = [p for p in phones if p in vocab]

        # Convert to label IDs
        batch["labels"] = [vocab[p] for p in phones]
        batch["phone_count"] = len(phones)

        return batch

    logger.info("Preprocessing training data...")
    train_ds = train_ds.map(
        prepare_dataset,
        remove_columns=train_ds.column_names,
        num_proc=4,
        desc="Preparing train",
    )

    logger.info("Preprocessing eval data...")
    eval_ds = eval_ds.map(
        prepare_dataset,
        remove_columns=eval_ds.column_names,
        num_proc=4,
        desc="Preparing eval",
    )

    # Filter out empty examples (no valid phones)
    train_ds = train_ds.filter(lambda x: x["phone_count"] > 0)
    eval_ds = eval_ds.filter(lambda x: x["phone_count"] > 0)
    logger.info(
        "After filtering: Train %d, Eval %d", len(train_ds), len(eval_ds)
    )

    # --- Load model ---
    logger.info("Loading w2v-bert-2.0 with new %d-class CTC head...", len(vocab))
    model = Wav2Vec2BertForCTC.from_pretrained(
        "facebook/w2v-bert-2.0",
        attention_dropout=0.0,
        hidden_dropout=0.0,
        feat_proj_dropout=0.0,
        mask_time_prob=0.0,
        layerdrop=0.0,
        ctc_loss_reduction="mean",
        add_adapter=True,
        pad_token_id=tokenizer.pad_token_id,
        vocab_size=len(tokenizer),
    )

    # Freeze the feature encoder (convolutional front-end)
    model.freeze_feature_encoder()
    logger.info(
        "Model loaded. Trainable params: %s",
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}",
    )

    # --- Data collator ---
    data_collator = DataCollatorCTCWithPadding(processor=processor)

    # --- Metrics ---
    per_metric = evaluate.load("wer")  # PER uses same computation as WER

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        # Replace -100 with pad token for decoding
        pred.label_ids[pred.label_ids == -100] = tokenizer.pad_token_id  # noqa: PLR2004

        # Decode predictions and labels to phone strings
        pred_str = tokenizer.batch_decode(pred_ids, group_tokens=True)
        label_str = tokenizer.batch_decode(pred.label_ids, group_tokens=False)

        # Compute PER (phone error rate = WER on phone sequences)
        per = per_metric.compute(predictions=pred_str, references=label_str)
        return {"per": per}

    # --- MLflow ---
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "https://mlflow.peacockery.studio")
    os.environ["MLFLOW_TRACKING_URI"] = mlflow_uri
    os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "peacock-asr-training")
    logger.info("MLflow tracking: %s", mlflow_uri)

    # --- Training arguments ---
    push = not args.no_push
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        group_by_length=True,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        eval_strategy="steps",
        num_train_epochs=args.num_epochs,
        gradient_checkpointing=True,
        bf16=True,  # A100 supports bf16 natively
        save_steps=2000,
        eval_steps=2000,
        logging_steps=100,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        save_total_limit=3,
        push_to_hub=push,
        hub_model_id=args.hub_repo if push else None,
        load_best_model_at_end=True,
        metric_for_best_model="per",
        greater_is_better=False,
        dataloader_num_workers=4,
        report_to="mlflow",
    )

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=processor.feature_extractor,
    )

    # --- Train ---
    logger.info("Starting training...")
    trainer.train()

    # --- Save final model ---
    logger.info("Saving final model to %s", args.output_dir)
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    if push:
        logger.info("Pushing to Hub: %s", args.hub_repo)
        trainer.push_to_hub()

    logger.info("Done.")


if __name__ == "__main__":
    main()
