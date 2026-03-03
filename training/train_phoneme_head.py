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
import io
import itertools
import json
import logging
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import evaluate
import mlflow
import numpy as np
import torch
from datasets import Audio, Dataset, concatenate_datasets, load_dataset
from transformers import (
    SeamlessM4TFeatureExtractor,
    Trainer,
    TrainingArguments,
    Wav2Vec2BertForCTC,
    Wav2Vec2BertProcessor,
    Wav2Vec2CTCTokenizer,
)

from peacock_asr.settings import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)
TARGET_SAMPLE_RATE = 16_000
HF_DATASET_REPO = "gilkeyio/librispeech-alignments"

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
        default=None,
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
    p.add_argument(
        "--dataloader-workers",
        type=int,
        default=4,
        help="Number of dataloader worker processes (default: 4)",
    )
    return p.parse_args()


def _numeric_metrics(metrics: dict[str, object]) -> dict[str, float]:
    numeric: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            numeric[key] = float(value)
    return numeric


def _load_split(
    split_name: str,
    *,
    max_samples: int | None = None,
) -> Dataset:
    hf_cache_dir = settings.data_dir / "hf-datasets"
    hf_cache_dir.mkdir(parents=True, exist_ok=True)
    if max_samples and max_samples > 0:
        logger.info(
            "  Streaming split: %s (taking first %d samples)",
            split_name,
            max_samples,
        )
        stream = load_dataset(
            HF_DATASET_REPO,
            split=split_name,
            streaming=True,
            cache_dir=str(hf_cache_dir),
        )
        stream = stream.cast_column("audio", Audio(decode=False))
        rows = list(itertools.islice(stream, max_samples))
        return Dataset.from_list(rows)

    logger.info("  Loading split: %s", split_name)
    return load_dataset(
        HF_DATASET_REPO,
        split=split_name,
        cache_dir=str(hf_cache_dir),
    )


def _load_audio_array(audio: dict[str, object]) -> tuple[np.ndarray, int]:
    import soundfile as sf  # noqa: PLC0415

    if "array" in audio and "sampling_rate" in audio:
        array = np.asarray(audio["array"], dtype=np.float32)
        sample_rate = int(audio["sampling_rate"])
    elif isinstance(audio.get("bytes"), (bytes, bytearray)):
        array, sample_rate = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
    elif isinstance(audio.get("path"), str):
        array, sample_rate = sf.read(audio["path"], dtype="float32")
    else:
        msg = (
            "Audio payload must contain either decoded 'array' data or raw "
            "'bytes'/'path' values."
        )
        raise ValueError(msg)

    if array.ndim > 1:
        array = np.mean(array, axis=1)
    return array.astype(np.float32, copy=False), sample_rate


def _resample_audio(array: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate == TARGET_SAMPLE_RATE:
        return array.astype(np.float32, copy=False)
    from scipy.signal import resample_poly  # noqa: PLC0415

    divisor = math.gcd(sample_rate, TARGET_SAMPLE_RATE)
    up = TARGET_SAMPLE_RATE // divisor
    down = sample_rate // divisor
    return resample_poly(array, up, down).astype(np.float32, copy=False)


def _freeze_feature_frontend(model: Wav2Vec2BertForCTC) -> None:
    if hasattr(model, "freeze_feature_encoder"):
        model.freeze_feature_encoder()
        return
    if hasattr(model, "freeze_feature_extractor"):
        model.freeze_feature_extractor()
        return
    # Fallback for transformers variants that don't expose freeze helpers.
    if hasattr(model, "wav2vec2_bert"):
        for parameter in model.wav2vec2_bert.feature_projection.parameters():
            parameter.requires_grad = False
        logger.warning(
            "Used fallback freezing for wav2vec2_bert.feature_projection; "
            "freeze_feature_encoder helper not found in this transformers version."
        )
        return
    logger.warning(
        "Could not freeze feature frontend automatically; model will train fully."
    )


def main() -> None:  # noqa: PLR0915
    args = parse_args()
    hub_repo = args.hub_repo or settings.hf_train_repo

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
    per_split_limit: int | None = None
    if args.max_train_samples:
        per_split_limit = max(
            1,
            math.ceil(args.max_train_samples / max(1, len(args.train_splits))),
        )
    for split_name in args.train_splits:
        ds = _load_split(split_name, max_samples=per_split_limit)
        train_splits.append(ds)
    train_ds = concatenate_datasets(train_splits)

    eval_ds = _load_split(args.eval_split, max_samples=args.max_eval_samples)

    if args.max_train_samples:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
    if args.max_eval_samples:
        eval_ds = eval_ds.select(range(min(args.max_eval_samples, len(eval_ds))))

    logger.info("Train: %d examples, Eval: %d examples", len(train_ds), len(eval_ds))

    # Disable datasets audio auto-decoding (torchcodec dependency). We decode with
    # soundfile in prepare_dataset instead.
    train_ds = train_ds.cast_column("audio", Audio(decode=False))
    eval_ds = eval_ds.cast_column("audio", Audio(decode=False))

    # --- Preprocessing: extract phone sequences and audio features ---
    def prepare_dataset(batch):
        audio_array, sample_rate = _load_audio_array(batch["audio"])
        if sample_rate != TARGET_SAMPLE_RATE:
            audio_array = _resample_audio(audio_array, sample_rate)
            sample_rate = TARGET_SAMPLE_RATE

        # Extract audio features
        batch["input_features"] = processor(
            audio_array, sampling_rate=sample_rate
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

    # Filter out invalid examples before Trainer batching.
    train_ds = train_ds.filter(
        lambda x: x["phone_count"] > 0 and x["input_length"] > 1
    )
    eval_ds = eval_ds.filter(
        lambda x: x["phone_count"] > 0 and x["input_length"] > 1
    )
    logger.info(
        "After filtering: Train %d, Eval %d", len(train_ds), len(eval_ds)
    )

    # --- Load model ---
    logger.info("Loading w2v-bert-2.0 with new %d-class CTC head...", len(vocab))
    model = Wav2Vec2BertForCTC.from_pretrained(
        "facebook/w2v-bert-2.0",
        torch_dtype=torch.float32,
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

    # Freeze the feature frontend for faster/stabler CTC head adaptation.
    _freeze_feature_frontend(model)
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
    mlflow_uri = settings.mlflow_tracking_uri
    experiment_name = settings.mlflow_experiment_name
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)

    enable_system_metrics = settings.mlflow_enable_system_metrics_logging
    sample_interval_sec = settings.mlflow_system_metrics_sampling_interval
    samples_before_log = settings.mlflow_system_metrics_samples_before_logging
    if enable_system_metrics:
        mlflow.enable_system_metrics_logging()
        mlflow.set_system_metrics_sampling_interval(sample_interval_sec)
        mlflow.set_system_metrics_samples_before_logging(samples_before_log)
    logger.info("MLflow tracking: %s", mlflow_uri)

    # --- Training arguments ---
    push = not args.no_push
    if push and settings.hf_token:
        import os  # noqa: PLC0415

        os.environ.setdefault("HF_TOKEN", settings.hf_token)
    primary_device_count = (
        torch.cuda.device_count() if torch.cuda.is_available() else 0
    )
    primary_device_name = (
        torch.cuda.get_device_name(0) if primary_device_count > 0 else "cpu"
    )
    effective_devices = max(primary_device_count, 1)
    effective_global_batch_size = (
        args.batch_size * args.gradient_accumulation * effective_devices
    )
    steps_per_epoch = max(
        1,
        math.ceil(len(train_ds) / effective_global_batch_size),
    )
    total_train_steps = max(1, int(args.num_epochs * steps_per_epoch))
    warmup_steps = max(1, int(total_train_steps * 0.1))

    # Use BF16 when the GPU supports it (covers A100, H100, L4, etc.).
    # Previous allowlist incorrectly excluded L4 (Ada Lovelace, 242 TFLOPS BF16).
    # The original CUBLAS crashes were from FP16 overflow, not missing BF16 support.
    use_bf16 = primary_device_count > 0 and torch.cuda.is_bf16_supported()
    use_fp16 = False
    if primary_device_count > 0 and not use_bf16:
        logger.info(
            "BF16 not supported on %s; using full precision training.",
            primary_device_name,
        )
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        train_sampling_strategy="group_by_length",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        eval_strategy="steps",
        num_train_epochs=args.num_epochs,
        gradient_checkpointing=True,
        bf16=use_bf16,
        fp16=use_fp16,
        save_steps=2000,
        eval_steps=2000,
        logging_steps=100,
        learning_rate=args.learning_rate,
        warmup_steps=warmup_steps,
        save_total_limit=3,
        push_to_hub=push,
        hub_model_id=hub_repo if push else None,
        load_best_model_at_end=True,
        metric_for_best_model="per",
        greater_is_better=False,
        dataloader_num_workers=args.dataloader_workers,
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

    run_name = f"train-phoneme-head-{Path(args.output_dir).name}"
    description = (
        "W2V-BERT 2.0 CTC phoneme-head training on "
        "gilkeyio/librispeech-alignments"
    )
    with mlflow.start_run(
        run_name=run_name,
        description=description,
        log_system_metrics=enable_system_metrics,
    ) as active_run:
        logger.info("MLflow run id: %s", active_run.info.run_id)
        mlflow.set_tags(
            {
                "project": "peacock-asr",
                "stage": "training",
                "task": "phoneme-head-ctc",
                "dataset_repo": "gilkeyio/librispeech-alignments",
                "base_model": "facebook/w2v-bert-2.0",
            },
        )
        mlflow.log_params(
            {
                "output_dir": args.output_dir,
                "hub_repo": hub_repo if push else "none",
                "push_to_hub": push,
                "num_epochs": args.num_epochs,
                "batch_size_per_device": args.batch_size,
                "gradient_accumulation": args.gradient_accumulation,
                "effective_global_batch_size": effective_global_batch_size,
                "learning_rate": args.learning_rate,
                "train_split_names": ",".join(args.train_splits),
                "eval_split_name": args.eval_split,
                "train_examples": len(train_ds),
                "eval_examples": len(eval_ds),
                "max_train_samples": args.max_train_samples or 0,
                "max_eval_samples": args.max_eval_samples or 0,
                "device_name": primary_device_name,
                "gpu_device_count": primary_device_count,
            },
        )

        if primary_device_count > 0:
            props = torch.cuda.get_device_properties(0)
            mlflow.log_params(
                {
                    "gpu_0_name": props.name,
                    "gpu_0_total_memory_gb": round(
                        props.total_memory / (1024 ** 3), 2,
                    ),
                    "gpu_0_compute_capability": f"{props.major}.{props.minor}",
                },
            )

        try:
            train_input = mlflow.data.from_huggingface(
                train_ds,
                path="gilkeyio/librispeech-alignments",
                name="librispeech-alignments-train",
            )
            eval_input = mlflow.data.from_huggingface(
                eval_ds,
                path="gilkeyio/librispeech-alignments",
                name="librispeech-alignments-eval",
            )
            mlflow.log_input(train_input, context="training")
            mlflow.log_input(eval_input, context="evaluation")
        except Exception:
            logger.exception("Failed to log datasets to MLflow inputs.")

        # --- Train ---
        logger.info("Starting training...")
        train_start = perf_counter()
        train_output = trainer.train()
        wall_time_sec = perf_counter() - train_start

        mlflow.log_metrics(_numeric_metrics(train_output.metrics))
        wall_time_hours = wall_time_sec / 3600.0
        mlflow.log_metrics(
            {
                "compute_wall_time_sec": wall_time_sec,
                "compute_wall_time_hours": wall_time_hours,
                "compute_machine_hours": wall_time_hours,
                "compute_gpu_hours": wall_time_hours * primary_device_count,
            },
        )

        # --- Save final model ---
        logger.info("Saving final model to %s", args.output_dir)
        trainer.save_model(args.output_dir)
        processor.save_pretrained(args.output_dir)

        if push:
            logger.info("Pushing to Hub: %s", hub_repo)
            trainer.push_to_hub()

    logger.info("Done.")


if __name__ == "__main__":
    main()
