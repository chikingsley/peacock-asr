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
# ]
# ///
"""Train a w2v-BERT 2.0 CTC phoneme head on LibriSpeech alignments.

Usage (RunPod / cloud GPU):
    uv run --project projects/P003-compact-backbones python \
        projects/P003-compact-backbones/code/training/train_phoneme_head.py \
        --hub-repo your-username/w2v-bert-phoneme-en

Usage (quick test, no hub push):
    uv run --project projects/P003-compact-backbones python \
        projects/P003-compact-backbones/code/training/train_phoneme_head.py \
        --no-push \
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
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import evaluate
import numpy as np
import torch
from datasets import Audio, Dataset, concatenate_datasets, load_dataset
from transformers import (
    AutoFeatureExtractor,
    AutoModelForCTC,
    Trainer,
    TrainingArguments,
    Wav2Vec2BertProcessor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2Processor,
)

from p003_compact.settings import settings

REPO_ROOT = Path(__file__).resolve().parents[4]
HF_HOME_DEFAULT = REPO_ROOT / ".cache" / "models" / "huggingface"
HF_MODULES_DEFAULT = HF_HOME_DEFAULT / "modules"
HF_EVALUATE_DEFAULT = HF_HOME_DEFAULT / "evaluate"
HF_METRICS_DEFAULT = HF_HOME_DEFAULT / "metrics"
os.environ.setdefault("HF_HOME", str(HF_HOME_DEFAULT))
os.environ.setdefault("HF_HUB_CACHE", str(HF_HOME_DEFAULT / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_HOME_DEFAULT / "hub"))
os.environ.setdefault("HF_MODULES_CACHE", str(HF_MODULES_DEFAULT))
os.environ.setdefault("HF_EVALUATE_CACHE", str(HF_EVALUATE_DEFAULT))
os.environ.setdefault("HF_METRICS_CACHE", str(HF_METRICS_DEFAULT))

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
    processor: Wav2Vec2BertProcessor | Wav2Vec2Processor
    padding: bool | str = True

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        # w2v-bert uses "input_features", wav2vec2/HuBERT use "input_values"
        feat_key = (
            "input_features"
            if "input_features" in features[0]
            else "input_values"
        )
        input_features = [{feat_key: f[feat_key]} for f in features]
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
        help="Path to vocab.json (default: project-local training/vocab.json)",
    )
    # On RunPod, default to NFS volume so checkpoints survive pod restarts.
    # Locally, use a relative path.
    default_output_dir = (
        "/runpod/w2v-bert-phoneme-en"
        if Path("/runpod").exists()
        else "w2v-bert-phoneme-en"
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=default_output_dir,
        help="Output directory for checkpoints (auto-detects RunPod NFS)",
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
        default=4,
        help="Per-device train batch size (default: 4)",
    )
    p.add_argument(
        "--gradient-accumulation",
        type=int,
        default=16,
        help="Gradient accumulation steps (default: 16)",
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
    p.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable gradient checkpointing to trade compute for memory (default: on)",
    )
    p.add_argument(
        "--torch-compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable torch.compile through Hugging Face Trainer (default: off)",
    )
    p.add_argument(
        "--tf32",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override TF32 matmul usage on CUDA (default: Trainer/PyTorch default)",
    )
    p.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint dir, or 'true' to auto-detect latest in output_dir",
    )
    p.add_argument(
        "--preprocessed-dataset",
        type=str,
        default=None,
        help=(
            "HF Hub repo with pre-extracted features "
            "(e.g. Peacockery/librispeech-phoneme-features). "
            "When set, skips audio decoding and feature extraction entirely."
        ),
    )
    p.add_argument(
        "--model-name",
        type=str,
        default="facebook/w2v-bert-2.0",
        help="HF model name/path for the backbone (default: facebook/w2v-bert-2.0). "
        "Supports wav2vec2-base, HuBERT-base, etc.",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional hard cap on optimizer steps (overrides num_epochs when set)",
    )
    p.add_argument(
        "--save-steps",
        type=int,
        default=500,
        help="Checkpoint save interval in optimizer steps (default: 500)",
    )
    p.add_argument(
        "--eval-steps",
        type=int,
        default=500,
        help="Evaluation interval in optimizer steps (default: 500)",
    )
    p.add_argument(
        "--logging-steps",
        type=int,
        default=100,
        help="Logging interval in optimizer steps (default: 100)",
    )
    p.add_argument(
        "--save-total-limit",
        type=int,
        default=3,
        help="Maximum number of checkpoints to retain locally (default: 3)",
    )
    p.add_argument(
        "--hub-strategy",
        type=str,
        default="end",
        help="Trainer hub push strategy when push_to_hub is enabled (default: end)",
    )
    p.add_argument(
        "--wandb-log-model",
        type=str,
        choices=("false", "end", "checkpoint"),
        default="false",
        help="W&B model artifact policy: false, end, or checkpoint (default: false)",
    )
    return p.parse_args()


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

    array_value = audio.get("array")
    sampling_rate_value = audio.get("sampling_rate")
    bytes_value = audio.get("bytes")
    path_value = audio.get("path")

    if array_value is not None and sampling_rate_value is not None:
        array = np.asarray(array_value, dtype=np.float32)
        sample_rate = int(cast("int", sampling_rate_value))
    elif isinstance(bytes_value, (bytes, bytearray)):
        array, sample_rate = sf.read(io.BytesIO(bytes_value), dtype="float32")
    elif isinstance(path_value, str):
        array, sample_rate = sf.read(path_value, dtype="float32")
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


def _freeze_feature_frontend(model: torch.nn.Module) -> None:
    freeze_feature_encoder = getattr(model, "freeze_feature_encoder", None)
    if callable(freeze_feature_encoder):
        freeze_feature_encoder()
        return
    freeze_feature_extractor = getattr(model, "freeze_feature_extractor", None)
    if callable(freeze_feature_extractor):
        freeze_feature_extractor()
        return
    # Fallback for transformers variants that don't expose freeze helpers.
    wav2vec2_bert = getattr(model, "wav2vec2_bert", None)
    feature_projection = getattr(wav2vec2_bert, "feature_projection", None)
    if isinstance(feature_projection, torch.nn.Module):
        for parameter in feature_projection.parameters():
            parameter.requires_grad = False
        logger.warning(
            "Used fallback freezing for wav2vec2_bert.feature_projection; "
            "freeze_feature_encoder helper not found in this transformers version."
        )
        return
    logger.warning(
        "Could not freeze feature frontend automatically; model will train fully."
    )


def _configure_hf_model_cache() -> Path:
    hf_home = settings.models_dir / "huggingface"
    hub_cache = hf_home / "hub"
    modules_cache = hf_home / "modules"
    evaluate_cache = hf_home / "evaluate"
    metrics_cache = hf_home / "metrics"
    hf_home.mkdir(parents=True, exist_ok=True)
    hub_cache.mkdir(parents=True, exist_ok=True)
    modules_cache.mkdir(parents=True, exist_ok=True)
    evaluate_cache.mkdir(parents=True, exist_ok=True)
    metrics_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_HUB_CACHE", str(hub_cache))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hub_cache))
    os.environ.setdefault("HF_MODULES_CACHE", str(modules_cache))
    os.environ.setdefault("HF_EVALUATE_CACHE", str(evaluate_cache))
    os.environ.setdefault("HF_METRICS_CACHE", str(metrics_cache))
    logger.info(
        "HF caches: HF_HOME=%s HF_HUB_CACHE=%s HF_MODULES_CACHE=%s",
        hf_home,
        hub_cache,
        modules_cache,
    )
    return hf_home


def load_split(
    split_name: str,
    *,
    max_samples: int | None = None,
) -> Dataset:
    return _load_split(split_name, max_samples=max_samples)


def load_audio_array(audio: dict[str, object]) -> tuple[np.ndarray, int]:
    return _load_audio_array(audio)


def resample_audio(array: np.ndarray, sample_rate: int) -> np.ndarray:
    return _resample_audio(array, sample_rate)


def freeze_feature_frontend(model: torch.nn.Module) -> None:
    _freeze_feature_frontend(model)


def main() -> None:  # noqa: PLR0912, PLR0915
    args = parse_args()
    hub_repo = args.hub_repo or settings.hf_train_repo
    _configure_hf_model_cache()

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

    model_name = args.model_name
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    # w2v-bert uses Wav2Vec2BertProcessor; wav2vec2/HuBERT use Wav2Vec2Processor
    is_w2v_bert = (
        "w2v-bert" in model_name.lower()
        or "wav2vec2-bert" in model_name.lower()
    )
    if is_w2v_bert:
        processor = Wav2Vec2BertProcessor(
            feature_extractor=feature_extractor, tokenizer=tokenizer
        )
    else:
        processor = Wav2Vec2Processor(
            feature_extractor=feature_extractor, tokenizer=tokenizer
        )

    # --- Load dataset ---
    use_preprocessed = args.preprocessed_dataset is not None
    if use_preprocessed:
        logger.info("Loading preprocessed dataset: %s", args.preprocessed_dataset)
        hf_cache_dir = settings.data_dir / "hf-datasets"
        hf_cache_dir.mkdir(parents=True, exist_ok=True)

        # Preprocessed dataset has splits: train_clean_100, train_clean_360,
        # train_other_500, eval. Already filtered, features pre-extracted.
        train_splits = []
        for split_name in args.train_splits:
            ds = load_dataset(
                args.preprocessed_dataset,
                split=split_name,
                cache_dir=str(hf_cache_dir),
            )
            train_splits.append(ds)
        train_ds = concatenate_datasets(train_splits)

        # Preprocessed eval split is named "eval" (not "dev_clean")
        eval_split = "eval" if args.eval_split == "dev_clean" else args.eval_split
        eval_ds = load_dataset(
            args.preprocessed_dataset,
            split=eval_split,
            cache_dir=str(hf_cache_dir),
        )

        if args.max_train_samples:
            train_ds = train_ds.select(
                range(min(args.max_train_samples, len(train_ds)))
            )
        if args.max_eval_samples:
            eval_ds = eval_ds.select(
                range(min(args.max_eval_samples, len(eval_ds)))
            )

        logger.info(
            "Train: %d examples, Eval: %d examples",
            len(train_ds),
            len(eval_ds),
        )
        logger.info("Preprocessed columns: %s", train_ds.column_names)

    else:
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
            train_ds = train_ds.select(
                range(min(args.max_train_samples, len(train_ds)))
            )
        if args.max_eval_samples:
            eval_ds = eval_ds.select(
                range(min(args.max_eval_samples, len(eval_ds)))
            )

        logger.info(
            "Train: %d examples, Eval: %d examples",
            len(train_ds),
            len(eval_ds),
        )

        # Disable datasets audio auto-decoding (torchcodec dependency). We decode with
        # soundfile in prepare_dataset instead.
        train_ds = train_ds.cast_column("audio", Audio(decode=False))
        eval_ds = eval_ds.cast_column("audio", Audio(decode=False))

        # --- Pre-filter: remove examples with no valid phones (cheap, no audio) ---
        def _has_valid_phones(example: dict[str, Any]) -> bool:
            phones = [strip_stress(p["phoneme"]) for p in example["phonemes"]]
            return any(p in vocab for p in phones)

        logger.info("Filtering examples with no valid phones...")
        train_ds = train_ds.filter(
            _has_valid_phones, num_proc=4, desc="Filtering train"
        )
        eval_ds = eval_ds.filter(
            _has_valid_phones, num_proc=4, desc="Filtering eval"
        )
        logger.info(
            "After filtering: Train %d, Eval %d", len(train_ds), len(eval_ds)
        )

        # --- On-the-fly preprocessing via set_transform ---
        # Feature extraction runs in DataLoader workers during training, not
        # upfront.  This avoids a multi-hour .map() blocking step on large
        # datasets (see: https://github.com/huggingface/datasets/issues/6789).
        def prepare_on_the_fly(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
            # set_transform always receives batches (dict of lists)
            all_features: list[Any] = []
            all_labels: list[list[int]] = []
            all_lengths: list[int] = []
            all_counts: list[int] = []

            for audio, phoneme_list in zip(
                batch["audio"],
                batch["phonemes"],
                strict=False,
            ):
                audio_array, sample_rate = _load_audio_array(audio)
                if sample_rate != TARGET_SAMPLE_RATE:
                    audio_array = _resample_audio(audio_array, sample_rate)
                    sample_rate = TARGET_SAMPLE_RATE

                processed = processor(audio_array, sampling_rate=sample_rate)
                # w2v-bert: input_features (mel); wav2vec2/HuBERT: input_values (raw)
                if hasattr(processed, "input_features"):
                    features = processed.input_features[0]
                else:
                    features = processed.input_values[0]

                phones = [strip_stress(p["phoneme"]) for p in phoneme_list]
                phones = [p for p in phones if p in vocab]
                labels = [vocab[p] for p in phones]

                all_features.append(features)
                all_labels.append(labels)
                all_lengths.append(len(features))
                all_counts.append(len(phones))

            # Use the key name that matches the model's expected input
            feat_key = "input_features" if is_w2v_bert else "input_values"
            return {
                feat_key: all_features,
                "labels": all_labels,
                "input_length": all_lengths,
                "phone_count": all_counts,
            }

        train_ds.set_transform(prepare_on_the_fly)
        eval_ds.set_transform(prepare_on_the_fly)

    # --- Load model ---
    logger.info("Loading %s with new %d-class CTC head...", model_name, len(vocab))
    model_kwargs = {
        "torch_dtype": torch.float32,
        "attention_dropout": 0.0,
        "hidden_dropout": 0.0,
        "feat_proj_dropout": 0.0,
        "mask_time_prob": 0.0,
        "layerdrop": 0.0,
        "ctc_loss_reduction": "mean",
        "pad_token_id": tokenizer.pad_token_id,
        "vocab_size": len(tokenizer),
        "ignore_mismatched_sizes": True,
    }
    # w2v-bert supports adapter layers; wav2vec2/HuBERT don't
    if is_w2v_bert:
        model_kwargs["add_adapter"] = True
    model = AutoModelForCTC.from_pretrained(model_name, **model_kwargs)

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
    per_metric_runner = cast("Any", per_metric)

    def compute_metrics(pred: Any) -> dict[str, Any]:
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        # Replace -100 with pad token for decoding
        pred.label_ids[pred.label_ids == -100] = tokenizer.pad_token_id  # noqa: PLR2004

        # Decode predictions and labels to phone strings
        pred_str = tokenizer.batch_decode(pred_ids, group_tokens=True)
        label_str = tokenizer.batch_decode(pred.label_ids, group_tokens=False)

        # Compute PER (phone error rate = WER on phone sequences)
        per = per_metric_runner.compute(predictions=pred_str, references=label_str)
        return {"per": per}

    # --- Training arguments ---
    push = not args.no_push
    if settings.hf_token:
        os.environ.setdefault("HF_TOKEN", settings.hf_token)
    if push and settings.hf_token:
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
    total_train_steps = (
        max(1, args.max_steps)
        if args.max_steps is not None
        else max(1, int(args.num_epochs * steps_per_epoch))
    )
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
    logger.info(
        (
            "Runtime knobs: gradient_checkpointing=%s torch_compile=%s "
            "tf32=%s bf16=%s optim=%s"
        ),
        args.gradient_checkpointing,
        args.torch_compile,
        args.tf32,
        use_bf16,
        "adamw_torch_fused",
    )
    # --- W&B setup ---
    # HF Trainer's WandbCallback calls wandb.init() itself (integration_utils.py:752).
    # It reads project from WANDB_PROJECT env var (default: "huggingface").
    # It reads run name from TrainingArguments.run_name.
    # We set env vars here so it works regardless of shell environment.
    os.environ.setdefault(
        "WANDB_PROJECT",
        hub_repo.split("/")[-1] if hub_repo else "peacock-asr-training",
    )
    os.environ.setdefault("WANDB_ENTITY", "peacockery")
    os.environ["WANDB_LOG_MODEL"] = args.wandb_log_model

    run_name = f"phoneme-head-{Path(args.output_dir).name}"
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        run_name=run_name,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        eval_strategy="steps",
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps if args.max_steps is not None else -1,
        gradient_checkpointing=args.gradient_checkpointing,
        remove_unused_columns=False,
        bf16=use_bf16,
        fp16=use_fp16,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        warmup_steps=warmup_steps,
        save_total_limit=args.save_total_limit,
        push_to_hub=push,
        hub_model_id=hub_repo if push else None,
        hub_strategy=args.hub_strategy,
        load_best_model_at_end=True,
        metric_for_best_model="per",
        greater_is_better=False,
        dataloader_num_workers=args.dataloader_workers,
        dataloader_persistent_workers=args.dataloader_workers > 0,
        report_to=["wandb"],
        torch_compile=args.torch_compile,
        tf32=args.tf32,
    )

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=cast("Any", processor).feature_extractor,
    )

    # --- Train ---
    resume = args.resume_from_checkpoint
    if resume and resume.lower() == "true":
        resume = True  # HF Trainer auto-detects latest checkpoint in output_dir
    if resume:
        logger.info("Resuming from checkpoint: %s", resume)
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=resume)

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
