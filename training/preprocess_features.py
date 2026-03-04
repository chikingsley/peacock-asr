"""Preprocess librispeech-alignments into pre-extracted features.

Runs SeamlessM4TFeatureExtractor (CPU-only) in parallel across examples,
producing a dataset with columns: input_features, labels, input_length,
phone_count. Pushes to HuggingFace Hub for fast loading in future training.

Usage (on RunPod, while GPU training runs):
    .venv/bin/python training/preprocess_features.py

Dry run (small subset, no push):
    .venv/bin/python training/preprocess_features.py \
        --max-samples 50 --no-push --num-proc 4
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import math
import os
import re
import sys
from pathlib import Path

from peacock_asr.settings import settings

import numpy as np
import soundfile as sf
from datasets import Audio, DatasetDict, concatenate_datasets, load_dataset
from scipy.signal import resample_poly
from transformers import SeamlessM4TFeatureExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16_000
HF_SOURCE_REPO = "gilkeyio/librispeech-alignments"
STRESS_RE = re.compile(r"[012]$")

# Module-level state — set in main(), used by worker processes via fork().
_feature_extractor: SeamlessM4TFeatureExtractor | None = None
_vocab: dict[str, int] | None = None


# ---------------------------------------------------------------------------
# Helper functions (same logic as train_phoneme_head.py)
# ---------------------------------------------------------------------------
def strip_stress(phone: str) -> str:
    return STRESS_RE.sub("", phone)


def _load_audio_array(audio: dict[str, object]) -> tuple[np.ndarray, int]:
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
    divisor = math.gcd(sample_rate, TARGET_SAMPLE_RATE)
    up = TARGET_SAMPLE_RATE // divisor
    down = sample_rate // divisor
    return resample_poly(array, up, down).astype(np.float32, copy=False)


def _has_valid_phones(example):
    phones = [strip_stress(p["phoneme"]) for p in example["phonemes"]]
    return any(p in _vocab for p in phones)


def process_example(example):
    """Process a single example: audio -> features, phonemes -> label IDs."""
    audio_array, sample_rate = _load_audio_array(example["audio"])
    if sample_rate != TARGET_SAMPLE_RATE:
        audio_array = _resample_audio(audio_array, sample_rate)

    features = _feature_extractor(
        audio_array, sampling_rate=TARGET_SAMPLE_RATE,
    ).input_features[0]

    phones = [strip_stress(p["phoneme"]) for p in example["phonemes"]]
    phones = [p for p in phones if p in _vocab]
    labels = [_vocab[p] for p in phones]

    return {
        "input_features": features,
        "labels": labels,
        "input_length": len(features),
        "phone_count": len(phones),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--vocab-json",
        type=Path,
        default=Path(__file__).parent / "vocab.json",
        help="Path to phoneme vocab JSON (default: training/vocab.json)",
    )
    p.add_argument(
        "--num-proc",
        type=int,
        default=80,
        help="Number of parallel workers for .map() (default: 80)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./processed-features"),
        help="Local save directory (default: ./processed-features)",
    )
    p.add_argument(
        "--hub-repo",
        type=str,
        default="Peacockery/librispeech-phoneme-features",
        help="HuggingFace Hub repo to push to",
    )
    p.add_argument(
        "--no-push",
        action="store_true",
        help="Skip pushing to HuggingFace Hub",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of examples (for testing)",
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    global _vocab, _feature_extractor  # noqa: PLW0603

    # --- HF cache: prefer NFS on RunPod to avoid filling 50GB container overlay ---
    hf_home = "/runpod/.cache/huggingface"
    if Path("/runpod").exists() and "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = hf_home
        logger.info("Auto-set HF_HOME=%s (RunPod NFS detected)", hf_home)

    # --- HF Token (from pydantic settings, reads .env automatically) ---
    hf_token = settings.hf_token
    if not args.no_push and not hf_token:
        logger.error("HF_TOKEN not set. Use --no-push or export HF_TOKEN.")
        sys.exit(1)

    # --- Load vocab ---
    with args.vocab_json.open() as f:
        _vocab = json.load(f)
    logger.info("Loaded vocab: %d tokens", len(_vocab))

    # --- Load feature extractor (CPU only) ---
    logger.info("Loading SeamlessM4TFeatureExtractor...")
    _feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(
        "facebook/w2v-bert-2.0",
    )

    # --- Load dataset splits ---
    # Use the generic "parquet" loader with hf:// URIs to download ONLY the
    # requested split's files. Loading via the dataset repo name downloads ALL
    # splits (~130GB) due to split verification in _split_generators().
    hf_prefix = f"hf://datasets/{HF_SOURCE_REPO}/data"
    train_splits = []
    for split_name in args.train_splits:
        logger.info("Loading %s...", split_name)
        ds = load_dataset(
            "parquet",
            data_files=f"{hf_prefix}/{split_name}-*.parquet",
            split="train",
        )
        ds = ds.cast_column("audio", Audio(decode=False))
        train_splits.append(ds)
    train_ds = concatenate_datasets(train_splits)

    logger.info("Loading %s...", args.eval_split)
    eval_ds = load_dataset(
        "parquet",
        data_files=f"{hf_prefix}/{args.eval_split}-*.parquet",
        split="train",
    )
    eval_ds = eval_ds.cast_column("audio", Audio(decode=False))

    if args.max_samples:
        train_ds = train_ds.select(range(min(args.max_samples, len(train_ds))))
        eval_ds = eval_ds.select(range(min(args.max_samples, len(eval_ds))))

    logger.info("Raw: Train %d, Eval %d", len(train_ds), len(eval_ds))

    # --- Filter (cheap, no audio decoding) ---
    logger.info("Filtering examples with no valid phones...")
    train_ds = train_ds.filter(
        _has_valid_phones, num_proc=args.num_proc, desc="Filtering train",
    )
    eval_ds = eval_ds.filter(
        _has_valid_phones, num_proc=args.num_proc, desc="Filtering eval",
    )
    logger.info("After filter: Train %d, Eval %d", len(train_ds), len(eval_ds))

    # --- Process features ---
    columns_to_remove = [
        c for c in train_ds.column_names
        if c not in {"input_features", "labels", "input_length", "phone_count"}
    ]

    logger.info("Processing train features with %d workers...", args.num_proc)
    train_ds = train_ds.map(
        process_example,
        num_proc=args.num_proc,
        remove_columns=columns_to_remove,
        desc="Processing train",
    )

    logger.info("Processing eval features with %d workers...", args.num_proc)
    eval_ds = eval_ds.map(
        process_example,
        num_proc=args.num_proc,
        remove_columns=columns_to_remove,
        desc="Processing eval",
    )

    # --- Save locally (checkpoint) ---
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    logger.info("Saving to disk: %s", out)
    train_ds.save_to_disk(str(out / "train"))
    eval_ds.save_to_disk(str(out / "eval"))
    logger.info("Local save complete.")

    # --- Push to Hub ---
    if not args.no_push:
        ds_dict = DatasetDict({"train": train_ds, "eval": eval_ds})
        logger.info("Pushing to Hub: %s", args.hub_repo)
        ds_dict.push_to_hub(args.hub_repo, token=hf_token)
        logger.info(
            "Done. Dataset at: https://huggingface.co/datasets/%s",
            args.hub_repo,
        )
    else:
        logger.info("Skipping Hub push (--no-push).")

    logger.info("Finished.")


if __name__ == "__main__":
    main()
