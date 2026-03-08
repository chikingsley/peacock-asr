#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "torchaudio",
#     "transformers>=4.40",
#     "datasets",
#     "accelerate",
#     "numpy",
#     "evaluate",
#     "jiwer",
#     "pydantic",
#     "pydantic-settings",
#     "soundfile",
#     "scipy",
# ]
# ///
"""Benchmark the phoneme-head training hot path.

This is a performance harness, not a training entrypoint. It measures:
- dataset load/filter/setup time
- DataLoader batch production time (includes on-the-fly preprocessing)
- model load time
- forward/backward/optimizer step time on a small number of batches

Example:
    uv run --project projects/P003-compact-backbones python \
        projects/P003-compact-backbones/code/training/benchmark_phoneme_head.py \
        --model-name facebook/hubert-base-ls960 \
        --max-train-samples 128 \
        --batch-size 4 \
        --dataloader-workers 4 \
        --num-batches 8 \
        --step-batches 4
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, cast

import torch
from torch.utils.data import DataLoader

from p003_compact.settings import settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

TRAINING_ROOT = Path(__file__).resolve().parent
VOCAB_JSON = TRAINING_ROOT / "vocab.json"


def load_training_module() -> Any:
    script_path = TRAINING_ROOT / "train_phoneme_head.py"
    spec = importlib.util.spec_from_file_location(
        "p003_train_phoneme_head",
        script_path,
    )
    if spec is None or spec.loader is None:
        msg = f"Could not load training module from {script_path}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-name",
        default="facebook/hubert-base-ls960",
        help="HF backbone to benchmark.",
    )
    parser.add_argument(
        "--train-splits",
        nargs="+",
        default=["train_clean_100"],
        help="Training split(s) to sample from.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=128,
        help="Limit examples used for the benchmark.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size used for the benchmark DataLoader.",
    )
    parser.add_argument(
        "--dataloader-workers",
        type=int,
        default=4,
        help="Number of worker processes for the DataLoader.",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=8,
        help="How many DataLoader batches to time.",
    )
    parser.add_argument(
        "--step-batches",
        type=int,
        default=4,
        help="How many timed train steps to run.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Benchmark device: auto, cpu, or cuda.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mirror training with gradient checkpointing enabled (default: on).",
    )
    parser.add_argument(
        "--torch-compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable torch.compile for the benchmarked model (default: off).",
    )
    parser.add_argument(
        "--precision",
        choices=("auto", "fp32", "bf16", "fp16"),
        default="auto",
        help="Autocast precision for timed steps (default: auto).",
    )
    parser.add_argument(
        "--dataloader-pin-memory",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override DataLoader pin_memory (default: auto based on device).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to persist the JSON summary.",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_vocab() -> dict[str, int]:
    with VOCAB_JSON.open(encoding="utf-8") as handle:
        return json.load(handle)


def configure_hf_env() -> Path:
    hf_home = settings.models_dir / "huggingface"
    hub_cache = hf_home / "hub"
    hf_home.mkdir(parents=True, exist_ok=True)
    hub_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_HUB_CACHE", str(hub_cache))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hub_cache))
    return hf_home


def build_processor(
    module: Any,
    model_name: str,
    vocab: dict[str, int],
) -> tuple[Any, Any, bool]:
    tokenizer_dir = (
        settings.cache_dir
        / "bench-tokenizer"
        / model_name.replace("/", "__")
    )
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    with (tokenizer_dir / "vocab.json").open("w", encoding="utf-8") as handle:
        json.dump(vocab, handle)

    tokenizer = module.Wav2Vec2CTCTokenizer(
        str(tokenizer_dir / "vocab.json"),
        unk_token="[UNK]",  # noqa: S106
        pad_token="[PAD]",  # noqa: S106
        word_delimiter_token=None,
    )
    feature_extractor = module.AutoFeatureExtractor.from_pretrained(model_name)
    is_w2v_bert = (
        "w2v-bert" in model_name.lower() or "wav2vec2-bert" in model_name.lower()
    )
    if is_w2v_bert:
        processor = module.Wav2Vec2BertProcessor(
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
        )
    else:
        processor = module.Wav2Vec2Processor(
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
        )
    return processor, tokenizer, is_w2v_bert


def load_train_dataset(
    module: Any,
    args: argparse.Namespace,
    vocab: dict[str, int],
    processor: Any,
    *,
    is_w2v_bert: bool,
):
    load_start = time.perf_counter()
    per_split_limit: int | None = None
    if args.max_train_samples:
        per_split_limit = max(
            1,
            math.ceil(args.max_train_samples / max(1, len(args.train_splits))),
        )
    train_splits = [
        module.load_split(split_name, max_samples=per_split_limit)
        for split_name in args.train_splits
    ]
    train_ds = module.concatenate_datasets(train_splits)
    if args.max_train_samples:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
    train_ds = train_ds.cast_column("audio", module.Audio(decode=False))

    def has_valid_phones(example: dict[str, Any]) -> bool:
        phones = [module.strip_stress(p["phoneme"]) for p in example["phonemes"]]
        return any(phone in vocab for phone in phones)

    train_ds = train_ds.filter(
        has_valid_phones,
        num_proc=min(4, max(1, os.cpu_count() or 1)),
        desc="Filtering benchmark train",
    )

    def prepare_on_the_fly(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        all_features: list[Any] = []
        all_labels: list[list[int]] = []
        all_lengths: list[int] = []
        all_counts: list[int] = []

        for audio, phoneme_list in zip(batch["audio"], batch["phonemes"], strict=False):
            audio_array, sample_rate = module.load_audio_array(audio)
            if sample_rate != module.TARGET_SAMPLE_RATE:
                audio_array = module.resample_audio(audio_array, sample_rate)
                sample_rate = module.TARGET_SAMPLE_RATE

            processed = processor(audio_array, sampling_rate=sample_rate)
            features = (
                processed.input_features[0]
                if hasattr(processed, "input_features")
                else processed.input_values[0]
            )
            phones = [module.strip_stress(p["phoneme"]) for p in phoneme_list]
            phones = [phone for phone in phones if phone in vocab]
            labels = [vocab[phone] for phone in phones]

            all_features.append(features)
            all_labels.append(labels)
            all_lengths.append(len(features))
            all_counts.append(len(phones))

        feat_key = "input_features" if is_w2v_bert else "input_values"
        return {
            feat_key: all_features,
            "labels": all_labels,
            "input_length": all_lengths,
            "phone_count": all_counts,
        }

    train_ds.set_transform(prepare_on_the_fly)
    return train_ds, time.perf_counter() - load_start


def benchmark_dataloader(
    loader: DataLoader,
    num_batches: int,
) -> tuple[list[dict[str, torch.Tensor]], list[float]]:
    batches: list[dict[str, torch.Tensor]] = []
    timings: list[float] = []
    iterator = iter(loader)
    for _ in range(num_batches):
        started = time.perf_counter()
        batch = next(iterator)
        timings.append(time.perf_counter() - started)
        batches.append(batch)
    return batches, timings


def build_model(
    module: Any,
    model_name: str,
    vocab_size: int,
    pad_token_id: int,
    *,
    gradient_checkpointing: bool,
) -> torch.nn.Module:
    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch.float32,
        "attention_dropout": 0.0,
        "hidden_dropout": 0.0,
        "feat_proj_dropout": 0.0,
        "mask_time_prob": 0.0,
        "layerdrop": 0.0,
        "ctc_loss_reduction": "mean",
        "pad_token_id": pad_token_id,
        "vocab_size": vocab_size,
    }
    is_w2v_bert = (
        "w2v-bert" in model_name.lower() or "wav2vec2-bert" in model_name.lower()
    )
    if is_w2v_bert:
        model_kwargs["add_adapter"] = True
    model = module.AutoModelForCTC.from_pretrained(model_name, **model_kwargs)
    module.freeze_feature_frontend(model)
    if gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    model.train()
    return model


def to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def resolve_autocast_dtype(
    device: torch.device,
    precision: str,
) -> torch.dtype | None:
    if device.type != "cuda":
        return None
    if precision == "fp32":
        return None
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return None


def benchmark_steps(
    model: torch.nn.Module,
    batches: list[dict[str, torch.Tensor]],
    device: torch.device,
    step_batches: int,
    *,
    autocast_dtype: torch.dtype | None,
) -> dict[str, list[float]]:
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=1e-4,
    )
    forward_times: list[float] = []
    backward_times: list[float] = []
    total_times: list[float] = []

    for batch in batches[:step_batches]:
        batch_on_device = to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        cuda_sync(device)
        t0 = time.perf_counter()
        autocast_context = (
            torch.autocast(device_type=device.type, dtype=autocast_dtype)
            if autocast_dtype is not None
            else contextlib.nullcontext()
        )
        with autocast_context:
            outputs = model(**batch_on_device)
        loss = outputs.loss
        cuda_sync(device)
        t1 = time.perf_counter()
        loss.backward()
        optimizer.step()
        cuda_sync(device)
        t2 = time.perf_counter()
        forward_times.append(t1 - t0)
        backward_times.append(t2 - t1)
        total_times.append(t2 - t0)
    return {
        "forward_s": forward_times,
        "backward_s": backward_times,
        "step_total_s": total_times,
    }


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": sum(values) / max(1, len(values)),
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    args = parse_args()
    module = load_training_module()
    hf_home = configure_hf_env()
    vocab = load_vocab()
    processor, tokenizer, is_w2v_bert = build_processor(module, args.model_name, vocab)
    train_ds, dataset_setup_s = load_train_dataset(
        module,
        args,
        vocab,
        processor,
        is_w2v_bert=is_w2v_bert,
    )
    collator = module.DataCollatorCTCWithPadding(processor=processor)
    device = resolve_device(args.device)
    pin_memory = (
        device.type == "cuda"
        if args.dataloader_pin_memory is None
        else args.dataloader_pin_memory
    )
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        collate_fn=collator,
        num_workers=args.dataloader_workers,
        pin_memory=pin_memory,
        persistent_workers=args.dataloader_workers > 0,
    )

    batches, dataloader_times = benchmark_dataloader(loader, args.num_batches)
    model_load_started = time.perf_counter()
    model = build_model(
        module,
        args.model_name,
        len(vocab),
        tokenizer.pad_token_id,
        gradient_checkpointing=args.gradient_checkpointing,
    ).to(device)
    if args.torch_compile:
        model = cast("torch.nn.Module", torch.compile(model))
    model_load_s = time.perf_counter() - model_load_started
    autocast_dtype = resolve_autocast_dtype(device, args.precision)
    step_timings = benchmark_steps(
        model,
        batches,
        device,
        args.step_batches,
        autocast_dtype=autocast_dtype,
    )

    summary = {
        "model_name": args.model_name,
        "device": str(device),
        "hf_home": str(hf_home),
        "dataset_examples": len(train_ds),
        "batch_size": args.batch_size,
        "dataloader_workers": args.dataloader_workers,
        "dataloader_pin_memory": pin_memory,
        "is_w2v_bert": is_w2v_bert,
        "gradient_checkpointing": args.gradient_checkpointing,
        "torch_compile": args.torch_compile,
        "precision": args.precision,
        "autocast_dtype": str(autocast_dtype) if autocast_dtype is not None else None,
        "dataset_setup_s": dataset_setup_s,
        "model_load_s": model_load_s,
        "dataloader_batch_s": summarize(dataloader_times),
        "forward_s": summarize(step_timings["forward_s"]),
        "backward_s": summarize(step_timings["backward_s"]),
        "step_total_s": summarize(step_timings["step_total_s"]),
    }
    if device.type == "cuda":
        summary["cuda_max_memory_bytes"] = torch.cuda.max_memory_allocated(device)

    text = json.dumps(summary, indent=2)
    sys.stdout.write(text + "\n")
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
        logger.info("Wrote benchmark summary to %s", args.output_json)


if __name__ == "__main__":
    try:
        main()
    except StopIteration:
        sys.stderr.write(
            "Not enough batches were available for the requested benchmark. "
            "Increase --max-train-samples or lower --num-batches.",
        )
        sys.stderr.write("\n")
        raise SystemExit(1) from None
