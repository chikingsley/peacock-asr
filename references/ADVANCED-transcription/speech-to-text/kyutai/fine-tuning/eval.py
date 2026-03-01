#!/usr/bin/env python3
"""
Kyutai STT Evaluation Script

Usage:
    # Evaluate on processed training data
    uv run eval.py --config config.yaml --model-path ./kyutai-finetuned

    # Evaluate on validation split (raw audio)
    uv run eval.py --config config.yaml --model-path ./kyutai-finetuned --split validation

    # Compare base vs trained on validation
    uv run eval.py --config config.yaml --split validation --compare
"""

# Disable XET before any HuggingFace imports (must be first)
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"

import argparse
import pickle
import sys
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf
import torch
import yaml
from jiwer import wer, transforms

# Normalize text before WER: lowercase, remove punctuation, collapse whitespace
wer_transform = transforms.Compose([
    transforms.ToLowerCase(),
    transforms.RemovePunctuation(),
    transforms.RemoveMultipleSpaces(),
    transforms.Strip(),
    transforms.ReduceToListOfListOfWords(),
])
from scipy import signal
from datasets import load_dataset, Audio
from transformers import (
    KyutaiSpeechToTextForConditionalGeneration,
    KyutaiSpeechToTextProcessor,
)

sys.path.insert(0, str(Path(__file__).parent / "utils"))
from data_collator import TimeAlignedDataCollator


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_device(device_config: str) -> str:
    if device_config == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device_config


def load_audio_from_sample(sample: Dict) -> np.ndarray:
    """Load and resample audio from dataset sample to 24kHz."""
    audio_data = sample["audio"]
    if isinstance(audio_data, dict) and "bytes" in audio_data:
        audio_array, sr = sf.read(BytesIO(audio_data["bytes"]))
    else:
        raise ValueError("Unexpected audio format")

    if len(audio_array.shape) > 1:
        audio_array = audio_array.mean(axis=1)
    audio_array = audio_array.astype(np.float32)

    if sr != 24000:
        audio_array = signal.resample(audio_array, int(len(audio_array) * 24000 / sr)).astype(np.float32)

    return audio_array


def evaluate_with_generate(model, processor, audio_array: np.ndarray, ground_truth: str, idx: int) -> Dict:
    """Evaluate a sample using generate() only."""
    print(f"\n{'='*60}")
    print(f"SAMPLE {idx}")
    print(f"{'='*60}")
    print(f"TRUTH: {ground_truth}")

    inputs = processor(audio_array)
    inputs.to(model.device)

    with torch.no_grad():
        generated = model.generate(**inputs)

    gen_text = processor.batch_decode(generated, skip_special_tokens=True)[0]
    print(f"GEN:   {gen_text}")

    # Calculate both raw and normalized WER
    if ground_truth and gen_text:
        raw_wer = wer(ground_truth, gen_text)
        norm_wer = wer(ground_truth, gen_text,
                       reference_transform=wer_transform,
                       hypothesis_transform=wer_transform)
    else:
        raw_wer = norm_wer = 1.0
    print(f"WER:   {norm_wer:.2%} (raw: {raw_wer:.2%})")

    return {"ground_truth": ground_truth, "generated": gen_text, "wer": norm_wer, "raw_wer": raw_wer}


def evaluate_processed_sample(model, processor, collator, sample: Dict, idx: int) -> Dict:
    """Evaluate a pre-processed sample with both forward and generate."""
    print(f"\n{'='*60}")
    print(f"SAMPLE {idx}")
    print(f"{'='*60}")

    ground_truth = sample["text"]
    print(f"TRUTH: {ground_truth}")

    # Teacher-forced (forward pass)
    model.eval()
    batch = collator([sample])
    # Remove loss_weights before passing to model
    batch.pop('loss_weights', None)

    with torch.no_grad():
        outputs = model.forward(**batch)

    predictions = outputs.logits.argmax(dim=-1)
    pred_text = processor.tokenizer.decode(predictions[0], skip_special_tokens=True)
    print(f"FORWARD: {pred_text}")

    # Accuracy
    input_tokens = batch['input_ids'][0, :, 0]
    pred_shifted = predictions[0, :-1]
    target_shifted = input_tokens[1:]
    non_pad_mask = target_shifted != 3
    accuracy = (pred_shifted[non_pad_mask] == target_shifted[non_pad_mask]).float().mean().item() if non_pad_mask.sum() > 0 else 0.0
    print(f"Accuracy: {accuracy:.2%}")

    # Generate
    audio_array = np.array(sample['processed_audio']['array'], dtype=np.float32)
    inputs = processor(audio_array)
    inputs.to(model.device)

    with torch.no_grad():
        generated = model.generate(**inputs)

    gen_text = processor.batch_decode(generated, skip_special_tokens=True)[0]
    print(f"GEN: {gen_text}")

    if ground_truth and gen_text:
        raw_wer = wer(ground_truth, gen_text)
        norm_wer = wer(ground_truth, gen_text,
                       reference_transform=wer_transform,
                       hypothesis_transform=wer_transform)
    else:
        raw_wer = norm_wer = 1.0
    print(f"WER: {norm_wer:.2%} (raw: {raw_wer:.2%})")

    return {"ground_truth": ground_truth, "forward": pred_text, "generated": gen_text, "accuracy": accuracy, "wer": norm_wer, "raw_wer": raw_wer}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model-path", default=None, help="Path to fine-tuned model (default: base model)")
    parser.add_argument("--split", default=None, help="Dataset split to evaluate (e.g., 'validation')")
    parser.add_argument("--compare", action="store_true", help="Compare base vs trained model")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device(config["model"]["device"])
    print(f"Device: {device}")

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    base_model_id = config["model"]["model_id"]
    trained_path = args.model_path or "./kyutai-finetuned"

    if args.compare:
        # Compare mode: load both models
        print(f"Loading base model: {base_model_id}")
        base_processor = KyutaiSpeechToTextProcessor.from_pretrained(base_model_id)
        base_model = KyutaiSpeechToTextForConditionalGeneration.from_pretrained(base_model_id, device_map=device)

        print(f"Loading trained model: {trained_path}")
        trained_processor = KyutaiSpeechToTextProcessor.from_pretrained(trained_path)
        trained_model = KyutaiSpeechToTextForConditionalGeneration.from_pretrained(trained_path, device_map=device)

        # Load dataset split
        split = args.split or "validation"
        print(f"Loading dataset split: {split}")
        dataset = load_dataset(config["dataset"]["name"], split=split)
        dataset = dataset.cast_column("audio", Audio(decode=False))

        max_samples = args.max_samples or len(dataset)

        print(f"\n{'='*70}")
        print(f"COMPARING BASE vs TRAINED on {split} ({min(max_samples, len(dataset))} samples)")
        print(f"{'='*70}")

        base_wers, trained_wers = [], []

        for i in range(min(max_samples, len(dataset))):
            sample = dataset[i]
            audio_array = load_audio_from_sample(sample)
            ground_truth = sample["text"]

            print(f"\n{'='*70}")
            print(f"SAMPLE {i}")
            print(f"{'='*70}")
            print(f"TRUTH:   {ground_truth}")

            # Base model
            inputs = base_processor(audio_array)
            inputs.to(base_model.device)
            with torch.no_grad():
                base_gen = base_model.generate(**inputs)
            base_text = base_processor.batch_decode(base_gen, skip_special_tokens=True)[0]
            if ground_truth and base_text:
                base_sample_wer = wer(ground_truth, base_text,
                                      reference_transform=wer_transform,
                                      hypothesis_transform=wer_transform)
            else:
                base_sample_wer = 1.0
            base_wers.append(base_sample_wer)
            print(f"BASE:    {base_text}")
            print(f"         WER: {base_sample_wer:.2%}")

            # Trained model
            inputs = trained_processor(audio_array)
            inputs.to(trained_model.device)
            with torch.no_grad():
                trained_gen = trained_model.generate(**inputs)
            trained_text = trained_processor.batch_decode(trained_gen, skip_special_tokens=True)[0]
            if ground_truth and trained_text:
                trained_sample_wer = wer(ground_truth, trained_text,
                                         reference_transform=wer_transform,
                                         hypothesis_transform=wer_transform)
            else:
                trained_sample_wer = 1.0
            trained_wers.append(trained_sample_wer)
            print(f"TRAINED: {trained_text}")
            print(f"         WER: {trained_sample_wer:.2%}")

        # Summary
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        avg_base_wer = sum(base_wers) / len(base_wers) if base_wers else 0
        avg_trained_wer = sum(trained_wers) / len(trained_wers) if trained_wers else 0
        print(f"Base model avg WER:    {avg_base_wer:.2%}")
        print(f"Trained model avg WER: {avg_trained_wer:.2%}")
        print(f"Improvement:           {avg_base_wer - avg_trained_wer:+.2%}")

    elif args.split:
        # Evaluate on raw dataset split
        model_path = args.model_path or base_model_id
        print(f"Loading model: {model_path}")
        processor = KyutaiSpeechToTextProcessor.from_pretrained(model_path)
        model = KyutaiSpeechToTextForConditionalGeneration.from_pretrained(model_path, device_map=device)

        print(f"Loading dataset split: {args.split}")
        dataset = load_dataset(config["dataset"]["name"], split=args.split)
        dataset = dataset.cast_column("audio", Audio(decode=False))

        max_samples = args.max_samples or len(dataset)
        results = []

        for i in range(min(max_samples, len(dataset))):
            sample = dataset[i]
            audio_array = load_audio_from_sample(sample)
            result = evaluate_with_generate(model, processor, audio_array, sample["text"], i)
            results.append(result)

        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"{'Sample':<8} {'WER':<10} {'Generated Preview'}")
        print("-" * 70)
        for i, r in enumerate(results):
            preview = r["generated"][:45] + "..." if len(r["generated"]) > 45 else r["generated"]
            print(f"{i:<8} {r['wer']:<10.2%} {preview}")

        avg_wer = sum(r["wer"] for r in results) / len(results) if results else 0
        print("-" * 70)
        print(f"{'Average':<8} {avg_wer:<10.2%}")

    else:
        # Evaluate on processed training data
        model_path = args.model_path or base_model_id
        print(f"Loading model: {model_path}")
        processor = KyutaiSpeechToTextProcessor.from_pretrained(model_path)
        model = KyutaiSpeechToTextForConditionalGeneration.from_pretrained(model_path, device_map=device)

        cache_path = config["dataset"].get("processed_data_path", "./processed_data.pkl")
        if not Path(cache_path).exists():
            print(f"Error: Processed data not found at {cache_path}")
            print("Run train.py first or use --split to evaluate on raw data.")
            return

        print(f"Loading processed data from {cache_path}...")
        with open(cache_path, "rb") as f:
            processed_samples = pickle.load(f)

        collator = TimeAlignedDataCollator(
            processor=processor, model=model,
            stt_delay=config["processing"]["stt_delay"],
            frame_hop_s=config["processing"]["frame_hop_s"],
        )

        max_samples = args.max_samples or len(processed_samples)
        results = []

        for i in range(min(max_samples, len(processed_samples))):
            result = evaluate_processed_sample(model, processor, collator, processed_samples[i], i)
            results.append(result)

        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"{'Sample':<8} {'Accuracy':<12} {'WER':<10} {'Generated Preview'}")
        print("-" * 80)
        for i, r in enumerate(results):
            preview = r["generated"][:35] + "..." if len(r["generated"]) > 35 else r["generated"]
            print(f"{i:<8} {r['accuracy']:<12.2%} {r['wer']:<10.2%} {preview}")

        avg_acc = sum(r["accuracy"] for r in results) / len(results) if results else 0
        avg_wer = sum(r["wer"] for r in results) / len(results) if results else 0
        print("-" * 80)
        print(f"{'Average':<8} {avg_acc:<12.2%} {avg_wer:<10.2%}")


if __name__ == "__main__":
    main()
