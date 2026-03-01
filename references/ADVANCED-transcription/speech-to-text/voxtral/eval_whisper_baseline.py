#!/usr/bin/env python3
# /// script
# dependencies = [
#   "transformers>=4.52.0",
#   "torch>=2.0",
#   "datasets[audio]>=3.4.1",
#   "jiwer",
#   "librosa",
#   "soundfile",
#   "torchcodec",
# ]
# ///
"""
Whisper Baseline Evaluation on Trelis/llm-lingo

Evaluates Whisper (base or fine-tuned) on the same validation set used
for Voxtral training, enabling direct WER comparison.

Usage:
    # Whisper-large-v3-turbo baseline
    uv run eval_whisper_baseline.py

    # Fine-tuned Whisper model
    uv run eval_whisper_baseline.py --model Trelis/llm-lingo

    # Different Whisper variant
    uv run eval_whisper_baseline.py --model openai/whisper-small
"""

import argparse
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
from jiwer import wer
import numpy as np


def extract_audio(audio_data):
    """Extract audio array and sample rate from various formats."""
    # torchcodec AudioDecoder
    if hasattr(audio_data, "get_all_samples"):
        audio_samples = audio_data.get_all_samples()
        if hasattr(audio_samples, "data"):
            audio_tensor = audio_samples.data.squeeze()
            if audio_tensor.dim() > 1:
                audio_tensor = audio_tensor.mean(dim=0)
            audio_array = audio_tensor.numpy()
        else:
            audio_array = np.array(audio_samples).squeeze()
        sr = audio_data.metadata.sample_rate if hasattr(audio_data, "metadata") else 16000
        return audio_array, sr

    # Standard HF datasets dict
    if isinstance(audio_data, dict) and "array" in audio_data:
        return audio_data["array"], audio_data.get("sampling_rate", 16000)

    # Legacy object
    if hasattr(audio_data, "array"):
        return audio_data.array, getattr(audio_data, "sampling_rate", 16000)

    return None, None


def resample_if_needed(audio_array, sr, target_sr=16000):
    """Resample audio to target sample rate."""
    if sr == target_sr:
        return audio_array
    import librosa
    if isinstance(audio_array, list):
        audio_array = np.array(audio_array, dtype=np.float32)
    return librosa.resample(
        audio_array.astype(np.float32), orig_sr=sr, target_sr=target_sr
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate Whisper on Trelis/llm-lingo")
    parser.add_argument("--model", default="openai/whisper-large-v3-turbo",
                        help="Whisper model ID (default: whisper-large-v3-turbo)")
    parser.add_argument("--dataset", default="Trelis/llm-lingo",
                        help="HuggingFace dataset name")
    parser.add_argument("--language", default="en", help="Language code")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of validation samples")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    print("=" * 60)
    print("WHISPER BASELINE EVALUATION")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {device}")
    print()

    # Load model
    print(f"Loading model: {args.model}...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model, torch_dtype=torch_dtype, low_cpu_mem_usage=True,
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(args.model)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        generate_kwargs={"language": args.language},
    )

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    ds = load_dataset(args.dataset)

    if "validation" in ds:
        val_ds = ds["validation"]
    elif "test" in ds:
        val_ds = ds["test"]
    else:
        split = ds["train"].train_test_split(test_size=0.3, seed=42)
        val_ds = split["test"]

    if args.max_samples:
        val_ds = val_ds.select(range(min(args.max_samples, len(val_ds))))

    print(f"Evaluating {len(val_ds)} validation samples...\n")

    predictions = []
    references = []

    for i in range(len(val_ds)):
        sample = val_ds[i]
        text = sample.get("text", sample.get("transcription", ""))
        audio_data = sample.get("audio")

        if audio_data is None:
            continue

        audio_array, sr = extract_audio(audio_data)
        if audio_array is None:
            continue

        audio_array = resample_if_needed(audio_array, sr, 16000)
        if isinstance(audio_array, list):
            audio_array = np.array(audio_array, dtype=np.float32)

        result = pipe({"raw": audio_array, "sampling_rate": 16000})
        prediction = result["text"].strip()

        predictions.append(prediction)
        references.append(text)
        print(f"  [{i}] REF: {text}")
        print(f"       HYP: {prediction}")

    # Compute WER
    if predictions:
        word_error_rate = wer(references, predictions)
        print(f"\n{'=' * 60}")
        print(f"Model: {args.model}")
        print(f"WER: {word_error_rate:.4f} ({word_error_rate*100:.1f}%)")
        print(f"Evaluated: {len(predictions)} / {len(val_ds)} samples")
        print(f"{'=' * 60}")
    else:
        print("No predictions generated")


if __name__ == "__main__":
    main()
