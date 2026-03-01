"""
Test TorchAudio forced alignment utility.
Run with: uv run python tests/test_forced_alignment.py
"""

import io
import sys
from pathlib import Path

import torch
import soundfile as sf
import librosa
from datasets import load_dataset, Audio

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from forced_alignment import (
    get_word_timestamps,
    get_word_timestamps_with_original,
    normalize_text_for_alignment
)


def load_audio_from_sample(sample, target_sr=16000):
    """Load audio from dataset sample, avoiding torchcodec issues."""
    audio_bytes = sample["audio"]["bytes"]
    audio_array, sr = sf.read(io.BytesIO(audio_bytes))

    # Convert to mono if stereo
    if len(audio_array.shape) > 1:
        audio_array = audio_array.mean(axis=1)

    # Resample if needed
    if sr != target_sr:
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=target_sr)

    return audio_array, target_sr


def load_dataset_no_decode(split="train"):
    """Load dataset with audio decoding disabled."""
    ds = load_dataset("Trelis/llm-lingo", split=split)
    ds = ds.cast_column("audio", Audio(decode=False))
    return ds


def test_normalization():
    """Test text normalization for alignment."""
    print("=== Test: Text Normalization ===")

    test_cases = [
        ("Hello World!", "hello world"),
        ("Fine-tuned model", "fine tuned model"),
        ("GPT-4 Turbo", "gpt four turbo"),  # Numbers converted to words
        ("Claude's API", "claude's api"),
        ("Phi-2", "phi two"),  # Numbers converted to words
    ]

    for input_text, expected in test_cases:
        result = normalize_text_for_alignment(input_text)
        status = "PASS" if result == expected else "FAIL"
        print(f"  {status}: '{input_text}' -> '{result}' (expected: '{expected}')")

    print()


def test_alignment_on_sample():
    """Test forced alignment on a real sample from the dataset."""
    print("=== Test: Forced Alignment on Real Audio ===")

    # Load a sample from the dataset
    print("Loading dataset sample...")
    ds = load_dataset_no_decode("train")

    sample = ds[0]
    audio_array, sr = load_audio_from_sample(sample)
    text = sample["text"]

    print(f"Text: {text}")
    print(f"Audio duration: {len(audio_array) / sr:.2f}s")
    print()

    # Convert to tensor
    waveform = torch.tensor(audio_array, dtype=torch.float32)

    # Get word timestamps
    print("Running forced alignment...")
    timestamps = get_word_timestamps(
        waveform,
        sr,
        text,
        device="cpu"
    )

    print(f"Got {len(timestamps)} word timestamps:")
    for i, ts in enumerate(timestamps[:10]):  # Show first 10
        print(f"  [{ts['start']:.2f}s - {ts['end']:.2f}s] {ts['word']}")

    if len(timestamps) > 10:
        print(f"  ... and {len(timestamps) - 10} more words")

    print()

    # Validate timestamps
    audio_duration = len(audio_array) / sr
    all_valid = True

    for ts in timestamps:
        if ts["start"] < 0 or ts["end"] > audio_duration + 0.5:  # 0.5s tolerance
            print(f"  WARNING: Invalid timestamp: {ts}")
            all_valid = False
        if ts["end"] < ts["start"]:
            print(f"  WARNING: End before start: {ts}")
            all_valid = False

    if all_valid:
        print("PASS: All timestamps are valid")
    else:
        print("FAIL: Some timestamps are invalid")

    print()


def test_alignment_with_original_words():
    """Test that original word forms are preserved."""
    print("=== Test: Preserve Original Words ===")

    ds = load_dataset_no_decode("train")

    sample = ds[0]
    audio_array, sr = load_audio_from_sample(sample)
    text = sample["text"]

    waveform = torch.tensor(audio_array, dtype=torch.float32)

    timestamps = get_word_timestamps_with_original(
        waveform,
        sr,
        text,
        device="cpu"
    )

    print("Original words preserved in timestamps:")
    for ts in timestamps[:5]:
        print(f"  [{ts['start']:.2f}s - {ts['end']:.2f}s] '{ts['word']}'")

    print()


def test_multiple_samples():
    """Test alignment on multiple samples to check consistency."""
    print("=== Test: Multiple Samples ===")

    ds = load_dataset_no_decode("train")

    num_samples = min(3, len(ds))
    all_passed = True

    for i in range(num_samples):
        sample = ds[i]
        audio_array, sr = load_audio_from_sample(sample)
        text = sample["text"]

        waveform = torch.tensor(audio_array, dtype=torch.float32)
        audio_duration = len(audio_array) / sr

        try:
            timestamps = get_word_timestamps(
                waveform,
                sr,
                text,
                device="cpu"
            )

            # Basic validation
            if len(timestamps) == 0:
                print(f"  Sample {i}: FAIL - No timestamps returned")
                all_passed = False
            elif timestamps[-1]["end"] > audio_duration + 1.0:
                print(f"  Sample {i}: FAIL - Timestamps exceed audio duration")
                all_passed = False
            else:
                print(f"  Sample {i}: PASS - {len(timestamps)} words aligned")

        except Exception as e:
            print(f"  Sample {i}: FAIL - {e}")
            all_passed = False

    print()
    if all_passed:
        print("All samples passed!")
    else:
        print("Some samples failed!")

    print()


if __name__ == "__main__":
    print("Testing TorchAudio Forced Alignment\n")
    print("=" * 50)
    print()

    test_normalization()
    test_alignment_on_sample()
    test_alignment_with_original_words()
    test_multiple_samples()

    print("=" * 50)
    print("Tests complete!")
