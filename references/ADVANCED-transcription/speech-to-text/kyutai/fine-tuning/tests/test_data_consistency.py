#!/usr/bin/env python3
"""
Test script to validate data consistency between:
1. Data collator (training)
2. model.forward()
3. model.generate() (inference)

Usage:
    uv run tests/test_data_consistency.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

from transformers import (
    KyutaiSpeechToTextForConditionalGeneration,
    KyutaiSpeechToTextProcessor,
)

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from data_collator import TimeAlignedDataCollator
from timestamp_alignment import schedule_tokens_discrete


def create_test_audio(duration_s: float = 2.0, sample_rate: int = 24000) -> np.ndarray:
    """Create a simple test audio signal."""
    t = np.linspace(0, duration_s, int(duration_s * sample_rate))
    # Simple sine wave at 440 Hz
    audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    return audio


def create_mock_sample(audio: np.ndarray, processor, model) -> dict:
    """Create a mock processed sample for testing."""
    # Encode audio
    audio_inputs = processor(audio, sampling_rate=24000)
    input_values = audio_inputs['input_values'].to(model.device)
    padding_mask = audio_inputs.get('padding_mask')
    if padding_mask is not None:
        padding_mask = padding_mask.to(model.device)

    with torch.no_grad():
        codec_output = model.codec_model.encode(input_values, padding_mask=padding_mask)
        audio_codes = codec_output.audio_codes.transpose(1, 2).cpu()

    seq_len = audio_codes.shape[1]
    frame_hop_s = 0.08
    audio_duration = seq_len * frame_hop_s

    # Create simple mock transcription
    mock_text = "hello world"
    mock_timestamps = [
        {"word": "hello", "start": 0.1, "end": 0.3},
        {"word": "world", "start": 0.4, "end": 0.6},
    ]

    def format_seconds(s):
        h = int(s // 3600)
        m = int((s % 3600) // 60)
        sec = s % 60
        return f"{h:02d}:{m:02d}:{sec:06.3f}"

    token_schedule = schedule_tokens_discrete(
        {
            "text": mock_text,
            "word_timestamps": mock_timestamps,
            "start_time": "00:00:00",
            "end_time": format_seconds(audio_duration),
        },
        processor.tokenizer,
        stt_delay=0.5,
        frame_hop_s=frame_hop_s,
        pad_to_segment_end=True,
        spillover="shift",
    )

    return {
        "text": mock_text,
        "word_timestamps": mock_timestamps,
        "token_schedule": token_schedule,
        "processed_audio": {"array": audio.tolist(), "sampling_rate": 24000},
        "audio_codes": audio_codes.squeeze(0).tolist(),
    }


def test_codec_consistency(processor, model, audio: np.ndarray):
    """Test that codec encoding is consistent."""
    print("\n" + "=" * 60)
    print("TEST 1: Codec Encoding Consistency")
    print("=" * 60)

    # Method 1: Direct encoding (what collator does)
    audio_inputs = processor(audio, sampling_rate=24000)
    input_values = audio_inputs['input_values'].to(model.device)
    padding_mask = audio_inputs.get('padding_mask')
    if padding_mask is not None:
        padding_mask = padding_mask.to(model.device)

    with torch.no_grad():
        codec_output_1 = model.codec_model.encode(input_values, padding_mask=padding_mask)
        audio_codes_1 = codec_output_1.audio_codes.transpose(1, 2)

    # Method 2: Encode again (should be identical - deterministic)
    with torch.no_grad():
        codec_output_2 = model.codec_model.encode(input_values, padding_mask=padding_mask)
        audio_codes_2 = codec_output_2.audio_codes.transpose(1, 2)

    match = torch.equal(audio_codes_1, audio_codes_2)
    print(f"Codec encoding deterministic: {match}")
    print(f"Audio codes shape: {audio_codes_1.shape}")
    print(f"  - Batch: {audio_codes_1.shape[0]}")
    print(f"  - Seq length (frames): {audio_codes_1.shape[1]}")
    print(f"  - Codebooks: {audio_codes_1.shape[2]}")

    return audio_codes_1


def test_input_format(processor, model, collator, sample: dict):
    """Test input format consistency between collator and generate."""
    print("\n" + "=" * 60)
    print("TEST 2: Input Format Consistency")
    print("=" * 60)

    # Get collator output
    batch = collator([sample])
    print(f"\nCollator output:")
    print(f"  input_ids shape: {batch['input_ids'].shape}")
    print(f"  labels shape: {batch['labels'].shape}")
    print(f"  attention_mask shape: {batch['attention_mask'].shape}")

    # Check expected shape
    batch_size, seq_len, channels = batch['input_ids'].shape
    print(f"\nExpected format: [batch, seq_len, 33]")
    print(f"  Channel 0: text tokens")
    print(f"  Channels 1-32: audio codes (32 codebooks)")
    print(f"  Actual channels: {channels}")
    assert channels == 33, f"Expected 33 channels, got {channels}"

    # Check text channel
    text_tokens = batch['input_ids'][0, :, 0]
    print(f"\nText tokens (channel 0):")
    print(f"  First 10: {text_tokens[:10].tolist()}")
    print(f"  Unique tokens: {torch.unique(text_tokens).tolist()}")

    # Check audio channels
    audio_tokens = batch['input_ids'][0, :, 1:]
    print(f"\nAudio tokens (channels 1-32):")
    print(f"  Shape: {audio_tokens.shape}")
    print(f"  Min value: {audio_tokens.min().item()}")
    print(f"  Max value: {audio_tokens.max().item()}")

    return batch


def test_special_tokens(processor, model, batch: dict):
    """Test special token handling."""
    print("\n" + "=" * 60)
    print("TEST 3: Special Token Handling")
    print("=" * 60)

    bos_id = model.config.bos_token_id
    audio_bos_id = getattr(model.config, 'audio_bos_token_id', None)
    audio_pad_id = getattr(model.config, 'audio_pad_token_id', None)
    pad_id = 3  # Standard PAD

    print(f"\nSpecial token IDs:")
    print(f"  BOS (text): {bos_id}")
    print(f"  Audio BOS: {audio_bos_id}")
    print(f"  Audio PAD: {audio_pad_id}")
    print(f"  PAD: {pad_id}")

    # Check position 0
    input_ids = batch['input_ids']
    pos0_text = input_ids[0, 0, 0].item()
    pos0_audio = input_ids[0, 0, 1:].tolist()

    print(f"\nPosition 0 tokens:")
    print(f"  Text channel: {pos0_text} (expected BOS: {bos_id})")
    print(f"  Audio channels (first 5): {pos0_audio[:5]}")

    if audio_bos_id is not None:
        audio_bos_correct = all(t == audio_bos_id for t in pos0_audio)
        print(f"  Audio BOS correct: {audio_bos_correct}")

    # Check BOS at position 0
    assert pos0_text == bos_id, f"Position 0 should be BOS ({bos_id}), got {pos0_text}"
    print("  [PASS] Text BOS at position 0")


def test_label_alignment(batch: dict):
    """Test label alignment for next-token prediction."""
    print("\n" + "=" * 60)
    print("TEST 4: Label Alignment")
    print("=" * 60)

    input_ids = batch['input_ids']
    labels = batch['labels']

    text_tokens = input_ids[0, :, 0]
    seq_len = len(text_tokens)

    print(f"\nSequence length: {seq_len}")
    print(f"Labels shape: {labels.shape}")

    # Check position 0 label (should be -100, ignore)
    label_0 = labels[0, 0].item()
    print(f"\nLabel at position 0: {label_0} (expected: -100)")
    assert label_0 == -100, f"Position 0 label should be -100 (ignore), got {label_0}"
    print("  [PASS] Position 0 label is -100")

    # Check last position label (should be -100, no target)
    label_last = labels[0, -1].item()
    print(f"Label at last position: {label_last} (expected: -100)")
    assert label_last == -100, f"Last position label should be -100 (ignore), got {label_last}"
    print("  [PASS] Last position label is -100")

    # Check alignment: labels should be same as inputs (model handles shifting internally)
    # Actually check the collator's approach
    valid_labels = labels[0][labels[0] != -100]
    valid_inputs = text_tokens[1:-1]  # Skip BOS and last

    print(f"\nValid labels count: {len(valid_labels)}")
    print(f"First 10 valid labels: {valid_labels[:10].tolist()}")
    print(f"Corresponding input tokens: {valid_inputs[:10].tolist()}")


def test_forward_pass(model, batch: dict):
    """Test forward pass with collator output."""
    print("\n" + "=" * 60)
    print("TEST 5: Forward Pass")
    print("=" * 60)

    model.eval()
    with torch.no_grad():
        outputs = model.forward(**batch)

    print(f"\nForward pass outputs:")
    print(f"  Logits shape: {outputs.logits.shape}")
    print(f"  Loss: {outputs.loss.item() if outputs.loss is not None else 'None'}")

    # Check logits make sense
    logits = outputs.logits
    predictions = logits.argmax(dim=-1)
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  First 10 predictions: {predictions[0, :10].tolist()}")

    return outputs


def test_generate_vs_forward(processor, model, audio: np.ndarray, batch: dict):
    """Compare generate() output structure with forward() expectations."""
    print("\n" + "=" * 60)
    print("TEST 6: Generate vs Forward Comparison")
    print("=" * 60)

    model.eval()

    # Run generate
    inputs = processor(audio, sampling_rate=24000)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=50, use_cache=False)

    print(f"\nGenerate output:")
    print(f"  Shape: {generated.shape}")
    print(f"  First 20 tokens: {generated[0, :20].tolist()}")

    # Decode
    decoded = processor.batch_decode(generated, skip_special_tokens=False)[0]
    print(f"  Decoded (with special): {decoded[:100]}...")

    # Compare sequence lengths
    collator_seq_len = batch['input_ids'].shape[1]
    generate_seq_len = generated.shape[1]
    print(f"\nSequence length comparison:")
    print(f"  Collator input_ids: {collator_seq_len}")
    print(f"  Generate output: {generate_seq_len}")

    # Note: These may differ - generate runs until EOS or max_tokens
    # Training uses fixed-length sequences from audio duration


def test_forward_predictions_quality(processor, model, batch: dict):
    """Test if forward pass predictions are reasonable."""
    print("\n" + "=" * 60)
    print("TEST 7: Forward Prediction Quality")
    print("=" * 60)

    model.eval()
    with torch.no_grad():
        outputs = model.forward(**batch)

    predictions = outputs.logits.argmax(dim=-1)
    labels = batch['labels']
    input_text = batch['input_ids'][0, :, 0]

    # Decode all
    input_decoded = processor.tokenizer.decode(input_text, skip_special_tokens=False)
    pred_decoded = processor.tokenizer.decode(predictions[0], skip_special_tokens=False)

    valid_labels = labels[0][labels[0] != -100]
    label_decoded = processor.tokenizer.decode(valid_labels, skip_special_tokens=False)

    print(f"\nDecoded sequences:")
    print(f"  Input:      {input_decoded}")
    print(f"  Labels:     {label_decoded}")
    print(f"  Predicted:  {pred_decoded}")

    # Calculate accuracy on valid positions
    valid_mask = labels[0] != -100
    valid_preds = predictions[0][valid_mask]
    valid_targets = labels[0][valid_mask]

    if len(valid_targets) > 0:
        accuracy = (valid_preds == valid_targets).float().mean().item()
        print(f"\nAccuracy on valid positions: {accuracy:.2%}")


def main():
    print("=" * 60)
    print("KYUTAI STT DATA CONSISTENCY TESTS")
    print("=" * 60)

    # Setup
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    model_id = "kyutai/stt-1b-en_fr-trfs"
    print(f"Loading model: {model_id}")

    processor = KyutaiSpeechToTextProcessor.from_pretrained(model_id)
    model = KyutaiSpeechToTextForConditionalGeneration.from_pretrained(model_id, device_map=device)

    # Create test data
    print("\nCreating test audio (2 seconds)...")
    audio = create_test_audio(duration_s=2.0)

    print("Creating mock sample...")
    sample = create_mock_sample(audio, processor, model)

    # Create collator
    collator = TimeAlignedDataCollator(
        processor=processor,
        model=model,
        stt_delay=0.5,
        frame_hop_s=0.08,
    )

    # Run tests
    audio_codes = test_codec_consistency(processor, model, audio)
    batch = test_input_format(processor, model, collator, sample)
    test_special_tokens(processor, model, batch)
    test_label_alignment(batch)
    test_forward_pass(model, batch)
    test_generate_vs_forward(processor, model, audio, batch)
    test_forward_predictions_quality(processor, model, batch)

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
