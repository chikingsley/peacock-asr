#!/usr/bin/env python3
"""
Test the custom voice pack with Kokoro.

This script:
1. Loads the custom voice pack created with create_voice_pack.py
2. Adds it to the Kokoro voices file
3. Generates test audio using the custom voice
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import soundfile as sf


def add_voice_to_npz(voice_pack_path: str, voices_bin_path: str, voice_name: str, output_path: str):
    """
    Add a custom voice pack to an existing Kokoro voices file.
    """
    # Load custom voice pack
    voice_pack = torch.load(voice_pack_path, map_location='cpu', weights_only=True)
    print(f"Loaded voice pack: shape={voice_pack.shape}, dtype={voice_pack.dtype}")

    # Convert to numpy float32
    voice_np = voice_pack.numpy().astype(np.float32)
    print(f"Converted to numpy: shape={voice_np.shape}, dtype={voice_np.dtype}")

    # Load existing voices
    existing_voices = np.load(voices_bin_path, allow_pickle=True)
    print(f"Existing voices: {list(existing_voices.files)[:10]}...")

    # Create new dict with all voices including the custom one
    new_voices = {name: existing_voices[name] for name in existing_voices.files}
    new_voices[voice_name] = voice_np

    # Save to new file
    np.savez(output_path, **new_voices)
    print(f"Saved combined voices to: {output_path}")
    print(f"Total voices: {len(new_voices)}")

    return output_path


def generate_sample(text: str, voice_name: str, voices_path: str, model_path: str, output_path: str):
    """
    Generate audio using Kokoro with the custom voice.
    """
    from kokoro_onnx import Kokoro

    print(f"\nGenerating audio with voice '{voice_name}'...")
    print(f"Text: {text}")

    kokoro = Kokoro(model_path, voices_path)
    samples, sr = kokoro.create(
        text,
        voice=voice_name,
        speed=1.0,
    )

    sf.write(output_path, samples, sr)
    print(f"Audio saved to: {output_path}")
    print(f"Duration: {len(samples)/sr:.2f}s")


def main():
    parser = argparse.ArgumentParser(description="Test custom voice pack with Kokoro")
    parser.add_argument('--voice-pack', type=str, default='./voice_pack.pt',
                        help='Path to custom voice pack')
    parser.add_argument('--voice-name', type=str, default='custom_trelis',
                        help='Name for the custom voice')
    parser.add_argument('--voices-bin', type=str, default='../models/voices-v1.0.bin',
                        help='Path to existing Kokoro voices file')
    parser.add_argument('--model', type=str, default='../models/kokoro-v1.0.onnx',
                        help='Path to Kokoro ONNX model')
    parser.add_argument('--output-voices', type=str, default='./custom_voices.npz',
                        help='Path to save combined voices file')
    parser.add_argument('--output-audio', type=str, default='./test_output.wav',
                        help='Path to save test audio')
    parser.add_argument('--text', type=str,
                        default="Hello! This is a test of the custom voice pack created from the Trelis Latent Space audio.",
                        help='Text to synthesize')

    args = parser.parse_args()

    # Add custom voice to voices file
    voices_path = add_voice_to_npz(
        args.voice_pack,
        args.voices_bin,
        args.voice_name,
        args.output_voices
    )

    # Generate sample audio
    generate_sample(
        args.text,
        args.voice_name,
        voices_path,
        args.model,
        args.output_audio
    )

    # Also generate with an existing voice for comparison
    compare_output = args.output_audio.replace('.wav', '_compare.wav')
    generate_sample(
        args.text,
        'af_bella',  # Use Bella as reference
        voices_path,
        args.model,
        compare_output
    )

    print("\n" + "="*50)
    print("Test complete!")
    print(f"Custom voice output: {args.output_audio}")
    print(f"Reference voice output: {compare_output}")
    print("Compare the two audio files to evaluate voice quality.")


if __name__ == "__main__":
    main()
