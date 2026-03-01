#!/usr/bin/env python3
"""
Audio Cleaning Utilities for Voice Pack Creation

Prepares audio files for StyleTTS2 style extraction:
- Resamples to 24kHz (StyleTTS2 requirement)
- Trims silence
- Normalizes volume
- Removes noise (optional)

Usage:
    python audio_cleaning.py --input-dir ./raw_audio --output-dir ./cleaned_audio

    # Or use as a module:
    from audio_cleaning import clean_audio, process_dataset
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import librosa
import soundfile as sf
from scipy import signal


def resample_audio(audio: np.ndarray, sr_orig: int, sr_target: int = 24000) -> np.ndarray:
    """Resample audio to target sample rate."""
    if sr_orig == sr_target:
        return audio
    return librosa.resample(audio, orig_sr=sr_orig, target_sr=sr_target)


def trim_silence(audio: np.ndarray, sr: int, top_db: int = 30) -> np.ndarray:
    """Trim silence from beginning and end of audio."""
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed


def normalize_audio(audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """Normalize audio to target dB level."""
    # Calculate current RMS
    rms = np.sqrt(np.mean(audio ** 2))
    if rms == 0:
        return audio

    # Calculate target RMS from dB
    target_rms = 10 ** (target_db / 20)

    # Scale audio
    return audio * (target_rms / rms)


def highpass_filter(audio: np.ndarray, sr: int, cutoff: int = 80) -> np.ndarray:
    """Apply highpass filter to remove low-frequency noise."""
    nyquist = sr / 2
    normalized_cutoff = cutoff / nyquist
    b, a = signal.butter(5, normalized_cutoff, btype='high')
    return signal.filtfilt(b, a, audio)


def clean_audio(
    audio: np.ndarray,
    sr: int,
    target_sr: int = 24000,
    trim: bool = True,
    normalize: bool = True,
    highpass: bool = True,
    target_db: float = -20.0,
    trim_db: int = 30,
    highpass_cutoff: int = 80
) -> Tuple[np.ndarray, int]:
    """
    Apply full audio cleaning pipeline.

    Args:
        audio: Input audio array
        sr: Sample rate of input
        target_sr: Target sample rate (24000 for StyleTTS2)
        trim: Whether to trim silence
        normalize: Whether to normalize volume
        highpass: Whether to apply highpass filter
        target_db: Target dB level for normalization
        trim_db: Threshold dB for silence trimming
        highpass_cutoff: Cutoff frequency for highpass filter

    Returns:
        Tuple of (cleaned_audio, sample_rate)
    """
    # Ensure mono
    if len(audio.shape) > 1:
        audio = librosa.to_mono(audio)

    # Resample to target
    audio = resample_audio(audio, sr, target_sr)

    # Apply highpass filter (removes rumble, DC offset)
    if highpass:
        audio = highpass_filter(audio, target_sr, highpass_cutoff)

    # Trim silence
    if trim:
        audio = trim_silence(audio, target_sr, trim_db)

    # Normalize volume
    if normalize:
        audio = normalize_audio(audio, target_db)

    return audio, target_sr


def process_file(
    input_path: Path,
    output_path: Path,
    **kwargs
) -> Optional[dict]:
    """
    Process a single audio file.

    Returns dict with metadata or None if processing failed.
    """
    try:
        # Load audio
        audio, sr = librosa.load(input_path, sr=None)

        # Get original duration
        orig_duration = len(audio) / sr

        # Clean
        cleaned, new_sr = clean_audio(audio, sr, **kwargs)

        # Get new duration
        new_duration = len(cleaned) / new_sr

        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, cleaned, new_sr)

        return {
            'input_file': str(input_path),
            'output_file': str(output_path),
            'original_sr': sr,
            'output_sr': new_sr,
            'original_duration': orig_duration,
            'output_duration': new_duration,
            'trimmed_seconds': orig_duration - new_duration
        }

    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return None


def process_dataset(
    input_dir: Path,
    output_dir: Path,
    extensions: list = None,
    **kwargs
) -> list:
    """
    Process all audio files in a directory.

    Returns list of metadata dicts for processed files.
    """
    if extensions is None:
        extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    results = []

    # Find all audio files
    audio_files = []
    for ext in extensions:
        audio_files.extend(input_dir.glob(f'**/*{ext}'))

    print(f"Found {len(audio_files)} audio files")

    for i, input_path in enumerate(audio_files):
        # Create output path (preserve relative structure, force .wav)
        rel_path = input_path.relative_to(input_dir)
        output_path = output_dir / rel_path.with_suffix('.wav')

        print(f"[{i+1}/{len(audio_files)}] Processing {input_path.name}...")

        result = process_file(input_path, output_path, **kwargs)
        if result:
            results.append(result)
            print(f"  -> {result['output_duration']:.2f}s (trimmed {result['trimmed_seconds']:.2f}s)")

    return results


def process_huggingface_dataset(
    dataset_name: str = "Trelis/latent-space-train",
    output_dir: str = "./cleaned_audio",
    **kwargs
) -> list:
    """
    Process audio from a HuggingFace dataset.

    Returns list of metadata dicts including transcripts.
    """
    import json
    from huggingface_hub import hf_hub_download
    import pandas as pd

    print(f"Loading dataset: {dataset_name}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download the parquet file directly
    try:
        parquet_path = hf_hub_download(
            repo_id=dataset_name,
            filename="data/train-00000-of-00001.parquet",
            repo_type="dataset"
        )
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        print(f"Error downloading parquet: {e}")
        print("Trying alternative method...")
        # Fallback: load dataset without audio decoding
        from datasets import load_dataset
        ds = load_dataset(dataset_name, split="train")
        # Convert to dataframe without decoding audio
        df = ds.to_pandas()

    results = []

    for i, row in df.iterrows():
        # Extract audio bytes and decode with librosa
        audio_data = row['audio']

        # Handle different formats
        if isinstance(audio_data, dict):
            if 'bytes' in audio_data:
                # Audio stored as bytes
                import io
                audio_bytes = audio_data['bytes']
                audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
            elif 'path' in audio_data:
                # Audio stored as path reference
                audio, sr = librosa.load(audio_data['path'], sr=None)
            elif 'array' in audio_data:
                audio = np.array(audio_data['array'])
                sr = audio_data.get('sampling_rate', 16000)
            else:
                print(f"Unknown audio format for sample {i}: {audio_data.keys()}")
                continue
        else:
            print(f"Unexpected audio data type for sample {i}: {type(audio_data)}")
            continue

        # Clean audio
        cleaned, new_sr = clean_audio(audio, sr, **kwargs)

        # Save
        output_path = output_dir / f"sample_{i:04d}.wav"
        sf.write(output_path, cleaned, new_sr)

        result = {
            'index': i,
            'output_file': str(output_path),
            'text': row['text'],
            'original_sr': sr,
            'output_sr': new_sr,
            'original_duration': len(audio) / sr,
            'output_duration': len(cleaned) / new_sr,
            'start_time': row.get('start_time', ''),
            'end_time': row.get('end_time', ''),
            'word_timestamps': row.get('word_timestamps', [])
        }
        results.append(result)

        text_preview = row['text'][:50] if isinstance(row['text'], str) else str(row['text'])[:50]
        print(f"[{i+1}/{len(df)}] {text_preview}... -> {result['output_duration']:.2f}s")

    # Save metadata
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved metadata to {metadata_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Clean audio files for voice pack creation")

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Local directory processing
    local_parser = subparsers.add_parser('local', help='Process local audio directory')
    local_parser.add_argument('--input-dir', type=str, required=True, help='Input directory')
    local_parser.add_argument('--output-dir', type=str, required=True, help='Output directory')

    # HuggingFace dataset processing
    hf_parser = subparsers.add_parser('huggingface', help='Process HuggingFace dataset')
    hf_parser.add_argument('--dataset', type=str, default='Trelis/latent-space-train',
                          help='HuggingFace dataset name')
    hf_parser.add_argument('--output-dir', type=str, default='./cleaned_audio',
                          help='Output directory')

    # Common options
    for p in [local_parser, hf_parser]:
        p.add_argument('--target-sr', type=int, default=24000, help='Target sample rate')
        p.add_argument('--no-trim', action='store_true', help='Disable silence trimming')
        p.add_argument('--no-normalize', action='store_true', help='Disable normalization')
        p.add_argument('--no-highpass', action='store_true', help='Disable highpass filter')
        p.add_argument('--target-db', type=float, default=-20.0, help='Target dB for normalization')

    args = parser.parse_args()

    kwargs = {
        'target_sr': args.target_sr,
        'trim': not args.no_trim,
        'normalize': not args.no_normalize,
        'highpass': not args.no_highpass,
        'target_db': args.target_db
    }

    if args.command == 'local':
        results = process_dataset(Path(args.input_dir), Path(args.output_dir), **kwargs)
    elif args.command == 'huggingface':
        results = process_huggingface_dataset(args.dataset, args.output_dir, **kwargs)
    else:
        parser.print_help()
        return

    print(f"\n{'='*50}")
    print(f"Processed {len(results)} files")
    total_duration = sum(r['output_duration'] for r in results)
    print(f"Total duration: {total_duration:.2f}s ({total_duration/60:.2f} min)")


if __name__ == "__main__":
    main()
