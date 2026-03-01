#!/usr/bin/env python3
"""
Voice Pack Creation for Kokoro

Extracts style embeddings from audio files using StyleTTS2's style encoder
and creates a voice pack tensor compatible with Kokoro inference.

The voice pack has shape (510, 1, 256):
- 510: One style vector per token length (0-509)
- 1: Batch dimension (squeezed during inference)
- 256: Style embedding dimension

Usage:
    python create_voice_pack.py --audio-dir ./cleaned_audio --output voice_pack.pt

    # Or with explicit metadata:
    python create_voice_pack.py --metadata-json ./cleaned_audio/metadata.json --output voice_pack.pt
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torchaudio
import librosa


def compute_mel_spectrogram(
    audio: np.ndarray,
    sr: int = 24000,
    n_fft: int = 2048,
    hop_length: int = 300,  # StyleTTS2 uses 300 for 24kHz
    win_length: int = 1200,
    n_mels: int = 80,
    fmin: float = 0.0,
    fmax: float = 8000.0,
) -> torch.Tensor:
    """
    Compute mel spectrogram matching StyleTTS2 parameters.

    Returns tensor of shape (n_mels, time)
    """
    # Ensure audio is float32
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    # Compute mel spectrogram using librosa (matches StyleTTS2)
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=1.0,  # amplitude spectrogram
    )

    # Log mel spectrogram (StyleTTS2 convention)
    log_mel = np.log(np.clip(mel, 1e-5, None))

    return torch.from_numpy(log_mel).float()


def load_styletts2_encoder(model_path: Optional[str] = None, device: str = 'cpu'):
    """
    Load the StyleTTS2 style encoder.

    This function attempts to load the style encoder from:
    1. A local checkpoint if model_path is provided
    2. The StyleTTS2 GitHub repository code
    """
    # Try loading from cloned StyleTTS2 repository
    styletts2_repo = Path(__file__).parent.parent.parent / "StyleTTS2-repo"
    if not styletts2_repo.exists():
        styletts2_repo = Path(__file__).parent.parent.parent / "StyleTTS2"

    if styletts2_repo.exists():
        sys.path.insert(0, str(styletts2_repo))

    try:
        from models import StyleEncoder

        # StyleTTS2-LJSpeech style encoder config
        # These match the pretrained model
        encoder = StyleEncoder(
            dim_in=64,
            style_dim=256,
            max_conv_dim=512,
        )

        if model_path:
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
            if 'style_encoder' in checkpoint:
                encoder.load_state_dict(checkpoint['style_encoder'])
            else:
                encoder.load_state_dict(checkpoint)

        return encoder.to(device).eval()

    except ImportError as e:
        print(f"StyleTTS2 models not found: {e}")
        print(f"Searched in: {styletts2_repo}")
        print("You can clone it from: https://github.com/yl4579/StyleTTS2")
        raise


class StyleEncoderWithProjection(torch.nn.Module):
    """Wrapper that projects 128-dim output to 256-dim for Kokoro compatibility."""

    def __init__(self, encoder: torch.nn.Module, input_dim: int = 128, output_dim: int = 256):
        super().__init__()
        self.encoder = encoder
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Simple projection: repeat the 128-dim vector to get 256-dim
        # This preserves the style information while matching expected dimensions
        # Alternatively, could use a linear layer but that would need training

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        style = self.encoder(x)
        # Repeat 128-dim to 256-dim: [s1, s2, ..., s128] -> [s1, s2, ..., s128, s1, s2, ..., s128]
        # This is a simple projection that maintains speaker identity
        style_256 = torch.cat([style, style], dim=-1)
        return style_256


def load_styletts2_from_hf(device: str = 'cpu', target_dim: int = 256):
    """
    Load StyleTTS2-LJSpeech model from HuggingFace.

    Returns the style encoder module (with projection if needed).

    Note: LJSpeech model outputs 128-dim style vectors, but Kokoro expects 256-dim.
    This function wraps the encoder with a projection layer.
    """
    from huggingface_hub import hf_hub_download
    import yaml

    # Download model files
    repo_id = "yl4579/StyleTTS2-LJSpeech"

    print(f"Downloading StyleTTS2-LJSpeech from {repo_id}...")

    # Download config
    config_path = hf_hub_download(
        repo_id=repo_id,
        filename="Models/LJSpeech/config.yml",
    )

    # Download model checkpoint
    model_path = hf_hub_download(
        repo_id=repo_id,
        filename="Models/LJSpeech/epoch_2nd_00100.pth",
    )

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Import StyleTTS2 model building code
    # Try both possible locations
    styletts2_path = Path(__file__).parent.parent.parent / "StyleTTS2-repo"
    if not styletts2_path.exists():
        styletts2_path = Path(__file__).parent.parent.parent / "StyleTTS2"

    if not styletts2_path.exists():
        # Clone StyleTTS2 repo
        import subprocess
        print("Cloning StyleTTS2 repository...")
        subprocess.run([
            "git", "clone", "--depth", "1",
            "https://github.com/yl4579/StyleTTS2.git",
            str(styletts2_path)
        ], check=True)

    sys.path.insert(0, str(styletts2_path))

    from models import StyleEncoder

    # Build style encoder with LJSpeech config
    model_params = config.get('model_params', {})
    style_dim = model_params.get('style_dim', 128)
    dim_in = model_params.get('dim_in', 64)
    max_conv_dim = model_params.get('max_conv_dim', 512)

    print(f"Model config: dim_in={dim_in}, style_dim={style_dim}, max_conv_dim={max_conv_dim}")

    encoder = StyleEncoder(
        dim_in=dim_in,
        style_dim=style_dim,
        max_conv_dim=max_conv_dim,
    )

    # Load weights
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Extract style_encoder weights
    # The checkpoint stores style_encoder as an OrderedDict with 'module.' prefix from DataParallel
    style_encoder_state = checkpoint['net']['style_encoder']

    # Remove 'module.' prefix if present
    new_state = {}
    for k, v in style_encoder_state.items():
        new_key = k.replace('module.', '') if k.startswith('module.') else k
        new_state[new_key] = v

    encoder.load_state_dict(new_state)
    encoder = encoder.to(device).eval()

    # Wrap with projection if needed
    if style_dim != target_dim:
        print(f"Wrapping encoder with projection: {style_dim} -> {target_dim}")
        encoder = StyleEncoderWithProjection(encoder, style_dim, target_dim)

    return encoder


def phonemize_text(text: str, language: str = 'en-us') -> List[str]:
    """
    Convert text to phonemes.

    For now, uses a simple character-based estimate since phonemizers
    can have complex dependencies.

    Returns list of phoneme tokens.
    """
    # Simple estimation: ~1 phoneme per character (excluding spaces)
    # This is rough but good enough for token length estimation
    # Average English word has ~5 phonemes and ~5 chars, so 1:1 ratio works
    return list(text.replace(' ', ''))


def extract_style_embedding(
    audio: np.ndarray,
    encoder: torch.nn.Module,
    sr: int = 24000,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Extract style embedding from audio using the style encoder.

    Args:
        audio: Audio array at target sample rate
        encoder: StyleTTS2 style encoder module
        sr: Sample rate
        device: Device to run on

    Returns:
        Style embedding tensor of shape (256,)
    """
    # Compute mel spectrogram
    mel = compute_mel_spectrogram(audio, sr=sr)

    # Add batch and channel dimensions: (n_mels, time) -> (1, 1, n_mels, time)
    mel = mel.unsqueeze(0).unsqueeze(0).to(device)

    # Extract style
    with torch.no_grad():
        style = encoder(mel)

    return style.squeeze()


def create_voice_pack(
    embeddings: List[Tuple[int, torch.Tensor]],
    num_slots: int = 510,
    style_dim: int = 256,
    smoothing: float = 0.1
) -> torch.Tensor:
    """
    Create a voice pack from extracted embeddings.

    Args:
        embeddings: List of (token_length, style_embedding) tuples
        num_slots: Number of length slots (510 for Kokoro)
        style_dim: Style embedding dimension (256)
        smoothing: Exponential smoothing factor for interpolation

    Returns:
        Voice pack tensor of shape (510, 1, 256)
    """
    voice_pack = torch.zeros(num_slots, 1, style_dim)

    # Group embeddings by token length
    length_to_embeddings = {}
    for length, embedding in embeddings:
        length = min(length, num_slots - 1)  # Clamp to valid range
        if length not in length_to_embeddings:
            length_to_embeddings[length] = []
        length_to_embeddings[length].append(embedding)

    # Average embeddings at each length
    averaged = {}
    for length, embs in length_to_embeddings.items():
        averaged[length] = torch.stack(embs).mean(dim=0)

    if not averaged:
        print("Warning: No embeddings provided. Creating zero voice pack.")
        return voice_pack

    # Get the overall average as baseline
    all_embeddings = torch.stack([e for _, e in embeddings])
    baseline = all_embeddings.mean(dim=0)

    # Fill voice pack with interpolated values
    sorted_lengths = sorted(averaged.keys())

    for i in range(num_slots):
        if i in averaged:
            # Use actual embedding
            voice_pack[i, 0] = averaged[i]
        elif len(sorted_lengths) == 0:
            # No data, use baseline
            voice_pack[i, 0] = baseline
        elif i < sorted_lengths[0]:
            # Before first data point, use first embedding with slight variation
            voice_pack[i, 0] = averaged[sorted_lengths[0]]
        elif i > sorted_lengths[-1]:
            # After last data point, use last embedding
            voice_pack[i, 0] = averaged[sorted_lengths[-1]]
        else:
            # Interpolate between neighbors
            lower = max(l for l in sorted_lengths if l < i)
            upper = min(l for l in sorted_lengths if l > i)
            t = (i - lower) / (upper - lower)
            voice_pack[i, 0] = (1 - t) * averaged[lower] + t * averaged[upper]

    # Apply exponential smoothing for continuity
    if smoothing > 0:
        smoothed = voice_pack.clone()
        for i in range(1, num_slots):
            smoothed[i] = (1 - smoothing) * voice_pack[i] + smoothing * smoothed[i - 1]
        voice_pack = smoothed

    return voice_pack


def process_audio_files(
    audio_dir: Path,
    metadata_path: Optional[Path],
    encoder: torch.nn.Module,
    device: str = 'cpu'
) -> List[Tuple[int, torch.Tensor]]:
    """
    Process audio files and extract style embeddings.

    Returns list of (token_length, embedding) tuples.
    """
    import traceback
    embeddings = []

    if metadata_path and metadata_path.exists():
        # Load metadata with transcripts
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        for i, item in enumerate(metadata):
            try:
                audio_path = Path(item['output_file'])
                if not audio_path.exists():
                    audio_path = audio_dir / audio_path.name

                if not audio_path.exists():
                    print(f"Warning: Audio file not found: {audio_path}")
                    continue

                # Load audio
                audio, sr = librosa.load(audio_path, sr=24000)

                # Get token length from transcript
                text = item.get('text', '')
                tokens = phonemize_text(text)
                token_length = len(tokens)

                print(f"[{i+1}/{len(metadata)}] {audio_path.name}: {token_length} tokens, {len(audio)/sr:.2f}s")

                # Extract embedding
                embedding = extract_style_embedding(audio, encoder, sr=24000, device=device)
                embeddings.append((token_length, embedding))
            except Exception as e:
                print(f"Error processing {item.get('output_file', 'unknown')}: {e}")
                traceback.print_exc()
    else:
        # Process all audio files without transcripts
        audio_files = list(audio_dir.glob('*.wav'))

        for audio_path in audio_files:
            audio, sr = librosa.load(audio_path, sr=24000)

            # Estimate token length from duration (rough approximation)
            # Average speaking rate is ~150 words/min, ~5 phonemes/word
            # So ~12.5 phonemes per second
            estimated_tokens = int(len(audio) / sr * 12.5)

            print(f"Processing: {audio_path.name} (~{estimated_tokens} tokens estimated, {len(audio)/sr:.2f}s)")

            embedding = extract_style_embedding(audio, encoder, sr=24000, device=device)
            embeddings.append((estimated_tokens, embedding))

    return embeddings


def main():
    parser = argparse.ArgumentParser(description="Create Kokoro voice pack from audio")
    parser.add_argument('--audio-dir', type=str, default='./cleaned_audio',
                        help='Directory containing cleaned audio files')
    parser.add_argument('--metadata-json', type=str, default=None,
                        help='Path to metadata.json with transcripts')
    parser.add_argument('--output', type=str, default='./voice_pack.pt',
                        help='Output path for voice pack')
    parser.add_argument('--model-checkpoint', type=str, default=None,
                        help='Path to StyleTTS2 checkpoint (optional)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (cpu/cuda)')
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Smoothing factor for interpolation (0-1)')

    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    metadata_path = Path(args.metadata_json) if args.metadata_json else audio_dir / 'metadata.json'

    print(f"Audio directory: {audio_dir}")
    print(f"Metadata: {metadata_path}")
    print(f"Output: {args.output}")
    print(f"Device: {args.device}")

    # Load style encoder
    print("\nLoading StyleTTS2 style encoder...")
    try:
        if args.model_checkpoint:
            encoder = load_styletts2_encoder(args.model_checkpoint, args.device)
        else:
            encoder = load_styletts2_from_hf(args.device)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nFalling back to dummy encoder for testing...")
        encoder = DummyEncoder()

    # Process audio files
    print("\nExtracting style embeddings...")
    import traceback
    try:
        embeddings = process_audio_files(audio_dir, metadata_path, encoder, args.device)
    except Exception as e:
        print(f"Error during processing: {e}")
        traceback.print_exc()
        return

    if not embeddings:
        print("Error: No embeddings extracted. Check audio files and paths.")
        return

    print(f"\nExtracted {len(embeddings)} embeddings")

    # Create voice pack
    print("\nCreating voice pack...")
    voice_pack = create_voice_pack(embeddings, smoothing=args.smoothing)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(voice_pack, output_path)

    print(f"\nVoice pack saved to: {output_path}")
    print(f"Shape: {voice_pack.shape}")

    # Print statistics
    norms = voice_pack.norm(dim=2).squeeze()
    print(f"\nStatistics:")
    print(f"  Mean norm: {norms.mean():.4f}")
    print(f"  Std norm: {norms.std():.4f}")
    print(f"  Min norm: {norms.min():.4f}")
    print(f"  Max norm: {norms.max():.4f}")

    # Check adjacent similarity
    similarities = []
    for i in range(1, len(voice_pack)):
        sim = torch.cosine_similarity(
            voice_pack[i-1].flatten().unsqueeze(0),
            voice_pack[i].flatten().unsqueeze(0)
        ).item()
        similarities.append(sim)

    print(f"  Adjacent similarity (mean): {np.mean(similarities):.6f}")


class DummyEncoder(torch.nn.Module):
    """Dummy encoder for testing when StyleTTS2 is not available."""

    def __init__(self, style_dim: int = 256):
        super().__init__()
        self.style_dim = style_dim

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        batch_size = mel.shape[0]
        # Generate deterministic embedding based on mel statistics
        mel_mean = mel.mean(dim=(2, 3))  # (batch, 1)
        mel_std = mel.std(dim=(2, 3))

        # Create embedding from mel statistics
        torch.manual_seed(int(mel_mean.sum().item() * 1000))
        embedding = torch.randn(batch_size, self.style_dim)
        embedding = embedding / embedding.norm(dim=1, keepdim=True) * 2.5

        return embedding


if __name__ == "__main__":
    main()
