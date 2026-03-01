#!/usr/bin/env python3
"""
Kokoro Voice Training - Adapted for Trelis Data

Trains a voice embedding delta using gradient descent on the voice pack.
This adapts the existing voice pack by learning corrections that make
the generated audio sound more like the target speaker.

Key insight: We're not training the decoder - we're training the voice
embedding (style vector) that conditions the decoder.

Usage:
    python train_voice.py \
        --data-dir ./cleaned_audio \
        --base-voice af_bella \
        --epochs 30 \
        --output trelis_voice_trained.pt
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import soundfile as sf


class MultiResolutionSTFTLoss(nn.Module):
    """Multi-resolution STFT loss for audio comparison."""

    def __init__(
        self,
        fft_sizes=[512, 1024, 2048],
        hop_sizes=[120, 240, 480],
        win_sizes=[480, 960, 1920],
    ):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_sizes = win_sizes

    def stft(self, x, fft_size, hop_size, win_size):
        """Compute STFT magnitude."""
        if x.dim() == 1:
            x = x.unsqueeze(0)

        window = torch.hann_window(win_size, device=x.device)

        stft_out = torch.stft(
            x, fft_size, hop_size, win_size, window,
            return_complex=True
        )
        return torch.abs(stft_out)

    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute multi-resolution STFT loss."""
        min_len = min(generated.shape[-1], target.shape[-1])
        generated = generated[..., :min_len]
        target = target[..., :min_len]

        total_loss = 0.0
        for fft_size, hop_size, win_size in zip(
            self.fft_sizes, self.hop_sizes, self.win_sizes
        ):
            gen_stft = self.stft(generated, fft_size, hop_size, win_size)
            tgt_stft = self.stft(target, fft_size, hop_size, win_size)

            # Spectral convergence loss
            sc_loss = torch.norm(tgt_stft - gen_stft, p='fro') / (torch.norm(tgt_stft, p='fro') + 1e-7)

            # Log magnitude loss
            log_gen = torch.log(gen_stft + 1e-7)
            log_tgt = torch.log(tgt_stft + 1e-7)
            mag_loss = F.l1_loss(log_gen, log_tgt)

            total_loss += sc_loss + mag_loss

        return total_loss / len(self.fft_sizes)


class SpeakerEmbeddingLoss(nn.Module):
    """Loss based on speaker embedding similarity using resemblyzer."""

    def __init__(self, device='cpu'):
        super().__init__()
        try:
            from resemblyzer import VoiceEncoder, preprocess_wav
            self.encoder = VoiceEncoder(device=device)
            self.preprocess_wav = preprocess_wav
            self.available = True
        except ImportError:
            print("Warning: resemblyzer not installed. Speaker loss disabled.")
            self.available = False

        self.device = device

    def forward(self, generated: torch.Tensor, target: torch.Tensor, sr: int = 24000) -> torch.Tensor:
        """Compute speaker embedding distance."""
        if not self.available:
            return torch.tensor(0.0, device=self.device)

        gen_np = generated.detach().cpu().numpy()
        tgt_np = target.detach().cpu().numpy()

        if gen_np.ndim > 1:
            gen_np = gen_np.squeeze()
        if tgt_np.ndim > 1:
            tgt_np = tgt_np.squeeze()

        gen_wav = self.preprocess_wav(gen_np, source_sr=sr)
        tgt_wav = self.preprocess_wav(tgt_np, source_sr=sr)

        gen_emb = self.encoder.embed_utterance(gen_wav)
        tgt_emb = self.encoder.embed_utterance(tgt_wav)

        similarity = np.dot(gen_emb, tgt_emb) / (np.linalg.norm(gen_emb) * np.linalg.norm(tgt_emb) + 1e-7)

        return torch.tensor(1.0 - similarity, device=self.device)


class KokoroVoiceTrainer:
    """Train voice embedding using ONNX Kokoro model."""

    def __init__(
        self,
        model_path: str,
        voices_path: str,
        base_voice: str = "af_bella",
        device: str = "cpu",
    ):
        self.device = device

        # Load ONNX model
        print(f"Loading Kokoro ONNX model from {model_path}...")
        from kokoro_onnx import Kokoro
        self.kokoro = Kokoro(model_path, voices_path)

        # Load base voice
        voices = np.load(voices_path, allow_pickle=True)
        if base_voice not in voices.files:
            raise ValueError(f"Voice '{base_voice}' not found. Available: {voices.files[:10]}")

        self.base_voice = torch.from_numpy(voices[base_voice]).float()
        print(f"Loaded base voice '{base_voice}': shape {self.base_voice.shape}")

        # Initialize losses
        self.stft_loss = MultiResolutionSTFTLoss()
        self.speaker_loss = SpeakerEmbeddingLoss(device=device)

        self.samples = []

    def load_training_data(self, data_dir: str, metadata_file: str = "metadata.json"):
        """Load training audio and transcripts from metadata.json."""
        data_path = Path(data_dir)
        metadata_path = data_path / metadata_file

        with open(metadata_path) as f:
            metadata = json.load(f)

        samples = []
        for item in metadata:
            audio_path = Path(item['output_file'])
            if not audio_path.exists():
                audio_path = data_path / audio_path.name

            if audio_path.exists():
                audio, sr = librosa.load(audio_path, sr=24000)
                audio_tensor = torch.from_numpy(audio).float()

                samples.append({
                    "audio": audio_tensor,
                    "text": item['text'],
                    "filename": audio_path.name,
                    "sr": sr,
                    "duration": len(audio) / sr
                })
                print(f"Loaded {audio_path.name}: {len(audio)/sr:.2f}s")

        print(f"\nTotal: {len(samples)} training samples")
        self.samples = samples
        return samples

    def generate_with_voice(self, text: str, voice: np.ndarray) -> Tuple[np.ndarray, int]:
        """Generate audio using Kokoro with given voice embedding."""
        # Create temporary voices file with our voice
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            np.savez(f.name, custom=voice)
            temp_path = f.name

        try:
            from kokoro_onnx import Kokoro
            kokoro_temp = Kokoro(
                self.kokoro.model_path if hasattr(self.kokoro, 'model_path') else "../models/kokoro-v1.0.onnx",
                temp_path
            )
            samples, sr = kokoro_temp.create(text, voice="custom", speed=1.0)
            return samples, sr
        finally:
            import os
            os.unlink(temp_path)

    def evaluate_voice(self, voice: torch.Tensor, samples: List[dict],
                        stft_weight: float = 1.0, speaker_weight: float = 0.3) -> float:
        """Evaluate a voice embedding on training samples."""
        total_loss = 0.0
        n_samples = 0

        voice_np = voice.numpy().astype(np.float32)

        for sample in samples:
            try:
                generated_np, sr = self.generate_with_voice(sample["text"], voice_np)
                generated = torch.from_numpy(generated_np).float()
                target = sample["audio"]

                stft_l = self.stft_loss(generated, target).item()

                if self.speaker_loss.available:
                    spk_l = self.speaker_loss(generated, target, sr).item()
                else:
                    spk_l = 0.0

                loss = stft_weight * stft_l + speaker_weight * spk_l
                total_loss += loss
                n_samples += 1

            except Exception as e:
                print(f"  Error evaluating {sample['filename']}: {e}")
                continue

        return total_loss / max(n_samples, 1)

    def train(
        self,
        epochs: int = 30,
        lr: float = 0.01,
        stft_weight: float = 1.0,
        speaker_weight: float = 0.3,
        subset_size: Optional[int] = None,
        population_size: int = 10,
    ) -> torch.Tensor:
        """Train voice embedding using evolution strategies.

        Since ONNX doesn't support backpropagation, we use a simple
        evolutionary approach:
        1. Generate population of delta vectors
        2. Evaluate each on subset of samples
        3. Keep best and generate new population around it
        """
        dim = self.base_voice.shape[-1]

        # Initialize with small random delta
        best_delta = torch.zeros(dim)
        best_loss = float('inf')

        train_samples = self.samples[:subset_size] if subset_size else self.samples

        # Use only first few samples for faster evaluation during training
        eval_samples = train_samples[:3] if len(train_samples) > 3 else train_samples

        print(f"\n{'='*60}")
        print("TRAINING (Evolution Strategy)")
        print(f"{'='*60}")
        print(f"Samples for training: {len(train_samples)}")
        print(f"Samples for eval: {len(eval_samples)}")
        print(f"Epochs: {epochs}")
        print(f"Population size: {population_size}")
        print(f"STFT weight: {stft_weight}")
        print(f"Speaker weight: {speaker_weight}")
        print(f"{'='*60}\n")

        # Initial evaluation with base voice
        print("Evaluating base voice...")
        voice = self.base_voice.clone()
        base_loss = self.evaluate_voice(voice, eval_samples, stft_weight, speaker_weight)
        print(f"Base voice loss: {base_loss:.4f}\n")

        sigma = 0.1  # Initial mutation strength

        for epoch in range(epochs):
            # Generate population
            population = []
            for _ in range(population_size):
                # Add gaussian noise to best delta
                noise = torch.randn(dim) * sigma
                candidate = best_delta + noise
                population.append(candidate)

            # Evaluate population
            losses = []
            for i, delta in enumerate(population):
                voice = self.base_voice.clone() + delta.view(1, 1, -1)
                loss = self.evaluate_voice(voice, eval_samples, stft_weight, speaker_weight)
                losses.append(loss)
                print(f"  Pop {i+1}/{population_size}: loss={loss:.4f}", end='\r')

            # Find best
            best_idx = np.argmin(losses)
            if losses[best_idx] < best_loss:
                best_loss = losses[best_idx]
                best_delta = population[best_idx].clone()
                sigma *= 1.1  # Increase exploration if improving
            else:
                sigma *= 0.9  # Decrease if not improving

            sigma = max(0.01, min(sigma, 1.0))  # Clamp sigma

            print(f"Epoch {epoch+1:3d}/{epochs}: best_loss={best_loss:.4f} "
                  f"epoch_best={losses[best_idx]:.4f} sigma={sigma:.4f}")

        # Final evaluation on all samples
        print(f"\nFinal evaluation on all {len(train_samples)} samples...")
        voice = self.base_voice.clone() + best_delta.view(1, 1, -1)
        final_loss = self.evaluate_voice(voice, train_samples, stft_weight, speaker_weight)
        print(f"Final loss: {final_loss:.4f}")
        print(f"Improvement over base: {base_loss - final_loss:.4f} ({(base_loss-final_loss)/base_loss*100:.1f}%)")

        return best_delta

    def create_voice_pack(self, delta: torch.Tensor) -> torch.Tensor:
        """Create full voice pack from trained delta."""
        voice = self.base_voice.clone()
        voice = voice + delta.view(1, 1, -1)
        return voice

    def save_voice(self, voice: torch.Tensor, output_path: str):
        """Save voice pack."""
        torch.save(voice, output_path)
        print(f"Saved voice to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Kokoro voice embedding")

    parser.add_argument("--data-dir", default="./cleaned_audio",
                        help="Training data directory with metadata.json")
    parser.add_argument("--model", default="../models/kokoro-v1.0.onnx",
                        help="Path to Kokoro ONNX model")
    parser.add_argument("--voices", default="../models/voices-v1.0.bin",
                        help="Path to voices file")
    parser.add_argument("--base-voice", default="af_bella",
                        help="Base voice to adapt from")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate")
    parser.add_argument("--stft-weight", type=float, default=1.0,
                        help="STFT loss weight")
    parser.add_argument("--speaker-weight", type=float, default=0.3,
                        help="Speaker embedding loss weight")
    parser.add_argument("--subset", type=int, default=None,
                        help="Use only first N samples")
    parser.add_argument("--output", default="trelis_voice_trained.pt",
                        help="Output voice file")

    args = parser.parse_args()

    # Initialize trainer
    trainer = KokoroVoiceTrainer(
        model_path=args.model,
        voices_path=args.voices,
        base_voice=args.base_voice,
    )

    # Load data
    trainer.load_training_data(args.data_dir)

    # Train
    best_delta = trainer.train(
        epochs=args.epochs,
        lr=args.lr,
        stft_weight=args.stft_weight,
        speaker_weight=args.speaker_weight,
        subset_size=args.subset,
    )

    # Create and save voice
    new_voice = trainer.create_voice_pack(best_delta)
    trainer.save_voice(new_voice, args.output)

    print(f"\n{'='*60}")
    print("COMPLETE")
    print(f"{'='*60}")
    print(f"Voice saved to: {args.output}")
    print("\nTo test:")
    print(f"  python test_voice_pack.py --voice-pack {args.output}")


if __name__ == "__main__":
    main()
