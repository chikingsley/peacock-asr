#!/usr/bin/env python3
"""
Loss Functions for Kokoro Voice Fine-Tuning

Implements:
- Mel Loss: L1 mel-spectrogram reconstruction
- Multi-Resolution STFT Loss: Spectral convergence + log magnitude
- Speaker Embedding Loss: Resemblyzer cosine distance
- WavLM Loss: WavLM feature matching for perceptual quality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple

# Check for optional dependencies
try:
    from resemblyzer import VoiceEncoder, preprocess_wav
    HAS_RESEMBLYZER = True
except ImportError:
    HAS_RESEMBLYZER = False
    print("Note: resemblyzer not installed. Speaker embedding loss disabled.")
    print("      Install with: pip install resemblyzer")

try:
    from transformers import WavLMModel
    HAS_WAVLM = True
except ImportError:
    HAS_WAVLM = False
    print("Note: transformers not installed. WavLM loss disabled.")
    print("      Install with: pip install transformers")


class MelLoss(nn.Module):
    """L1 Mel-spectrogram reconstruction loss."""

    def __init__(
        self,
        sample_rate: int = 24000,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max or sample_rate / 2

        # Create mel filterbank
        self.register_buffer(
            "mel_basis",
            self._create_mel_filterbank()
        )

    def _create_mel_filterbank(self) -> torch.Tensor:
        """Create mel filterbank matrix."""
        import librosa
        mel_basis = librosa.filters.mel(
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.f_min,
            fmax=self.f_max,
        )
        return torch.from_numpy(mel_basis).float()

    def audio_to_mel(self, audio: torch.Tensor) -> torch.Tensor:
        """Convert audio waveform to mel spectrogram."""
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        # Compute STFT
        window = torch.hann_window(self.n_fft, device=audio.device)
        stft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            return_complex=True,
        )
        magnitude = torch.abs(stft)

        # Apply mel filterbank
        mel_basis = self.mel_basis.to(audio.device)
        mel = torch.matmul(mel_basis, magnitude)

        # Log scale
        mel = torch.log(torch.clamp(mel, min=1e-5))

        return mel

    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute L1 mel loss between generated and target audio."""
        # Align lengths
        min_len = min(generated.shape[-1], target.shape[-1])
        generated = generated[..., :min_len]
        target = target[..., :min_len]

        # Convert to mel
        gen_mel = self.audio_to_mel(generated)
        tgt_mel = self.audio_to_mel(target)

        # L1 loss
        return F.l1_loss(gen_mel, tgt_mel)


class MultiResolutionSTFTLoss(nn.Module):
    """Multi-resolution STFT loss for audio comparison.

    Combines spectral convergence loss and log magnitude loss
    at multiple FFT resolutions for robust frequency matching.
    """

    def __init__(
        self,
        fft_sizes: List[int] = None,
        hop_sizes: List[int] = None,
        win_sizes: List[int] = None,
    ):
        super().__init__()
        self.fft_sizes = fft_sizes if fft_sizes is not None else [512, 1024, 2048]
        self.hop_sizes = hop_sizes if hop_sizes is not None else [120, 240, 480]
        self.win_sizes = win_sizes if win_sizes is not None else [480, 960, 1920]

    def stft(
        self,
        x: torch.Tensor,
        fft_size: int,
        hop_size: int,
        win_size: int
    ) -> torch.Tensor:
        """Compute STFT magnitude."""
        if x.dim() == 1:
            x = x.unsqueeze(0)

        window = torch.hann_window(win_size, device=x.device)

        stft_out = torch.stft(
            x,
            fft_size,
            hop_size,
            win_size,
            window,
            return_complex=True
        )
        return torch.abs(stft_out)

    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute multi-resolution STFT loss."""
        # Align lengths
        min_len = min(generated.shape[-1], target.shape[-1])
        generated = generated[..., :min_len]
        target = target[..., :min_len]

        total_loss = 0.0
        for fft_size, hop_size, win_size in zip(
            self.fft_sizes, self.hop_sizes, self.win_sizes
        ):
            gen_stft = self.stft(generated, fft_size, hop_size, win_size)
            tgt_stft = self.stft(target, fft_size, hop_size, win_size)

            # Spectral convergence loss: ||target - generated|| / ||target||
            sc_loss = torch.norm(tgt_stft - gen_stft, p='fro') / (torch.norm(tgt_stft, p='fro') + 1e-7)

            # Log magnitude loss: L1 on log spectrograms
            log_gen = torch.log(gen_stft + 1e-7)
            log_tgt = torch.log(tgt_stft + 1e-7)
            mag_loss = F.l1_loss(log_gen, log_tgt)

            total_loss += sc_loss + mag_loss

        return total_loss / len(self.fft_sizes)


class SpeakerEmbeddingLoss(nn.Module):
    """Speaker embedding loss using resemblyzer.

    Measures cosine distance between speaker embeddings
    of generated and target audio. Helps preserve voice identity.
    """

    def __init__(self, device: str = 'cpu'):
        super().__init__()
        if not HAS_RESEMBLYZER:
            raise ImportError(
                "resemblyzer required for speaker embedding loss. "
                "Install with: pip install resemblyzer"
            )

        self.encoder = VoiceEncoder(device=device)
        self.device = device

    def forward(
        self,
        generated: torch.Tensor,
        target: torch.Tensor,
        sr: int = 24000
    ) -> torch.Tensor:
        """Compute speaker embedding cosine distance."""
        # Convert to numpy for resemblyzer
        gen_np = generated.detach().cpu().numpy()
        tgt_np = target.detach().cpu().numpy()

        if gen_np.ndim > 1:
            gen_np = gen_np.squeeze()
        if tgt_np.ndim > 1:
            tgt_np = tgt_np.squeeze()

        # Preprocess and embed
        gen_wav = preprocess_wav(gen_np, source_sr=sr)
        tgt_wav = preprocess_wav(tgt_np, source_sr=sr)

        gen_emb = self.encoder.embed_utterance(gen_wav)
        tgt_emb = self.encoder.embed_utterance(tgt_wav)

        # Cosine distance (1 - similarity)
        similarity = np.dot(gen_emb, tgt_emb) / (
            np.linalg.norm(gen_emb) * np.linalg.norm(tgt_emb) + 1e-7
        )

        return torch.tensor(1.0 - similarity, device=self.device)


class WavLMLoss(nn.Module):
    """WavLM feature matching loss for perceptual quality.

    Uses WavLM (wav2vec 2.0 successor) to extract deep features
    and computes L1 distance. This captures high-level perceptual
    aspects that spectral losses may miss.
    """

    def __init__(
        self,
        model_name: str = "microsoft/wavlm-base",
        layer_weights: Optional[List[float]] = None,
        device: str = 'cpu',
    ):
        super().__init__()
        if not HAS_WAVLM:
            raise ImportError(
                "transformers required for WavLM loss. "
                "Install with: pip install transformers"
            )

        self.device = device
        self.model = WavLMModel.from_pretrained(model_name)
        self.model.eval()
        self.model.to(device)

        # Freeze WavLM parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Layer weights for combining hidden states
        # Default: use last 4 layers equally
        if layer_weights is None:
            self.layer_weights = [0.25, 0.25, 0.25, 0.25]
            self.layer_indices = [-4, -3, -2, -1]
        else:
            self.layer_weights = layer_weights
            self.layer_indices = list(range(-len(layer_weights), 0))

    def extract_features(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract WavLM features from audio."""
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        audio = audio.to(self.device)

        with torch.no_grad():
            outputs = self.model(audio, output_hidden_states=True)

        # Weighted combination of selected layers
        hidden_states = outputs.hidden_states
        features = torch.zeros_like(hidden_states[self.layer_indices[0]])

        for idx, weight in zip(self.layer_indices, self.layer_weights):
            features = features + weight * hidden_states[idx]

        return features

    def forward(
        self,
        generated: torch.Tensor,
        target: torch.Tensor,
        sr: int = 24000
    ) -> torch.Tensor:
        """Compute WavLM feature matching loss."""
        # WavLM expects 16kHz audio
        if sr != 16000:
            import torchaudio.functional as AF
            generated = AF.resample(generated, sr, 16000)
            target = AF.resample(target, sr, 16000)

        # Align lengths
        min_len = min(generated.shape[-1], target.shape[-1])
        generated = generated[..., :min_len]
        target = target[..., :min_len]

        # Extract features
        gen_features = self.extract_features(generated)
        tgt_features = self.extract_features(target)

        # L1 loss on features
        return F.l1_loss(gen_features, tgt_features)


class CombinedLoss(nn.Module):
    """Combined loss for voice fine-tuning.

    Combines multiple loss functions with configurable weights:
    - Mel Loss: Reconstruction quality
    - Multi-res STFT: Spectral accuracy
    - Speaker Embedding: Voice identity preservation
    - WavLM: Perceptual quality
    - L2 Regularization: Keep voice embedding close to base
    """

    def __init__(
        self,
        mel_weight: float = 1.0,
        stft_weight: float = 1.0,
        speaker_weight: float = 0.5,
        wavlm_weight: float = 1.0,
        reg_weight: float = 0.01,
        device: str = 'cpu',
        use_speaker_loss: bool = True,
        use_wavlm_loss: bool = True,
    ):
        super().__init__()

        self.mel_weight = mel_weight
        self.stft_weight = stft_weight
        self.speaker_weight = speaker_weight
        self.wavlm_weight = wavlm_weight
        self.reg_weight = reg_weight
        self.device = device

        # Initialize loss functions
        self.mel_loss = MelLoss()
        self.stft_loss = MultiResolutionSTFTLoss()

        # Optional losses
        self.speaker_loss = None
        self.wavlm_loss = None

        if use_speaker_loss and HAS_RESEMBLYZER:
            try:
                self.speaker_loss = SpeakerEmbeddingLoss(device=device)
                print("Speaker embedding loss enabled")
            except Exception as e:
                print(f"Speaker embedding loss disabled: {e}")

        if use_wavlm_loss and HAS_WAVLM:
            try:
                self.wavlm_loss = WavLMLoss(device=device)
                print("WavLM loss enabled")
            except Exception as e:
                print(f"WavLM loss disabled: {e}")

    def forward(
        self,
        generated: torch.Tensor,
        target: torch.Tensor,
        voice_embedding: Optional[torch.Tensor] = None,
        base_embedding: Optional[torch.Tensor] = None,
        sr: int = 24000
    ) -> Tuple[torch.Tensor, dict]:
        """Compute combined loss.

        Args:
            generated: Generated audio tensor
            target: Target audio tensor
            voice_embedding: Current voice embedding (for regularization)
            base_embedding: Base voice embedding (for regularization)
            sr: Sample rate

        Returns:
            total_loss: Combined weighted loss
            loss_dict: Dictionary with individual loss values
        """
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=self.device)

        # Mel loss
        mel_l = self.mel_loss(generated, target)
        loss_dict['mel'] = mel_l.item()
        total_loss = total_loss + self.mel_weight * mel_l

        # Multi-resolution STFT loss
        stft_l = self.stft_loss(generated, target)
        loss_dict['stft'] = stft_l.item()
        total_loss = total_loss + self.stft_weight * stft_l

        # Speaker embedding loss
        if self.speaker_loss is not None:
            spk_l = self.speaker_loss(generated, target, sr)
            loss_dict['speaker'] = spk_l.item()
            total_loss = total_loss + self.speaker_weight * spk_l

        # WavLM loss
        if self.wavlm_loss is not None:
            wavlm_l = self.wavlm_loss(generated, target, sr)
            loss_dict['wavlm'] = wavlm_l.item()
            total_loss = total_loss + self.wavlm_weight * wavlm_l

        # L2 regularization on voice embedding
        if voice_embedding is not None and base_embedding is not None:
            reg_l = F.mse_loss(voice_embedding, base_embedding)
            loss_dict['regularization'] = reg_l.item()
            total_loss = total_loss + self.reg_weight * reg_l

        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict


if __name__ == "__main__":
    # Test the loss functions
    print("Testing loss functions...")

    # Create dummy audio
    sr = 24000
    duration = 1.0
    t = torch.linspace(0, duration, int(sr * duration))

    # Two slightly different sine waves
    generated = torch.sin(2 * np.pi * 440 * t)  # 440 Hz
    target = torch.sin(2 * np.pi * 445 * t)     # 445 Hz

    print(f"Audio shape: {generated.shape}")

    # Test Mel loss
    mel_loss = MelLoss()
    mel_l = mel_loss(generated, target)
    print(f"Mel loss: {mel_l.item():.4f}")

    # Test STFT loss
    stft_loss = MultiResolutionSTFTLoss()
    stft_l = stft_loss(generated, target)
    print(f"STFT loss: {stft_l.item():.4f}")

    # Test combined loss
    combined = CombinedLoss(
        use_speaker_loss=HAS_RESEMBLYZER,
        use_wavlm_loss=HAS_WAVLM
    )
    total, losses = combined(generated, target)
    print(f"Combined loss: {total.item():.4f}")
    print(f"Loss breakdown: {losses}")

    print("\nAll loss functions working!")
