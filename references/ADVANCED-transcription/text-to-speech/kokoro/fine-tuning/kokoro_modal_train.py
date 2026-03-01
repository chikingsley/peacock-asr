#!/usr/bin/env python3
"""
Kokoro Voice Fine-Tuning on Modal

Trains a custom voice for Kokoro TTS using:
- Learnable voice embedding initialized from blended base
- Combined loss: Mel + STFT + Speaker + WavLM + Regularization

Usage:
    # Run training on Modal
    cd fine-tuning && uv run modal run kokoro_modal_train.py

    # With custom parameters
    uv run modal run kokoro_modal_train.py --epochs 100 --lr 1e-4
"""

import modal
from pathlib import Path

# Modal app configuration
app = modal.App("kokoro-voice-training")

# Docker image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")  # For audio decoding
    .pip_install(
        "torch>=2.0",
        "torchaudio>=2.0",
        "numpy",
        "librosa",
        "soundfile",
        "transformers",
        "resemblyzer",
        "huggingface_hub",
        "datasets",  # For validation dataset
        "python-dotenv",  # For .env file support
        "wandb",  # For experiment tracking
        "torchcodec",  # Required by datasets to decode audio
    )
    .pip_install("kokoro>=0.8")  # Kokoro TTS
)

# Volume for persistent storage
volume = modal.Volume.from_name("kokoro-training-data", create_if_missing=True)


# ============================================================================
# Inlined Loss Functions (to avoid mount issues)
# ============================================================================

def create_loss_functions(device: str = "cpu"):
    """Create and return loss function instances."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from typing import Optional, Tuple, List

    class MelLoss(nn.Module):
        """L1 Mel-spectrogram reconstruction loss."""

        def __init__(
            self,
            sample_rate: int = 24000,
            n_fft: int = 1024,
            hop_length: int = 256,
            n_mels: int = 80,
        ):
            super().__init__()
            self.sample_rate = sample_rate
            self.n_fft = n_fft
            self.hop_length = hop_length
            self.n_mels = n_mels

            import librosa
            mel_basis = librosa.filters.mel(
                sr=sample_rate, n_fft=n_fft, n_mels=n_mels,
                fmin=0.0, fmax=sample_rate / 2,
            )
            self.register_buffer("mel_basis", torch.from_numpy(mel_basis).float())

        def audio_to_mel(self, audio: torch.Tensor) -> torch.Tensor:
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            window = torch.hann_window(self.n_fft, device=audio.device)
            stft = torch.stft(
                audio, n_fft=self.n_fft, hop_length=self.hop_length,
                win_length=self.n_fft, window=window, return_complex=True,
            )
            magnitude = torch.abs(stft)
            mel_basis = self.mel_basis.to(audio.device)
            mel = torch.matmul(mel_basis, magnitude)
            return torch.log(torch.clamp(mel, min=1e-5))

        def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            min_len = min(generated.shape[-1], target.shape[-1])
            generated = generated[..., :min_len]
            target = target[..., :min_len]
            gen_mel = self.audio_to_mel(generated)
            tgt_mel = self.audio_to_mel(target)
            return F.l1_loss(gen_mel, tgt_mel)

    class MultiResolutionSTFTLoss(nn.Module):
        """Multi-resolution STFT loss."""

        def __init__(
            self,
            fft_sizes: List[int] = [512, 1024, 2048],
            hop_sizes: List[int] = [120, 240, 480],
            win_sizes: List[int] = [480, 960, 1920],
        ):
            super().__init__()
            self.fft_sizes = fft_sizes
            self.hop_sizes = hop_sizes
            self.win_sizes = win_sizes

        def stft(self, x, fft_size, hop_size, win_size):
            if x.dim() == 1:
                x = x.unsqueeze(0)
            window = torch.hann_window(win_size, device=x.device)
            stft_out = torch.stft(x, fft_size, hop_size, win_size, window, return_complex=True)
            return torch.abs(stft_out)

        def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            min_len = min(generated.shape[-1], target.shape[-1])
            generated = generated[..., :min_len]
            target = target[..., :min_len]

            total_loss = 0.0
            for fft_size, hop_size, win_size in zip(self.fft_sizes, self.hop_sizes, self.win_sizes):
                gen_stft = self.stft(generated, fft_size, hop_size, win_size)
                tgt_stft = self.stft(target, fft_size, hop_size, win_size)
                sc_loss = torch.norm(tgt_stft - gen_stft, p='fro') / (torch.norm(tgt_stft, p='fro') + 1e-7)
                log_gen = torch.log(gen_stft + 1e-7)
                log_tgt = torch.log(tgt_stft + 1e-7)
                mag_loss = F.l1_loss(log_gen, log_tgt)
                total_loss += sc_loss + mag_loss

            return total_loss / len(self.fft_sizes)

    class SpeakerEmbeddingLoss(nn.Module):
        """Speaker embedding loss using resemblyzer."""

        def __init__(self, device: str = 'cpu'):
            super().__init__()
            from resemblyzer import VoiceEncoder, preprocess_wav
            self.encoder = VoiceEncoder(device=device)
            self.preprocess_wav = preprocess_wav
            self.device = device

        def forward(self, generated: torch.Tensor, target: torch.Tensor, sr: int = 24000) -> torch.Tensor:
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

    class WavLMLoss(nn.Module):
        """WavLM feature matching loss."""

        def __init__(self, model_name: str = "microsoft/wavlm-base", device: str = 'cpu'):
            super().__init__()
            from transformers import WavLMModel
            self.device = device
            self.model = WavLMModel.from_pretrained(model_name)
            self.model.eval()
            self.model.to(device)
            for param in self.model.parameters():
                param.requires_grad = False

        def extract_features(self, audio: torch.Tensor) -> torch.Tensor:
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            audio = audio.to(self.device)
            with torch.no_grad():
                outputs = self.model(audio, output_hidden_states=True)
            # Use last 4 layers
            hidden_states = outputs.hidden_states
            features = (hidden_states[-4] + hidden_states[-3] + hidden_states[-2] + hidden_states[-1]) / 4
            return features

        def forward(self, generated: torch.Tensor, target: torch.Tensor, sr: int = 24000) -> torch.Tensor:
            import torchaudio.functional as AF
            if sr != 16000:
                generated = AF.resample(generated, sr, 16000)
                target = AF.resample(target, sr, 16000)
            min_len = min(generated.shape[-1], target.shape[-1])
            generated = generated[..., :min_len]
            target = target[..., :min_len]
            gen_features = self.extract_features(generated)
            tgt_features = self.extract_features(target)
            return F.l1_loss(gen_features, tgt_features)

    class CombinedLoss(nn.Module):
        """Combined loss for voice fine-tuning."""

        def __init__(
            self,
            mel_weight: float = 1.0,
            stft_weight: float = 1.0,
            speaker_weight: float = 0.5,
            wavlm_weight: float = 1.0,
            reg_weight: float = 0.01,
            device: str = 'cpu',
        ):
            super().__init__()
            self.mel_weight = mel_weight
            self.stft_weight = stft_weight
            self.speaker_weight = speaker_weight
            self.wavlm_weight = wavlm_weight
            self.reg_weight = reg_weight
            self.device = device

            self.mel_loss = MelLoss()
            self.stft_loss = MultiResolutionSTFTLoss()

            try:
                self.speaker_loss = SpeakerEmbeddingLoss(device=device)
                print("Speaker embedding loss enabled")
            except Exception as e:
                self.speaker_loss = None
                print(f"Speaker embedding loss disabled: {e}")

            try:
                self.wavlm_loss = WavLMLoss(device=device)
                print("WavLM loss enabled")
            except Exception as e:
                self.wavlm_loss = None
                print(f"WavLM loss disabled: {e}")

        def forward(
            self,
            generated: torch.Tensor,
            target: torch.Tensor,
            voice_embedding: Optional[torch.Tensor] = None,
            base_embedding: Optional[torch.Tensor] = None,
            sr: int = 24000
        ) -> Tuple[torch.Tensor, dict]:
            loss_dict = {}
            total_loss = torch.tensor(0.0, device=self.device)

            mel_l = self.mel_loss(generated, target)
            loss_dict['mel'] = mel_l.item()
            total_loss = total_loss + self.mel_weight * mel_l

            stft_l = self.stft_loss(generated, target)
            loss_dict['stft'] = stft_l.item()
            total_loss = total_loss + self.stft_weight * stft_l

            if self.speaker_loss is not None:
                spk_l = self.speaker_loss(generated, target, sr)
                loss_dict['speaker'] = spk_l.item()
                total_loss = total_loss + self.speaker_weight * spk_l

            if self.wavlm_loss is not None:
                wavlm_l = self.wavlm_loss(generated, target, sr)
                loss_dict['wavlm'] = wavlm_l.item()
                total_loss = total_loss + self.wavlm_weight * wavlm_l

            if voice_embedding is not None and base_embedding is not None:
                reg_l = F.mse_loss(voice_embedding, base_embedding)
                loss_dict['regularization'] = reg_l.item()
                total_loss = total_loss + self.reg_weight * reg_l

            loss_dict['total'] = total_loss.item()
            return total_loss, loss_dict

    return CombinedLoss(device=device)


# ============================================================================
# LoRA (Low-Rank Adaptation) for Decoder
# ============================================================================

def create_lora_linear(original_layer, rank=16, alpha=16.0, dropout=0.0):
    """Create a LoRA-wrapped linear layer."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import math

    class LoRALinear(nn.Module):
        def __init__(self, original, rank, alpha, dropout):
            super().__init__()
            self.in_features = original.in_features
            self.out_features = original.out_features
            self.rank = rank
            self.scaling = alpha / rank

            # Get device from original layer
            device = original.weight.device

            # Keep original frozen
            self.original = original
            for param in self.original.parameters():
                param.requires_grad = False

            # LoRA matrices - on same device as original
            self.lora_A = nn.Parameter(torch.zeros(rank, self.in_features, device=device))
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank, device=device))
            self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

            # Initialize
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

        def forward(self, x):
            result = self.original(x)
            lora_out = F.linear(self.dropout(x), self.lora_B @ self.lora_A) * self.scaling
            return result + lora_out

    return LoRALinear(original_layer, rank, alpha, dropout)


def apply_lora_to_decoder(model, rank=16, alpha=16.0, dropout=0.0):
    """Apply LoRA to all Linear layers in the decoder."""
    import torch.nn as nn

    adapted = []
    decoder = model.decoder

    for name, module in decoder.named_modules():
        if isinstance(module, nn.Linear):
            # Get parent and attr name
            parts = name.split('.')
            if len(parts) == 1:
                parent = decoder
                attr = parts[0]
            else:
                parent = decoder.get_submodule('.'.join(parts[:-1]))
                attr = parts[-1]

            # Replace with LoRA version
            lora_layer = create_lora_linear(module, rank=rank, alpha=alpha, dropout=dropout)
            setattr(parent, attr, lora_layer)
            adapted.append(f"decoder.{name}")

    return adapted


def get_lora_params(model):
    """Get all LoRA parameters from the model."""
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            lora_params.append(param)
    return lora_params


def get_lora_state_dict(model):
    """Extract only LoRA parameters from model."""
    import torch
    lora_state = {}
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            lora_state[name] = param.data.clone()
    return lora_state


# ============================================================================
# Trainable Forward Pass (no @torch.no_grad)
# ============================================================================

def trainable_forward(model, input_ids, ref_s, speed=1.0):
    """Forward pass WITH gradients - mirrors KModel.forward_with_tokens without @no_grad."""
    import torch

    device = model.device

    input_lengths = torch.full(
        (input_ids.shape[0],), input_ids.shape[-1],
        device=input_ids.device, dtype=torch.long
    )

    text_mask = torch.arange(input_lengths.max(), device=device).unsqueeze(0).expand(
        input_lengths.shape[0], -1
    ).type_as(input_lengths)
    text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1)).to(device)

    # BERT encoding
    bert_dur = model.bert(input_ids, attention_mask=(~text_mask).int())
    d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

    # Style split: prosody (second half)
    s = ref_s[:, 128:]

    # Duration prediction
    d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
    x, _ = model.predictor.lstm(d)
    duration = model.predictor.duration_proj(x)
    duration = torch.sigmoid(duration).sum(axis=-1) / speed
    pred_dur = torch.round(duration).clamp(min=1).long().squeeze()

    if pred_dur.dim() == 0:
        pred_dur = pred_dur.unsqueeze(0)

    # Alignment
    indices = torch.repeat_interleave(
        torch.arange(input_ids.shape[1], device=device), pred_dur
    )
    pred_aln_trg = torch.zeros(
        (input_ids.shape[1], indices.shape[0]), device=device
    )
    pred_aln_trg[indices, torch.arange(indices.shape[0], device=device)] = 1
    pred_aln_trg = pred_aln_trg.unsqueeze(0)

    en = d.transpose(-1, -2) @ pred_aln_trg

    # F0 and noise prediction
    F0_pred, N_pred = model.predictor.F0Ntrain(en, s)

    # Text encoding
    t_en = model.text_encoder(input_ids, input_lengths, text_mask)
    asr = t_en @ pred_aln_trg

    # Decode to audio (acoustic style = first 128 dims)
    audio = model.decoder(asr, F0_pred, N_pred, ref_s[:, :128]).squeeze()

    return audio, pred_dur


def phonemes_to_ids(phonemes, vocab):
    """Convert phoneme string to token IDs."""
    import torch
    ids = [vocab.get(p) for p in phonemes if vocab.get(p) is not None]
    return torch.LongTensor([[0] + ids + [0]])


# ============================================================================
# Training Function
# ============================================================================

@app.function(
    image=image,
    gpu="H100",
    timeout=3600,
    volumes={"/data": volume},
)
def train_voice(
    data_dir: str = "/data/training_audio",
    hf_token: str = None,  # HuggingFace token for private datasets
    wandb_api_key: str = None,  # WandB API key for logging
    epochs: int = 3,  # Few epochs since this isn't diffusion
    lr: float = 1e-3,  # Higher LR for faster convergence
    mel_weight: float = 1.0,
    stft_weight: float = 1.0,
    speaker_weight: float = 0.5,
    wavlm_weight: float = 1.0,
    reg_weight: float = 0.01,
    save_every: int = 1,  # Save every epoch with few epochs
):
    """Main training function with GRADIENT FLOW through voice embedding."""
    import json
    import torch
    import torch.nn.functional as F
    import numpy as np
    import librosa
    import soundfile as sf
    from huggingface_hub import hf_hub_download
    from pathlib import Path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Download voice packs (with caching on volume)
    print("Loading voice packs...")
    cache_dir = Path("/data/model_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    voices_to_download = ["af_bella", "bm_george", "bf_emma", "am_michael"]
    voices = {}
    for voice_name in voices_to_download:
        cached_path = cache_dir / f"{voice_name}.pt"
        if cached_path.exists():
            voices[voice_name] = torch.load(cached_path, map_location=device, weights_only=True)
            print(f"  Loaded {voice_name} from cache: {voices[voice_name].shape}")
        else:
            voice_path = hf_hub_download("hexgrad/Kokoro-82M", f"voices/{voice_name}.pt")
            voices[voice_name] = torch.load(voice_path, map_location=device, weights_only=True)
            # Cache for next time
            torch.save(voices[voice_name], cached_path)
            print(f"  Downloaded {voice_name}: {voices[voice_name].shape}")

    volume.commit()  # Persist cached files

    # Create blended base voice
    print("Creating blended base voice...")
    blended_base = sum(voices.values()) / len(voices)
    blended_base = blended_base.to(device)
    print(f"Blended base shape: {blended_base.shape}")

    # Initialize trainable voice embedding
    voice_embedding = blended_base.clone().requires_grad_(True)

    # Load KModel DIRECTLY (not pipeline) for gradient flow
    print("Loading KModel directly (for gradient flow)...")
    from kokoro import KModel
    model = KModel(repo_id='hexgrad/Kokoro-82M').to(device)
    model.train()  # Train mode required for cuDNN LSTM backward pass

    # Freeze model weights first
    for param in model.parameters():
        param.requires_grad = False

    # Apply LoRA to decoder linear layers
    # Lower rank and alpha for more stable/less noisy adaptation
    print("Applying LoRA to decoder...")
    lora_rank = 8  # Reduced from 16 for stability
    lora_alpha = 8.0  # Reduced for more conservative adaptation
    adapted_modules = apply_lora_to_decoder(model, rank=lora_rank, alpha=lora_alpha, dropout=0.15)
    print(f"  LoRA applied to {len(adapted_modules)} layers:")
    for m in adapted_modules[:5]:
        print(f"    - {m}")
    if len(adapted_modules) > 5:
        print(f"    ... and {len(adapted_modules) - 5} more")

    # Count trainable params
    lora_params = get_lora_params(model)
    lora_param_count = sum(p.numel() for p in lora_params)
    voice_param_count = voice_embedding.numel()
    print(f"Trainable params: {lora_param_count:,} (LoRA) + {voice_param_count:,} (voice) = {lora_param_count + voice_param_count:,} total")

    # Setup G2P for phoneme conversion
    print("Setting up G2P...")
    from misaki import en
    g2p_model = en.G2P(trf=False, british=False)

    def text_to_phonemes(text):
        _, tokens = g2p_model(text)
        # Filter out tokens with None phonemes
        return ''.join(
            (t.phonemes or '') + (' ' if t.whitespace else '')
            for t in tokens
        ).strip()

    # Initialize loss functions
    print("Initializing loss functions...")
    criterion = create_loss_functions(device=str(device))

    # Load training data
    print(f"Loading training data from {data_dir}...")
    data_path = Path(data_dir)

    metadata_file = data_path / "metadata.json"
    transcripts_file = data_path / "transcripts.json"

    samples = []
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)
        for item in metadata:
            audio_path = Path(item.get('output_file', item.get('audio_file', '')))
            if not audio_path.exists():
                audio_path = data_path / audio_path.name
            if audio_path.exists():
                audio, sr = librosa.load(audio_path, sr=24000)
                samples.append({
                    "audio": torch.from_numpy(audio).float().to(device),
                    "text": item['text'],
                    "filename": audio_path.name,
                })
    elif transcripts_file.exists():
        with open(transcripts_file) as f:
            transcripts = json.load(f)
        for filename, text in transcripts.items():
            audio_path = data_path / filename
            if audio_path.exists():
                audio, sr = librosa.load(audio_path, sr=24000)
                samples.append({
                    "audio": torch.from_numpy(audio).float().to(device),
                    "text": text,
                    "filename": filename,
                })
    else:
        for audio_file in data_path.glob("*.wav"):
            txt_file = audio_file.with_suffix(".txt")
            if txt_file.exists():
                audio, sr = librosa.load(audio_file, sr=24000)
                text = txt_file.read_text().strip()
                samples.append({
                    "audio": torch.from_numpy(audio).float().to(device),
                    "text": text,
                    "filename": audio_file.name,
                })

    print(f"Loaded {len(samples)} training samples")
    if len(samples) == 0:
        raise ValueError(f"No training data found in {data_dir}")

    # Load validation dataset from HuggingFace
    print("Loading validation dataset...")
    val_samples = []
    try:
        from datasets import load_dataset, Audio
        import soundfile as sf
        import io
        import numpy as np

        # Load dataset with token passed from local environment
        print(f"  Loading Trelis/latent-space-validation (token: {'provided' if hf_token else 'none'})...")
        val_dataset = load_dataset(
            "Trelis/latent-space-validation",
            split="validation",
            token=hf_token,  # Token passed from main()
        )
        print(f"  Dataset loaded: {len(val_dataset)} samples")
        print(f"  Columns: {val_dataset.column_names}")

        # Use numpy format to get decoded audio arrays directly
        # This avoids the torchcodec AudioDecoder object issue
        print("  Setting format to get decoded audio...")
        val_dataset = val_dataset.with_format("numpy")

        for idx, item in enumerate(val_dataset):
            try:
                # Debug: show structure of first sample
                if idx == 0:
                    print(f"  Sample 0 keys: {item.keys()}")
                    audio_data = item.get('audio')
                    print(f"  Audio type: {type(audio_data)}")
                    if isinstance(audio_data, dict):
                        print(f"  Audio dict keys: {audio_data.keys()}")
                    else:
                        # Introspect the AudioDecoder
                        print(f"  AudioDecoder attrs: {[a for a in dir(audio_data) if not a.startswith('_')]}")

                # Get audio
                audio_data = item.get('audio')
                if audio_data is None:
                    print(f"  Sample {idx}: no 'audio' key")
                    continue

                # Handle torchcodec AudioDecoder
                if hasattr(audio_data, 'get_all_samples'):
                    audio_samples = audio_data.get_all_samples()
                    # audio_samples is typically a FrameBatch with .data tensor
                    if hasattr(audio_samples, 'data'):
                        audio_tensor = audio_samples.data.squeeze()
                        # Convert to mono if stereo
                        if audio_tensor.dim() > 1:
                            audio_tensor = audio_tensor.mean(dim=0)
                        audio_array = audio_tensor.numpy()
                    else:
                        audio_array = np.array(audio_samples).squeeze()
                    sr = audio_data.metadata.sample_rate if hasattr(audio_data, 'metadata') else 24000
                    if idx == 0:
                        print(f"  Sample 0: AudioDecoder decoded, shape={audio_array.shape}, sr={sr}")
                # datasets with numpy format returns dict with 'array' and 'sampling_rate'
                elif isinstance(audio_data, dict) and 'array' in audio_data:
                    audio_array = np.array(audio_data['array'])
                    sr = audio_data.get('sampling_rate', 24000)
                    if idx == 0:
                        print(f"  Sample 0: array shape={audio_array.shape}, sr={sr}")
                elif isinstance(audio_data, np.ndarray):
                    audio_array = audio_data
                    sr = 24000
                    if idx == 0:
                        print(f"  Sample 0: ndarray shape={audio_array.shape}")
                else:
                    print(f"  Sample {idx}: unexpected audio format, type={type(audio_data)}")
                    continue

                # Resample if needed
                if sr != 24000:
                    import torchaudio.functional as F_audio
                    audio_tensor = torch.from_numpy(audio_array).float()
                    if audio_tensor.dim() == 1:
                        audio_tensor = audio_tensor.unsqueeze(0)
                    audio_tensor = F_audio.resample(audio_tensor, int(sr), 24000)
                    audio_array = audio_tensor.squeeze().numpy()

                val_samples.append({
                    "audio": torch.from_numpy(audio_array).float().to(device),
                    "text": item['text'],
                })
            except Exception as e:
                print(f"  Sample {idx} error: {e}")
                import traceback
                traceback.print_exc()
                continue

        print(f"Loaded {len(val_samples)} validation samples")
    except Exception as e:
        print(f"Could not load validation dataset: {e}")
        import traceback
        traceback.print_exc()
        print("Continuing without validation...")

    # Setup optimizer for both voice embedding and LoRA params
    trainable_params = [voice_embedding] + lora_params
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.02)  # Increased for regularization
    # Using constant LR (no scheduler) - simpler and sufficient for few epochs

    # Initialize WandB
    use_wandb = False
    if wandb_api_key:
        try:
            import wandb
            wandb.login(key=wandb_api_key)
            wandb.init(
                project="kokoro-voice-finetuning",
                config={
                    "epochs": epochs,
                    "lr": lr,
                    "lora_rank": lora_rank,
                    "lora_alpha": lora_alpha,
                    "n_train_samples": len(samples),
                    "n_val_samples": len(val_samples),
                    "voice_param_count": voice_param_count,
                    "lora_param_count": lora_param_count,
                },
            )
            use_wandb = True
            print("WandB logging enabled")
        except Exception as e:
            print(f"WandB init failed: {e}")
            use_wandb = False
    else:
        print("WandB logging disabled (no API key)")

    print(f"\n{'='*60}")
    print("TRAINING (LoRA + Voice Embedding)")
    print(f"{'='*60}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {lr}")
    print(f"Train samples: {len(samples)}, Val samples: {len(val_samples)}")
    print(f"Voice embedding shape: {voice_embedding.shape}")
    print(f"LoRA rank: {lora_rank}, layers: {len(adapted_modules)}")
    print(f"Total trainable params: {lora_param_count + voice_param_count:,}")
    print(f"{'='*60}\n")

    best_loss = float('inf')
    best_voice = voice_embedding.detach().clone()

    output_dir = Path("/data/output")
    output_dir.mkdir(exist_ok=True)

    for epoch in range(epochs):
        epoch_losses = {'total': 0.0, 'mel': 0.0, 'stft': 0.0, 'speaker': 0.0, 'wavlm': 0.0}
        n_samples = 0
        total_grad_norm = 0.0

        for sample in samples:
            optimizer.zero_grad()

            try:
                # Convert text to phonemes
                phonemes = text_to_phonemes(sample["text"])

                # Convert phonemes to token IDs
                input_ids = phonemes_to_ids(phonemes, model.vocab).to(device)

                # Get style vector for this phoneme length (index by token count)
                # voice_embedding shape: (510, 1, 256), indexing gives (1, 256)
                token_len = min(input_ids.shape[1] - 1, voice_embedding.shape[0] - 1)
                ref_s = voice_embedding[token_len]  # (1, 256) - already 2D, don't unsqueeze

                # Forward WITH gradients
                generated, _ = trainable_forward(model, input_ids, ref_s)
                target = sample["audio"]

                # Compute loss
                loss, loss_dict = criterion(
                    generated, target,
                    voice_embedding=voice_embedding,
                    base_embedding=blended_base,
                )

                # Backprop
                loss.backward()

                # Track gradient norm BEFORE clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                total_grad_norm += grad_norm.item()  # clip_grad_norm_ returns the unclipped norm

                optimizer.step()

                for key in epoch_losses:
                    if key in loss_dict:
                        epoch_losses[key] += loss_dict[key]
                n_samples += 1

            except Exception as e:
                print(f"  Error on {sample['filename']}: {e}")
                import traceback
                traceback.print_exc()
                continue

        avg_losses = {k: v / max(n_samples, 1) for k, v in epoch_losses.items()}
        avg_grad = total_grad_norm / max(n_samples, 1)

        if avg_losses['total'] < best_loss:
            best_loss = avg_losses['total']
            best_voice = voice_embedding.detach().clone()

        # Compute validation loss every epoch (since we have few epochs)
        val_loss_value = None
        if val_samples:
            val_losses = {'total': 0.0, 'mel': 0.0}
            n_val = 0
            with torch.no_grad():
                for val_sample in val_samples[:10]:  # Use first 10 for speed
                    try:
                        val_phonemes = text_to_phonemes(val_sample["text"])
                        if not val_phonemes.strip():
                            continue
                        val_input_ids = phonemes_to_ids(val_phonemes, model.vocab).to(device)
                        val_token_len = min(val_input_ids.shape[1] - 1, voice_embedding.shape[0] - 1)
                        val_ref_s = voice_embedding[val_token_len]
                        val_generated, _ = trainable_forward(model, val_input_ids, val_ref_s)
                        val_target = val_sample["audio"]
                        val_loss, val_loss_dict = criterion(val_generated, val_target)
                        val_losses['total'] += val_loss.item()
                        val_losses['mel'] += val_loss_dict.get('mel', 0)
                        n_val += 1
                    except Exception:
                        continue
            if n_val > 0:
                val_loss_value = val_losses['total'] / n_val

        val_loss_str = f" | val={val_loss_value:.4f}" if val_loss_value is not None else ""
        print(f"Epoch {epoch+1:3d}/{epochs}: "
              f"loss={avg_losses['total']:.4f} "
              f"(mel={avg_losses['mel']:.4f}, stft={avg_losses['stft']:.4f}, "
              f"spk={avg_losses.get('speaker', 0):.4f}, wavlm={avg_losses.get('wavlm', 0):.4f}) "
              f"grad={avg_grad:.6f} lr={lr:.6f}{val_loss_str}")

        # Log to WandB
        if use_wandb:
            log_dict = {
                "train/loss": avg_losses['total'],
                "train/mel": avg_losses['mel'],
                "train/stft": avg_losses['stft'],
                "train/speaker": avg_losses.get('speaker', 0),
                "train/wavlm": avg_losses.get('wavlm', 0),
                "train/grad_norm": avg_grad,
                "epoch": epoch + 1,
            }
            if val_loss_value is not None:
                log_dict["val/loss"] = val_loss_value
            wandb.log(log_dict)

        if (epoch + 1) % save_every == 0:
            checkpoint_path = output_dir / f"voice_epoch_{epoch+1}.pt"
            torch.save(voice_embedding.detach().cpu(), checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")
            volume.commit()

    print(f"\n{'='*60}")
    print("SAVING RESULTS")
    print(f"{'='*60}")

    final_voice_path = output_dir / "custom_voice.pt"
    torch.save(best_voice, final_voice_path)
    print(f"Saved voice: {final_voice_path}")

    # Save LoRA weights
    lora_path = output_dir / "lora_weights.pt"
    lora_state = get_lora_state_dict(model)
    torch.save({
        'lora_state': lora_state,
        'rank': lora_rank,
        'alpha': lora_alpha,
        'adapted_modules': adapted_modules,
    }, lora_path)
    print(f"Saved LoRA: {lora_path} ({len(lora_state)} tensors)")

    # Generate test sample using trainable_forward
    print("\nGenerating test sample...")
    test_text = "Hello, this is a test of my new custom voice."
    try:
        test_phonemes = text_to_phonemes(test_text)
        test_input_ids = phonemes_to_ids(test_phonemes, model.vocab).to(device)
        token_len = test_input_ids.shape[1]
        test_ref_s = best_voice[min(token_len - 1, 509)].to(device)

        with torch.no_grad():
            test_audio, _ = trainable_forward(model, test_input_ids, test_ref_s)

        test_audio_np = test_audio.cpu().numpy()
        test_path = output_dir / "test_sample.wav"
        sf.write(test_path, test_audio_np, 24000)
        print(f"Saved test: {test_path}")
    except Exception as e:
        print(f"Could not generate test sample: {e}")

    volume.commit()

    # Finish WandB run
    if use_wandb:
        wandb.log({"best_loss": best_loss})
        wandb.finish()
        print("WandB run finished")

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Best loss: {best_loss:.4f}")

    return {"best_loss": best_loss, "voice_path": str(final_voice_path)}


@app.function(image=image, volumes={"/data": volume})
def upload_training_data(local_files: list[tuple[str, bytes]]):
    """Upload training data to Modal volume."""
    from pathlib import Path

    remote = Path("/data/training_audio")
    remote.mkdir(parents=True, exist_ok=True)

    for filename, content in local_files:
        (remote / filename).write_bytes(content)
        print(f"Uploaded: {filename}")

    volume.commit()
    print(f"\nUploaded {len(local_files)} files")
    return str(remote)


@app.function(image=image, volumes={"/data": volume})
def download_results() -> list[tuple[str, bytes]]:
    """Download training results from Modal volume."""
    from pathlib import Path

    remote = Path("/data/output")
    files = []
    for f in remote.iterdir():
        if f.is_file():
            files.append((f.name, f.read_bytes()))
            print(f"Prepared: {f.name}")
    return files


@app.local_entrypoint()
def main(
    data_dir: str = "./cleaned_audio",
    epochs: int = 3,  # Few epochs since this isn't diffusion
    lr: float = 1e-3,  # Higher LR for faster convergence
    upload: bool = True,
):
    """Local entrypoint for running training."""
    from pathlib import Path
    from dotenv import load_dotenv
    import os

    # Load .env file for HF token and WandB
    load_dotenv()
    hf_token = os.environ.get("HF_TOKEN")
    wandb_api_key = os.environ.get("WANDB_API_KEY")

    if hf_token:
        print(f"HF token found: {hf_token[:10]}...")
    else:
        print("Warning: No HF_TOKEN found in .env - validation dataset may not load")

    if wandb_api_key:
        print(f"WandB API key found: {wandb_api_key[:10]}...")
    else:
        print("Note: No WANDB_API_KEY found in .env - logging disabled")

    if upload:
        print("Uploading training data...")
        local_path = Path(data_dir)
        files = []
        for f in local_path.iterdir():
            if f.is_file():
                files.append((f.name, f.read_bytes()))
        remote_path = upload_training_data.remote(files)
        print(f"Data uploaded to: {remote_path}")

    print("\nStarting training...")
    result = train_voice.remote(
        data_dir="/data/training_audio",
        hf_token=hf_token,  # Pass token to Modal
        wandb_api_key=wandb_api_key,  # Pass WandB key
        epochs=epochs,
        lr=lr,
    )
    print(f"\nTraining result: {result}")

    print("\nDownloading results...")
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)
    files = download_results.remote()
    for filename, content in files:
        (output_dir / filename).write_bytes(content)
        print(f"Downloaded: {filename}")

    print(f"\nResults saved to: {output_dir}")
