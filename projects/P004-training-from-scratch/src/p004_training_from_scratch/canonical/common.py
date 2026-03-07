"""Canonical audio model helpers.

References
- Conformer paper: https://arxiv.org/abs/2005.08100
- PyTorch MultiheadAttention:
  https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
- PyTorch FlexAttention:
  https://pytorch.org/docs/main/nn.attention.flex_attention.html
- PyTorch Conv1d:
  https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
"""

from __future__ import annotations

import importlib
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_NUM_MELS = 80
CanonicalModelType = Literal["tiny", "conformer_like"]
CanonicalAttentionBackend = Literal["mha", "flex_auto", "flex_triton", "flex_flash"]


@dataclass(frozen=True, slots=True)
class CanonicalModelConfig:
    model_type: CanonicalModelType = "tiny"
    hidden_dim: int = 256
    encoder_layers: int = 4
    attention_heads: int = 4
    conv_kernel_size: int = 15
    dropout: float = 0.1
    attention_backend: CanonicalAttentionBackend = "mha"


def build_canonical_ctc_model(
    *,
    torch: Any,
    input_dim: int,
    vocab_size: int,
    config: CanonicalModelConfig,
) -> Any:
    if config.hidden_dim <= 0:
        msg = "hidden_dim must be positive"
        raise ValueError(msg)
    if config.encoder_layers <= 0:
        msg = "encoder_layers must be positive"
        raise ValueError(msg)
    if config.attention_heads <= 0:
        msg = "attention_heads must be positive"
        raise ValueError(msg)
    if config.hidden_dim % config.attention_heads != 0:
        msg = "hidden_dim must be divisible by attention_heads"
        raise ValueError(msg)
    if config.conv_kernel_size <= 0 or config.conv_kernel_size % 2 == 0:
        msg = "conv_kernel_size must be a positive odd integer"
        raise ValueError(msg)
    if not 0.0 <= config.dropout < 1.0:
        msg = "dropout must be in the range [0.0, 1.0)"
        raise ValueError(msg)
    if config.model_type == "tiny" and config.attention_backend != "mha":
        msg = "tiny model only supports attention_backend='mha'"
        raise ValueError(msg)

    if config.model_type == "tiny":
        return build_tiny_canonical_ctc(
            torch=torch,
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            vocab_size=vocab_size,
        )
    if config.model_type == "conformer_like":
        return build_conformer_like_canonical_ctc(
            torch=torch,
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            vocab_size=vocab_size,
            encoder_layers=config.encoder_layers,
            attention_heads=config.attention_heads,
            conv_kernel_size=config.conv_kernel_size,
            dropout=config.dropout,
            attention_backend=config.attention_backend,
        )

    msg = f"unsupported canonical model type: {config.model_type}"
    raise ValueError(msg)


def build_tiny_canonical_ctc(
    *,
    torch: Any,
    input_dim: int,
    hidden_dim: int,
    vocab_size: int,
) -> Any:
    class TinyCanonicalCtc(torch.nn.Module):
        """A tiny compile-friendly CTC stack for canonical-lane smoke tests."""

        def __init__(self) -> None:
            super().__init__()
            self.input_proj = torch.nn.Linear(input_dim, hidden_dim)
            self.conv = torch.nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                padding=1,
            )
            self.norm = torch.nn.LayerNorm(hidden_dim)
            self.ffn = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim * 2),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim * 2, hidden_dim),
            )
            self.output = torch.nn.Linear(hidden_dim, vocab_size)

        def forward(self, features: Any) -> Any:
            x = self.input_proj(features)
            residual = x
            x = self.conv(x.transpose(1, 2)).transpose(1, 2)
            x = self.norm(x + residual)
            x = x + self.ffn(x)
            return self.output(x)

    return TinyCanonicalCtc()


def build_conformer_like_canonical_ctc(
    *,
    torch: Any,
    input_dim: int,
    hidden_dim: int,
    vocab_size: int,
    encoder_layers: int,
    attention_heads: int,
    conv_kernel_size: int,
    dropout: float,
    attention_backend: CanonicalAttentionBackend,
) -> Any:
    kernel_options = _resolve_attention_kernel_options(attention_backend)

    class MultiheadSelfAttention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = torch.nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=attention_heads,
                dropout=dropout,
                batch_first=True,
            )

        def forward(self, features: Any) -> Any:
            attn_output, _ = self.attn(
                features,
                features,
                features,
                need_weights=False,
            )
            return attn_output

    class FlexSelfAttention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            try:
                flex_module = importlib.import_module(
                    "torch.nn.attention.flex_attention"
                )
            except Exception as exc:
                msg = (
                    "attention_backend requires torch.nn.attention.flex_attention; "
                    "use the nightly env for flex_attention backends"
                )
                raise RuntimeError(msg) from exc

            self.flex_attention = flex_module.flex_attention
            self.head_dim = hidden_dim // attention_heads
            self.num_heads = attention_heads
            self.kernel_options = kernel_options
            self.in_proj = torch.nn.Linear(hidden_dim, hidden_dim * 3)
            self.out_proj = torch.nn.Linear(hidden_dim, hidden_dim)

        def forward(self, features: Any) -> Any:
            batch_size, seq_len, _ = features.shape
            qkv = self.in_proj(features)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
            k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
            v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()
            attn_output = self.flex_attention(
                q,
                k,
                v,
                kernel_options=self.kernel_options,
            )
            # Guard against compiled CUDAGraph output reuse in the nightly lane.
            attn_output = attn_output.clone()
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(batch_size, seq_len, hidden_dim)
            return self.out_proj(attn_output)

    class StructuredEncoderBlock(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            ffn_dim = hidden_dim * 4
            self.attn_norm = torch.nn.LayerNorm(hidden_dim)
            if attention_backend == "mha":
                self.attn = MultiheadSelfAttention()
            else:
                self.attn = FlexSelfAttention()
            self.attn_dropout = torch.nn.Dropout(dropout)
            self.conv_norm = torch.nn.LayerNorm(hidden_dim)
            self.conv_in = torch.nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=1)
            self.depthwise = torch.nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_size=conv_kernel_size,
                padding=conv_kernel_size // 2,
                groups=hidden_dim,
            )
            self.conv_out = torch.nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
            self.conv_dropout = torch.nn.Dropout(dropout)
            self.ffn_norm = torch.nn.LayerNorm(hidden_dim)
            self.ffn = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, ffn_dim),
                torch.nn.SiLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(ffn_dim, hidden_dim),
                torch.nn.Dropout(dropout),
            )

        def forward(self, features: Any) -> Any:
            attn_input = self.attn_norm(features)
            attn_output = self.attn(attn_input)
            features = features + self.attn_dropout(attn_output)

            conv_input = self.conv_norm(features).transpose(1, 2)
            conv_input = self.conv_in(conv_input)
            conv_input = torch.nn.functional.glu(conv_input, dim=1)
            conv_input = self.depthwise(conv_input)
            conv_input = torch.nn.functional.silu(conv_input)
            conv_output = self.conv_out(conv_input).transpose(1, 2)
            features = features + self.conv_dropout(conv_output)

            return features + self.ffn(self.ffn_norm(features))

    class ConformerLikeCanonicalCtc(torch.nn.Module):
        """A compact structured encoder with attention and depthwise conv blocks."""

        def __init__(self) -> None:
            super().__init__()
            self.input_proj = torch.nn.Linear(input_dim, hidden_dim)
            self.input_dropout = torch.nn.Dropout(dropout)
            self.blocks = torch.nn.ModuleList(
                StructuredEncoderBlock() for _ in range(encoder_layers)
            )
            self.output_norm = torch.nn.LayerNorm(hidden_dim)
            self.output = torch.nn.Linear(hidden_dim, vocab_size)

        def forward(self, features: Any) -> Any:
            hidden = self.input_dropout(self.input_proj(features))
            for block in self.blocks:
                hidden = block(hidden)
            return self.output(self.output_norm(hidden))

    return ConformerLikeCanonicalCtc()


def _resolve_attention_kernel_options(
    attention_backend: CanonicalAttentionBackend,
) -> dict[str, str] | None:
    if attention_backend == "mha":
        return None
    if attention_backend == "flex_auto":
        return None
    if attention_backend == "flex_triton":
        return {"BACKEND": "TRITON"}
    if attention_backend == "flex_flash":
        return {"BACKEND": "FLASH"}
    msg = f"unsupported attention backend: {attention_backend}"
    raise ValueError(msg)


def read_local_wav(*, path: Path, torch: Any) -> tuple[Any, int]:
    with wave.open(str(path), "rb") as handle:
        sample_rate = handle.getframerate()
        sample_width = handle.getsampwidth()
        channel_count = handle.getnchannels()
        frame_count = handle.getnframes()
        pcm_bytes = handle.readframes(frame_count)

    if sample_width != 2:
        msg = f"unsupported WAV sample width for smoke loader: {sample_width}"
        raise ValueError(msg)

    pcm_buffer = bytearray(pcm_bytes)
    waveform = torch.frombuffer(pcm_buffer, dtype=torch.int16).clone()
    waveform = waveform.view(-1, channel_count).transpose(0, 1).float() / 32768.0
    return waveform, sample_rate


def load_log_mel_features(
    *,
    path: Path,
    torch: Any,
    torchaudio: Any,
    device: Any | None = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    n_mels: int = DEFAULT_NUM_MELS,
) -> Any:
    waveform, source_sample_rate = read_local_wav(path=path, torch=torch)
    waveform = waveform.mean(dim=0, keepdim=True)
    if source_sample_rate != sample_rate:
        waveform = torchaudio.functional.resample(
            waveform,
            source_sample_rate,
            sample_rate,
        )
    if device is not None:
        waveform = waveform.to(device)

    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=400,
        win_length=400,
        hop_length=160,
        n_mels=n_mels,
        center=True,
    )
    if device is not None:
        transform = transform.to(device)

    mels = transform(waveform).clamp_min(1e-5).log()
    return mels.transpose(1, 2).squeeze(0).contiguous()


__all__ = [
    "DEFAULT_NUM_MELS",
    "DEFAULT_SAMPLE_RATE",
    "CanonicalAttentionBackend",
    "CanonicalModelConfig",
    "CanonicalModelType",
    "_resolve_attention_kernel_options",
    "build_canonical_ctc_model",
    "build_conformer_like_canonical_ctc",
    "build_tiny_canonical_ctc",
    "load_log_mel_features",
    "read_local_wav",
]
