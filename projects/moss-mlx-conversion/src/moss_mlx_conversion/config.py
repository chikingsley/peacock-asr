from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast


def _filtered(cls: type[Any], params: dict[str, Any]) -> dict[str, Any]:
    allowed = inspect.signature(cls).parameters
    return {key: value for key, value in params.items() if key in allowed}


@dataclass
class AudioEncoderConfig:
    num_mel_bins: int = 128
    encoder_layers: int = 32
    encoder_attention_heads: int = 20
    encoder_ffn_dim: int = 5120
    d_model: int = 1280
    dropout: float = 0.0
    attention_dropout: float = 0.0
    activation_function: str = "gelu"
    activation_dropout: float = 0.0
    scale_embedding: bool = False
    initializer_range: float = 0.02
    max_source_positions: int = 1500
    n_window: int = 50
    output_dim: int = 2048
    n_window_infer: int = 800
    conv_chunksize: int = 500
    downsample_hidden_size: int = 480
    model_type: str = "qwen3_omni_moe_audio_encoder"

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> AudioEncoderConfig:
        return cls(**_filtered(cls, params))


@dataclass
class TextConfig:
    model_type: str = "qwen3"
    vocab_size: int = 151936
    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 40960
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = True
    rope_theta: float = 1_000_000.0
    rope_scaling: dict[str, Any] | None = None
    attention_bias: bool = False
    attention_dropout: float = 0.0

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> TextConfig:
        return cls(**_filtered(cls, params))


@dataclass
class MossModelConfig:
    audio_config: AudioEncoderConfig = field(default_factory=AudioEncoderConfig)
    text_config: TextConfig = field(default_factory=TextConfig)
    model_type: str = "moss_transcribe"
    adapter_hidden_size: int = 8192
    ignore_index: int = -100
    audio_placeholder_id: int = 0
    start_token_id: int = 151644
    end_token_id: int = 151645
    audio_start_token_id: int = 151669
    audio_end_token_id: int = 151670
    sample_rate: int = 16_000
    mel_dim: int = 128
    mel_n_fft: int = 400
    mel_hop_length: int = 160

    def __post_init__(self) -> None:
        if isinstance(self.audio_config, dict):
            self.audio_config = AudioEncoderConfig.from_dict(
                cast("dict[str, Any]", self.audio_config)
            )
        if isinstance(self.text_config, dict):
            self.text_config = TextConfig.from_dict(cast("dict[str, Any]", self.text_config))

    @classmethod
    def from_moss_dict(cls, params: dict[str, Any]) -> MossModelConfig:
        audio_config = AudioEncoderConfig.from_dict(params.get("audio_config", {}))
        text_config = TextConfig.from_dict(
            params.get("language_config", params.get("text_config", {}))
        )
        return cls(
            audio_config=audio_config,
            text_config=text_config,
            adapter_hidden_size=params.get("adapter_hidden_size", 8192),
            ignore_index=params.get("ignore_index", -100),
            audio_placeholder_id=params.get("audio_placeholder_id", 0),
            start_token_id=params.get("start_token_id", 151644),
            end_token_id=params.get("end_token_id", 151645),
            audio_start_token_id=params.get("audio_start_token_id", 151669),
            audio_end_token_id=params.get("audio_end_token_id", 151670),
            sample_rate=params.get("sample_rate", 16_000),
            mel_dim=params.get("mel_dim", 128),
            mel_n_fft=params.get("mel_n_fft", 400),
            mel_hop_length=params.get("mel_hop_length", 160),
        )

    @classmethod
    def from_json(cls, path: Path) -> MossModelConfig:
        return cls.from_moss_dict(json.loads(path.read_text(encoding="utf-8")))

    def to_mlx_dict(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "audio_config": self.audio_config.__dict__,
            "text_config": self.text_config.__dict__,
            "adapter_hidden_size": self.adapter_hidden_size,
            "ignore_index": self.ignore_index,
            "audio_placeholder_id": self.audio_placeholder_id,
            "start_token_id": self.start_token_id,
            "end_token_id": self.end_token_id,
            "audio_start_token_id": self.audio_start_token_id,
            "audio_end_token_id": self.audio_end_token_id,
            "sample_rate": self.sample_rate,
            "mel_dim": self.mel_dim,
            "mel_n_fft": self.mel_n_fft,
            "mel_hop_length": self.mel_hop_length,
        }
