from __future__ import annotations

from importlib import import_module
from typing import Any

import numpy as np

from moss_mlx_conversion.config import MossModelConfig
from moss_mlx_conversion.mlx_compat import mx, nn, require_mlx
from moss_mlx_conversion.model.adapter import MossGatedMLP
from moss_mlx_conversion.model.audio_encoder import Qwen3OmniMoeAudioEncoder


class MossMLXModel(nn.Module):
    def __init__(self, config: MossModelConfig) -> None:
        require_mlx()
        super().__init__()
        cache_module: Any = import_module("mlx_lm.models.cache")
        qwen3_module: Any = import_module("mlx_lm.models.qwen3")
        cache_cls = cache_module.KVCache
        qwen3_args_cls = qwen3_module.ModelArgs
        qwen3_model_cls = qwen3_module.Qwen3Model

        self._cache_cls = cache_cls
        self.config = config
        text_args = qwen3_args_cls.from_dict(
            {
                "model_type": "qwen3",
                "hidden_size": config.text_config.hidden_size,
                "num_hidden_layers": config.text_config.num_hidden_layers,
                "intermediate_size": config.text_config.intermediate_size,
                "num_attention_heads": config.text_config.num_attention_heads,
                "num_key_value_heads": config.text_config.num_key_value_heads,
                "head_dim": config.text_config.head_dim,
                "rms_norm_eps": config.text_config.rms_norm_eps,
                "vocab_size": config.text_config.vocab_size,
                "max_position_embeddings": config.text_config.max_position_embeddings,
                "rope_theta": config.text_config.rope_theta,
                "rope_scaling": config.text_config.rope_scaling,
                "tie_word_embeddings": config.text_config.tie_word_embeddings,
            }
        )
        self.audio_model = Qwen3OmniMoeAudioEncoder(config.audio_config)
        self.audio_adapter = MossGatedMLP(
            input_size=config.audio_config.output_dim,
            hidden_size=config.adapter_hidden_size,
            output_size=config.text_config.hidden_size,
        )
        self.model = qwen3_model_cls(text_args)
        if not config.text_config.tie_word_embeddings:
            self.lm_head = nn.Linear(
                config.text_config.hidden_size,
                config.text_config.vocab_size,
                bias=False,
            )
        else:
            self.lm_head = None

    @property
    def layers(self) -> list[Any]:
        return self.model.layers

    def make_cache(self) -> list[Any]:
        return [self._cache_cls() for _ in range(self.config.text_config.num_hidden_layers)]

    def get_audio_features(self, audio_data: mx.array, audio_data_seqlens: mx.array) -> mx.array:
        audio_hidden = self.audio_model(audio_data, audio_data_seqlens)
        return self.audio_adapter(audio_hidden)

    def build_inputs_embeds(
        self,
        input_ids: mx.array,
        audio_embeds: mx.array,
        audio_input_mask: mx.array,
    ) -> mx.array:
        inputs_embeds = self.model.embed_tokens(input_ids)
        audio_embeds = audio_embeds.astype(inputs_embeds.dtype)
        batch_size, seq_len, hidden_dim = inputs_embeds.shape
        flat_mask = np.array(audio_input_mask.flatten())
        audio_indices = np.where(flat_mask)[0]
        flat_embeds = inputs_embeds.reshape(-1, hidden_dim)

        result = []
        audio_idx = 0
        for idx in range(flat_embeds.shape[0]):
            if audio_idx < len(audio_indices) and idx == audio_indices[audio_idx]:
                result.append(audio_embeds[audio_idx])
                audio_idx += 1
            else:
                result.append(flat_embeds[idx])
        return mx.stack(result, axis=0).reshape(batch_size, seq_len, hidden_dim)

    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: list[Any] | None = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        hidden_states = self.model(input_ids, cache=cache, input_embeddings=input_embeddings)
        if self.lm_head is not None:
            return self.lm_head(hidden_states)
        return self.model.embed_tokens.as_linear(hidden_states)
