from __future__ import annotations

import math
from typing import Any

import numpy as np

from moss_mlx_conversion.config import AudioEncoderConfig
from moss_mlx_conversion.mlx_compat import mx, nn


def _floor_div(a: mx.array, b: int) -> mx.array:
    return mx.floor(a.astype(mx.float32) / b).astype(mx.int32)


def get_feat_extract_output_lengths(input_lengths: mx.array) -> mx.array:
    input_lengths_leave = input_lengths % 100
    feat_lengths = _floor_div(input_lengths_leave - 1, 2) + 1
    return _floor_div(_floor_div(feat_lengths - 1, 2) + 1 - 1, 2) + 1 + (
        input_lengths // 100
    ) * 13


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, length: int, channels: int, max_timescale: float = 10000.0) -> None:
        super().__init__()
        if channels % 2 != 0:
            raise ValueError("SinusoidalPositionEmbedding requires an even channel count")

        log_timescale_increment = math.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = mx.exp(
            -log_timescale_increment * mx.arange(channels // 2, dtype=mx.float32)
        )
        positions = mx.arange(length, dtype=mx.float32)[:, None]
        scaled_time = positions * inv_timescales[None, :]
        self._positional_embedding = mx.concatenate(
            [mx.sin(scaled_time), mx.cos(scaled_time)],
            axis=1,
        )

    def __call__(self, seqlen: int) -> mx.array:
        return self._positional_embedding[:seqlen, :]


class AudioAttention(nn.Module):
    def __init__(self, config: AudioEncoderConfig) -> None:
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def __call__(self, hidden_states: mx.array, mask: mx.array | None = None) -> mx.array:
        batch_size, seq_len, _ = hidden_states.shape
        query_states = self.q_proj(hidden_states) * self.scaling
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        key_states = key_states.reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        value_states = value_states.reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)

        output = mx.fast.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            scale=1.0,
            mask=mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)
        return self.out_proj(output)


class AudioEncoderLayer(nn.Module):
    def __init__(self, config: AudioEncoderConfig) -> None:
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = AudioAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def __call__(self, hidden_states: mx.array, mask: mx.array | None = None) -> mx.array:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, mask=mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = nn.gelu(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        return residual + hidden_states


class Qwen3OmniMoeAudioEncoder(nn.Module):
    def __init__(self, config: AudioEncoderConfig) -> None:
        super().__init__()
        self.config = config
        embed_dim = config.d_model
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.n_window = config.n_window
        self.n_window_infer = config.n_window_infer
        self.conv_chunksize = config.conv_chunksize

        self.conv2d1 = nn.Conv2d(
            1,
            config.downsample_hidden_size,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.conv2d2 = nn.Conv2d(
            config.downsample_hidden_size,
            config.downsample_hidden_size,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.conv2d3 = nn.Conv2d(
            config.downsample_hidden_size,
            config.downsample_hidden_size,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        freq_after_conv = ((((config.num_mel_bins + 1) // 2) + 1) // 2 + 1) // 2
        self.conv_out = nn.Linear(
            config.downsample_hidden_size * freq_after_conv,
            embed_dim,
            bias=False,
        )
        self.positional_embedding = SinusoidalPositionEmbedding(
            self.max_source_positions,
            embed_dim,
        )
        self.layers = [AudioEncoderLayer(config) for _ in range(config.encoder_layers)]
        self.ln_post = nn.LayerNorm(embed_dim)
        self.proj1 = nn.Linear(embed_dim, embed_dim)
        self.proj2 = nn.Linear(embed_dim, config.output_dim)

    def _create_block_attention_mask(
        self,
        seq_len: int,
        cu_seqlens: list[int],
        dtype: Any,
    ) -> mx.array:
        mask = mx.full((seq_len, seq_len), -1e9, dtype=dtype)
        for idx in range(len(cu_seqlens) - 1):
            start = cu_seqlens[idx]
            end = cu_seqlens[idx + 1]
            mask[start:end, start:end] = 0.0
        return mask

    def _chunk_features(
        self,
        input_features: mx.array,
        feature_lens_np: np.ndarray[Any, Any],
        chunk_num: np.ndarray[Any, Any],
        chunk_size: int,
    ) -> tuple[list[mx.array], list[int]]:
        chunks = []
        chunk_lengths: list[int] = []
        for item_idx, feat_len_raw in enumerate(feature_lens_np):
            feat_len = int(feat_len_raw)
            pos = 0
            for chunk_idx in range(int(chunk_num[item_idx])):
                if chunk_idx == int(chunk_num[item_idx]) - 1:
                    remainder = feat_len % chunk_size
                    clen = chunk_size if remainder == 0 else remainder
                else:
                    clen = chunk_size
                chunk_lengths.append(clen)
                chunks.append(input_features[item_idx][:, pos : pos + clen])
                pos += clen
        return chunks, chunk_lengths

    def _pad_chunks(self, chunks: list[mx.array], chunk_lengths: list[int]) -> mx.array:
        max_chunk_len = int(max(chunk_lengths))
        padded_chunks = []
        for chunk, clen in zip(chunks, chunk_lengths, strict=True):
            padded_chunk = chunk
            if clen < max_chunk_len:
                padded_chunk = mx.pad(chunk, [(0, 0), (0, max_chunk_len - clen)])
            padded_chunks.append(padded_chunk)
        return mx.stack(padded_chunks, axis=0)

    def _encode_convolutional_chunks(self, padded_feature: mx.array) -> mx.array:
        x = padded_feature[:, :, :, None]
        x = nn.gelu(self.conv2d1(x))
        x = nn.gelu(self.conv2d2(x))
        x = nn.gelu(self.conv2d3(x))

        batch, freq, frames, channels = x.shape
        x = x.transpose(0, 2, 3, 1).reshape(batch, frames, channels * freq)
        x = self.conv_out(x)
        return x + self.positional_embedding(x.shape[1])[None, :, :]

    def _pack_valid_frames(
        self,
        x: mx.array,
        feature_lens_after_cnn_np: np.ndarray[Any, Any],
    ) -> mx.array:
        hidden_list = []
        for idx in range(x.shape[0]):
            valid_len = int(feature_lens_after_cnn_np[idx])
            hidden_list.append(x[idx, :valid_len])
        return mx.concatenate(hidden_list, axis=0)

    def _build_cu_seqlens(
        self,
        aftercnn_lens_np: np.ndarray[Any, Any],
        window_aftercnn: int,
    ) -> list[int]:
        cu_chunk_lens = [0]
        for cnn_len_raw in aftercnn_lens_np:
            cnn_len = int(cnn_len_raw)
            cu_chunk_lens.extend([window_aftercnn] * (cnn_len // window_aftercnn))
            remainder = cnn_len % window_aftercnn
            if remainder != 0:
                cu_chunk_lens.append(remainder)
        return np.cumsum(cu_chunk_lens).tolist()

    def _run_transformer(self, hidden_states: mx.array, attention_mask: mx.array) -> mx.array:
        hidden_states = hidden_states[None, :, :]
        for layer in self.layers:
            hidden_states = layer(hidden_states, mask=attention_mask)
        return hidden_states[0]

    def __call__(self, input_features: mx.array, feature_lens: mx.array) -> mx.array:
        feature_lens_np = np.array(feature_lens)
        aftercnn_lens = get_feat_extract_output_lengths(feature_lens)
        chunk_size = self.n_window * 2
        chunk_num = np.ceil(feature_lens_np / chunk_size).astype(np.int32)

        chunks, chunk_lengths = self._chunk_features(
            input_features,
            feature_lens_np,
            chunk_num,
            chunk_size,
        )
        padded_feature = self._pad_chunks(chunks, chunk_lengths)
        chunk_lens = mx.array(chunk_lengths)
        feature_lens_after_cnn = get_feat_extract_output_lengths(chunk_lens)
        feature_lens_after_cnn_np = np.array(feature_lens_after_cnn)
        max_len_after_cnn = int(feature_lens_after_cnn_np.max())

        x = self._encode_convolutional_chunks(padded_feature)
        hidden_states = self._pack_valid_frames(x, feature_lens_after_cnn_np)

        aftercnn_lens_np = np.array(aftercnn_lens)
        window_aftercnn = max_len_after_cnn * (self.n_window_infer // (self.n_window * 2))
        cu_seqlens = self._build_cu_seqlens(aftercnn_lens_np, window_aftercnn)
        seq_len = hidden_states.shape[0]
        attention_mask = self._create_block_attention_mask(
            seq_len,
            cu_seqlens,
            hidden_states.dtype,
        )[None, None, :, :]

        hidden_states = self._run_transformer(hidden_states, attention_mask)
        hidden_states = self.ln_post(hidden_states)
        hidden_states = nn.gelu(self.proj1(hidden_states))
        return self.proj2(hidden_states)
