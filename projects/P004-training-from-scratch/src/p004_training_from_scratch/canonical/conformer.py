"""Compact Conformer encoder used by the stable canonical lane.

References
- Gulati et al. (2020) Conformer: https://arxiv.org/abs/2005.08100
- PyTorch MultiheadAttention:
  https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
"""

from __future__ import annotations

import math
from typing import Any


def build_conformer_canonical_ctc(
    *,
    torch: Any,
    input_dim: int,
    hidden_dim: int,
    vocab_size: int,
    encoder_layers: int,
    attention_heads: int,
    conv_kernel_size: int,
    dropout: float,
    attention_backend: str,
) -> Any:
    if attention_backend != "mha":
        msg = (
            "real conformer currently only supports attention_backend='mha' "
            "in the stable canonical lane"
        )
        raise ValueError(msg)

    class SinusoidalPositionalEncoding(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dropout = torch.nn.Dropout(dropout)

        def forward(self, features: Any) -> Any:
            seq_len = features.size(1)
            position = torch.arange(
                seq_len,
                device=features.device,
                dtype=torch.float32,
            ).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(
                    0,
                    hidden_dim,
                    2,
                    device=features.device,
                    dtype=torch.float32,
                )
                * (-math.log(10000.0) / hidden_dim)
            )
            pos_encoding = torch.zeros(
                (seq_len, hidden_dim),
                device=features.device,
                dtype=features.dtype,
            )
            pos_encoding[:, 0::2] = torch.sin(position * div_term).to(features.dtype)
            pos_encoding[:, 1::2] = torch.cos(position * div_term).to(features.dtype)
            return self.dropout(features + pos_encoding.unsqueeze(0))

    class PositionwiseFeedForward(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            ffn_dim = hidden_dim * 4
            self.linear1 = torch.nn.Linear(hidden_dim, ffn_dim)
            self.dropout1 = torch.nn.Dropout(dropout)
            self.linear2 = torch.nn.Linear(ffn_dim, hidden_dim)
            self.dropout2 = torch.nn.Dropout(dropout)

        def forward(self, features: Any) -> Any:
            features = self.linear1(features)
            features = torch.nn.functional.silu(features)
            features = self.dropout1(features)
            features = self.linear2(features)
            return self.dropout2(features)

    class ConformerConvModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.pointwise_in = torch.nn.Conv1d(hidden_dim, hidden_dim * 2, 1)
            self.depthwise = torch.nn.Conv1d(
                hidden_dim,
                hidden_dim,
                conv_kernel_size,
                padding=conv_kernel_size // 2,
                groups=hidden_dim,
            )
            self.batch_norm = torch.nn.BatchNorm1d(hidden_dim)
            self.pointwise_out = torch.nn.Conv1d(hidden_dim, hidden_dim, 1)
            self.dropout = torch.nn.Dropout(dropout)

        def forward(
            self,
            features: Any,
            *,
            key_padding_mask: Any | None,
        ) -> Any:
            if key_padding_mask is not None:
                features = features.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
            hidden = self.pointwise_in(features.transpose(1, 2))
            hidden = torch.nn.functional.glu(hidden, dim=1)
            hidden = self.depthwise(hidden)
            hidden = self.batch_norm(hidden)
            hidden = torch.nn.functional.silu(hidden)
            hidden = self.pointwise_out(hidden).transpose(1, 2)
            hidden = self.dropout(hidden)
            if key_padding_mask is not None:
                hidden = hidden.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
            return hidden

    class ConformerBlock(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.ffn1_norm = torch.nn.LayerNorm(hidden_dim)
            self.ffn1 = PositionwiseFeedForward()
            self.attn_norm = torch.nn.LayerNorm(hidden_dim)
            self.attn = torch.nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=attention_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.attn_dropout = torch.nn.Dropout(dropout)
            self.conv_norm = torch.nn.LayerNorm(hidden_dim)
            self.conv = ConformerConvModule()
            self.ffn2_norm = torch.nn.LayerNorm(hidden_dim)
            self.ffn2 = PositionwiseFeedForward()
            self.final_norm = torch.nn.LayerNorm(hidden_dim)

        def forward(
            self,
            features: Any,
            *,
            key_padding_mask: Any | None,
        ) -> Any:
            hidden = features + 0.5 * self.ffn1(self.ffn1_norm(features))
            attn_input = self.attn_norm(hidden)
            attn_output, _ = self.attn(
                attn_input,
                attn_input,
                attn_input,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            hidden = hidden + self.attn_dropout(attn_output)
            hidden = hidden + self.conv(
                self.conv_norm(hidden),
                key_padding_mask=key_padding_mask,
            )
            hidden = hidden + 0.5 * self.ffn2(self.ffn2_norm(hidden))
            hidden = self.final_norm(hidden)
            if key_padding_mask is not None:
                hidden = hidden.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
            return hidden

    class ConformerCanonicalCtc(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_proj = torch.nn.Linear(input_dim, hidden_dim)
            self.input_dropout = torch.nn.Dropout(dropout)
            self.positional_encoding = SinusoidalPositionalEncoding()
            self.blocks = torch.nn.ModuleList(
                ConformerBlock() for _ in range(encoder_layers)
            )
            self.output_norm = torch.nn.LayerNorm(hidden_dim)
            self.output = torch.nn.Linear(hidden_dim, vocab_size)

        def forward(
            self,
            features: Any,
            input_lengths: Any | None = None,
        ) -> Any:
            key_padding_mask = _build_key_padding_mask(
                torch=torch,
                input_lengths=input_lengths,
                max_length=features.size(1),
            )
            hidden = self.input_proj(features)
            hidden = self.input_dropout(hidden)
            hidden = self.positional_encoding(hidden)
            if key_padding_mask is not None:
                hidden = hidden.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
            for block in self.blocks:
                hidden = block(hidden, key_padding_mask=key_padding_mask)
            logits = self.output(self.output_norm(hidden))
            if key_padding_mask is not None:
                logits = logits.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
            return logits

    return ConformerCanonicalCtc()


def _build_key_padding_mask(
    *,
    torch: Any,
    input_lengths: Any | None,
    max_length: int,
) -> Any | None:
    if input_lengths is None:
        return None
    positions = torch.arange(max_length, device=input_lengths.device)
    return positions.unsqueeze(0) >= input_lengths.unsqueeze(1)


__all__ = ["build_conformer_canonical_ctc"]
