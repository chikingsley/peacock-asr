"""Building blocks for HierCB: BlockCNN, MLP, Attention, utilities.

Ported from ConPCO/src/models/gopt_ssl_3m_bfr_cat_utt_clap.py.

Modernizations vs reference:
- trunc_normal_: hand-rolled 80-line implementation → nn.init.trunc_normal_ (PyTorch 1.11+).
- Attention.forward: manual QKV matmul → F.scaled_dot_product_attention (PyTorch 2.0+).
  On RTX 5070 this automatically enables FlashAttention2 via CUDA kernels.
- pad_list: kept as-is (simpler than pad_sequence for this shape).
- get_sinusoid_encoding: kept as-is (functionally correct, not a hotpath).
- MultiHeadedAttention: kept (needed for word-level masking with batch-specific mask tensors;
  nn.MultiheadAttention doesn't easily handle [B, T, T] batch masks without reshaping tricks).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Utilities ─────────────────────────────────────────────────────────────────

def pad_list(xs: list[torch.Tensor], pad_value: float) -> torch.Tensor:
    """Pad a list of variable-length tensors to the same length.

    Args:
        xs: List of tensors each shape (T_i, *).
        pad_value: Fill value for padding positions.

    Returns:
        Padded tensor of shape (B, T_max, *).
    """
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)
    for i in range(n_batch):
        pad[i, : xs[i].size(0)] = xs[i]
    return pad


def get_sinusoid_encoding(n_position: int, d_hid: int) -> torch.Tensor:
    """Standard sinusoidal position encoding table, shape (1, n_position, d_hid)."""
    def get_position_angle_vec(position: int) -> list[float]:
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    table = np.array([get_position_angle_vec(pos) for pos in range(n_position)])
    table[:, 0::2] = np.sin(table[:, 0::2])
    table[:, 1::2] = np.cos(table[:, 1::2])
    return torch.FloatTensor(table).unsqueeze(0)


# ── Multi-headed attention (espnet-style, used for word-level masking) ─────────

class MultiHeadedAttention(nn.Module):
    """Multi-head attention with batch-specific attention masks.

    Used at word level where each utterance has a different phone→word grouping mask.
    The mask is 0 where positions should be ignored (filled with -inf).
    """

    def __init__(self, n_head: int, n_feat: int, dropout_rate: float) -> None:
        super().__init__()
        assert n_feat % n_head == 0
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k).transpose(1, 2)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k).transpose(1, 2)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            # mask: (B, T, T) — 0 = ignore, 1 = attend
            mask = mask.unsqueeze(1).eq(0)  # (B, 1, T, T) — True = fill with -inf
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)

        attn = self.dropout(torch.softmax(scores, dim=-1))
        if mask is not None:
            attn = attn.masked_fill(mask, 0.0)

        x = torch.matmul(attn, v)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        return self.linear_out(x)


# ── Standard self-attention (used inside BlockCNN) ────────────────────────────

class Attention(nn.Module):
    """Self-attention with F.scaled_dot_product_attention (FlashAttention2-compatible)."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop_p = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each: (B, heads, N, head_dim)
        # F.scaled_dot_product_attention selects FlashAttention2 / math / mem-efficient
        # automatically based on input shape and device capability.
        dropout_p = self.attn_drop_p if self.training else 0.0
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


# ── MLP ───────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


# ── Standard Transformer block (for pretraining, HierTFR ref [41]) ───────────

class TransformerBlock(nn.Module):
    """Standard pre-norm Transformer encoder block.

    Used during pretraining (HierTFR, Yan et al. ACL 2024). The parameter names
    (norm1, attn, norm2, mlp) match BlockCNN's attention sub-branch, so pretrained
    weights transfer via strict=False into the attention pathway of BlockCNN.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ── Convolutional Branchformer block ──────────────────────────────────────────

class BlockCNN(nn.Module):
    """One Convolutional Branchformer block.

    Two parallel branches:
      1. Self-attention branch (Transformer-style)
      2. Depthwise conv branch (Conformer-style)
    Branches are merged via learned attention-weighted average.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.size = dim
        mlp_hidden_dim = int(dim * mlp_ratio)

        # Self-attention branch
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

        # Conv branch
        self.cnn_norm1 = nn.LayerNorm(dim)
        self.cnn_norm2 = nn.LayerNorm(dim)
        self.conv1d = nn.Conv1d(dim, dim * 2, kernel_size=1)   # pointwise expansion
        self.glu_act = nn.GLU()
        self.dep_cnn1 = nn.Conv1d(dim, dim, kernel_size=3, groups=dim, padding=1)  # depthwise
        self.dep_cnn2 = nn.Conv1d(dim, dim, kernel_size=1)   # pointwise projection
        self.conv1d_2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.cnn_drop = nn.Dropout(0.2)

        # Branch merging: attention pooling → scalar gate → weighted average
        self.pooling_proj1 = nn.Linear(dim, 1)
        self.pooling_proj2 = nn.Linear(dim, 1)
        self.weight_proj1 = nn.Linear(dim, 1)
        self.weight_proj2 = nn.Linear(dim, 1)
        self.mlp2 = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── Self-attention branch ──────────────────────────────────────────────
        x_att = x + self.attn(self.norm1(x))
        x_att = x_att + self.mlp(self.norm2(x_att))

        # ── Conv branch ───────────────────────────────────────────────────────
        # (B, T, dim) → (B, dim, T) for Conv1d
        x_cnn = self.cnn_norm1(x).transpose(1, 2)
        x_cnn = self.glu_act(self.conv1d(x_cnn).transpose(1, 2))   # GLU halves dim
        x_cnn = x_cnn.transpose(1, 2)
        x_cnn = self.dep_cnn2(self.dep_cnn1(x_cnn)).transpose(1, 2)
        x_cnn = F.relu(self.cnn_norm2(x_cnn))
        x_cnn = self.cnn_drop(self.conv1d_2(x_cnn.transpose(1, 2)).transpose(1, 2))
        x_cnn = x + x_cnn

        # ── Learned branch merge ───────────────────────────────────────────────
        # Attention-pool each branch to a single vector, then compute scalar gates
        score1 = torch.softmax(self.pooling_proj1(x_att).transpose(1, 2) / self.size ** 0.5, dim=-1)
        pooled1 = torch.matmul(score1, x_att).squeeze(1)

        score2 = torch.softmax(self.pooling_proj2(x_cnn).transpose(1, 2) / self.size ** 0.5, dim=-1)
        pooled2 = torch.matmul(score2, x_cnn).squeeze(1)

        merge_w = torch.softmax(torch.cat([self.weight_proj1(pooled1), self.weight_proj2(pooled2)], dim=-1), dim=-1)
        w1, w2 = merge_w[:, 0:1].unsqueeze(-1), merge_w[:, 1:2].unsqueeze(-1)  # (B, 1, 1)

        return x + self.dropout(self.mlp2(w1 * x_att + w2 * x_cnn))


# ── Word-to-utterance feature aggregation ─────────────────────────────────────

class W2UFeatGen(nn.Module):
    """Combine 3 word-level feature streams into a single utterance-level representation."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0) -> None:
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.size = dim
        self.w1_proj = MLP(dim, mlp_hidden_dim, drop=drop)
        self.w2_proj = MLP(dim, mlp_hidden_dim, drop=drop)
        self.w3_proj = MLP(dim, mlp_hidden_dim, drop=drop)
        self.pooling_proj1 = nn.Linear(dim, 1)
        self.pooling_proj2 = nn.Linear(dim, 1)
        self.pooling_proj3 = nn.Linear(dim, 1)
        self.weight_proj1 = nn.Linear(dim, 1)
        self.weight_proj2 = nn.Linear(dim, 1)
        self.weight_proj3 = nn.Linear(dim, 1)
        self.mlp = MLP(dim, mlp_hidden_dim, drop=drop)
        self.dropout = nn.Dropout(0.2)

    def forward(self, w1: torch.Tensor, w2: torch.Tensor, w3: torch.Tensor) -> torch.Tensor:
        p1, p2, p3 = self.w1_proj(w1), self.w2_proj(w2), self.w3_proj(w3)

        def _pool(proj: nn.Linear, x: torch.Tensor) -> torch.Tensor:
            score = torch.softmax(proj(x).transpose(1, 2) / self.size ** 0.5, dim=-1)
            return torch.matmul(score, x).squeeze(1)  # (B, dim)

        pooled1 = _pool(self.pooling_proj1, p1)
        pooled2 = _pool(self.pooling_proj2, p2)
        pooled3 = _pool(self.pooling_proj3, p3)
        gates = torch.softmax(
            torch.stack([self.weight_proj1(pooled1), self.weight_proj2(pooled2), self.weight_proj3(pooled3)], dim=-1),
            dim=-1,
        )  # (B, 1, 3)
        g1, g2, g3 = gates[:, :, 0].unsqueeze(-1), gates[:, :, 1].unsqueeze(-1), gates[:, :, 2].unsqueeze(-1)
        return self.dropout(self.mlp(g1 * p1 + g2 * p2 + g3 * p3))


# ── Attention pooling ─────────────────────────────────────────────────────────

class AttentionPooling(nn.Module):
    """Soft attention pooling: collapse sequence dimension to a single vector."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0) -> None:
        super().__init__()
        self.pooling_proj = nn.Linear(dim, 1)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=drop)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        score = torch.softmax(self.pooling_proj(x).transpose(1, 2) / x.size(-1) ** 0.5, dim=-1)
        return torch.matmul(score, x).squeeze(1)  # (B, dim)
