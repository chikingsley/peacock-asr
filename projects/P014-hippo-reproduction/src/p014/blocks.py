from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    """Root-mean-square LayerNorm (Zhang & Sennrich 2019), used inside Conv-LLaMA."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, inputs: Tensor) -> Tensor:
        variance = inputs.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = inputs * torch.rsqrt(variance + self.eps).to(inputs.dtype)
        return normalized * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary position encoding (Su et al. 2024 / RoFormer) for Conv-LLaMA MHSA."""

    def __init__(self, head_dim: int, max_seq_len: int = 4096, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE requires even head_dim, got {head_dim}")
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        cos_cache, sin_cache = self._build_cache(max_seq_len)
        self.register_buffer("cos_cache", cos_cache, persistent=False)
        self.register_buffer("sin_cache", sin_cache, persistent=False)

    def _build_cache(self, seq_len: int) -> tuple[Tensor, Tensor]:
        positions = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq)
        return freqs.cos(), freqs.sin()

    def get_cos_sin(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if seq_len > self.cos_cache.size(0):
            cos_cache, sin_cache = self._build_cache(seq_len)
            self.cos_cache = cos_cache.to(device=self.cos_cache.device)
            self.sin_cache = sin_cache.to(device=self.sin_cache.device)
        cos = self.cos_cache[:seq_len].to(device=device, dtype=dtype)
        sin = self.sin_cache[:seq_len].to(device=device, dtype=dtype)
        return cos, sin


def rotate_queries_and_keys(query: Tensor, key: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    cos_full = cos.repeat_interleave(2, dim=-1)
    sin_full = sin.repeat_interleave(2, dim=-1)
    cos_full = cos_full.unsqueeze(0).unsqueeze(0)
    sin_full = sin_full.unsqueeze(0).unsqueeze(0)

    def rotate_half(x: Tensor) -> Tensor:
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        stacked = torch.stack((-x2, x1), dim=-1)
        return stacked.flatten(-2)

    query_rot = query * cos_full + rotate_half(query) * sin_full
    key_rot = key * cos_full + rotate_half(key) * sin_full
    return query_rot, key_rot


class SwiGLU(nn.Module):
    """SwiGLU feed-forward (Touvron et al. 2023) used inside Conv-LLaMA MHSA branch."""

    def __init__(self, dim: int, hidden_dim: int | None = None, drop: float = 0.0) -> None:
        super().__init__()
        if hidden_dim is None:
            base_hidden = int(dim * 4 * 2 / 3)
            hidden_dim = max(8, ((base_hidden + 7) // 8) * 8)
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(drop)

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = F.silu(self.gate_proj(inputs)) * self.up_proj(inputs)
        return self.dropout(self.down_proj(hidden))


class Attention(nn.Module):
    """MHSA with optional RoPE (Conv-LLaMA attention branch, eq. implicit in §2.2)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        rope: RotaryEmbedding | None = None,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.rope = rope
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, inputs: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        batch_size, seq_len, dim = inputs.shape
        qkv = self.qkv(inputs).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        if self.rope is not None:
            cos, sin = self.rope.get_cos_sin(seq_len, inputs.device, inputs.dtype)
            query, key = rotate_queries_and_keys(query, key, cos, sin)
        attn_mask: Tensor | None = None
        if key_padding_mask is not None:
            attn_mask = key_padding_mask.view(batch_size, 1, 1, seq_len).expand(-1, self.num_heads, seq_len, -1)
        dropout = self.attn_drop if self.training else 0.0
        attended = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=dropout)
        attended = attended.transpose(1, 2).reshape(batch_size, seq_len, dim)
        return self.proj_drop(self.proj(attended))


class ConvBranch(nn.Module):
    """CNN branch of Conv-LLaMA (Figure 4): pointwise -> depthwise -> RMSNorm -> SiLU."""

    def __init__(self, dim: int, kernel_size: int = 3, drop: float = 0.0) -> None:
        super().__init__()
        self.norm_in = RMSNorm(dim)
        self.pointwise = nn.Conv1d(dim, dim, kernel_size=1)
        self.depthwise = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm_mid = RMSNorm(dim)
        self.dropout = nn.Dropout(drop)

    def forward(self, inputs: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        normalized = self.norm_in(inputs).transpose(1, 2)
        hidden = self.pointwise(normalized)
        hidden = self.depthwise(hidden)
        hidden = hidden.transpose(1, 2)
        hidden = self.norm_mid(hidden)
        hidden = F.silu(hidden)
        hidden = self.dropout(hidden)
        if key_padding_mask is not None:
            hidden = hidden * key_padding_mask.unsqueeze(-1)
        return inputs + hidden


class ConvLlamaBlock(nn.Module):
    """Conv-LLaMA block (Figure 4): MHSA+SwiGLU branch and CNN branch combined via weighted average."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        rope: RotaryEmbedding | None = None,
        mlp_hidden: int | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attention = Attention(dim, num_heads=num_heads, rope=rope, attn_drop=attn_drop, proj_drop=drop)
        self.ffn_norm = RMSNorm(dim)
        self.swiglu = SwiGLU(dim, hidden_dim=mlp_hidden, drop=drop)
        self.conv_branch = ConvBranch(dim, drop=drop)
        self.fusion = nn.Linear(dim * 2, 2)

    def forward(self, inputs: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        attn_out = inputs + self.attention(self.attn_norm(inputs), key_padding_mask=key_padding_mask)
        if key_padding_mask is not None:
            attn_out = attn_out * key_padding_mask.unsqueeze(-1)
        attn_out = attn_out + self.swiglu(self.ffn_norm(attn_out))
        if key_padding_mask is not None:
            attn_out = attn_out * key_padding_mask.unsqueeze(-1)

        conv_out = self.conv_branch(inputs, key_padding_mask=key_padding_mask)
        if key_padding_mask is not None:
            conv_out = conv_out * key_padding_mask.unsqueeze(-1)

        pooled_attn = _masked_mean(attn_out, key_padding_mask)
        pooled_conv = _masked_mean(conv_out, key_padding_mask)
        weights = torch.softmax(self.fusion(torch.cat([pooled_attn, pooled_conv], dim=-1)), dim=-1)
        w_attn = weights[:, 0].view(-1, 1, 1)
        w_conv = weights[:, 1].view(-1, 1, 1)
        merged = w_attn * attn_out + w_conv * conv_out
        if key_padding_mask is not None:
            merged = merged * key_padding_mask.unsqueeze(-1)
        return merged


@dataclass(frozen=True)
class AttentionPoolingConfig:
    dim: int
    num_heads: int = 3
    num_queries: int = 1
    kernel_size: int = 3
    drop: float = 0.0


class AttentionPooling(nn.Module):
    """Attention pooling (paper §2.2 + App. B): depthwise conv -> MHA -> average."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 3,
        num_queries: int = 1,
        kernel_size: int = 3,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.depthwise = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.queries = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=drop,
            batch_first=True,
        )

    def forward(self, inputs: Tensor, mask: Tensor | None = None) -> Tensor:
        conv_in = inputs.transpose(1, 2)
        conv_out = self.depthwise(conv_in).transpose(1, 2)
        if mask is not None:
            conv_out = conv_out * mask.unsqueeze(-1)
        batch_size = inputs.size(0)
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)
        key_padding_mask = None if mask is None else ~mask
        if key_padding_mask is not None and bool(key_padding_mask.all(dim=-1).any()):
            all_masked = key_padding_mask.all(dim=-1)
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_masked, 0] = False
        attended, _ = self.attention(
            query=queries,
            key=conv_out,
            value=conv_out,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return attended.mean(dim=1)


def _masked_mean(inputs: Tensor, mask: Tensor | None) -> Tensor:
    if mask is None:
        return inputs.mean(dim=1)
    weights = mask.to(dtype=inputs.dtype).unsqueeze(-1)
    total = (inputs * weights).sum(dim=1)
    denom = weights.sum(dim=1).clamp(min=1.0)
    return total / denom
