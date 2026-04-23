from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from p014.blocks import AttentionPooling, ConvLlamaBlock, RotaryEmbedding, _masked_mean


@dataclass(frozen=True)
class ReadAloudPredictions:
    phone_scores: Tensor
    word_scores: Tensor
    utterance_scores: Tensor
    utterance_feature: Tensor


class Regressor(nn.Module):
    """Two-layer FFN regressor used for phone/word/utterance heads (paper App. B)."""

    def __init__(self, dim: int, hidden_dim: int | None = None, drop: float = 0.1) -> None:
        super().__init__()
        inner = hidden_dim if hidden_dim is not None else dim
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, inner)
        self.dropout = nn.Dropout(drop)
        self.fc2 = nn.Linear(inner, 1)

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = F.silu(self.fc1(self.norm(inputs)))
        hidden = self.dropout(hidden)
        return self.fc2(hidden)


class ConvLlamaStack(nn.Module):
    """Sequence of Conv-LLaMA blocks sharing one RoPE cache (paper eq. 5, 8, 11)."""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        drop: float,
    ) -> None:
        super().__init__()
        head_dim = dim // num_heads
        self.rope = RotaryEmbedding(head_dim=head_dim)
        self.blocks = nn.ModuleList(
            [ConvLlamaBlock(dim=dim, num_heads=num_heads, rope=self.rope, drop=drop) for _ in range(depth)]
        )

    def forward(self, inputs: Tensor, mask: Tensor | None) -> Tensor:
        hidden = inputs
        for block in self.blocks:
            hidden = block(hidden, key_padding_mask=mask)
        return hidden


class WordGroupedAttentionPool(nn.Module):
    """Per-word attention pool over phone sequence (paper eq. 6): AttPool applied within phone groups of each word."""

    def __init__(self, dim: int, num_heads: int = 3, drop: float = 0.0) -> None:
        super().__init__()
        self.pool = AttentionPooling(dim=dim, num_heads=num_heads, drop=drop)

    def forward(
        self,
        phone_features: Tensor,
        phone_to_word: Tensor,
        phone_mask: Tensor,
        word_mask: Tensor,
    ) -> Tensor:
        batch_size, num_phones, dim = phone_features.shape
        num_words = word_mask.size(1)
        word_indices = torch.arange(num_words, device=phone_to_word.device).view(1, num_words, 1)
        group_mask = phone_to_word.unsqueeze(1).eq(word_indices) & phone_mask.unsqueeze(1)

        flat_inputs = phone_features.unsqueeze(1).expand(batch_size, num_words, num_phones, dim)
        flat_inputs = flat_inputs.reshape(batch_size * num_words, num_phones, dim)
        flat_mask = group_mask.reshape(batch_size * num_words, num_phones)

        pooled = self.pool(flat_inputs, mask=flat_mask)
        pooled = pooled.view(batch_size, num_words, dim)
        return pooled * word_mask.unsqueeze(-1)


class WordAspectHead(nn.Module):
    """Depthwise-conv branch producing one aspect-specific word representation (paper eq. 9)."""

    def __init__(self, dim: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim
        )

    def forward(self, inputs: Tensor, mask: Tensor) -> Tensor:
        hidden = self.depthwise(inputs.transpose(1, 2)).transpose(1, 2)
        return hidden * mask.unsqueeze(-1)


class HiPPOReadAloud(nn.Module):
    """HiPPO read-aloud assessment model (Yan et al. 2025, §2.2, eq. 3-12)."""

    def __init__(
        self,
        num_phone_tokens: int,
        hidden_dim: int = 24,
        num_heads: int = 1,
        phone_depth: int = 3,
        word_depth: int = 2,
        utterance_depth: int = 1,
        gop_dim: int = 41,
        ssl_utterance_dim: int = 3072,
        word_embedding_dim: int = 768,
        attn_pool_heads: int = 3,
        drop: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim % attn_pool_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by attn_pool_heads={attn_pool_heads}"
            )

        self.phone_input_proj = nn.Linear(gop_dim, hidden_dim)
        self.phone_embedding = nn.Embedding(num_phone_tokens + 1, hidden_dim, padding_idx=0)
        self.phone_encoder = ConvLlamaStack(
            dim=hidden_dim, depth=phone_depth, num_heads=num_heads, drop=drop
        )
        self.phone_regressor = Regressor(hidden_dim, drop=drop)

        self.word_raw_pool = WordGroupedAttentionPool(
            dim=hidden_dim, num_heads=attn_pool_heads, drop=drop
        )
        self.word_ctx_pool = WordGroupedAttentionPool(
            dim=hidden_dim, num_heads=attn_pool_heads, drop=drop
        )
        self.word_input_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.word_embedding_proj = nn.Linear(word_embedding_dim, hidden_dim)
        self.word_encoder = ConvLlamaStack(
            dim=hidden_dim, depth=word_depth, num_heads=num_heads, drop=drop
        )

        self.word_acc_branch = WordAspectHead(hidden_dim)
        self.word_stress_branch = WordAspectHead(hidden_dim)
        self.word_total_branch = WordAspectHead(hidden_dim)
        self.word_acc_regressor = Regressor(hidden_dim, drop=drop)
        self.word_stress_regressor = Regressor(hidden_dim, drop=drop)
        self.word_total_regressor = Regressor(hidden_dim, drop=drop)

        self.word_merge_gate = nn.Linear(hidden_dim, 3)

        self.utt_phone_branch = WordAspectHead(hidden_dim)
        self.utt_context_branch = WordAspectHead(hidden_dim)
        self.utt_word_branch = WordAspectHead(hidden_dim)
        self.utt_input_proj = nn.Linear(hidden_dim * 3, hidden_dim)
        self.utt_encoder = ConvLlamaStack(
            dim=hidden_dim, depth=utterance_depth, num_heads=num_heads, drop=drop
        )

        self.ssl_residual_proj = nn.Linear(ssl_utterance_dim, hidden_dim)
        self.utt_pools = nn.ModuleList(
            [AttentionPooling(dim=hidden_dim, num_heads=attn_pool_heads, drop=drop) for _ in range(5)]
        )
        self.utt_regressors = nn.ModuleList([Regressor(hidden_dim, drop=drop) for _ in range(5)])

    def forward(
        self,
        gop_features: Tensor,
        ssl_utterance: Tensor,
        phone_ids: Tensor,
        phone_to_word: Tensor,
        word_embeddings: Tensor,
        phone_mask: Tensor,
        word_mask: Tensor,
    ) -> ReadAloudPredictions:
        phone_mask_f = phone_mask.unsqueeze(-1)
        word_mask_f = word_mask.unsqueeze(-1)

        phone_input = self.phone_input_proj(gop_features) + self.phone_embedding(phone_ids)
        phone_input = phone_input * phone_mask_f
        phone_hidden = self.phone_encoder(phone_input, mask=phone_mask) * phone_mask_f
        phone_scores = self.phone_regressor(phone_hidden).squeeze(-1)

        # Utterance-level feature Z: mean-pool phone-encoder outputs over phones (CONO input per §2.3).
        utterance_feature = _masked_mean(phone_hidden, phone_mask)

        pooled_raw = self.word_raw_pool(phone_input, phone_to_word, phone_mask, word_mask)
        pooled_ctx = self.word_ctx_pool(phone_hidden, phone_to_word, phone_mask, word_mask)
        word_input = self.word_input_proj(torch.cat([pooled_raw, pooled_ctx], dim=-1))
        word_input = word_input + self.word_embedding_proj(word_embeddings)
        word_input = word_input * word_mask_f
        word_hidden = self.word_encoder(word_input, mask=word_mask) * word_mask_f

        word_acc = self.word_acc_branch(word_hidden, word_mask)
        word_stress = self.word_stress_branch(word_hidden, word_mask)
        word_total = self.word_total_branch(word_hidden, word_mask)
        word_scores = torch.cat(
            [
                self.word_acc_regressor(word_acc),
                self.word_stress_regressor(word_stress),
                self.word_total_regressor(word_total),
            ],
            dim=-1,
        )

        merged_word = self._merge_word_aspects(word_acc, word_stress, word_total, mask=word_mask)
        per_phone_word = gather_word_features(merged_word, phone_to_word, phone_mask)

        utt_sequence = torch.cat(
            [
                self.utt_phone_branch(phone_input, phone_mask),
                self.utt_context_branch(phone_hidden, phone_mask),
                self.utt_word_branch(per_phone_word, phone_mask),
            ],
            dim=-1,
        )
        utt_hidden = self.utt_input_proj(utt_sequence) * phone_mask_f
        utt_hidden = self.utt_encoder(utt_hidden, mask=phone_mask) * phone_mask_f

        ssl_residual = self.ssl_residual_proj(ssl_utterance)
        utterance_scores_list: list[Tensor] = []
        for pool, regressor in zip(self.utt_pools, self.utt_regressors, strict=True):
            pooled = pool(utt_hidden, mask=phone_mask) + ssl_residual
            utterance_scores_list.append(regressor(pooled))
        utterance_scores = torch.cat(utterance_scores_list, dim=-1)

        return ReadAloudPredictions(
            phone_scores=phone_scores,
            word_scores=word_scores,
            utterance_scores=utterance_scores,
            utterance_feature=utterance_feature,
        )

    def _merge_word_aspects(
        self, acc: Tensor, stress: Tensor, total: Tensor, mask: Tensor
    ) -> Tensor:
        stacked = torch.stack([acc, stress, total], dim=2)
        pooled_ctx = _masked_mean(stacked.mean(dim=2), mask)
        gate_logits = self.word_merge_gate(pooled_ctx)
        weights = torch.softmax(gate_logits, dim=-1).view(-1, 1, 3, 1)
        merged = (stacked * weights).sum(dim=2)
        return merged * mask.unsqueeze(-1)


def gather_word_features(word_features: Tensor, phone_to_word: Tensor, phone_mask: Tensor) -> Tensor:
    batch_size, phone_steps = phone_to_word.shape
    safe_indices = phone_to_word.clamp(min=0)
    gathered = torch.gather(
        word_features,
        dim=1,
        index=safe_indices.unsqueeze(-1).expand(batch_size, phone_steps, word_features.size(-1)),
    )
    return gathered * phone_mask.unsqueeze(-1)
