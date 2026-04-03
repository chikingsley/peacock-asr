from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
from torch import Tensor

from .modules import BiMamba


class AttentionPooling(nn.Module):
    def __init__(self, in_dim: int, temperature: float = 1.0) -> None:
        super().__init__()
        self.attention = nn.Linear(in_dim, 1, bias=False)
        self.temperature = temperature

    def forward(self, x: Tensor, score: Tensor, mask: Tensor) -> Tensor:
        weights = self.attention(score).float() / self.temperature
        weights = weights.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        weights = torch.softmax(weights, dim=1)
        return torch.sum(weights * x, dim=1)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_features: int,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class SequenceBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 4,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.model = BiMamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.model(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class HMamba(nn.Module):
    def __init__(
        self,
        embed_dim: int = 24,
        gop_dim: int | None = None,
        ssl_dim: list[int] | int | None = None,
        raw_dim: int | None = None,
        kernel_size: int = 3,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 4,
        drop: float = 0.0,
        feat_drop: float = 0.0,
        max_len: int = 50,
        vocab_size: int = 81,
        use_bies: bool = False,
        use_cano: bool = True,
        use_pos: bool = True,
        use_conv: bool = False,
    ) -> None:
        super().__init__()
        self.gop_dim = gop_dim
        self.ssl_dim = ssl_dim
        self.raw_dim = raw_dim
        self.embed_dim = embed_dim
        self.use_bies = use_bies
        self.use_cano = use_cano
        self.use_pos = use_pos
        self.use_conv = use_conv
        self.max_len = max_len
        self.vocab_size = vocab_size

        self.phn_block_1 = SequenceBlock(
            dim=embed_dim,
            drop=drop,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.phn_block_2 = SequenceBlock(
            dim=embed_dim,
            drop=drop,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.phn_block_3 = SequenceBlock(
            dim=embed_dim,
            drop=drop,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.wrd_block_4 = SequenceBlock(
            dim=embed_dim,
            drop=drop,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.utt_block_5 = SequenceBlock(
            dim=embed_dim,
            drop=drop,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.resolved_mamba_backend = self.phn_block_1.model.backend_name

        frontend_dim = gop_dim or 0
        if isinstance(ssl_dim, Sequence):
            frontend_dim += sum(ssl_dim)
        elif ssl_dim is not None:
            frontend_dim += ssl_dim
        if raw_dim is not None:
            frontend_dim += raw_dim
        self.frontend_dim = frontend_dim

        self.feat_drop = nn.Dropout(feat_drop)
        self.in_proj: nn.Module
        if self.frontend_dim == self.gop_dim:
            self.in_proj = nn.Identity()
        else:
            self.in_proj = nn.Linear(self.frontend_dim, embed_dim)

        if self.use_pos:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.max_len, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        if self.use_cano:
            self.phn_embed = nn.Embedding(self.vocab_size + 1, embed_dim, padding_idx=0)
            self.canophn_proj = nn.Linear(embed_dim, embed_dim)

        if self.use_bies:
            self.bies_embed = nn.Embedding(7, embed_dim, padding_idx=0)
            self.bies_proj = nn.Linear(embed_dim, embed_dim)

        self.phn_mlp_recog = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, self.vocab_size),
            nn.Dropout(0.1),
        )
        self.phn_mlp_score = nn.Sequential(nn.LayerNorm(embed_dim), nn.Dropout(0.0), nn.Linear(embed_dim, 1))

        if self.use_conv:
            self.wrd_conv = nn.Sequential(
                nn.Conv1d(
                    embed_dim,
                    2 * embed_dim,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=(kernel_size - 1) // 2,
                    groups=embed_dim,
                ),
                nn.Conv1d(2 * embed_dim, embed_dim, kernel_size=1, stride=1, padding=0),
            )

        self.wrd_mlp_score_1 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Dropout(0.0), nn.Linear(embed_dim, 1))
        self.wrd_mlp_score_2 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Dropout(0.0), nn.Linear(embed_dim, 1))
        self.wrd_mlp_score_3 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Dropout(0.0), nn.Linear(embed_dim, 1))

        self.utt_pool = AttentionPooling(4)
        self.utt_mlp_score_1 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Dropout(0.0), nn.Linear(embed_dim, 1))
        self.utt_mlp_score_2 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Dropout(0.0), nn.Linear(embed_dim, 1))
        self.utt_mlp_score_3 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Dropout(0.0), nn.Linear(embed_dim, 1))
        self.utt_mlp_score_4 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Dropout(0.0), nn.Linear(embed_dim, 1))
        self.utt_mlp_score_5 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Dropout(0.0), nn.Linear(embed_dim, 1))

    def forward(
        self,
        x: Tensor,
        x2: list[Tensor] | tuple[Tensor, ...],
        x3: Tensor | None,
        canophn: Tensor,
        bies: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        if isinstance(x2, (list, tuple)):
            for index, ssl in enumerate(x2):
                ssl = self.feat_drop(ssl)
                if self.gop_dim is None and index == 0:
                    x = ssl
                    continue
                x = torch.cat((x, ssl), dim=2)

        if x3 is not None:
            dur = x3[:, :, 0:1]
            eng = x3[:, :, 1:8]
            x = torch.cat((x, dur, eng), dim=2)

        x = self.in_proj(x)

        if self.use_cano:
            canophn_embed = self.canophn_proj(self.phn_embed(canophn.long() + 1).float())
            x = x + canophn_embed

        if self.use_pos:
            x = x + self.pos_embed[:, : x.shape[1], :]

        if self.use_bies:
            if bies is None:
                raise ValueError("BIES labels are required when use_bies=True.")
            bies_embed = self.bies_proj(self.bies_embed(bies.long() + 1).float())
            x = x + bies_embed

        x = self.phn_block_1(x)
        x = self.phn_block_2(x)
        x = self.phn_block_3(x)

        logits = self.phn_mlp_recog(x)
        p = self.phn_mlp_score(x)

        x = self.wrd_block_4(x)
        word_conv = self.wrd_conv(x.transpose(1, 2)).transpose(1, 2) if self.use_conv else x
        w1 = self.wrd_mlp_score_1(word_conv)
        w2 = self.wrd_mlp_score_2(word_conv)
        w3 = self.wrd_mlp_score_3(word_conv)

        word_conv = self.utt_block_5(word_conv)
        if mask is None:
            raise ValueError("A padding mask is required for utterance pooling.")
        scores = torch.cat((p, w1, w2, w3), dim=-1)
        pooled = self.utt_pool(word_conv, scores, mask)

        u1 = self.utt_mlp_score_1(pooled)
        u2 = self.utt_mlp_score_2(pooled)
        u3 = self.utt_mlp_score_3(pooled)
        u4 = self.utt_mlp_score_4(pooled)
        u5 = self.utt_mlp_score_5(pooled)
        return u1, u2, u3, u4, u5, p, w1, w2, w3, logits
