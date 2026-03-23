"""HierCB — Hierarchical Convolutional Branchformer for pronunciation assessment.

Ported from ConPCO/src/models/gopt_ssl_3m_bfr_cat_utt_clap.py (HierCB class, lines 411-607).

Adds MDD heads (binary correct/incorrect per phone) for the full MuFFIN configuration.
MDD heads are parallel to mlp_head_phn and are gated by settings.use_mdd.

Modernizations vs reference:
- trunc_normal_: nn.init.trunc_normal_ instead of hand-rolled version.
- make_word_pos_mask: nested loop → vectorized with broadcasting.
- nn.DataParallel removed (single GPU; wrap externally if needed).
- one_hot indices: reference uses phn+1 (0 → pad class); kept to match exactly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from p010.models.blocks import (
    MLP,
    AttentionPooling,
    BlockCNN,
    MultiHeadedAttention,
    W2UFeatGen,
)

# SpeechOcean762 vocabulary sizes.
# Phone classes: 42 = 39 CMU phones + pad(0) + shift(+1) + mask(41) for pretraining.
# Extra classes (40-41) are inactive during fine-tuning but their phn_proj weights
# must exist for pretraining weight transfer (HierTFR ref [41]).
NUM_PHN_CLASSES: int = 42
NUM_WORD_CLASSES: int = 2607  # 2605 word IDs + 2 for pad/shift
NUM_DIAG_CLASSES: int = 39   # CMU phone set for diagnosis prediction (MuFFIN §III.B Eq.5)
PHN_MASK_TOKEN: int = 41     # mask token index for pretraining (after +1 shift)
WORD_MASK_TOKEN: int = 2606  # mask token index for pretraining (after +1 shift)


class HierCB(nn.Module):
    """Hierarchical Convolutional Branchformer.

    Architecture (embed_dim=24 default):
      Input: [B, 50, 92 + 3072] = GOP(84) + energy(7) + dur(1) + SSL(3072)
      Phone  : 3 × BlockCNN → phone score (1) + optional MDD logit (1)
      Word   : 2 × BlockCNN → word scores (3): accuracy, stress, total
      Utt    : 1 × BlockCNN → utt scores (5): accuracy, completeness, fluency, prosodic, total

    Returns (u1..u5, p, w1..w3, phn_audio_feats, phn_text_feats)
    plus mdd_logit if use_mdd=True.
    """

    def __init__(
        self,
        embed_dim: int = 24,
        num_heads: int = 1,
        p_depth: int = 3,
        w_depth: int = 2,
        u_depth: int = 1,
        ssl_drop: float = 0.2,
        input_dim: int = 92,     # 84 GOP + 7 energy + 1 dur
        ssl_dim: int = 3072,     # 3 × 1024 (wav2vec2 + HuBERT + WavLM)
        use_mdd: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.use_mdd = use_mdd

        # Phone / word / utterance transformer stacks
        self.phn_blocks = nn.ModuleList([BlockCNN(dim=embed_dim, num_heads=num_heads) for _ in range(p_depth)])
        self.word_blocks = nn.ModuleList([BlockCNN(dim=embed_dim, num_heads=num_heads) for _ in range(w_depth)])
        self.utt_blocks = nn.ModuleList([BlockCNN(dim=embed_dim, num_heads=num_heads) for _ in range(u_depth)])

        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, 50, embed_dim))
        self.word_pos_embed = nn.Embedding(50, embed_dim, padding_idx=0)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Input projections — same weights applied independently at phone/word/utt level
        total_in = input_dim + ssl_dim
        self.p_in_proj = nn.Linear(total_in, embed_dim)
        self.w_in_proj = nn.Linear(total_in, embed_dim)
        self.u_in_proj = nn.Linear(total_in, embed_dim)
        self.ssl_drop = nn.Dropout(ssl_drop)

        # Phone-level outputs
        self.mlp_head_phn = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        if use_mdd:
            # Error detector: binary per-phone (MuFFIN Eq.4)
            self.mlp_head_mdd = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
            # Diagnosis predictor: which phoneme was actually spoken (MuFFIN Eq.5)
            self.mlp_head_diag = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, NUM_DIAG_CLASSES))

        # Canonical phone and word embeddings
        self.phn_proj = nn.Linear(NUM_PHN_CLASSES, embed_dim)
        self.word_proj = nn.Linear(NUM_WORD_CLASSES, embed_dim)

        # Word-level components
        self.word_input_att = MultiHeadedAttention(num_heads, embed_dim, 0.1)
        self.word_input_att1 = MultiHeadedAttention(num_heads, embed_dim, 0.1)
        self.w_in_cat_proj = MLP(embed_dim * 2, embed_dim, embed_dim, drop=0.1)
        self.w_proj_ln1 = MLP(embed_dim, embed_dim, drop=0.1)
        self.w_proj_ln2 = MLP(embed_dim, embed_dim, drop=0.1)
        self.w_proj_ln3 = MLP(embed_dim, embed_dim, drop=0.1)
        self.w_proj_cnn1 = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim * 2, kernel_size=3, groups=embed_dim, padding=1),
            nn.Conv1d(embed_dim * 2, embed_dim, kernel_size=1),
        )
        self.w_proj_cnn2 = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim * 2, kernel_size=3, groups=embed_dim, padding=1),
            nn.Conv1d(embed_dim * 2, embed_dim, kernel_size=1),
        )
        self.mlp_head_word1 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.mlp_head_word2 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.mlp_head_word3 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))

        # Utterance-level components
        self.utt_feat_ext = W2UFeatGen(embed_dim)
        self.u_in_cat_proj = MLP(embed_dim * 3, embed_dim, embed_dim, drop=0.1)
        # u_in_proj1/2/3 removed: defined in reference (lines 499-503) but never called
        # in forward — vestigial from an earlier draft of the utterance path.
        self.u_proj_cnn1 = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim * 2, kernel_size=3, groups=embed_dim, padding=1),
            nn.Conv1d(embed_dim * 2, embed_dim, kernel_size=1),
        )
        self.u_proj_cnn2 = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim * 2, kernel_size=3, groups=embed_dim, padding=1),
            nn.Conv1d(embed_dim * 2, embed_dim, kernel_size=1),
        )
        self.u1_att_pooling = AttentionPooling(embed_dim)
        self.u2_att_pooling = AttentionPooling(embed_dim)
        self.u3_att_pooling = AttentionPooling(embed_dim)
        self.u4_att_pooling = AttentionPooling(embed_dim)
        self.u5_att_pooling = AttentionPooling(embed_dim)
        self.mlp_head_utt1 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.mlp_head_utt2 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.mlp_head_utt3 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.mlp_head_utt4 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.mlp_head_utt5 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))

        # ConPCO audio/text projection heads
        self.phn_audio_proj = MLP(embed_dim, embed_dim, drop=0.1)
        self.phn_text_proj = MLP(embed_dim, embed_dim, drop=0.1)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _make_word_pos_mask(self, word_pos: torch.Tensor) -> torch.Tensor:
        """Build per-utterance phone→word grouping mask.

        For each position i, allow attending to all positions j in the same word.
        Vectorized vs reference's nested loop.

        Args:
            word_pos: [B, T] integer word-position ids (0-based, -1 = padding).

        Returns:
            mask: [B, T, T] float — 1.0 where same word, 0.0 elsewhere (padding cols = 0).
        """
        # (B, T, 1) == (B, 1, T) → True where same word id
        same_word = word_pos.unsqueeze(2) == word_pos.unsqueeze(1)  # (B, T, T)
        # Zero out padding positions (word_pos == -1)
        valid = (word_pos >= 0).unsqueeze(1)  # (B, 1, T)
        return (same_word & valid).float()

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        gop: torch.Tensor,       # [B, 50, 84]
        energy: torch.Tensor,    # [B, 50, 7]
        dur: torch.Tensor,       # [B, 50, 1]
        ssl: torch.Tensor,       # [B, 50, 3072]  (ssl_drop applied here)
        phn: torch.Tensor,       # [B, 50]  phone ids, -1 = pad
        word_pos: torch.Tensor,  # [B, 50]  word position ids, -1 = pad
        word: torch.Tensor,      # [B, 50]  word ids, -1 = pad
    ) -> tuple[torch.Tensor, ...]:
        # Concatenate all input features: [B, 50, 84 + 7 + 1 + 3072]
        x = torch.cat([gop, self.ssl_drop(ssl), dur, energy], dim=-1)

        # Project to embed_dim independently for phone, word, utterance paths
        p_x = self.p_in_proj(x)   # [B, 50, embed_dim]
        w_x = self.w_in_proj(x)
        u_x = self.u_in_proj(x)

        # Phone embedding: one-hot over 40 classes (phn+1 shifts -1 pad to 0)
        phn_one_hot = F.one_hot((phn.long() + 1).clamp(min=0), num_classes=NUM_PHN_CLASSES).float()
        phn_embed = self.phn_proj(phn_one_hot)   # [B, 50, embed_dim]

        # Add phone embedding + positional embedding
        p_x = p_x + phn_embed + self.pos_embed   # [B, 50, embed_dim]

        # ── Phone-level blocks ────────────────────────────────────────────────
        p_tmp_feat = []
        for blk in self.phn_blocks:
            p_x = blk(p_x)
            p_tmp_feat.append(p_x)

        p = self.mlp_head_phn(p_x)  # [B, 50, 1]

        mdd_logit = self.mlp_head_mdd(p_x) if self.use_mdd else None  # [B, 50, 1] or None
        diag_logit = self.mlp_head_diag(p_x) if self.use_mdd else None  # [B, 50, 39] or None

        # ── Word-level blocks ─────────────────────────────────────────────────
        # Word one-hot (word+1 shifts -1 pad to 0)
        word_one_hot = F.one_hot((word.long() + 1).clamp(min=0), num_classes=NUM_WORD_CLASSES).float()
        word_embed = self.word_proj(word_one_hot)

        phn2word_msk = self._make_word_pos_mask(word_pos)

        # Mix phone-path output back into word path via cross-attention masking
        w_p_x = self.w_proj_cnn1(p_x.transpose(1, 2)).transpose(1, 2)
        w_x = self.w_proj_cnn2(w_x.transpose(1, 2)).transpose(1, 2)

        w_x_att = self.word_input_att(w_x, w_x, w_x, phn2word_msk)
        w_p_x_att = self.word_input_att1(w_p_x, w_p_x, w_p_x, phn2word_msk)
        x_word = self.w_in_cat_proj(torch.cat([w_x_att, w_p_x_att], dim=-1))

        x_word = x_word + word_embed + self.word_pos_embed((word_pos.int() + 1).clamp(min=0)) + p_x

        for blk in self.word_blocks:
            x_word = blk(x_word)

        w1_proj = self.w_proj_ln1(x_word)
        w2_proj = self.w_proj_ln2(x_word)
        w3_proj = self.w_proj_ln3(x_word)

        w1 = self.mlp_head_word1(w1_proj)  # accuracy
        w2 = self.mlp_head_word2(w2_proj)  # stress
        w3 = self.mlp_head_word3(w3_proj)  # total

        # ── Utterance-level blocks ────────────────────────────────────────────
        u_w_feats = self.utt_feat_ext(w1_proj, w2_proj, w3_proj)
        u_p_feats = self.u_proj_cnn1(p_x.transpose(1, 2)).transpose(1, 2)
        u_w_feats = self.u_proj_cnn2(u_w_feats.transpose(1, 2)).transpose(1, 2)
        utt_feats = self.u_in_cat_proj(torch.cat([u_p_feats, u_w_feats, u_x], dim=-1)) + p_x

        for blk in self.utt_blocks:
            utt_feats = blk(utt_feats)

        u1 = self.mlp_head_utt1(self.u1_att_pooling(utt_feats))  # accuracy
        u2 = self.mlp_head_utt2(self.u2_att_pooling(utt_feats))  # completeness
        u3 = self.mlp_head_utt3(self.u3_att_pooling(utt_feats))  # fluency
        u4 = self.mlp_head_utt4(self.u4_att_pooling(utt_feats))  # prosodic
        u5 = self.mlp_head_utt5(self.u5_att_pooling(utt_feats))  # total

        # ConPCO features: audio feats from first phone block, text feats from phone embed
        phn_audio_feats = self.phn_audio_proj(p_tmp_feat[0])
        phn_text_feats = self.phn_text_proj(phn_embed + self.pos_embed)

        if mdd_logit is not None and diag_logit is not None:
            return u1, u2, u3, u4, u5, p, w1, w2, w3, mdd_logit, diag_logit, phn_audio_feats, phn_text_feats
        return u1, u2, u3, u4, u5, p, w1, w2, w3, phn_audio_feats, phn_text_feats
