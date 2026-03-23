"""Loss functions for P010: masked MSE, ConPCO, MDD BCE.

Ported from ConPCO/src/models/conPCO_norm.py and the training script.

Modernizations vs reference:
- euclidean_dist: addmm_(1, -2, x, y.t()) deprecated 3-arg form → torch.cdist (cleaner, correct).
- Removed unused `import random` and `import numpy as np` from conPCO_norm.py.
- up_triu: kept as-is (correct and simple).
- ContrastivePhonemicOrdinalRegularizer: class structure kept; internal variable names cleaned up.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Utilities ─────────────────────────────────────────────────────────────────

def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE over valid (non-padding) positions, corrected for batch × seq scale.

    The correction factor (total_positions / valid_positions) matches the reference
    training script (lines 139-142): avoids the loss being artificially small when
    many positions are padding.

    Args:
        pred:   [B, T] or [B, T, 1] predictions.
        target: same shape as pred.
        mask:   [B, T] bool, True = valid position.
    """
    pred = pred.squeeze(-1)
    target = target.squeeze(-1)
    pred = pred * mask
    target = target * mask
    loss = F.mse_loss(pred, target)
    # Scale to account for padding dilution
    n_total = mask.shape[0] * mask.shape[1]
    n_valid = mask.sum().clamp(min=1)
    return loss * (n_total / n_valid)


def _up_triu(x: torch.Tensor) -> torch.Tensor:
    """Return upper-triangular off-diagonal elements as a 1-D tensor."""
    n, m = x.shape
    assert n == m
    idx = torch.triu(torch.ones(n, n, dtype=torch.bool, device=x.device), diagonal=1)
    return x[idx]


# ── Contrastive Phonemic Ordinal Regularizer ──────────────────────────────────

class ConPCOLoss(nn.Module):
    """Contrastive Phonemic Ordinal Regularizer.

    Two terms returned separately so the caller can weight them independently:
      loss_oe   : L_d (diversity) - L_t (tightness)  ← combined ordinal-entropy term
      loss_clap : L_clap (audio ↔ text contrastive alignment)

    Args:
        lambda_d:  Weight for diversity term (push phone centers apart). Default 0.5.
        lambda_t:  Weight for tightness term (pull features to center). Default 0.1.
        clap_t2a:  Text-to-audio weight in CLAP loss (1 - clap_t2a for audio-to-text). Default 0.5.
        margin:    Ordinal margin added to tightness weight. Default 1.0.
    """

    def __init__(
        self,
        lambda_d: float = 0.5,
        lambda_t: float = 0.1,
        clap_t2a: float = 0.5,
        margin: float = 1.0,
    ) -> None:
        super().__init__()
        self.lambda_d = lambda_d
        self.lambda_t = lambda_t
        self.clap_t2a = clap_t2a
        self.margin = margin

    def forward(
        self,
        audio_feats: torch.Tensor,  # [B, T, C]
        text_feats: torch.Tensor,   # [B, T, C]
        scores: torch.Tensor,       # [B, T]  phone accuracy scores
        phn_ids: torch.Tensor,      # [B, T]  phone class ids
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, C = audio_feats.shape
        flat_scores = scores.reshape(-1)
        flat_phn_ids = phn_ids.reshape(-1)
        flat_audio = audio_feats.reshape(-1, C)
        flat_text = text_feats.reshape(-1, C)

        # Mask: exclude padding (score <= 0) and ensure every phone class present has a score-2 sample
        valid_mask = flat_scores > 0
        high_mask = flat_scores == 2.0
        u_high, _, _ = torch.unique(flat_phn_ids[high_mask], return_inverse=True, return_counts=True)

        # Skip phone classes that have no score-2 exemplars (can't form ordinal triplets)
        skip_mask = torch.tensor(
            [i.item() not in u_high for i in flat_phn_ids[valid_mask]], device=scores.device
        )
        if skip_mask.any():
            keep = ~skip_mask
            _scores = flat_scores[valid_mask][keep]
            _audio = flat_audio[valid_mask][keep]
            _text = flat_text[valid_mask][keep]
            _phn_ids = flat_phn_ids[valid_mask][keep]
        else:
            _scores = flat_scores[valid_mask]
            _audio = flat_audio[valid_mask]
            _text = flat_text[valid_mask]
            _phn_ids = flat_phn_ids[valid_mask]

        # Guard: if filtering left no samples, return zeros rather than NaN from empty .mean()
        if len(_scores) == 0:
            zero = audio_feats.new_tensor(0.0)
            return zero, zero

        u_phns, u_idx, u_counts = torch.unique(_phn_ids, return_inverse=True, return_counts=True)
        n_phns = len(u_phns)

        # Compute per-phone centroids (L2-normalized)
        center_audio = torch.zeros(n_phns, C, device=_audio.device)
        center_audio.index_add_(0, u_idx, _audio)
        center_audio = F.normalize(center_audio / u_counts.unsqueeze(1), dim=1)

        center_text = torch.zeros(n_phns, C, device=_text.device)
        center_text.index_add_(0, u_idx, _text)
        center_text = F.normalize(center_text / u_counts.unsqueeze(1), dim=1)

        # ── CLAP loss: InfoNCE between audio and text centroids ────────────────
        cos = torch.matmul(center_audio, center_text.T)  # (n_phns, n_phns)
        log_softmax_a2t = F.log_softmax(cos, dim=1)
        log_softmax_t2a = F.log_softmax(cos.T, dim=1)
        loss_clap = (
            self.clap_t2a * (-torch.diagonal(log_softmax_a2t).mean())
            + (1 - self.clap_t2a) * (-torch.diagonal(log_softmax_t2a).mean())
        )

        # ── Diversity term: maximize inter-phone centroid distances ───────────
        # torch.cdist replaces the deprecated addmm_(1, -2, x, y.t()) form
        dists = torch.cdist(center_audio, center_audio)  # (n_phns, n_phns)
        diversity = _up_triu(dists).mean()

        # ── Tightness term: ordinal-weighted distance to centroid ─────────────
        _audio_norm = F.normalize(_audio, dim=1)
        per_feature_center = center_audio[u_idx]  # center for each sample's phone class
        sq_dist = (_audio_norm - per_feature_center).pow(2).sum(dim=1)
        valid_tight = sq_dist > 0
        high_score = torch.full_like(_scores[valid_tight], 2.0)
        ordinal_weight = (high_score - _scores[valid_tight]) + self.margin
        tightness = (sq_dist[valid_tight].sqrt() * ordinal_weight).mean()

        loss_oe = self.lambda_t * tightness - self.lambda_d * diversity

        return loss_oe, loss_clap
