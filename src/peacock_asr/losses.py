"""ConPCO: Contrastive Phonemic Ordinal loss components.

Ported from Yan & Chen (ICASSP 2025):
    references/ConPCO/src/models/conPCO_norm.py

Three loss terms:
    L_d (diversity): push phoneme cluster centers apart
    L_t (tightness): pull features toward their phoneme center,
        weighted by ordinal score (lower scores penalized more)
    L_clap: contrastive alignment between audio and text phoneme centers

Adapted for our pipeline:
    - Padding convention: phn_id == -1 (ours) vs gt == 0 (theirs)
    - Scores: continuous 0.0-2.0 in 0.2 steps (same as paper's data)
    - Edge cases: return 0 for degenerate batches instead of NaN
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers (from conPCO_norm.py lines 12-33)
# ---------------------------------------------------------------------------


def _euclidean_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Pairwise Euclidean distance. x: [m, d], y: [n, d] → [m, n]."""
    xx = x.pow(2).sum(1, keepdim=True)
    yy = y.pow(2).sum(1, keepdim=True)
    dist = xx + yy.t()
    # Use addmm for efficiency: dist = xx + yy^T - 2*x*y^T
    dist = dist - 2.0 * torch.mm(x, y.t())
    return dist.clamp(min=1e-12).sqrt()


def _upper_triangular(x: torch.Tensor) -> torch.Tensor:
    """Flatten upper-triangular elements (above diagonal) of square matrix."""
    n = x.shape[0]
    mask = torch.triu(torch.ones(n, n, device=x.device, dtype=torch.bool), diagonal=1)
    return x[mask]


def _filter_valid_tokens(
    scores: torch.Tensor,
    phn_ids: torch.Tensor,
    *feature_tensors: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
    """Filter to non-padding tokens whose phoneme has at least one score==2.0 sample.

    This matches the reference (conPCO_norm.py lines 56-93).

    Args:
        scores: [B, T] — continuous phone scores, -1 for padding
        phn_ids: [B, T] — phone IDs 0..38, -1 for padding
        *feature_tensors: [B, T, D] tensors to filter in parallel

    Returns:
        (filtered_scores, filtered_phn_ids, tuple of filtered feature tensors)
        All are 1D/2D with only valid tokens remaining.
        Returns empty tensors if no valid tokens.
    """
    flat_scores = scores.view(-1)
    flat_phn_ids = phn_ids.view(-1)

    # Step 1: non-padding mask (our padding = -1, so scores >= 0)
    mask_valid = flat_scores >= 0

    # Step 2: find phonemes that have at least one score == 2.0
    valid_phn_ids = flat_phn_ids[mask_valid]
    valid_scores = flat_scores[mask_valid]

    mask_high = valid_scores == 2.0
    if mask_high.any():
        high_phn_ids = valid_phn_ids[mask_high]
        high_phn_set = torch.unique(high_phn_ids)

        # Step 3: further filter to only phonemes in high_phn_set
        # For each valid token, check if its phn_id is in high_phn_set
        keep = torch.isin(valid_phn_ids, high_phn_set)
    else:
        # No score-2.0 tokens at all — nothing to compute on
        keep = torch.zeros_like(valid_scores, dtype=torch.bool)

    # Apply both filters to features
    b, t = scores.shape
    filtered_feats = []
    for feat in feature_tensors:
        flat_feat = feat.reshape(-1, feat.shape[-1])
        filtered_feats.append(flat_feat[mask_valid][keep])

    return (
        valid_scores[keep],
        valid_phn_ids[keep],
        tuple(filtered_feats),
    )


# ---------------------------------------------------------------------------
# Ordinal Entropy Loss (diversity + tightness)
# ---------------------------------------------------------------------------


class OrdinalEntropyLoss(nn.Module):
    """PCO loss: diversity pushes phoneme centers apart, tightness pulls
    features toward centers weighted by ordinal score.

    From Yan & Chen (ASRU 2023, ICASSP 2025).
    Reference: conPCO_norm.py lines 124-153.

    loss_oe = λ_t * tightness - λ_d * diversity
    """

    def __init__(
        self,
        lambda_d: float = 0.5,
        lambda_t: float = 0.1,
        margin: float = 1.0,
    ) -> None:
        super().__init__()
        self.lambda_d = lambda_d
        self.lambda_t = lambda_t
        self.margin = margin

    def forward(
        self,
        features: torch.Tensor,
        scores: torch.Tensor,
        phn_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ordinal entropy loss.

        Args:
            features: [B, T, D] — projected audio features
            scores: [B, T] — continuous phone scores, -1 for padding
            phn_ids: [B, T] — phone IDs 0..38, -1 for padding

        Returns:
            Scalar loss: λ_t * tightness - λ_d * diversity
        """
        gt, phn_id, (feats,) = _filter_valid_tokens(scores, phn_ids, features)

        if len(gt) == 0:
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        u_values, u_inverse, u_counts = torch.unique(
            phn_id, return_inverse=True, return_counts=True,
        )

        if len(u_values) < 2:
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        # Phoneme centers (ref lines 97-102)
        center = torch.zeros(len(u_values), feats.shape[1], device=feats.device)
        center.index_add_(0, u_inverse, feats)
        center = center / u_counts.unsqueeze(1)
        center = F.normalize(center, dim=1)

        # Diversity: mean pairwise Euclidean distance between centers (ref lines 124-134)
        p = F.normalize(center, dim=1)
        dist = _euclidean_dist(p, p)
        diversity = torch.mean(_upper_triangular(dist))

        # Tightness: ordinal-weighted distance to center (ref lines 135-151)
        feats_norm = F.normalize(feats, dim=1)
        feat_centers = p[u_inverse]
        diff_sq = (feats_norm - feat_centers).pow(2).sum(dim=1)

        mask_nonzero = diff_sq > 0
        if not mask_nonzero.any():
            return -self.lambda_d * diversity

        ordinal_weight = (2.0 - gt[mask_nonzero]) + self.margin
        tightness = torch.mean(torch.sqrt(diff_sq[mask_nonzero]) * ordinal_weight)

        return self.lambda_t * tightness - self.lambda_d * diversity


# ---------------------------------------------------------------------------
# CLAP Contrastive Loss
# ---------------------------------------------------------------------------


class CLAPContrastiveLoss(nn.Module):
    """Contrastive audio-text phoneme center alignment.

    Aligns per-phoneme audio centers with per-phoneme text centers using
    bidirectional cross-entropy on cosine similarity matrix.

    From Yan & Chen (ICASSP 2025).
    Reference: conPCO_norm.py lines 110-122.
    """

    def __init__(self, lambda_clap_t2a: float = 0.5) -> None:
        super().__init__()
        self.lambda_clap_t2a = lambda_clap_t2a

    def forward(
        self,
        audio_features: torch.Tensor,
        text_features: torch.Tensor,
        scores: torch.Tensor,
        phn_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute bidirectional contrastive loss.

        Args:
            audio_features: [B, T, D] — projected audio phoneme features
            text_features: [B, T, D] — projected text phoneme features
            scores: [B, T] — continuous phone scores, -1 for padding
            phn_ids: [B, T] — phone IDs 0..38, -1 for padding

        Returns:
            Scalar contrastive loss
        """
        gt, phn_id, (audio_feats, text_feats) = _filter_valid_tokens(
            scores, phn_ids, audio_features, text_features,
        )

        if len(gt) == 0:
            return torch.tensor(0.0, device=audio_features.device, requires_grad=True)

        u_values, u_inverse, u_counts = torch.unique(
            phn_id, return_inverse=True, return_counts=True,
        )

        if len(u_values) < 2:
            return torch.tensor(0.0, device=audio_features.device, requires_grad=True)

        # Audio phoneme centers (ref lines 97-102)
        d = audio_feats.shape[1]
        center_audio = torch.zeros(len(u_values), d, device=audio_feats.device)
        center_audio.index_add_(0, u_inverse, audio_feats)
        center_audio = center_audio / u_counts.unsqueeze(1)
        center_audio = F.normalize(center_audio, dim=1)

        # Text phoneme centers (ref lines 104-108)
        center_text = torch.zeros(len(u_values), d, device=text_feats.device)
        center_text.index_add_(0, u_inverse, text_feats)
        center_text = center_text / u_counts.unsqueeze(1)
        center_text = F.normalize(center_text, dim=1)

        # Cosine similarity matrix + log-softmax (ref lines 112-122)
        cos_matrix = torch.matmul(center_audio, center_text.t())

        # Audio→text direction
        log_probs_a2t = F.log_softmax(cos_matrix, dim=1)
        loss_a2t = -torch.diagonal(log_probs_a2t).mean()

        # Text→audio direction
        log_probs_t2a = F.log_softmax(cos_matrix.t(), dim=1)
        loss_t2a = -torch.diagonal(log_probs_t2a).mean()

        return self.lambda_clap_t2a * loss_a2t + (1 - self.lambda_clap_t2a) * loss_t2a
