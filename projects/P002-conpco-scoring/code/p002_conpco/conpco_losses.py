"""ConPCO: Contrastive Phonemic Ordinal loss components.

Ported from Yan & Chen (ICASSP 2025):
    projects/P002-conpco-scoring/third_party/ConPCO/src/models/conPCO_norm.py

These losses are project-local to P002.
"""

from __future__ import annotations

import logging

import torch
from torch import nn
from torch.nn import functional

logger = logging.getLogger(__name__)

MAX_PHONE_SCORE = 2.0
MIN_UNIQUE_PHONEMES = 2


def _euclidean_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Pairwise Euclidean distance. x: [m, d], y: [n, d] -> [m, n]."""
    xx = x.pow(2).sum(1, keepdim=True)
    yy = y.pow(2).sum(1, keepdim=True)
    dist = xx + yy.t()
    dist = dist - 2.0 * torch.mm(x, y.t())
    return dist.clamp(min=1e-12).sqrt()


def _upper_triangular(x: torch.Tensor) -> torch.Tensor:
    """Flatten upper-triangular elements (above diagonal) of square matrix."""
    n = x.shape[0]
    mask = torch.triu(
        torch.ones(n, n, device=x.device, dtype=torch.bool),
        diagonal=1,
    )
    return x[mask]


def _filter_valid_tokens(
    scores: torch.Tensor,
    phn_ids: torch.Tensor,
    *feature_tensors: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
    """Filter non-padding tokens whose phoneme has a max-score sample."""
    flat_scores = scores.view(-1)
    flat_phn_ids = phn_ids.view(-1)

    mask_valid = flat_scores >= 0
    valid_phn_ids = flat_phn_ids[mask_valid]
    valid_scores = flat_scores[mask_valid]

    mask_high = valid_scores == MAX_PHONE_SCORE
    if mask_high.any():
        high_phn_ids = valid_phn_ids[mask_high]
        high_phn_set = torch.unique(high_phn_ids)
        keep = torch.isin(valid_phn_ids, high_phn_set)
    else:
        keep = torch.zeros_like(valid_scores, dtype=torch.bool)

    filtered_feats = []
    for feat in feature_tensors:
        flat_feat = feat.reshape(-1, feat.shape[-1])
        filtered_feats.append(flat_feat[mask_valid][keep])

    return valid_scores[keep], valid_phn_ids[keep], tuple(filtered_feats)


class OrdinalEntropyLoss(nn.Module):
    """PCO loss with diversity and tightness terms."""

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
        gt, phn_id, (feats,) = _filter_valid_tokens(scores, phn_ids, features)

        if len(gt) == 0:
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        u_values, u_inverse, u_counts = torch.unique(
            phn_id,
            return_inverse=True,
            return_counts=True,
        )
        if len(u_values) < MIN_UNIQUE_PHONEMES:
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        center = torch.zeros(len(u_values), feats.shape[1], device=feats.device)
        center.index_add_(0, u_inverse, feats)
        center = center / u_counts.unsqueeze(1)
        center = functional.normalize(center, dim=1)

        p = functional.normalize(center, dim=1)
        dist = _euclidean_dist(p, p)
        diversity = torch.mean(_upper_triangular(dist))

        feats_norm = functional.normalize(feats, dim=1)
        feat_centers = p[u_inverse]
        diff_sq = (feats_norm - feat_centers).pow(2).sum(dim=1)

        mask_nonzero = diff_sq > 0
        if not mask_nonzero.any():
            return -self.lambda_d * diversity

        ordinal_weight = (MAX_PHONE_SCORE - gt[mask_nonzero]) + self.margin
        tightness = torch.mean(torch.sqrt(diff_sq[mask_nonzero]) * ordinal_weight)

        return self.lambda_t * tightness - self.lambda_d * diversity


class CLAPContrastiveLoss(nn.Module):
    """Contrastive audio-text phoneme center alignment."""

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
        gt, phn_id, (audio_feats, text_feats) = _filter_valid_tokens(
            scores,
            phn_ids,
            audio_features,
            text_features,
        )

        if len(gt) == 0:
            return torch.tensor(0.0, device=audio_features.device, requires_grad=True)

        u_values, u_inverse, u_counts = torch.unique(
            phn_id,
            return_inverse=True,
            return_counts=True,
        )
        if len(u_values) < MIN_UNIQUE_PHONEMES:
            return torch.tensor(0.0, device=audio_features.device, requires_grad=True)

        d = audio_feats.shape[1]
        center_audio = torch.zeros(len(u_values), d, device=audio_feats.device)
        center_audio.index_add_(0, u_inverse, audio_feats)
        center_audio = center_audio / u_counts.unsqueeze(1)
        center_audio = functional.normalize(center_audio, dim=1)

        center_text = torch.zeros(len(u_values), d, device=text_feats.device)
        center_text.index_add_(0, u_inverse, text_feats)
        center_text = center_text / u_counts.unsqueeze(1)
        center_text = functional.normalize(center_text, dim=1)

        cos_matrix = torch.matmul(center_audio, center_text.t())
        log_probs_a2t = functional.log_softmax(cos_matrix, dim=1)
        loss_a2t = -torch.diagonal(log_probs_a2t).mean()
        log_probs_t2a = functional.log_softmax(cos_matrix.t(), dim=1)
        loss_t2a = -torch.diagonal(log_probs_t2a).mean()
        return self.lambda_clap_t2a * loss_a2t + (1 - self.lambda_clap_t2a) * loss_t2a
