"""CONO regularizer from Yan et al. 2025 (eqs. 14-16).

Operates on utterance-level features `Z` (shape `(B, D)`) produced by averaging
the phone-encoder outputs. Two terms are combined:

* Diversity `L_d` — a negative sum (so minimizing it spreads centroids apart)
  of pairwise centroid distances weighted by the absolute score gap.
* Tightness `L_t` — the mean distance between each sample's feature and the
  centroid of its own score bucket.

Centroids stay on the autograd graph: we build a `(K, B)` soft assignment
matrix (a one-hot equality mask divided by per-bucket counts) and compute the
centroids via matmul, which is fully differentiable w.r.t. `features`.
"""

from __future__ import annotations

import torch
from torch import Tensor


def cono_loss(
    features: Tensor,
    utterance_accuracy: Tensor,
    lambda_diversity: float = 1.0,
    lambda_tightness: float = 1.0,
) -> Tensor:
    """Compute the CONO regularizer.

    Args:
        features: `(B, D)` utterance-level representation `Z` per sample.
        utterance_accuracy: `(B,)` target accuracy scores (the paper uses
            accuracy as the bucketing signal; see §2.3 around eq. 14).
        lambda_diversity: Weight on the diversity term.
        lambda_tightness: Weight on the tightness term.

    Returns:
        A scalar tensor equal to
        ``lambda_diversity * L_d + lambda_tightness * L_t``.
        When the batch contains fewer than two unique scores, the diversity
        term is zero.
    """

    if features.ndim != 2:
        raise ValueError(f"`features` must be (B, D), got shape {tuple(features.shape)}")
    if utterance_accuracy.ndim != 1 or utterance_accuracy.size(0) != features.size(0):
        raise ValueError(
            f"`utterance_accuracy` must be (B,) matching features; got "
            f"features={tuple(features.shape)} accuracy={tuple(utterance_accuracy.shape)}"
        )

    batch_size = features.size(0)
    if batch_size == 0:
        return features.new_zeros(())

    # torch.unique is non-differentiable but we only use it to define cluster
    # assignments; the centroid values themselves remain differentiable.
    unique_scores, inverse_index = torch.unique(
        utterance_accuracy.detach(), sorted=True, return_inverse=True
    )
    num_clusters = unique_scores.size(0)

    # (K, B) hard assignment, then average -> differentiable centroids via matmul.
    assignment = (
        torch.arange(num_clusters, device=features.device).unsqueeze(1)
        == inverse_index.unsqueeze(0)
    ).to(dtype=features.dtype)
    counts = assignment.sum(dim=1, keepdim=True).clamp(min=1.0)
    normalized_assignment = assignment / counts
    centroids = normalized_assignment @ features  # (K, D)

    # Tightness: mean squared L2 distance from each sample to its own centroid
    # (paper eq. 16).
    per_sample_centroid = centroids[inverse_index]  # (B, D)
    tightness = (features - per_sample_centroid).pow(2).sum(dim=-1).mean()

    if num_clusters < 2:
        diversity = features.new_zeros(())
    else:
        # Pairwise diff tensor: (K, K, D)
        centroid_i = centroids.unsqueeze(1)
        centroid_j = centroids.unsqueeze(0)
        pairwise_dist = (centroid_i - centroid_j).pow(2).sum(dim=-1)

        # Score gap w_{ij} = |y_i - y_j|.
        score_i = unique_scores.unsqueeze(1)
        score_j = unique_scores.unsqueeze(0)
        pairwise_weight = (score_i - score_j).abs().to(dtype=features.dtype)

        # Mask out the diagonal (i == j) and collapse.
        off_diagonal = 1.0 - torch.eye(
            num_clusters, device=features.device, dtype=features.dtype
        )
        weighted = pairwise_weight * pairwise_dist * off_diagonal
        # Per eq. (15): M * (M - 1) normalizer, negative sign so the loss
        # decreases as centroids spread.
        normalizer = float(num_clusters * (num_clusters - 1))
        diversity = -weighted.sum() / normalizer

    return lambda_diversity * diversity + lambda_tightness * tightness
