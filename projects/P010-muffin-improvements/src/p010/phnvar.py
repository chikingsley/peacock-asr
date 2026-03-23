"""Phoneme-Specific Variation (PhnVar) for MDD class imbalance (MuFFIN §IV cont, Eq.24-27).

PhnVar perturbs diagnosis predictor logits during training with Gaussian noise
scaled by two phoneme-dependent factors:
  - QF_k (Quantity Factor): larger for rare phonemes (log inverse frequency)
  - DF_k (Difficulty Factor): larger for frequently mispronounced phonemes

With alpha=beta=1 (paper default), the perturbation simplifies to:
  g_tilde_k = g_k + N(0, sigma) * sqrt(QF_k * DF_k)
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


def compute_phnvar_stats(
    dataset: Dataset,
    mdd_threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-phoneme QF and DF from training data.

    Args:
        dataset:       GoPDataset instance (must have .phn_id and .phn_score attributes).
        mdd_threshold: Score below which a phone is considered mispronounced.

    Returns:
        QF: [39] quantity factor per phoneme (0-indexed), normalized to (0, 1].
        DF: [39] difficulty factor per phoneme, normalized to (0, 1].
    """
    phn_id = dataset.phn_id  # type: ignore[attr-defined]   # [N, 50]
    phn_score = dataset.phn_score  # type: ignore[attr-defined]  # [N, 50]

    valid = phn_id >= 0
    ids = phn_id[valid].long()
    scores = phn_score[valid]

    n_classes = 39  # CMU phone set

    # Count instances per phoneme
    q = torch.zeros(n_classes)
    q.index_add_(0, ids, torch.ones_like(ids, dtype=torch.float))
    q = q.clamp(min=1)  # avoid log(0)

    # QF: log inverse frequency, normalized (Eq. 26)
    total = q.sum()
    c = torch.log(total / q)
    QF = c / c.max()

    # Count mispronounced and correct per phoneme
    is_mispronounced = scores < mdd_threshold
    mp = torch.zeros(n_classes)
    cp = torch.zeros(n_classes)
    mp.index_add_(0, ids[is_mispronounced], torch.ones(is_mispronounced.sum()))
    cp.index_add_(0, ids[~is_mispronounced], torch.ones((~is_mispronounced).sum()))

    # DF: mispronunciation rate, normalized (Eq. 27)
    d = mp / (mp + cp).clamp(min=1)
    DF = d / d.max().clamp(min=1e-8)

    return QF, DF


def perturb_diag_logits(
    logits: torch.Tensor,
    qf: torch.Tensor,
    df: torch.Tensor,
    sigma: float = 1.0,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> torch.Tensor:
    """Apply PhnVar perturbation to diagnosis logits during training (Eq. 25).

    Args:
        logits: [B, T, 39] raw diagnosis logits.
        qf:     [39] quantity factor per phoneme class.
        df:     [39] difficulty factor per phoneme class.
        sigma:  Gaussian noise standard deviation.
        alpha:  QF weight (default 1.0).
        beta:   DF weight (default 1.0).

    Returns:
        Perturbed logits, same shape as input.
    """
    # Eq. 25: perturbation = N(0, sigma) * exp((alpha*log(QF) + beta*log(DF)) / (alpha+beta))
    # With alpha=beta=1: = N(0, sigma) * sqrt(QF * DF)
    qf = qf.to(logits.device)
    df = df.to(logits.device)

    log_scale = (alpha * torch.log(qf.clamp(min=1e-8)) + beta * torch.log(df.clamp(min=1e-8))) / (alpha + beta)
    scale = torch.exp(log_scale)  # [39]

    noise = torch.randn_like(logits) * sigma  # [B, T, 39]
    return logits + noise * scale.unsqueeze(0).unsqueeze(0)
