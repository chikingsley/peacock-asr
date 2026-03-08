"""Alternative scalar score variants built from frame posteriors.

The default benchmark score is GOP-SF scalar from `gop.py`. This module adds
fast, non-invasive scalar variants that can be swapped at runtime without
changing CTC dynamic-programming internals.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

ScoreVariant = Literal["gop_sf", "logit_margin", "logit_combined"]

_EPS = 1e-12
_POSTERIORS_RANK = 2
_MIN_MARGIN_VOCAB = 2


def apply_score_variant(
    *,
    variant: ScoreVariant,
    score_alpha: float,
    posteriors: np.ndarray,
    phone_indices: list[int],
    baseline_scores: list[float],
) -> list[float]:
    """Return per-phone scores for the selected variant.

    Args:
        variant: Scoring variant to apply.
        score_alpha: Mixture weight for `logit_combined` in [0, 1].
        posteriors: [T, V] posterior matrix.
        phone_indices: canonical phone indices (length = n_phones).
        baseline_scores: GOP-SF scalar scores (length = n_phones).
    """
    if variant == "gop_sf":
        return list(baseline_scores)

    margin_scores = _compute_logit_margin_scores(posteriors, phone_indices)
    if variant == "logit_margin":
        return margin_scores

    if variant == "logit_combined":
        alpha = float(score_alpha)
        if not (0.0 <= alpha <= 1.0):
            msg = "score_alpha must be in [0, 1] for logit_combined"
            raise ValueError(msg)
        return [
            alpha * margin + (1.0 - alpha) * base
            for margin, base in zip(margin_scores, baseline_scores, strict=True)
        ]

    msg = f"Unknown score variant: {variant}"
    raise ValueError(msg)


def _compute_logit_margin_scores(
    posteriors: np.ndarray,
    phone_indices: list[int],
) -> list[float]:
    """Compute a soft, frame-weighted logit-margin proxy score per phone.

    Per frame, margin is `log p(target) - max log p(other)`. We aggregate using
    posterior weight on the target token, which keeps the computation alignment-
    free while emphasizing frames where the token is likely active.
    """
    if len(phone_indices) == 0:
        return []
    if posteriors.ndim != _POSTERIORS_RANK:
        msg = f"Expected posteriors shape [T, V], got {posteriors.shape}"
        raise ValueError(msg)

    probs = np.clip(posteriors, _EPS, 1.0)
    log_probs = np.log(probs)
    num_frames, vocab_size = log_probs.shape
    if vocab_size < _MIN_MARGIN_VOCAB:
        msg = "Need at least 2 vocab classes to compute margin scores."
        raise ValueError(msg)

    max_idx = np.argmax(log_probs, axis=1)
    max_val = log_probs[np.arange(num_frames), max_idx]
    second_val = np.partition(log_probs, -2, axis=1)[:, -2]

    scores: list[float] = []
    for idx in phone_indices:
        if idx < 0 or idx >= vocab_size:
            msg = f"Phone index {idx} out of range for vocab size {vocab_size}"
            raise ValueError(msg)

        target = log_probs[:, idx]
        best_other = np.where(max_idx == idx, second_val, max_val)
        margin = target - best_other
        weights = probs[:, idx]
        denom = float(weights.sum())
        if denom <= _EPS:
            score = float(margin.mean())
        else:
            score = float((weights * margin).sum() / denom)
        scores.append(score)

    return scores
