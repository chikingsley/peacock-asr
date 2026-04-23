from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from torch import Tensor

from p014.data import UTTERANCE_SCORE_KEYS, WORD_SCORE_KEYS


@dataclass(frozen=True)
class EvaluationMetrics:
    phone_mse: float
    phone_pcc: float
    word_pcc: dict[str, float]
    utterance_pcc: dict[str, float]
    loss: float


def safe_pcc(prediction: np.ndarray, target: np.ndarray) -> float:
    pred = prediction.reshape(-1)
    gold = target.reshape(-1)
    if pred.size < 2 or gold.size < 2:
        return 0.0
    if np.allclose(pred, pred[0]) or np.allclose(gold, gold[0]):
        return 0.0
    corr = float(np.corrcoef(pred, gold)[0, 1])
    return corr if np.isfinite(corr) else 0.0


def masked_mse(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    diff = (prediction - target) ** 2
    diff = diff * mask.to(dtype=prediction.dtype)
    denom = mask.sum().clamp(min=1).to(dtype=prediction.dtype)
    return diff.sum() / denom


def evaluate_predictions(
    phone_scores: Tensor,
    phone_targets: Tensor,
    phone_mask: Tensor,
    word_scores: Tensor,
    word_targets: Tensor,
    word_mask: Tensor,
    utterance_scores: Tensor,
    utterance_targets: Tensor,
    loss: float,
) -> EvaluationMetrics:
    phone_pred = phone_scores[phone_mask].detach().cpu().numpy()
    phone_gold = phone_targets[phone_mask].detach().cpu().numpy()
    phone_mse = float(np.mean((phone_pred - phone_gold) ** 2))
    phone_pcc = safe_pcc(phone_pred, phone_gold)

    word_metrics = {}
    word_mask_3d = word_mask.unsqueeze(-1).expand_as(word_scores)
    for aspect_index, aspect_name in enumerate(WORD_SCORE_KEYS):
        aspect_mask = word_mask_3d[:, :, aspect_index]
        pred = word_scores[:, :, aspect_index][aspect_mask].detach().cpu().numpy()
        gold = word_targets[:, :, aspect_index][aspect_mask].detach().cpu().numpy()
        word_metrics[aspect_name] = safe_pcc(pred, gold)

    utterance_metrics = {}
    for aspect_index, aspect_name in enumerate(UTTERANCE_SCORE_KEYS):
        pred = utterance_scores[:, aspect_index].detach().cpu().numpy()
        gold = utterance_targets[:, aspect_index].detach().cpu().numpy()
        utterance_metrics[aspect_name] = safe_pcc(pred, gold)

    return EvaluationMetrics(
        phone_mse=phone_mse,
        phone_pcc=phone_pcc,
        word_pcc=word_metrics,
        utterance_pcc=utterance_metrics,
        loss=loss,
    )
