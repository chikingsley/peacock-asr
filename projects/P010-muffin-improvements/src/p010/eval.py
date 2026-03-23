"""Evaluation metrics for SpeechOcean762.

Ported from ConPCO/src/traintest_eng_dur_ssl_3m_HierBFR_conPCO_norm.py
(valid_phn, valid_utt, valid_word, lines 300-367).

Modernizations vs reference:
- valid_phn: nested loop collecting valid tokens → vectorized mask + np.corrcoef.
- valid_utt: kept (already vectorized enough for 5 aspects × 2500 samples).
- valid_word: kept sequential (word-boundary logic inherently sequential; outer loop is
  only ~2500 utterances, inner loop ~10 words each — not a hotpath).
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score


def _safe_pcc(pred: np.ndarray, target: np.ndarray) -> float:
    """Return Pearson correlation when defined, else 0.0 for tiny/constant inputs."""
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)

    if pred.size < 2 or target.size < 2:
        return 0.0
    if np.allclose(pred, pred[0]) or np.allclose(target, target[0]):
        return 0.0

    pcc = float(np.corrcoef(pred, target)[0, 1])
    return pcc if np.isfinite(pcc) else 0.0


# ── Phone-level ───────────────────────────────────────────────────────────────

def eval_phn(pred: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    """Phone-level MSE and PCC.

    Args:
        pred:   [N_utt, 50] or [N_utt, 50, 1] predicted scores.
        target: [N_utt, 50] target scores; negative = padding.

    Returns:
        (mse, pcc)
    """
    pred = pred.squeeze(-1).numpy()
    target = target.numpy()
    mask = target >= 0
    pred_valid = pred[mask]
    target_valid = target[mask]
    mse = float(np.mean((target_valid - pred_valid) ** 2))
    pcc = _safe_pcc(pred_valid, target_valid)
    return mse, pcc


# ── Utterance-level ───────────────────────────────────────────────────────────

def eval_utt(pred: torch.Tensor, target: torch.Tensor) -> tuple[list[float], list[float]]:
    """Utterance-level MSE and PCC per aspect.

    Args:
        pred:   [N_utt, 5] — 5 utterance aspects
        target: [N_utt, 5]

    Returns:
        (mse_list, pcc_list) each of length 5: [accuracy, completeness, fluency, prosodic, total]
    """
    mse_list, pcc_list = [], []
    for i in range(5):
        p = pred[:, i].numpy()
        t = target[:, i].numpy()
        mse_list.append(float(np.mean((t - p) ** 2)))
        pcc_list.append(_safe_pcc(p, t))
    return mse_list, pcc_list


# ── Word-level ────────────────────────────────────────────────────────────────

def eval_word(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> tuple[list[float], list[float], np.ndarray, np.ndarray]:
    """Word-level MSE and PCC for [accuracy, stress, total].

    Aggregates phone-level predictions to word level by averaging over phones
    belonging to the same word (identified by word_pos in target[:, :, -1]).

    Args:
        pred:   [N_utt, 50, 3]  word score predictions (accuracy, stress, total)
        target: [N_utt, 50, 4]  [accuracy, stress, total, word_pos]; negative = padding

    Returns:
        (mse_list, pcc_list, pred_array, target_array)
        pred_array / target_array: [N_words, 3] aggregated word-level scores
    """
    word_pos = target[:, :, -1]
    target_scores = target[:, :, :3]

    agg_pred, agg_target = [], []
    for i in range(target_scores.shape[0]):  # each utterance
        prev_w_id = 0
        start_id = 0
        for j in range(target_scores.shape[1]):  # each phone position
            cur_w_id = int(word_pos[i, j].item())
            if cur_w_id != prev_w_id:
                agg_pred.append(pred[i, start_id:j, :].numpy().mean(axis=0))
                agg_target.append(target_scores[i, start_id:j, :].numpy().mean(axis=0))
                if cur_w_id == -1:
                    break
                prev_w_id = cur_w_id
                start_id = j

    agg_pred_arr = np.array(agg_pred)
    agg_target_arr = np.array(agg_target).round(2)

    mse_list, pcc_list = [], []
    for i in range(3):
        mse_list.append(float(np.mean((agg_target_arr[:, i] - agg_pred_arr[:, i]) ** 2)))
        pcc_list.append(_safe_pcc(agg_pred_arr[:, i], agg_target_arr[:, i]))

    return mse_list, pcc_list, agg_pred_arr, agg_target_arr


# ── MDD metrics ───────────────────────────────────────────────────────────────

def eval_mdd(
    logit: torch.Tensor,
    label: torch.Tensor,
    threshold: float = 0.5,
    diag_logit: torch.Tensor | None = None,
    diag_label: torch.Tensor | None = None,
    canon_phn_id: torch.Tensor | None = None,
) -> dict[str, float]:
    """MDD detection + diagnosis metrics (MuFFIN §V.C, Table V).

    Detection: F1, precision, recall on binary mispronunciation labels.
    Diagnosis: FAR, FRR, DER, PER on phoneme predictions (if diag provided).

    Per MuFFIN §IV: "the canonical phonemes are masked during the softmax computation
    of the predictor" — for detected mispronunciations, the canonical phone class is
    set to -inf before argmax so the model can't just echo the prompt.

    Args:
        logit:        [N_utt, 50, 1] raw detection logits (before sigmoid).
        label:        [N_utt, 50]    1 = mispronounced, 0 = correct, -1 = padding.
        threshold:    Sigmoid threshold for positive classification.
        diag_logit:   [N_utt, 50, 39] diagnosis logits (optional).
        diag_label:   [N_utt, 50] ground-truth phoneme IDs for diagnosis (optional).
        canon_phn_id: [N_utt, 50] canonical phone IDs for softmax masking (optional).

    Returns:
        dict with keys: f1, precision, recall, and optionally far, frr, der, per
    """
    logit = logit.squeeze(-1)
    mask = label >= 0
    pred_detect = (logit[mask].sigmoid() >= threshold).long().numpy()
    true_detect = label[mask].long().numpy()

    metrics: dict[str, float] = {
        "f1": float(f1_score(true_detect, pred_detect, zero_division=0)),
        "precision": float(precision_score(true_detect, pred_detect, zero_division=0)),
        "recall": float(recall_score(true_detect, pred_detect, zero_division=0)),
    }

    # Diagnosis metrics (MuFFIN Table V: FAR, FRR, DER, PER)
    if diag_logit is not None and diag_label is not None:
        # Mask canonical phone before argmax (MuFFIN §IV: "canonical phonemes are
        # masked during the softmax computation of the predictor")
        if canon_phn_id is not None:
            masked_logit = diag_logit.clone()
            # canon_phn_id is 0-based (0-38). Set canonical class to -inf.
            canon_idx = canon_phn_id.long().clamp(min=0)  # [N_utt, 50]
            masked_logit.scatter_(2, canon_idx.unsqueeze(-1), float("-inf"))
            diag_pred = masked_logit.argmax(dim=-1)
        else:
            diag_pred = diag_logit.argmax(dim=-1)  # [N_utt, 50]

        # For detected mispronunciations, check if diagnosis matches
        detected = pred_detect == 1
        true_mispr = true_detect == 1
        true_correct = true_detect == 1 - 1  # == 0

        # FAR: false acceptance rate = mispronounced phones predicted as correct
        n_mispr = int(true_mispr.sum())
        far = float((~detected[true_mispr]).sum() / max(n_mispr, 1))

        # FRR: false rejection rate = correct phones predicted as mispronounced
        n_correct = int(true_correct.sum())
        frr = float(detected[true_correct].sum() / max(n_correct, 1))

        # PER: phoneme error rate among detected mispronunciations.
        # Compare diagnosis prediction against the ground-truth *actual* phone
        # (diag_label), NOT the canonical phone. The canonical phone is masked
        # before argmax, so comparing against it would be internally inconsistent.
        # Also exclude positions where diag_label == -1 (non-CMU sounds that
        # training ignores via ignore_index=-1).
        diag_flat = diag_pred.numpy().reshape(-1)[mask.numpy().reshape(-1)]
        diag_label_flat = diag_label.long().numpy().reshape(-1)[mask.numpy().reshape(-1)]
        valid_diag = diag_label_flat >= 0  # exclude non-CMU labels

        # PER over detected phones with valid diagnosis labels
        detected_valid = detected & valid_diag
        if detected_valid.sum() > 0:
            per = float((diag_flat[detected_valid] != diag_label_flat[detected_valid]).sum()
                        / max(detected_valid.sum(), 1))
        else:
            per = 0.0

        # DER: diagnostic error rate among correctly-detected true mispronunciations
        true_and_detected = detected & true_mispr & valid_diag
        if true_and_detected.sum() > 0:
            der = float((diag_flat[true_and_detected] != diag_label_flat[true_and_detected]).sum()
                        / max(true_and_detected.sum(), 1))
        else:
            der = 0.0

        metrics.update({"far": far, "frr": frr, "der": der, "per": per})

    return metrics


def grid_search_mdd_threshold(
    logit: torch.Tensor,
    label: torch.Tensor,
    lo: float = 0.0,
    hi: float = 1.0,
    step: float = 0.1,
) -> float:
    """Find MDD threshold that maximizes F1 on held-out data (MuFFIN §V.B).

    Args:
        logit: [N_utt, 50, 1] raw logits.
        label: [N_utt, 50] ground-truth labels.
        lo, hi, step: Grid search range and stride.

    Returns:
        Best threshold value.
    """
    best_f1, best_thresh = -1.0, 0.5
    thresh = lo
    while thresh <= hi + 1e-9:
        result = eval_mdd(logit, label, threshold=thresh)
        if result["f1"] > best_f1:
            best_f1 = result["f1"]
            best_thresh = thresh
        thresh += step
    return best_thresh
