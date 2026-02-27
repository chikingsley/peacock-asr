"""Evaluation: compute PCC between GOP scores and human annotations.

Supports two modes:
  1. Scalar GOP scores → per-phone polynomial regression (original)
  2. Feature vectors → per-phone SVR (following CTC-based-GOP reference)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR

logger = logging.getLogger(__name__)

POLY_ORDER = 2


@dataclass
class EvalResult:
    pcc: float
    pcc_low: float
    pcc_high: float
    mse: float
    n_phones: int
    per_phone_pcc: dict[str, float] = field(default_factory=dict)


def balanced_sampling(
    features: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Oversample minority classes to balance the dataset."""
    unique_labels = np.unique(labels)
    max_count = max(np.sum(labels == label) for label in unique_labels)

    all_features = []
    all_labels = []

    rng = np.random.default_rng(42)

    for label in unique_labels:
        mask = labels.ravel() == label
        class_features = features[mask]
        class_labels = labels[mask]
        n_samples = len(class_labels)

        if n_samples < max_count:
            indices = rng.choice(n_samples, size=max_count - n_samples, replace=True)
            class_features = np.concatenate([class_features, class_features[indices]])
            class_labels = np.concatenate([class_labels, class_labels[indices]])

        all_features.append(class_features)
        all_labels.append(class_labels)

    return np.concatenate(all_features), np.concatenate(all_labels)


def _train_poly_model(gop_scores: np.ndarray, labels: np.ndarray) -> LinearRegression:
    """Train a polynomial regression model for a single phone."""
    model = LinearRegression()
    gops = gop_scores.reshape(-1, 1)
    poly_gops = PolynomialFeatures(POLY_ORDER).fit_transform(gops)
    balanced_gops, balanced_labels = balanced_sampling(poly_gops, labels.reshape(-1, 1))
    model.fit(balanced_gops, balanced_labels)
    return model


def _round_score(
    score: np.ndarray,
    floor: float = 0.1,
    min_val: float = 0.0,
    max_val: float = 2.0,
) -> np.ndarray:
    score = np.clip(score, min_val, max_val)
    return np.round(score / floor) * floor


def evaluate_gop(
    train_data: list[tuple[str, float, float]],
    test_data: list[tuple[str, float, float]],
) -> EvalResult:
    """Evaluate GOP scores against human annotations.

    Args:
        train_data: list of (phone_name, gop_score, human_label) for training
        test_data: list of (phone_name, gop_score, human_label) for testing

    Returns:
        EvalResult with PCC and per-phone PCC
    """
    # Group by phone
    train_by_phone: dict[str, tuple[list[float], list[float]]] = {}
    for phone, gop, label in train_data:
        train_by_phone.setdefault(phone, ([], []))
        train_by_phone[phone][0].append(gop)
        train_by_phone[phone][1].append(label)

    test_by_phone: dict[str, tuple[list[float], list[float]]] = {}
    for phone, gop, label in test_data:
        test_by_phone.setdefault(phone, ([], []))
        test_by_phone[phone][0].append(gop)
        test_by_phone[phone][1].append(label)

    # Train per-phone polynomial regression
    models: dict[str, LinearRegression] = {}
    for phone, (gops, labels) in train_by_phone.items():
        gop_arr = np.array(gops)
        label_arr = np.array(labels, dtype=int)
        min_unique_labels = 2
        if len(np.unique(label_arr)) < min_unique_labels:
            continue
        models[phone] = _train_poly_model(gop_arr, label_arr)

    # Predict on test
    all_refs = []
    all_hyps = []
    per_phone_pcc: dict[str, float] = {}

    for phone, (gops, labels) in test_by_phone.items():
        if phone not in models:
            continue

        gop_arr = np.array(gops).reshape(-1, 1)
        ref = np.array(labels, dtype=int)
        poly_gops = PolynomialFeatures(POLY_ORDER).fit_transform(gop_arr)
        hyp = models[phone].predict(poly_gops).reshape(-1)
        hyp = _round_score(hyp, floor=1.0)

        all_refs.extend(ref.tolist())
        all_hyps.extend(hyp.tolist())

        min_samples = 3
        if len(ref) >= min_samples and np.std(ref) > 0 and np.std(hyp) > 0:
            per_phone_pcc[phone] = float(np.corrcoef(ref, hyp)[0, 1])

    all_refs_arr = np.array(all_refs)
    all_hyps_arr = np.array(all_hyps)

    min_eval_samples = 2
    if len(all_refs_arr) < min_eval_samples:
        logger.warning("Too few samples (%d) for PCC computation", len(all_refs_arr))
        return EvalResult(
            pcc=float("nan"),
            pcc_low=float("nan"),
            pcc_high=float("nan"),
            mse=float("nan"),
            n_phones=len(all_refs),
            per_phone_pcc=per_phone_pcc,
        )

    if np.std(all_refs_arr) == 0 or np.std(all_hyps_arr) == 0:
        logger.warning("Zero variance in scores, PCC undefined")
        mse = float(np.mean((all_refs_arr - all_hyps_arr) ** 2))
        return EvalResult(
            pcc=float("nan"),
            pcc_low=float("nan"),
            pcc_high=float("nan"),
            mse=mse,
            n_phones=len(all_refs),
            per_phone_pcc=per_phone_pcc,
        )

    res = stats.pearsonr(all_refs_arr, all_hyps_arr)
    ci = res.confidence_interval(confidence_level=0.95)
    mse = float(np.mean((all_refs_arr - all_hyps_arr) ** 2))

    return EvalResult(
        pcc=float(res.statistic),
        pcc_low=float(ci.low),
        pcc_high=float(ci.high),
        mse=mse,
        n_phones=len(all_refs),
        per_phone_pcc=per_phone_pcc,
    )


# ---------------------------------------------------------------------------
# Feature-vector evaluation (SVR)
# ---------------------------------------------------------------------------


def _add_cross_phone_negatives(
    train_by_phone: dict[str, tuple[list[list[float]], list[int]]],
) -> dict[str, tuple[list[list[float]], list[int]]]:
    """Balance classes by adding correctly-pronounced examples from other phones.

    For phones where score-2 (correct) examples dominate, add score-2 examples
    from OTHER phones as score-0 (mispronounced) negatives. This helps SVR
    learn that a correct realization of phone X does not sound like phone Y.

    Follows CTC-based-GOP/is24/evaluation/spo762/evaluate_gop_feats.py.
    """
    # Collect all score-2 examples across phones
    all_correct: list[tuple[str, list[float]]] = []
    for phone, (feats, labels) in train_by_phone.items():
        for f, lab in zip(feats, labels, strict=True):
            if lab == 2:  # noqa: PLR2004
                all_correct.append((phone, f))

    rng = np.random.default_rng(42)

    for phone, (feats, labels) in train_by_phone.items():
        n_correct = sum(1 for lab in labels if lab == 2)  # noqa: PLR2004
        n_needed = 2 * n_correct - len(labels)
        if n_needed <= 0:
            continue

        # Sample correct examples from OTHER phones as negatives (label 0)
        candidates = [f for p, f in all_correct if p != phone]
        if not candidates:
            continue
        n_sample = min(n_needed, len(candidates))
        need_replace = n_sample > len(candidates)
        indices = rng.choice(
            len(candidates), size=n_sample, replace=need_replace
        )
        for idx in indices:
            feats.append(candidates[idx])
            labels.append(0)

    return train_by_phone


def evaluate_gop_feats(
    train_data: list[tuple[str, list[float], float]],
    test_data: list[tuple[str, list[float], float]],
) -> EvalResult:
    """Evaluate GOP feature vectors against human annotations using SVR.

    This follows the reference evaluation protocol from
    CTC-based-GOP/is24/evaluation/spo762/evaluate_gop_feats.py:
    train one SVR per phone, predict on test, compute PCC.

    Args:
        train_data: list of (phone_name, feature_vector, human_label)
        test_data: list of (phone_name, feature_vector, human_label)

    Returns:
        EvalResult with PCC and per-phone PCC
    """
    # Group by phone
    train_by_phone: dict[str, tuple[list[list[float]], list[int]]] = {}
    for phone, feats, label in train_data:
        train_by_phone.setdefault(phone, ([], []))
        train_by_phone[phone][0].append(feats)
        train_by_phone[phone][1].append(int(label))

    test_by_phone: dict[str, tuple[list[list[float]], list[int]]] = {}
    for phone, feats, label in test_data:
        test_by_phone.setdefault(phone, ([], []))
        test_by_phone[phone][0].append(feats)
        test_by_phone[phone][1].append(int(label))

    # Balance training data
    train_by_phone = _add_cross_phone_negatives(train_by_phone)

    # Train per-phone SVR
    models: dict[str, SVR] = {}
    for phone, (feats, labels) in train_by_phone.items():
        feat_arr = np.array(feats)
        label_arr = np.array(labels)
        min_unique_labels = 2
        if len(np.unique(label_arr)) < min_unique_labels:
            continue
        model = SVR()
        model.fit(feat_arr, label_arr)
        models[phone] = model

    # Predict on test
    all_refs: list[float] = []
    all_hyps: list[float] = []
    per_phone_pcc: dict[str, float] = {}

    for phone, (feats, labels) in test_by_phone.items():
        if phone not in models:
            continue

        feat_arr = np.array(feats)
        ref = np.array(labels, dtype=int)
        hyp = models[phone].predict(feat_arr)
        hyp = _round_score(hyp, floor=1.0)

        all_refs.extend(ref.tolist())
        all_hyps.extend(hyp.tolist())

        min_samples = 3
        if len(ref) >= min_samples and np.std(ref) > 0 and np.std(hyp) > 0:
            per_phone_pcc[phone] = float(np.corrcoef(ref, hyp)[0, 1])

    all_refs_arr = np.array(all_refs)
    all_hyps_arr = np.array(all_hyps)

    min_eval_samples = 2
    if len(all_refs_arr) < min_eval_samples:
        logger.warning("Too few samples (%d) for PCC computation", len(all_refs_arr))
        return EvalResult(
            pcc=float("nan"),
            pcc_low=float("nan"),
            pcc_high=float("nan"),
            mse=float("nan"),
            n_phones=len(all_refs),
            per_phone_pcc=per_phone_pcc,
        )

    if np.std(all_refs_arr) == 0 or np.std(all_hyps_arr) == 0:
        logger.warning("Zero variance in scores, PCC undefined")
        mse = float(np.mean((all_refs_arr - all_hyps_arr) ** 2))
        return EvalResult(
            pcc=float("nan"),
            pcc_low=float("nan"),
            pcc_high=float("nan"),
            mse=mse,
            n_phones=len(all_refs),
            per_phone_pcc=per_phone_pcc,
        )

    res = stats.pearsonr(all_refs_arr, all_hyps_arr)
    ci = res.confidence_interval(confidence_level=0.95)
    mse = float(np.mean((all_refs_arr - all_hyps_arr) ** 2))

    return EvalResult(
        pcc=float(res.statistic),
        pcc_low=float(ci.low),
        pcc_high=float(ci.high),
        mse=mse,
        n_phones=len(all_refs),
        per_phone_pcc=per_phone_pcc,
    )
