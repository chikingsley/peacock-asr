"""Tests for eval.py — PCC vs scipy reference, shapes, MDD metrics."""

from __future__ import annotations

import pytest
import torch
from scipy import stats


def test_eval_phn_pcc_matches_scipy() -> None:
    """eval_phn PCC must match scipy.stats.pearsonr on valid positions."""
    from p010.eval import eval_phn

    rng = torch.Generator()
    rng.manual_seed(0)
    N = 100
    pred = torch.randn(N, 50, generator=rng)
    target = pred + torch.randn(N, 50, generator=rng) * 0.3

    # Padding: last 20 positions
    target[:, 30:] = -1.0

    mse, pcc = eval_phn(pred, target)

    # Reference: scipy
    mask = target.numpy() >= 0
    r, _ = stats.pearsonr(pred.numpy()[mask], target.numpy()[mask])

    assert pcc == pytest.approx(r, abs=1e-4)
    assert isinstance(mse, float)
    assert isinstance(pcc, float)


def test_eval_utt_five_aspects() -> None:
    """eval_utt must return 5 MSE and 5 PCC values."""
    from p010.eval import eval_utt

    N = 50
    pred = torch.randn(N, 5)
    target = pred + torch.randn(N, 5) * 0.1

    mse_list, pcc_list = eval_utt(pred, target)
    assert len(mse_list) == 5
    assert len(pcc_list) == 5

    for pcc in pcc_list:
        assert 0.5 < pcc <= 1.0, "PCC should be high when pred ≈ target"


def test_eval_utt_perfect_prediction() -> None:
    """Perfect predictions must give MSE=0 and PCC=1."""
    from p010.eval import eval_utt

    N = 50
    pred = torch.randn(N, 5)
    mse_list, pcc_list = eval_utt(pred, pred)

    for mse in mse_list:
        assert mse == pytest.approx(0.0, abs=1e-5)
    for pcc in pcc_list:
        assert pcc == pytest.approx(1.0, abs=1e-5)


def test_eval_word_aggregation_shape() -> None:
    """eval_word must return 3 MSE and 3 PCC values, and array shapes."""
    from p010.eval import eval_word

    N, T = 20, 50
    pred = torch.randn(N, T, 3)
    # Build target with word_pos in last column
    target = torch.full((N, T, 4), -1.0)
    target[:, :20, :3] = torch.randn(N, 20, 3)
    for i in range(N):
        target[i, :20, 3] = torch.arange(20, dtype=torch.float) // 4  # word positions

    mse_list, pcc_list, pred_arr, target_arr = eval_word(pred, target)

    assert len(mse_list) == 3
    assert len(pcc_list) == 3
    assert pred_arr.shape[1] == 3
    assert target_arr.shape[1] == 3


def test_eval_mdd_known_labels() -> None:
    """eval_mdd with known binary predictions must return correct F1."""
    from p010.eval import eval_mdd

    # 10 valid positions: 5 true=1, 5 true=0, all correctly predicted
    N, T = 2, 10
    logit = torch.zeros(N, T, 1)
    label = torch.zeros(N, T)

    # First 5 → mispronounced (label=1, logit >> 0)
    label[:, :5] = 1.0
    logit[:, :5, 0] = 10.0    # sigmoid → 1
    logit[:, 5:, 0] = -10.0   # sigmoid → 0 → predicted correct

    result = eval_mdd(logit, label)
    assert result["f1"] == pytest.approx(1.0, abs=1e-4)
    assert result["precision"] == pytest.approx(1.0, abs=1e-4)
    assert result["recall"] == pytest.approx(1.0, abs=1e-4)


def test_eval_mdd_padding_excluded() -> None:
    """Padding positions (label == -1) must not affect MDD metrics."""
    from p010.eval import eval_mdd

    N, T = 2, 10
    logit = torch.zeros(N, T, 1)
    label = torch.full((N, T), -1.0)  # all padding

    # Set only a few valid positions
    label[:, :4] = 0.0
    logit[:, :4, 0] = -10.0  # sigmoid → 0 → predicted correctly as 0

    result = eval_mdd(logit, label)
    # All correct → recall of positive class is 0 (no positives), no crash
    assert "f1" in result
    assert "precision" in result
    assert "recall" in result


def test_eval_phn_mse_zero_perfect() -> None:
    """eval_phn with pred == target must give MSE=0."""
    from p010.eval import eval_phn

    pred = torch.rand(10, 50) * 2.0
    target = pred.clone()

    mse, _ = eval_phn(pred, target)
    assert mse == pytest.approx(0.0, abs=1e-5)


def test_eval_phn_single_sample_constant_pcc_is_finite() -> None:
    """Single-sample constant inputs have undefined Pearson correlation; return 0.0."""
    from p010.eval import eval_phn

    pred = torch.ones(1, 50)
    target = torch.ones(1, 50)

    mse, pcc = eval_phn(pred, target)
    assert mse == pytest.approx(0.0, abs=1e-5)
    assert pcc == pytest.approx(0.0, abs=1e-5)


def test_eval_utt_single_sample_constant_pcc_is_finite() -> None:
    """Utterance PCC should stay finite for 1-sample debug runs."""
    from p010.eval import eval_utt

    pred = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    target = pred.clone()

    mse_list, pcc_list = eval_utt(pred, target)
    assert mse_list == pytest.approx([0.0] * 5, abs=1e-5)
    assert pcc_list == pytest.approx([0.0] * 5, abs=1e-5)


def test_eval_word_single_word_constant_pcc_is_finite() -> None:
    """Word-level PCC should stay finite when a tiny debug batch aggregates to one word."""
    from p010.eval import eval_word

    pred = torch.ones(1, 50, 3)
    target = torch.full((1, 50, 4), -1.0)
    target[0, :2, :3] = 1.0
    target[0, 0, 3] = 0.0
    target[0, 1, 3] = 1.0

    mse_list, pcc_list, _, _ = eval_word(pred, target)
    assert mse_list == pytest.approx([0.0] * 3, abs=1e-5)
    assert pcc_list == pytest.approx([0.0] * 3, abs=1e-5)
