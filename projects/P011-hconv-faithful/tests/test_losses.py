"""Tests for loss functions — masking, scale correction, gradient flow."""

from __future__ import annotations

import pytest
import torch


def test_masked_mse_padding_excluded() -> None:
    """Masked positions (mask=False) must not contribute to the loss."""
    from p011.losses import masked_mse_loss

    B, T = 4, 10
    pred = torch.zeros(B, T)
    target = torch.zeros(B, T)

    # Make mask=False positions have large pred-target error
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[:, 5:] = False
    pred[:, 5:] = 100.0   # huge error in padding — should be ignored

    loss = masked_mse_loss(pred, target, mask)
    assert loss.item() == pytest.approx(0.0, abs=1e-5), "Padding must not affect loss"


def test_masked_mse_scale_correction() -> None:
    """With 50% padding, scale factor = 2× raw MSE."""
    from p011.losses import masked_mse_loss

    B, T = 2, 10
    pred = torch.ones(B, T)
    target = torch.zeros(B, T)  # constant error of 1.0 at all valid positions

    # Mask: first half valid, second half padding
    mask = torch.zeros(B, T, dtype=torch.bool)
    mask[:, :5] = True  # 50% valid

    # Raw MSE (over all T) = (1.0² * 5 + 0 * 5) / (B*T) = 5/(B*T)
    # Scale factor = (B*T) / (B*5) = 2
    # Expected loss = 1.0 (since the scale-corrected MSE normalises to valid positions)
    loss = masked_mse_loss(pred, target, mask)
    assert loss.item() == pytest.approx(1.0, rel=1e-4)


def test_masked_mse_gradients() -> None:
    from p011.losses import masked_mse_loss

    pred = torch.randn(4, 10, requires_grad=True)
    target = torch.randn(4, 10)
    mask = torch.ones(4, 10, dtype=torch.bool)
    mask[:, 5:] = False

    loss = masked_mse_loss(pred, target, mask)
    loss.backward()
    assert pred.grad is not None


def test_conpco_returns_two_loss_terms() -> None:
    """ConPCOLoss must return two separate tensors (loss_oe, loss_clap)."""
    from p011.losses import ConPCOLoss

    loss_fn = ConPCOLoss()
    B, T, C = 4, 20, 24
    audio = torch.randn(B, T, C)
    text = torch.randn(B, T, C)
    scores = torch.rand(B, T) * 2.0  # [0, 2]
    phn_ids = torch.randint(0, 10, (B, T))

    # Make sure some positions have score == 2.0 (needed for ordinal triplets)
    scores[:, :5] = 2.0

    loss_oe, loss_clap = loss_fn(audio, text, scores, phn_ids)
    assert loss_oe.shape == (), "loss_oe must be a scalar"
    assert loss_clap.shape == (), "loss_clap must be a scalar"


def test_conpco_gradients_flow() -> None:
    """Gradients must flow through both ConPCO loss terms."""
    from p011.losses import ConPCOLoss

    loss_fn = ConPCOLoss()
    B, T, C = 4, 20, 24
    audio = torch.randn(B, T, C, requires_grad=True)
    text = torch.randn(B, T, C, requires_grad=True)
    scores = torch.rand(B, T) * 2.0
    scores[:, :5] = 2.0
    phn_ids = torch.randint(0, 10, (B, T))

    loss_oe, loss_clap = loss_fn(audio, text, scores, phn_ids)
    (loss_oe + loss_clap).backward()

    assert audio.grad is not None, "Gradient must flow to audio features"
    assert text.grad is not None, "Gradient must flow to text features"


def test_conpco_all_padding_skipped() -> None:
    """ConPCOLoss with all-zero scores must return zeros, not NaN.

    When valid_mask (scores > 0) is empty, means over empty tensors would produce NaN
    which silently poisons training. Guard added at losses.py returns (0, 0) instead.
    """
    from p011.losses import ConPCOLoss

    loss_fn = ConPCOLoss()
    B, T, C = 2, 10, 24
    audio = torch.randn(B, T, C)
    text = torch.randn(B, T, C)
    scores = torch.zeros(B, T)  # all zero → valid_mask is empty after filtering
    phn_ids = torch.randint(0, 10, (B, T))

    loss_oe, loss_clap = loss_fn(audio, text, scores, phn_ids)
    assert not torch.isnan(loss_oe), "loss_oe must not be NaN on empty batch"
    assert not torch.isnan(loss_clap), "loss_clap must not be NaN on empty batch"
    assert loss_oe.item() == pytest.approx(0.0)
    assert loss_clap.item() == pytest.approx(0.0)


def test_up_triu_off_diagonal() -> None:
    """_up_triu must return only upper-triangular off-diagonal elements."""
    from p011.losses import _up_triu

    x = torch.arange(9, dtype=torch.float).reshape(3, 3)
    result = _up_triu(x)
    # Upper-triangular (diagonal=1): positions (0,1)=1, (0,2)=2, (1,2)=5
    assert result.tolist() == pytest.approx([1.0, 2.0, 5.0])
