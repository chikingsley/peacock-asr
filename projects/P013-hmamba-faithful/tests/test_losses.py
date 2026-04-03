from __future__ import annotations

import torch

from p012.losses import cross_entropy_lsm, decoupled_cross_entropy_lsm


def test_decoupled_loss_uses_dataset_level_priors() -> None:
    logits = torch.tensor([[[2.0, 0.5], [0.5, 2.0]]], dtype=torch.float32)
    realphns = torch.tensor([[0, 1]])
    canophns = torch.tensor([[0, 0]])

    loss = decoupled_cross_entropy_lsm(
        logits,
        realphns,
        canophns,
        a=0.5,
        training=True,
        num_correct=9,
        num_mispronounced=1,
        lsm_prob_m=0.0,
        lsm_prob_c=0.0,
    )

    correct_only = cross_entropy_lsm(logits, torch.tensor([[0, -1]]), lsm_prob=0.0, ignore_index=-1, training=True)
    mis_only = cross_entropy_lsm(logits, torch.tensor([[-1, 1]]), lsm_prob=0.0, ignore_index=-1, training=True)
    expected = correct_only + (9.0**0.5) * mis_only
    assert torch.allclose(loss, expected)


def test_decoupled_loss_handles_no_mispronunciations() -> None:
    logits = torch.randn(1, 3, 4)
    realphns = torch.tensor([[1, 2, 3]])
    canophns = torch.tensor([[1, 2, 3]])
    loss = decoupled_cross_entropy_lsm(
        logits,
        realphns,
        canophns,
        training=True,
        num_correct=100,
        num_mispronounced=0,
    )
    assert torch.isfinite(loss)
