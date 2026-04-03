from __future__ import annotations

import torch
import torch.nn.functional as F


def cross_entropy_lsm(
    logits: torch.Tensor,
    ys: torch.Tensor,
    lsm_prob: float,
    ignore_index: int,
    training: bool,
    normalize_length: bool = False,
) -> torch.Tensor:
    bs, _, vocab = logits.size()
    ys = ys.view(-1)
    logits = logits.view((-1, vocab))

    if lsm_prob == 0 or not training:
        loss = F.cross_entropy(logits, ys, ignore_index=ignore_index, reduction="mean")
        if not normalize_length:
            loss *= (ys != ignore_index).sum() / float(bs)
        return loss

    with torch.no_grad():
        target_dist = logits.new_zeros(logits.size())
        target_dist.fill_(lsm_prob / (vocab - 1))
        mask = ys == ignore_index
        ys_masked = ys.masked_fill(mask, 0)
        target_dist.scatter_(1, ys_masked.unsqueeze(1), 1 - lsm_prob)

    log_probs = torch.log_softmax(logits, dim=-1)
    loss_sum = -torch.mul(target_dist, log_probs)
    n_tokens = len(ys) - mask.sum().item()
    denom = n_tokens if normalize_length else bs
    return loss_sum.masked_fill(mask.unsqueeze(1), 0).sum() / denom


def decoupled_cross_entropy_lsm(
    logits: torch.Tensor,
    realphns: torch.Tensor,
    canophns: torch.Tensor,
    lsm_prob_m: float = 0.1,
    lsm_prob_c: float = 0.0,
    a: float = 0.70,
    ignore_index: int = -1,
    training: bool = True,
    num_correct: int | None = None,
    num_mispronounced: int | None = None,
) -> torch.Tensor:
    cor_mask = canophns == realphns
    mis_mask = canophns != realphns
    cor_realphns = realphns.masked_fill(mis_mask, ignore_index)
    mis_realphns = realphns.masked_fill(cor_mask, ignore_index)

    if num_correct is None:
        num_correct = int(cor_mask.sum().item())
    if num_mispronounced is None:
        num_mispronounced = int(mis_mask.sum().item())

    if num_mispronounced <= 0:
        w_mis = logits.new_tensor(0.0)
    else:
        w_mis = logits.new_tensor((num_correct / num_mispronounced) ** a)

    loss_cor = cross_entropy_lsm(
        logits,
        cor_realphns.long(),
        lsm_prob=lsm_prob_c,
        ignore_index=ignore_index,
        training=training,
    )
    loss_mis = cross_entropy_lsm(
        logits,
        mis_realphns.long(),
        lsm_prob=lsm_prob_m,
        ignore_index=ignore_index,
        training=training,
    )
    return loss_cor + w_mis * loss_mis
