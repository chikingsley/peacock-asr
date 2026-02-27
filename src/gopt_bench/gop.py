"""GOP-SF-SD-Norm: Segmentation-Free Goodness of Pronunciation.

Adapted from CTC-based-GOP/taslpro26/gop_sf_sd_norm.py
Paper: "Segmentation-Free Goodness of Pronunciation" (IEEE TASLP 2026)

The algorithm computes GOP scores using CTC forward-backward without
forced alignment. For each phoneme position, it compares:
  - ll_self: CTC log-likelihood of the canonical sequence
  - ll_denom: CTC log-likelihood with the target position replaced
              by an "arbitrary" (any-phone) state

GOP = -ll_self + ll_denom  (log-likelihood ratio)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Force single thread to avoid deadlocks in multiprocessing
torch.set_num_threads(1)


@dataclass
class GOPResult:
    phones: list[str]
    scores: list[float]
    occupancies: list[float]
    features: np.ndarray | None = None  # [N_phones, 1 + V] feature matrix


def _ctc_forward(params: torch.Tensor, seq: torch.Tensor, blank: int = 0) -> float:
    """Standard CTC forward pass. Returns negative log-likelihood.

    Args:
        params: [V, T] posterior matrix (V=vocab size, T=frames)
        seq: 1-D tensor of phone indices
        blank: index of the blank token
    """
    seq_len = seq.shape[0]
    big_l = 2 * seq_len + 1
    big_t = params.shape[1]

    alphas = torch.zeros((big_l, big_t), dtype=torch.float64)
    alpha_bar = torch.zeros(big_t, dtype=torch.float64)

    alphas[0, 0] = params[blank, 0]
    alphas[1, 0] = params[seq[0], 0]
    alpha_bar[0] = torch.sum(alphas[:, 0])
    alphas[:, 0] = alphas[:, 0] / alpha_bar[0]

    for t in range(1, big_t):
        start = max(0, big_l - 1 - 2 * (big_t - t))
        for s in range(start, big_l):
            li = (s - 1) // 2
            if s % 2 == 0:
                if s == 0:
                    alphas[s, t] = alphas[s, t - 1] * params[blank, t]
                else:
                    prev = alphas[s, t - 1] + alphas[s - 1, t - 1]
                    alphas[s, t] = prev * params[blank, t]
            elif s == 1 or seq[li] == seq[li - 1]:
                prev = alphas[s, t - 1] + alphas[s - 1, t - 1]
                alphas[s, t] = prev * params[seq[li], t]
            else:
                prev = alphas[s, t - 1] + alphas[s - 1, t - 1] + alphas[s - 2, t - 1]
                alphas[s, t] = prev * params[seq[li], t]

        alpha_bar[t] = torch.sum(alphas[:, t])
        alphas[:, t] = alphas[:, t] / alpha_bar[t]

    return -torch.log(alpha_bar).sum().item()


def _check_arbitrary(
    in_alphas: torch.Tensor,
    s: int,
    t: int,
    pos: int,
    zero_pos: list[int] | None = None,
) -> float | None:
    """Check if state s is the arbitrary position."""
    if (s - 1) / 2 == pos:
        if zero_pos:
            mask = torch.ones_like(in_alphas[s, t])
            for i in zero_pos:
                mask[i] = 0
            return (in_alphas[s, t] * mask).sum().item()
        return in_alphas[s, t].sum().item()
    return None


def _ctc_forward_denom(
    params: torch.Tensor,
    seq: torch.Tensor,
    pos: int,
    blank: int = 0,
) -> tuple[float, float]:
    """CTC forward with position `pos` as arbitrary (any token).

    Returns (negative log-likelihood, occupancy).
    """
    seq_len = seq.shape[0]
    big_l = 2 * seq_len + 1
    big_t = params.shape[1]
    big_p = params.shape[0]

    alphas = torch.zeros((big_l, big_t, big_p), dtype=torch.float64)
    alpha_bar = torch.zeros(big_t, dtype=torch.float64)

    nli = seq[pos + 1].item() if pos < seq_len - 1 else None

    mask_ins = torch.eye(big_p, dtype=torch.float64)
    if nli is not None:
        mask_ins[nli, nli] = 0

    # Initialize
    if pos == 0:
        alphas[0, 0, 0] = params[blank, 0]
        alphas[2, 0, 0] = 0
        if len(seq) > 1:
            alphas[3, 0, 0] = params[seq[1], 0]
        alphas[1, 0] = params[:, 0]
        alphas[1, 0, blank] = 0
        if nli is not None:
            alphas[1, 0, nli] = 0
        alpha_bar[0] = alphas[:, 0, :].sum()
    else:
        alphas[0, 0, 0] = params[blank, 0]
        alphas[1, 0, 0] = params[seq[0], 0]
        alpha_bar[0] = alphas[0, 0, 0] + alphas[1, 0, 0]

    alphas[:, 0, :] = alphas[:, 0, :] / alpha_bar[0]

    for t in range(1, big_t):
        lo = big_l - 1 - 2 * (big_t - t)
        if (lo - 1) / 2 == pos:
            lo -= 2
        start = max(0, lo)

        for s in range(start, big_l):
            li = (s - 1) // 2
            _step_denom(
                alphas,
                params,
                seq,
                mask_ins,
                s,
                t,
                li,
                pos,
                nli,
                blank,
                big_p,
            )

        alpha_bar[t] = alphas[:, t, :].sum()
        alphas[:, t, :] = alphas[:, t, :] / alpha_bar[t]

    occ = alphas[2 * pos + 1, :, :].sum().item()
    ll_forward = -torch.log(alpha_bar).sum().item()
    return ll_forward, occ


def _step_denom(  # noqa: PLR0913
    alphas: torch.Tensor,
    params: torch.Tensor,
    seq: torch.Tensor,
    mask_ins: torch.Tensor,
    s: int,
    t: int,
    li: int,
    pos: int,
    nli: int | None,
    blank: int,
    big_p: int,  # noqa: ARG001
) -> None:
    """Single step of the denom forward pass."""
    if s % 2 == 0:
        _step_blank(alphas, params, s, t, pos, blank)
    elif pos != li and pos != li - 1:
        _step_normal(alphas, params, seq, s, t, li)
    elif pos == li - 1:
        _step_after_arb(alphas, params, seq, s, t, li, pos, blank)
    else:
        _step_arb(alphas, params, seq, mask_ins, s, t, li, nli, blank)


def _step_blank(
    alphas: torch.Tensor,
    params: torch.Tensor,
    s: int,
    t: int,
    pos: int,
    blank: int,
) -> None:
    if s == 0:
        alphas[s, t, 0] = alphas[s, t - 1, 0] * params[blank, t]
    else:
        arb = _check_arbitrary(alphas, s - 1, t - 1, pos, [blank])
        if arb is not None:
            val = (alphas[s, t - 1, 0] + arb) * params[blank, t]
            alphas[s, t, 0] = val
        else:
            prev = alphas[s, t - 1, 0] + alphas[s - 1, t - 1, 0]
            alphas[s, t, 0] = prev * params[blank, t]


def _step_normal(
    alphas: torch.Tensor,
    params: torch.Tensor,
    seq: torch.Tensor,
    s: int,
    t: int,
    li: int,
) -> None:
    if s == 1 or seq[li] == seq[li - 1]:
        prev = alphas[s, t - 1, 0] + alphas[s - 1, t - 1, 0]
        alphas[s, t, 0] = prev * params[seq[li], t]
    else:
        prev = alphas[s, t - 1, 0] + alphas[s - 1, t - 1, 0] + alphas[s - 2, t - 1, 0]
        alphas[s, t, 0] = prev * params[seq[li], t]


def _step_after_arb(
    alphas: torch.Tensor,
    params: torch.Tensor,
    seq: torch.Tensor,
    s: int,
    t: int,
    li: int,
    pos: int,
    blank: int,
) -> None:
    arb = _check_arbitrary(alphas, s - 2, t - 1, pos, [blank, seq[li].item()])
    if li - 2 < 0 or seq[li - 2] == seq[li]:
        skip_tok = 0
    else:
        skip_tok = alphas[s - 4, t - 1, 0] * params[seq[li], t]
    skip_emp = alphas[s - 3, t - 1, 0] * params[seq[li], t]
    prev = alphas[s, t - 1, 0] + alphas[s - 1, t - 1, 0] + arb
    alphas[s, t, 0] = prev * params[seq[li], t] + skip_emp + skip_tok


def _step_arb(  # noqa: PLR0913
    alphas: torch.Tensor,
    params: torch.Tensor,
    seq: torch.Tensor,
    mask_ins: torch.Tensor,
    s: int,
    t: int,
    li: int,
    nli: int | None,
    blank: int,
) -> None:
    self_trans = (
        alphas[s, t - 1, :].unsqueeze(0) * params[:, t].unsqueeze(1) * mask_ins
    ).sum(-1)

    if s == 1:
        empty = alphas[s - 1, t - 1, 0] * params[:, t]
        empty[blank] = 0
        if nli is not None:
            empty[nli] = 0
        alphas[s, t, :] = self_trans + empty
    else:
        skip = alphas[s - 2, t - 1, 0] * params[:, t]
        skip[seq[li - 1]] = 0
        skip[blank] = 0

        empty = alphas[s - 1, t - 1, 0] * params[:, t]
        empty[blank] = 0

        if nli is not None:
            empty[nli] = 0
            skip[nli] = 0

        alphas[s, t, :] = self_trans + skip + empty


def _compute_lpr_features(
    params: torch.Tensor,
    seq: torch.Tensor,
    ll_self: float,
    blank: int,
) -> np.ndarray:
    """Compute LPP + LPR feature vectors for all phone positions.

    For each phone position i, computes:
      - LPP: CTC log-likelihood of the canonical sequence
      - LPR[k] for k=0..V-1: log-likelihood ratio for substituting
        phone i with token k (k=blank means deletion)

    Follows the reference implementation in
    CTC-based-GOP/taslpro26/gop_feats_sf_sd_norm.py

    Args:
        params: [V, T] posterior matrix
        seq: 1-D tensor of phone indices
        ll_self: pre-computed CTC NLL of the canonical sequence
        blank: blank token index

    Returns:
        Feature matrix of shape [N_phones, 1 + V] where:
          col 0 = LPP (same for all phones in an utterance)
          col 1..V = LPR for each token substitution/deletion
    """
    n_phones = seq.shape[0]
    vocab_size = params.shape[0]
    features = np.zeros((n_phones, 1 + vocab_size), dtype=np.float32)

    for i in range(n_phones):
        features[i, 0] = ll_self  # LPP (stored as NLL, same as reference)

        for sub_id in range(vocab_size):
            if sub_id == blank:
                # Deletion: remove phone i from the sequence
                if n_phones == 1:
                    # Can't delete the only phone — set LPR to 0
                    features[i, 1 + sub_id] = 0.0
                    continue
                new_seq = torch.cat([seq[:i], seq[i + 1 :]])
            else:
                # Substitution: replace phone i with sub_id
                new_seq = seq.clone()
                new_seq[i] = sub_id

            ctc_modified = _ctc_forward(params, new_seq, blank=blank)
            # LPR = -LPP + CTC(modified) = difference in NLL
            features[i, 1 + sub_id] = -ll_self + ctc_modified

    return features


def compute_gop(
    posteriors: np.ndarray,
    phone_indices: list[int],
    blank: int = 0,
    *,
    extract_features: bool = False,
) -> GOPResult:
    """Compute GOP-SF-SD-Norm scores for a sequence of phones.

    Args:
        posteriors: [T, V] frame-level posterior matrix
        phone_indices: list of vocab indices for canonical phones
        blank: blank token index
        extract_features: if True, also compute the full LPP+LPR feature
            vectors per phone (slower but needed for SVR/GOPT evaluation)

    Returns:
        GOPResult with per-phone scores, occupancies, and optionally features
    """
    if len(phone_indices) == 0:
        return GOPResult(phones=[], scores=[], occupancies=[])

    post_mat = torch.from_numpy(posteriors).double()
    params = post_mat.transpose(0, 1)  # [V, T]
    seq = torch.tensor(phone_indices, dtype=torch.int32)

    ll_self = _ctc_forward(params, seq, blank=blank)

    scores = []
    occupancies = []

    for i in range(len(phone_indices)):
        ll_denom, occ = _ctc_forward_denom(params, seq, i, blank=blank)
        gop = -ll_self + ll_denom
        scores.append(gop)
        occupancies.append(occ)

    features = None
    if extract_features:
        features = _compute_lpr_features(params, seq, ll_self, blank)

    return GOPResult(
        phones=[str(idx) for idx in phone_indices],
        scores=scores,
        occupancies=occupancies,
        features=features,
    )
