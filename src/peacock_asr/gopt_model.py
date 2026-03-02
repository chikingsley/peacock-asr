"""GOPT: Goodness Of Pronunciation Transformer (phone-level).

Adapted from references/gopt-transformer/src/models/gopt.py (Gong et al.)
and references/gopt-transformer/src/traintest.py.

Paper: "GOPT: Generalized Goodness of Pronunciation with Transformer"
       (Gong et al., ICASSP 2022)

This is a phone-level-only adaptation — no utterance-level CLS tokens
or word-level heads. Those can be added later.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from scipy import stats
from torch import nn
from torch.utils.data import DataLoader, Dataset

from peacock_asr.backends.ctc_gop_original import ARPABET_VOCAB
from peacock_asr.evaluate import EvalResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Phone-to-ID mapping
# ---------------------------------------------------------------------------

# ARPABET_VOCAB[0] = "<pad>", [1:] = 39 phones.
# PHONE_TO_ID maps phone name → 0..38. Padding uses -1.
# In forward(), one_hot(phn + 1, 40) maps: -1→class0 (pad), 0→class1 (AA), etc.
PHONE_TO_ID: dict[str, int] = {
    phone: i for i, phone in enumerate(ARPABET_VOCAB[1:])
}

SEQ_LEN = 50  # max phones per utterance (matches reference)
NUM_PHONE_CLASSES = 40  # 39 phones + 1 pad class (after +1 shift)
FEATURE_RANK = 2


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class UtteranceFeats:
    """Per-utterance GOP features collected during process_split."""

    phones: list[str]  # ARPABET names (valid only)
    feat_vecs: list[list[float]]  # parallel, each [1+V+1] dim
    scores: list[float]  # parallel, human scores 0-2


# ---------------------------------------------------------------------------
# Model components (adapted from reference gopt.py)
# ---------------------------------------------------------------------------


class Attention(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, dim: int, num_heads: int = 1) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = (
            self.qkv(x)
            .reshape(b, n, 3, self.num_heads, c // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj(x)


class Mlp(nn.Module):
    """Two-layer MLP with GELU activation."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    """Pre-LN transformer block."""

    def __init__(self, dim: int, num_heads: int = 1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class GOPTModel(nn.Module):
    """Phone-level GOPT transformer.

    Takes padded sequences of GOP feature vectors + canonical phone IDs,
    outputs per-phone score predictions.
    """

    def __init__(
        self,
        input_dim: int = 42,
        embed_dim: int = 24,
        num_heads: int = 1,
        depth: int = 3,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.in_proj = nn.Linear(input_dim, embed_dim)
        self.phn_proj = nn.Linear(NUM_PHONE_CLASSES, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, SEQ_LEN, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads) for _ in range(depth)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1),
        )

    def forward(
        self, x: torch.Tensor, phn: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, SEQ_LEN, input_dim] — padded GOP features
            phn: [B, SEQ_LEN] — phone IDs (0..38), -1 for padding

        Returns:
            [B, SEQ_LEN, 1] — per-phone score predictions
        """
        phn_one_hot = torch.nn.functional.one_hot(
            (phn + 1).clamp(min=0).long(), num_classes=NUM_PHONE_CLASSES,
        ).float()
        phn_embed = self.phn_proj(phn_one_hot)
        x = self.in_proj(x) + phn_embed
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class GoptDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Pads utterances to SEQ_LEN and normalizes features."""

    def __init__(
        self,
        utts: list[UtteranceFeats],
        norm_mean: float,
        norm_std: float,
        input_dim: int,
    ) -> None:
        self.feats = torch.zeros(len(utts), SEQ_LEN, input_dim)
        self.phn_ids = torch.full((len(utts), SEQ_LEN), -1, dtype=torch.long)
        self.scores = torch.full(
            (len(utts), SEQ_LEN), -1.0, dtype=torch.float,
        )
        width_mismatch_count = 0

        for i, utt in enumerate(utts):
            n = min(
                len(utt.phones), len(utt.feat_vecs), len(utt.scores), SEQ_LEN,
            )
            if n == 0:
                continue

            feat_arr = torch.tensor(utt.feat_vecs[:n], dtype=torch.float)
            if feat_arr.ndim != FEATURE_RANK:
                msg = (
                    "GOPT feature array must be rank-2 "
                    f"(got shape {tuple(feat_arr.shape)})"
                )
                raise ValueError(msg)

            feat_width = int(feat_arr.shape[1])
            if feat_width != input_dim:
                width_mismatch_count += 1
                adjusted = torch.zeros(n, input_dim, dtype=torch.float)
                copy_width = min(feat_width, input_dim)
                adjusted[:, :copy_width] = feat_arr[:, :copy_width]
                feat_arr = adjusted

            # z-score normalize (single scalar mean/std, matching reference)
            self.feats[i, :n] = (feat_arr - norm_mean) / norm_std

            for j in range(n):
                pid = PHONE_TO_ID.get(utt.phones[j], -1)
                self.phn_ids[i, j] = pid
                self.scores[i, j] = utt.scores[j]

        if width_mismatch_count > 0:
            logger.warning(
                "GOPT dataset adjusted feature width for %d utterances "
                "(expected input_dim=%d)",
                width_mismatch_count, input_dim,
            )

    def __len__(self) -> int:
        return self.feats.shape[0]

    def __getitem__(
        self,
        index: object,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(index, int):
            msg = f"GoptDataset index must be int, got {type(index).__name__}"
            raise TypeError(msg)
        return self.feats[index], self.phn_ids[index], self.scores[index]


# ---------------------------------------------------------------------------
# Normalization stats
# ---------------------------------------------------------------------------


def _compute_norm_stats(utts: list[UtteranceFeats]) -> tuple[float, float]:
    """Compute mean and std over all valid feature values in training set."""
    all_vals: list[float] = []
    for utt in utts:
        for fv in utt.feat_vecs:
            all_vals.extend(fv)
    arr = np.array(all_vals, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------


def _warmup_lr(
    optimizer: torch.optim.Optimizer,
    step: int,
    target_lr: float,
    warmup_steps: int = 100,
) -> None:
    """Linear warmup for the first `warmup_steps` steps."""
    if step < warmup_steps:
        lr = target_lr * (step + 1) / warmup_steps
        for pg in optimizer.param_groups:
            pg["lr"] = lr


def train_and_evaluate_gopt(  # noqa: PLR0913
    train_utts: list[UtteranceFeats],
    test_utts: list[UtteranceFeats],
    *,
    input_dim: int,
    embed_dim: int = 24,
    num_heads: int = 1,
    depth: int = 3,
    n_epochs: int = 100,
    batch_size: int = 25,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> EvalResult:
    """Train GOPT and evaluate on test set.

    Returns:
        EvalResult with PCC and per-phone PCC.
    """
    if device is None:
        device = torch.device("cpu")

    # Normalization stats from training data
    norm_mean, norm_std = _compute_norm_stats(train_utts)
    if norm_std == 0:
        norm_std = 1.0
    logger.info("Norm stats: mean=%.3f, std=%.3f", norm_mean, norm_std)

    # Datasets and loaders
    train_ds = GoptDataset(train_utts, norm_mean, norm_std, input_dim)
    test_ds = GoptDataset(test_utts, norm_mean, norm_std, input_dim)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Model
    model = GOPTModel(
        input_dim=input_dim,
        embed_dim=embed_dim,
        num_heads=num_heads,
        depth=depth,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "GOPT: %d params, embed=%d, heads=%d, depth=%d",
        n_params, embed_dim, num_heads, depth,
    )

    # Optimizer and scheduler (matching reference)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=5e-7, betas=(0.95, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(range(20, n_epochs, 5)), gamma=0.5,
    )
    loss_fn = nn.MSELoss()

    # Training loop
    global_step = 0
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for feats_cpu, phns_cpu, scores_cpu in train_loader:
            feats = feats_cpu.to(device)
            phns = phns_cpu.to(device)
            scores = scores_cpu.to(device)

            _warmup_lr(optimizer, global_step, lr)

            pred = model(feats, phns).squeeze(2)

            # Mask padding (scores == -1)
            mask = (scores >= 0).float()
            pred_masked = pred * mask
            scores_masked = scores * mask

            loss = loss_fn(pred_masked, scores_masked)
            # Rescale: MSE averages over all positions, correct for padding
            n_valid = mask.sum()
            if n_valid > 0:
                loss = loss * (mask.numel() / n_valid)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

        scheduler.step()

        if (epoch + 1) % 20 == 0 or epoch == 0:
            avg_loss = epoch_loss / max(n_batches, 1)
            logger.info(
                "Epoch %d/%d  loss=%.4f  lr=%.2e",
                epoch + 1, n_epochs, avg_loss,
                optimizer.param_groups[0]["lr"],
            )

    # Evaluation
    return _evaluate_gopt(model, test_loader, device)


def _evaluate_gopt(
    model: GOPTModel,
    test_loader: DataLoader,
    device: torch.device,
) -> EvalResult:
    """Evaluate trained GOPT on test set, return EvalResult."""
    model.eval()
    all_preds: list[float] = []
    all_refs: list[float] = []
    # For per-phone PCC, track by phone ID
    per_phone_preds: dict[int, list[float]] = {}
    per_phone_refs: dict[int, list[float]] = {}

    with torch.no_grad():
        for feats_cpu, phns_cpu, scores_cpu in test_loader:
            feats = feats_cpu.to(device)
            phns = phns_cpu.to(device)

            pred = model(feats, phns).squeeze(2).cpu()

            for i in range(pred.shape[0]):
                for j in range(pred.shape[1]):
                    if scores_cpu[i, j] >= 0:
                        p = float(pred[i, j])
                        r = float(scores_cpu[i, j])
                        all_preds.append(p)
                        all_refs.append(r)

                        pid = int(phns_cpu[i, j])
                        per_phone_preds.setdefault(pid, []).append(p)
                        per_phone_refs.setdefault(pid, []).append(r)

    preds_arr = np.array(all_preds)
    refs_arr = np.array(all_refs)

    min_samples = 2
    if len(refs_arr) < min_samples:
        logger.warning("Too few samples for PCC")
        return EvalResult(
            pcc=float("nan"), pcc_low=float("nan"),
            pcc_high=float("nan"), mse=float("nan"),
            n_phones=len(all_refs), per_phone_pcc={},
        )

    mse = float(np.mean((refs_arr - preds_arr) ** 2))

    if np.std(refs_arr) == 0 or np.std(preds_arr) == 0:
        return EvalResult(
            pcc=float("nan"), pcc_low=float("nan"),
            pcc_high=float("nan"), mse=mse,
            n_phones=len(all_refs), per_phone_pcc={},
        )

    res = stats.pearsonr(refs_arr, preds_arr)
    ci = res.confidence_interval(confidence_level=0.95)

    # Per-phone PCC (map phone ID back to name)
    id_to_phone = {v: k for k, v in PHONE_TO_ID.items()}
    per_phone_pcc: dict[str, float] = {}
    min_per_phone = 3
    for pid, preds in per_phone_preds.items():
        refs = per_phone_refs[pid]
        if len(refs) >= min_per_phone and np.std(refs) > 0 and np.std(preds) > 0:
            phone_name = id_to_phone.get(pid, str(pid))
            per_phone_pcc[phone_name] = float(
                np.corrcoef(refs, preds)[0, 1],
            )

    return EvalResult(
        pcc=float(res.statistic),
        pcc_low=float(ci.low),
        pcc_high=float(ci.high),
        mse=mse,
        n_phones=len(all_refs),
        per_phone_pcc=per_phone_pcc,
    )
