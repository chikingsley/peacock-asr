"""Phone-level HMamba-style scorer adapted to the P001 feature contract.

This is not a full reproduction of the upstream HMamba training recipe.
P001 only has phone-level GOP feature tensors plus canonical phone IDs, so this
module ports the scorer architecture down to the assets we actually have:

- one GOP feature stream per phone
- canonical phone sequence
- phone-level score targets

The goal is to compare a Mamba-inspired scorer head against GOPT on the same
cached features and labels.
"""

from __future__ import annotations

import json
import logging
import random
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
import torch
from scipy import stats
from torch import nn
from torch.utils.data import DataLoader

from p001_gop.evaluate import EvalResult
from p001_gop.gopt_model import (
    NUM_PHONE_CLASSES,
    SEQ_LEN,
    GoptDataset,
    UtteranceFeats,
    _compute_norm_stats,
    _warmup_lr,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

logger = logging.getLogger(__name__)

MIN_PCC_SAMPLES = 2
MIN_PER_PHONE_PCC_SAMPLES = 3


class PredictionHead(nn.Module):
    """Upstream HMamba-style prediction head."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.dense = nn.Linear(in_dim, in_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return self.linear(x)


class Attention(nn.Module):
    """Single-head attention block used by the upstream transformer fallback."""

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
    """Two-layer feed-forward block."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class HMambaBlock(nn.Module):
    """Transformer-mode block from the upstream HMamba architecture."""

    def __init__(self, dim: int, num_heads: int = 1, drop: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.model = Attention(dim=dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim=dim, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.model(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class HMambaPhoneModel(nn.Module):
    """Phone-level HMamba adaptation for P001.

    This keeps the canonical phone embedding and deeper HMamba-style block
    stack, but only retains the phone-level scoring head because P001 does not
    carry word- or utterance-level supervision.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        embed_dim: int = 24,
        num_heads: int = 1,
        depth: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(input_dim, embed_dim)
        self.phn_embed = nn.Embedding(
            NUM_PHONE_CLASSES,
            embed_dim,
            padding_idx=0,
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, SEQ_LEN, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.feat_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            HMambaBlock(dim=embed_dim, num_heads=num_heads, drop=dropout)
            for _ in range(depth)
        )
        self.phn_score = PredictionHead(embed_dim, 1, dropout=dropout)

    def forward(self, x: torch.Tensor, phn: torch.Tensor) -> torch.Tensor:
        phn_embed = self.phn_embed((phn + 1).clamp(min=0).long())
        x = self.feat_drop(self.in_proj(x)) + phn_embed + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return self.phn_score(x)


def train_and_evaluate_hmamba(  # noqa: PLR0913, PLR0915
    train_utts: list[UtteranceFeats],
    test_utts: list[UtteranceFeats],
    *,
    input_dim: int,
    embed_dim: int = 24,
    num_heads: int = 1,
    depth: int = 5,
    n_epochs: int = 100,
    batch_size: int = 25,
    lr: float = 1e-3,
    dropout: float = 0.1,
    device: torch.device | None = None,
    seed: int | None = None,
    history_out: list[dict[str, float]] | None = None,
    checkpoint_dir: Path | None = None,
    on_epoch_end: Callable[[int, int, float, float, float], None] | None = None,
) -> EvalResult:
    """Train the phone-level HMamba scorer and evaluate on the test set."""
    if device is None:
        device = torch.device("cpu")

    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    norm_mean, norm_std = _compute_norm_stats(train_utts)
    if norm_std == 0:
        norm_std = 1.0
    logger.info("Norm stats: mean=%.3f, std=%.3f", norm_mean, norm_std)

    train_ds = GoptDataset(train_utts, norm_mean, norm_std, input_dim)
    test_ds = GoptDataset(test_utts, norm_mean, norm_std, input_dim)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = HMambaPhoneModel(
        input_dim=input_dim,
        embed_dim=embed_dim,
        num_heads=num_heads,
        depth=depth,
        dropout=dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "HMamba: %d params, embed=%d, heads=%d, depth=%d, dropout=%.2f",
        n_params,
        embed_dim,
        num_heads,
        depth,
        dropout,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=5e-7,
        betas=(0.95, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=list(range(20, n_epochs, 5)),
        gamma=0.5,
    )
    loss_fn = nn.MSELoss()

    global_step = 0
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        epoch_start = perf_counter()

        for feats_cpu, phns_cpu, scores_cpu in train_loader:
            feats = feats_cpu.to(device)
            phns = phns_cpu.to(device)
            scores = scores_cpu.to(device)

            _warmup_lr(optimizer, global_step, lr)

            pred = model(feats, phns).squeeze(2)
            mask = (scores >= 0).float()
            pred_masked = pred * mask
            scores_masked = scores * mask

            loss = loss_fn(pred_masked, scores_masked)
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
        avg_loss = epoch_loss / max(n_batches, 1)
        epoch_time_sec = perf_counter() - epoch_start
        epoch_lr = float(optimizer.param_groups[0]["lr"])

        if history_out is not None:
            history_out.append(
                {
                    "epoch": float(epoch + 1),
                    "loss": float(avg_loss),
                    "lr": epoch_lr,
                    "epoch_time_sec": float(epoch_time_sec),
                }
            )

        if on_epoch_end is not None:
            on_epoch_end(
                epoch + 1,
                n_epochs,
                float(avg_loss),
                epoch_lr,
                float(epoch_time_sec),
            )

        if (epoch + 1) % 20 == 0 or epoch == 0:
            logger.info(
                "Epoch %d/%d  loss=%.4f  lr=%.2e",
                epoch + 1,
                n_epochs,
                avg_loss,
                epoch_lr,
            )

    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "hmamba_model.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "input_dim": input_dim,
                "embed_dim": embed_dim,
                "num_heads": num_heads,
                "depth": depth,
                "n_epochs": n_epochs,
                "batch_size": batch_size,
                "lr": lr,
                "dropout": dropout,
                "seed": seed,
                "norm_mean": norm_mean,
                "norm_std": norm_std,
            },
            checkpoint_path,
        )
        config_path = checkpoint_dir / "hmamba_config.json"
        config_path.write_text(
            json.dumps(
                {
                    "input_dim": input_dim,
                    "embed_dim": embed_dim,
                    "num_heads": num_heads,
                    "depth": depth,
                    "n_epochs": n_epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "dropout": dropout,
                    "seed": seed,
                    "norm_mean": norm_mean,
                    "norm_std": norm_std,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    eval_result = _evaluate_hmamba(model, test_loader, device)

    if checkpoint_dir is not None:
        metrics_path = checkpoint_dir / "eval_metrics.json"
        metrics_path.write_text(
            json.dumps(
                {
                    "pcc": eval_result.pcc,
                    "pcc_low": eval_result.pcc_low,
                    "pcc_high": eval_result.pcc_high,
                    "mse": eval_result.mse,
                    "n_phones": eval_result.n_phones,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    return eval_result


def _evaluate_hmamba(
    model: HMambaPhoneModel,
    test_loader: DataLoader,
    device: torch.device,
) -> EvalResult:
    model.eval()
    all_preds: list[float] = []
    all_refs: list[float] = []
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
    if len(refs_arr) < MIN_PCC_SAMPLES:
        logger.warning("Too few samples for PCC")
        return EvalResult(
            pcc=float("nan"),
            pcc_low=float("nan"),
            pcc_high=float("nan"),
            mse=float("nan"),
            n_phones=len(all_refs),
            per_phone_pcc={},
        )

    mse = float(np.mean((refs_arr - preds_arr) ** 2))
    if np.std(refs_arr) == 0 or np.std(preds_arr) == 0:
        return EvalResult(
            pcc=float("nan"),
            pcc_low=float("nan"),
            pcc_high=float("nan"),
            mse=mse,
            n_phones=len(all_refs),
            per_phone_pcc={},
        )

    res = stats.pearsonr(refs_arr, preds_arr)
    ci = res.confidence_interval(confidence_level=0.95)
    per_phone_pcc: dict[str, float] = {}

    from p001_gop.gopt_model import PHONE_TO_ID  # noqa: PLC0415

    id_to_phone = {v: k for k, v in PHONE_TO_ID.items()}
    for pid, preds in per_phone_preds.items():
        refs = per_phone_refs[pid]
        if (
            len(refs) >= MIN_PER_PHONE_PCC_SAMPLES
            and np.std(refs) > 0
            and np.std(preds) > 0
        ):
            phone_name = id_to_phone.get(pid, str(pid))
            per_phone_pcc[phone_name] = float(np.corrcoef(refs, preds)[0, 1])

    return EvalResult(
        pcc=float(res.statistic),
        pcc_low=float(ci.low),
        pcc_high=float(ci.high),
        mse=mse,
        n_phones=len(all_refs),
        per_phone_pcc=per_phone_pcc,
    )
