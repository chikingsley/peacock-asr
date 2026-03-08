"""P002-local GOPT extensions for Track 09 ConPCO ablations."""

from __future__ import annotations

import logging
import random
from time import perf_counter
from typing import TYPE_CHECKING, Literal, overload

import torch
from torch import nn
from torch.utils.data import DataLoader

from p002_conpco.conpco_losses import CLAPContrastiveLoss, OrdinalEntropyLoss
from p002_conpco.gopt_model import (
    NUM_PHONE_CLASSES,
    SEQ_LEN,
    Block,
    GoptDataset,
    UtteranceFeats,
    _compute_norm_stats,
    _evaluate_gopt,
    _warmup_lr,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from p002_conpco.evaluate import EvalResult

logger = logging.getLogger(__name__)


class GOPTConPCOModel(nn.Module):
    """Project-local GOPT variant with ConPCO projection heads."""

    def __init__(
        self,
        input_dim: int = 42,
        embed_dim: int = 24,
        num_heads: int = 1,
        depth: int = 3,
        *,
        use_conpco: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.use_conpco = use_conpco
        self.in_proj = nn.Linear(input_dim, embed_dim)
        self.phn_proj = nn.Linear(NUM_PHONE_CLASSES, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, SEQ_LEN, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads) for _ in range(depth)],
        )
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1),
        )
        if use_conpco:
            self.phn_audio_proj = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(embed_dim, embed_dim),
                nn.Dropout(0.1),
            )
            self.phn_text_proj = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(embed_dim, embed_dim),
                nn.Dropout(0.1),
            )

    @overload
    def forward(
        self,
        x: torch.Tensor,
        phn: torch.Tensor,
        *,
        return_projections: Literal[False] = False,
    ) -> torch.Tensor: ...

    @overload
    def forward(
        self,
        x: torch.Tensor,
        phn: torch.Tensor,
        *,
        return_projections: Literal[True],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

    def forward(
        self,
        x: torch.Tensor,
        phn: torch.Tensor,
        *,
        return_projections: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        phn_one_hot = torch.nn.functional.one_hot(
            (phn + 1).clamp(min=0).long(),
            num_classes=NUM_PHONE_CLASSES,
        ).float()
        phn_embed = self.phn_proj(phn_one_hot)
        x = self.in_proj(x) + phn_embed
        x = x + self.pos_embed

        first_block_out = None
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i == 0 and return_projections:
                first_block_out = x

        scores = self.head(x)

        if return_projections:
            audio_proj = self.phn_audio_proj(first_block_out)
            text_proj = self.phn_text_proj(phn_embed + self.pos_embed)
            return scores, audio_proj, text_proj

        return scores


def train_and_evaluate_gopt_conpco(  # noqa: PLR0912, PLR0913, PLR0915
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
    seed: int | None = None,
    use_ordinal_entropy: bool = True,
    use_clap: bool = False,
    w_mse: float = 1.0,
    w_oe: float = 1.0,
    w_clap: float = 1.0,
    lambda_d: float = 0.5,
    lambda_t: float = 0.1,
    margin: float = 1.0,
    lambda_clap_t2a: float = 0.5,
    history_out: list[dict[str, float]] | None = None,
    on_epoch_end: Callable[..., None] | None = None,
) -> EvalResult:
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

    use_conpco = use_ordinal_entropy or use_clap
    model = GOPTConPCOModel(
        input_dim=input_dim,
        embed_dim=embed_dim,
        num_heads=num_heads,
        depth=depth,
        use_conpco=use_conpco,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "GOPT: %d params, embed=%d, heads=%d, depth=%d, conpco=%s",
        n_params,
        embed_dim,
        num_heads,
        depth,
        use_conpco,
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
    mse_fn = nn.MSELoss()
    oe_fn = (
        OrdinalEntropyLoss(lambda_d=lambda_d, lambda_t=lambda_t, margin=margin)
        if use_ordinal_entropy
        else None
    )
    clap_fn = (
        CLAPContrastiveLoss(lambda_clap_t2a=lambda_clap_t2a)
        if use_clap
        else None
    )

    global_step = 0
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_oe = 0.0
        epoch_clap = 0.0
        n_batches = 0
        epoch_start = perf_counter()

        for feats_cpu, phns_cpu, scores_cpu in train_loader:
            feats = feats_cpu.to(device)
            phns = phns_cpu.to(device)
            scores = scores_cpu.to(device)

            _warmup_lr(optimizer, global_step, lr)

            if use_conpco:
                pred, audio_proj, text_proj = model(
                    feats,
                    phns,
                    return_projections=True,
                )
            else:
                pred = model(feats, phns)
                audio_proj = text_proj = None

            pred = pred.squeeze(2)
            mask = (scores >= 0).float()
            pred_masked = pred * mask
            scores_masked = scores * mask
            loss_mse = mse_fn(pred_masked, scores_masked)
            n_valid = mask.sum()
            if n_valid > 0:
                loss_mse = loss_mse * (mask.numel() / n_valid)

            loss = w_mse * loss_mse

            loss_oe_val = torch.tensor(0.0, device=device)
            if oe_fn is not None:
                loss_oe_val = oe_fn(audio_proj, scores, phns)
                loss = loss + w_oe * loss_oe_val

            loss_clap_val = torch.tensor(0.0, device=device)
            if clap_fn is not None:
                loss_clap_val = clap_fn(audio_proj, text_proj, scores, phns)
                loss = loss + w_clap * loss_clap_val

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_mse += loss_mse.item()
            epoch_oe += loss_oe_val.item()
            epoch_clap += loss_clap_val.item()
            n_batches += 1
            global_step += 1

        scheduler.step()
        n_b = max(n_batches, 1)
        avg_loss = epoch_loss / n_b
        avg_mse = epoch_mse / n_b
        avg_oe = epoch_oe / n_b
        avg_clap = epoch_clap / n_b
        epoch_time_sec = perf_counter() - epoch_start
        epoch_lr = float(optimizer.param_groups[0]["lr"])

        if history_out is not None:
            history_out.append(
                {
                    "epoch": float(epoch + 1),
                    "loss": avg_loss,
                    "loss_mse": avg_mse,
                    "loss_oe": avg_oe,
                    "loss_clap": avg_clap,
                    "lr": epoch_lr,
                    "epoch_time_sec": epoch_time_sec,
                },
            )

        if on_epoch_end is not None:
            on_epoch_end(epoch + 1, n_epochs, avg_loss, epoch_lr, epoch_time_sec)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            logger.info(
                "Epoch %d/%d  loss=%.4f (mse=%.4f oe=%.4f clap=%.4f)  lr=%.2e",
                epoch + 1,
                n_epochs,
                avg_loss,
                avg_mse,
                avg_oe,
                avg_clap,
                epoch_lr,
            )

    return _evaluate_gopt(model, test_loader, device)
