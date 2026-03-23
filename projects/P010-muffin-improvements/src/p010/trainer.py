"""Training loop for P010.

Ported from ConPCO/src/traintest_eng_dur_ssl_3m_HierBFR_conPCO_norm.py (train + validate functions).

Modernizations vs reference:
- Warmup: manual `global_step % 5 == 0` every-5-steps discrete jumps → smooth per-step warmup.
- Logging: np.savetxt CSV → W&B (wandb.log per epoch).
- DataParallel: removed; single GPU assumed.
- Bug fix: reference prints loss_phn_pco unconditionally even when conpco=False (would crash).
  Our version only logs ConPCO losses when use_conpco=True.
- Checkpoint: saves best by phone MSE, matches reference.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from p010.eval import eval_mdd, eval_phn, eval_utt, eval_word
from p010.losses import ConPCOLoss, masked_mse_loss
from p010.models.hiercb import HierCB
from p010.settings import Settings


def train_one_config(
    settings: Settings,
    model: HierCB,
    train_loader: DataLoader,
    test_loader: DataLoader,
    run_name: str | None = None,
    checkpoint_dir: Path | None = None,
    pretrained: Path | None = None,
) -> float:
    """Train HierCB for one seed configuration. Returns best phone PCC.

    Args:
        settings:        Experiment hyperparameters.
        model:           HierCB instance (on CPU; moved to device internally).
        train_loader:    Training DataLoader.
        test_loader:     Evaluation DataLoader.
        run_name:        W&B run name. Defaults to "seed{settings.seed}".
        checkpoint_dir:  Directory to save best model. Defaults to ./checkpoints/.
    """
    device = torch.device(settings.device if torch.cuda.is_available() else "cpu")
    run_name = run_name or f"seed{settings.seed}"
    checkpoint_dir = checkpoint_dir or Path("checkpoints") / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load pretrained weights (MuFFIN §V.B, ref [41])
    if pretrained is not None:
        state_dict = torch.load(pretrained, map_location="cpu", weights_only=True)
        # Strip 'module.' prefix if saved with DataParallel
        cleaned = {k.removeprefix("module."): v for k, v in state_dict.items()}
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        print(f"Loaded pretrained weights from {pretrained}")
        if missing:
            print(f"  Missing keys (expected): {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"  Unexpected keys (dropped): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    model = model.to(device)

    # PhnVar: compute stats once from training data (MuFFIN §IV cont)
    phnvar_qf: torch.Tensor | None = None
    phnvar_df: torch.Tensor | None = None
    if settings.use_phnvar and settings.use_mdd:
        from p010.phnvar import compute_phnvar_stats
        phnvar_qf, phnvar_df = compute_phnvar_stats(train_loader.dataset)
        print(f"PhnVar stats computed: QF range [{phnvar_qf.min():.3f}, {phnvar_qf.max():.3f}], "
              f"DF range [{phnvar_df.min():.3f}, {phnvar_df.max():.3f}]")

    wandb.init(
        project=settings.wandb_project,
        entity=settings.wandb_entity,
        name=run_name,
        config=settings.model_dump(),
        reinit="finish_previous",
    )

    trainables = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in model.parameters()) / 1e3
    n_trainable = sum(p.numel() for p in trainables) / 1e3
    print(f"Parameters: {n_params:.1f}k total, {n_trainable:.1f}k trainable")

    optimizer = torch.optim.Adam(trainables, lr=settings.lr, weight_decay=5e-7, betas=(0.95, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
    loss_fn = nn.MSELoss()
    loss_pco = ConPCOLoss(
        lambda_d=settings.pco_ld,
        lambda_t=settings.pco_lt,
        clap_t2a=settings.clap_t2a,
        margin=settings.pco_mg,
    ) if settings.use_conpco else None
    loss_mdd_fn = nn.BCEWithLogitsLoss() if settings.use_mdd else None
    loss_diag_fn = nn.CrossEntropyLoss(ignore_index=-1) if settings.use_mdd else None

    warm_up_steps = 100
    global_step = 0
    best_mse = float("inf")
    best_pcc = float("-inf")

    epoch_bar = trange(settings.n_epochs, desc=run_name, unit="ep")
    for epoch in epoch_bar:
        model.train()
        running_loss = 0.0
        batch_bar = tqdm(train_loader, desc=f"ep{epoch:03d}", leave=False, unit="batch")
        for i, batch in enumerate(batch_bar):
            gop, ssl, energy, dur, phn_score, phn_id, utt_label, word_label, word_id, mdd_label, diag_label = (
                t.to(device, non_blocking=True) for t in batch
            )

            # Warmup: linear ramp over first warm_up_steps steps
            if global_step < warm_up_steps:
                lr_now = settings.lr * (global_step + 1) / warm_up_steps
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_now

            # Input noise augmentation on GOP features.
            # Always create the rand tensor to stay in sync with the reference RNG trajectory,
            # even when noise=0.0 (adding zeros is a no-op but rand() must be called).
            gop = gop + (torch.rand_like(gop) - 1) * settings.noise

            word_scores = word_label[:, :, :3]   # [B, 50, 3]
            word_pos = word_label[:, :, 3]        # [B, 50] within-utterance word position (small int)
            # word_id: lexical vocab IDs (0..2607), loaded separately from tr_word_id.npy

            outputs = model(gop, energy, dur, ssl, phn_id, word_pos, word_id)

            if settings.use_mdd:
                u1, u2, u3, u4, u5, p, w1, w2, w3, mdd_logit, diag_logit, phn_audio_feats, phn_text_feats = outputs
            else:
                u1, u2, u3, u4, u5, p, w1, w2, w3, phn_audio_feats, phn_text_feats = outputs
                mdd_logit = None
                diag_logit = None

            # ── Phone loss ────────────────────────────────────────────────────
            phn_mask = phn_id >= 0
            loss_phn = masked_mse_loss(p, phn_score, phn_mask)

            # ── Utterance loss ────────────────────────────────────────────────
            utt_pred = torch.cat([u1, u2, u3, u4, u5], dim=1)  # [B, 5]
            loss_utt = loss_fn(utt_pred, utt_label)

            # ── Word loss ─────────────────────────────────────────────────────
            word_mask = word_scores >= 0
            word_pred = torch.cat([w1, w2, w3], dim=2)  # [B, 50, 3]
            word_pred_masked = word_pred * word_mask
            word_target_masked = word_scores * word_mask
            loss_word = loss_fn(word_pred_masked, word_target_masked)
            n_word_total = word_mask.shape[0] * word_mask.shape[1] * word_mask.shape[2]
            loss_word = loss_word * n_word_total / word_mask.sum().clamp(min=1)

            # ── Total loss ────────────────────────────────────────────────────
            loss = (
                settings.loss_w_phn * loss_phn
                + settings.loss_w_utt * loss_utt
                + settings.loss_w_word * loss_word
            )

            if settings.use_conpco and loss_pco is not None:
                loss_oe, loss_center_clap = loss_pco(phn_audio_feats, phn_text_feats, phn_score, phn_id)
                loss = loss + settings.loss_w_pco * loss_oe + settings.loss_w_clap * loss_center_clap

            if settings.use_mdd and mdd_logit is not None and loss_mdd_fn is not None:
                # L_det: binary detection loss (MuFFIN Eq.16)
                mdd_mask = mdd_label >= 0
                mdd_logit_valid = mdd_logit.squeeze(-1)[mdd_mask]
                mdd_label_valid = mdd_label[mdd_mask]
                loss_mdd = loss_mdd_fn(mdd_logit_valid, mdd_label_valid)
                loss = loss + settings.loss_w_mdd * loss_mdd

            if settings.use_mdd and diag_logit is not None and loss_diag_fn is not None:
                # L_diag: diagnosis cross-entropy loss (MuFFIN Eq.17)
                # diag_logit: [B, 50, 39], diag_label: [B, 50] (long, -1=pad)
                _diag = diag_logit
                if phnvar_qf is not None and phnvar_df is not None:
                    from p010.phnvar import perturb_diag_logits
                    _diag = perturb_diag_logits(
                        _diag, phnvar_qf, phnvar_df,
                        sigma=settings.phnvar_sigma, alpha=settings.phnvar_alpha, beta=settings.phnvar_beta,
                    )
                loss_diag = loss_diag_fn(_diag.reshape(-1, _diag.shape[-1]), diag_label.long().reshape(-1))
                loss = loss + settings.loss_w_diag * loss_diag

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1

            running_loss += loss.item()
            batch_bar.set_postfix(loss=f"{running_loss / (i + 1):.4f}")

        # ── Validation ────────────────────────────────────────────────────────
        tr_metrics = _evaluate(model, train_loader, device, settings)
        te_metrics = _evaluate(model, test_loader, device, settings)

        te_phn_mse = te_metrics["phn_mse"]
        te_phn_pcc = te_metrics["phn_pcc"]

        log_dict: dict = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"train/{k}": v for k, v in tr_metrics.items()},
            **{f"test/{k}": v for k, v in te_metrics.items()},
        }
        wandb.log(log_dict, step=epoch)

        is_best = te_phn_mse < best_mse
        epoch_bar.set_postfix(
            pcc=f"{te_phn_pcc:.4f}",
            mse=f"{te_phn_mse:.4f}",
            best=f"{best_pcc:.4f}" if best_pcc > float('-inf') else "–",
            lr=f"{optimizer.param_groups[0]['lr']:.1e}",
            new="✓" if is_best else "",
        )

        if is_best:
            best_mse = te_phn_mse
            best_pcc = te_phn_pcc
            torch.save(model.state_dict(), checkpoint_dir / "best_model.pth")
            wandb.log({"best/phn_mse": best_mse, "best/phn_pcc": best_pcc}, step=epoch)

        if global_step > warm_up_steps:
            scheduler.step(best_mse)

    wandb.finish()
    return best_pcc


@torch.no_grad()
def _evaluate(
    model: HierCB, loader: DataLoader, device: torch.device, settings: Settings,
    mdd_threshold: float = 0.5,
) -> dict[str, float]:
    """Run full evaluation pass, return flat metric dict."""
    model.eval()

    all_p, all_phn_score = [], []
    all_u: list[list[torch.Tensor]] = [[], [], [], [], []]
    all_utt_label = []
    all_w, all_word_label = [], []
    all_mdd_logit, all_mdd_label = [], []
    all_diag_logit, all_diag_label = [], []

    for batch in loader:
        gop, ssl, energy, dur, phn_score, phn_id, utt_label, word_label, word_id, mdd_label, diag_label = (
            t.to(device, non_blocking=True) for t in batch
        )
        word_pos = word_label[:, :, 3]  # within-utterance word position
        # word_id: lexical vocab IDs, already unpacked separately

        outputs = model(gop, energy, dur, ssl, phn_id, word_pos, word_id)

        if settings.use_mdd:
            u1, u2, u3, u4, u5, p, w1, w2, w3, mdd_logit, diag_logit_out, *_ = outputs
            all_mdd_logit.append(mdd_logit.cpu())
            all_mdd_label.append(mdd_label.cpu())
            all_diag_logit.append(diag_logit_out.cpu())
            all_diag_label.append(diag_label.cpu())
        else:
            u1, u2, u3, u4, u5, p, w1, w2, w3, *_ = outputs

        all_p.append(p.cpu())
        all_phn_score.append(phn_score.cpu())
        for i, u in enumerate([u1, u2, u3, u4, u5]):
            all_u[i].append(u.cpu())
        all_utt_label.append(utt_label.cpu())
        all_w.append(torch.cat([w1, w2, w3], dim=2).cpu())
        all_word_label.append(word_label.cpu())

    cat_p = torch.cat(all_p)
    cat_phn_score = torch.cat(all_phn_score)
    cat_utt_pred = torch.cat([torch.cat(all_u[i]) for i in range(5)], dim=1)
    cat_utt_label = torch.cat(all_utt_label)
    cat_word_pred = torch.cat(all_w)
    cat_word_label = torch.cat(all_word_label)

    phn_mse, phn_pcc = eval_phn(cat_p, cat_phn_score)
    utt_mse, utt_pcc = eval_utt(cat_utt_pred, cat_utt_label)
    word_mse, word_pcc, _, _ = eval_word(cat_word_pred, cat_word_label)

    utt_names = ["accuracy", "completeness", "fluency", "prosodic", "total"]
    word_names = ["accuracy", "stress", "total"]

    metrics: dict[str, float] = {
        "phn_mse": phn_mse,
        "phn_pcc": phn_pcc,
        **{f"utt_mse_{n}": v for n, v in zip(utt_names, utt_mse, strict=True)},
        **{f"utt_pcc_{n}": v for n, v in zip(utt_names, utt_pcc, strict=True)},
        **{f"word_mse_{n}": v for n, v in zip(word_names, word_mse, strict=True)},
        **{f"word_pcc_{n}": v for n, v in zip(word_names, word_pcc, strict=True)},
    }

    if settings.use_mdd and all_mdd_logit:
        mdd_result = eval_mdd(
            torch.cat(all_mdd_logit), torch.cat(all_mdd_label),
            threshold=mdd_threshold,
            diag_logit=torch.cat(all_diag_logit) if all_diag_logit else None,
            diag_label=torch.cat(all_diag_label) if all_diag_label else None,
        )
        metrics.update({f"mdd_{k}": v for k, v in mdd_result.items()})

    model.train()
    return metrics
