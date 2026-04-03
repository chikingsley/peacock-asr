"""Training loop for P011."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import wandb
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from p011.data import Batch, FrameBatch
from p011.eval import eval_mdd, eval_phn, eval_utt, eval_word
from p011.losses import ConPCOLoss, masked_mse_loss
from p011.settings import Settings, SSLInterfaceMode
from p011.ssl_features import SSLModelKey


@dataclass(frozen=True)
class DeviceBatch:
    """Batch tensors moved to the target device."""

    gop: Tensor
    ssl: Tensor | dict[SSLModelKey, Tensor]
    energy: Tensor
    dur: Tensor
    phn_score: Tensor
    phn_id: Tensor
    utt_label: Tensor
    word_label: Tensor
    word_id: Tensor
    mdd_label: Tensor
    diag_label: Tensor
    frame_lengths: dict[SSLModelKey, Tensor] | None


def _move_batch_to_device(batch: Batch, device: torch.device) -> DeviceBatch:
    """Move one batch to the target device."""
    if isinstance(batch, FrameBatch):
        return DeviceBatch(
            gop=batch.gop.to(device, non_blocking=True),
            ssl={key: value.to(device, non_blocking=True) for key, value in batch.ssl_frames.items()},
            energy=batch.energy.to(device, non_blocking=True),
            dur=batch.dur.to(device, non_blocking=True),
            phn_score=batch.phn_score.to(device, non_blocking=True),
            phn_id=batch.phn_id.to(device, non_blocking=True),
            utt_label=batch.utt_label.to(device, non_blocking=True),
            word_label=batch.word_label.to(device, non_blocking=True),
            word_id=batch.word_id.to(device, non_blocking=True),
            mdd_label=batch.mdd_label.to(device, non_blocking=True),
            diag_label=batch.diag_label.to(device, non_blocking=True),
            frame_lengths={key: value.to(device, non_blocking=True) for key, value in batch.frame_lengths.items()},
        )

    gop, ssl, energy, dur, phn_score, phn_id, utt_label, word_label, word_id, mdd_label, diag_label = (
        tensor.to(device, non_blocking=True) for tensor in batch
    )
    return DeviceBatch(
        gop=gop,
        ssl=ssl,
        energy=energy,
        dur=dur,
        phn_score=phn_score,
        phn_id=phn_id,
        utt_label=utt_label,
        word_label=word_label,
        word_id=word_id,
        mdd_label=mdd_label,
        diag_label=diag_label,
        frame_lengths=None,
    )


def _forward_model(model: nn.Module, batch: DeviceBatch, settings: Settings) -> tuple[Tensor, ...]:
    """Run the correct forward path for the configured SSL interface."""
    word_pos = batch.word_label[:, :, 3]

    if settings.ssl_interface is SSLInterfaceMode.NONE:
        if not isinstance(batch.ssl, torch.Tensor):
            raise TypeError("Expected phone-level SSL tensor for ssl_interface=none")
        return model(batch.gop, batch.energy, batch.dur, batch.ssl, batch.phn_id, word_pos, batch.word_id)

    if isinstance(batch.ssl, torch.Tensor) or batch.frame_lengths is None:
        raise TypeError("Expected frame-level SSL tensors and frame lengths for frame-level SSL interfaces")
    return model(
        batch.gop,
        batch.energy,
        batch.dur,
        batch.ssl,
        batch.phn_id,
        word_pos,
        batch.word_id,
        batch.frame_lengths,
    )


def _remap_pretrained_keys(model: nn.Module, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    """Map plain HierCB checkpoint keys onto wrapper-model downstream keys when needed."""
    model_keys = set(model.state_dict().keys())
    remapped: dict[str, Tensor] = {}
    for key, value in state_dict.items():
        if key in model_keys:
            remapped[key] = value
            continue
        downstream_key = f"downstream.{key}"
        if downstream_key in model_keys:
            remapped[downstream_key] = value
            continue
        remapped[key] = value
    return remapped


def train_one_config(
    settings: Settings,
    model: nn.Module,
    train_loader: DataLoader[Batch],
    test_loader: DataLoader[Batch],
    run_name: str | None = None,
    checkpoint_dir: Path | None = None,
    pretrained: Path | None = None,
) -> float:
    """Train one seed/configuration and return best phone PCC."""
    device = torch.device(settings.device if torch.cuda.is_available() else "cpu")
    run_name = run_name or f"seed{settings.seed}"
    checkpoint_dir = checkpoint_dir or Path("checkpoints") / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if pretrained is not None:
        state_dict = torch.load(pretrained, map_location="cpu", weights_only=True)
        cleaned = {key.removeprefix("module."): value for key, value in state_dict.items()}
        cleaned = _remap_pretrained_keys(model, cleaned)
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        print(f"Loaded pretrained weights from {pretrained}")
        if missing:
            print(f"  Missing keys (expected): {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"  Unexpected keys (dropped): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    model = model.to(device)

    phnvar_qf: Tensor | None = None
    phnvar_df: Tensor | None = None
    if settings.use_phnvar and settings.use_mdd:
        from p011.phnvar import compute_phnvar_stats

        phnvar_qf, phnvar_df = compute_phnvar_stats(train_loader.dataset)
        print(
            f"PhnVar stats computed: QF range [{phnvar_qf.min():.3f}, {phnvar_qf.max():.3f}], "
            f"DF range [{phnvar_df.min():.3f}, {phnvar_df.max():.3f}]"
        )

    wandb.init(
        project=settings.wandb_project,
        entity=settings.wandb_entity,
        name=run_name,
        config=settings.model_dump(),
        reinit="finish_previous",
    )

    trainables = [parameter for parameter in model.parameters() if parameter.requires_grad]
    total_params = sum(parameter.numel() for parameter in model.parameters()) / 1e3
    trainable_params = sum(parameter.numel() for parameter in trainables) / 1e3
    print(f"Parameters: {total_params:.1f}k total, {trainable_params:.1f}k trainable")

    optimizer = torch.optim.Adam(trainables, lr=settings.lr, weight_decay=5e-7, betas=(0.95, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
    loss_fn = nn.MSELoss()
    loss_pco = (
        ConPCOLoss(
            lambda_d=settings.pco_ld,
            lambda_t=settings.pco_lt,
            clap_t2a=settings.clap_t2a,
            margin=settings.pco_mg,
        )
        if settings.use_conpco
        else None
    )
    loss_mdd_fn = nn.BCEWithLogitsLoss() if settings.use_mdd else None
    loss_diag_fn = nn.CrossEntropyLoss(ignore_index=-1) if settings.use_mdd else None

    warm_up_steps = 100
    global_step = 0
    best_mse = float("inf")
    best_pcc = float("-inf")
    grad_accum_steps = settings.grad_accum_steps

    epoch_bar = trange(settings.n_epochs, desc=run_name, unit="ep")
    for epoch in epoch_bar:
        model.train()
        running_loss = 0.0
        batch_bar = tqdm(train_loader, desc=f"ep{epoch:03d}", leave=False, unit="batch")
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(batch_bar):
            device_batch = _move_batch_to_device(batch, device)

            if global_step <= warm_up_steps and global_step % 5 == 0:
                lr_now = settings.lr * (global_step / warm_up_steps)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr_now

            gop = device_batch.gop + (torch.rand_like(device_batch.gop) - 1) * settings.noise
            device_batch = DeviceBatch(
                gop=gop,
                ssl=device_batch.ssl,
                energy=device_batch.energy,
                dur=device_batch.dur,
                phn_score=device_batch.phn_score,
                phn_id=device_batch.phn_id,
                utt_label=device_batch.utt_label,
                word_label=device_batch.word_label,
                word_id=device_batch.word_id,
                mdd_label=device_batch.mdd_label,
                diag_label=device_batch.diag_label,
                frame_lengths=device_batch.frame_lengths,
            )

            outputs = _forward_model(model, device_batch, settings)

            if settings.use_mdd:
                u1, u2, u3, u4, u5, p, w1, w2, w3, mdd_logit, diag_logit, phn_audio_feats, phn_text_feats = outputs
            else:
                u1, u2, u3, u4, u5, p, w1, w2, w3, phn_audio_feats, phn_text_feats = outputs
                mdd_logit = None
                diag_logit = None

            phn_mask = device_batch.phn_id >= 0
            loss_phn = masked_mse_loss(p, device_batch.phn_score, phn_mask)

            utt_pred = torch.cat([u1, u2, u3, u4, u5], dim=1)
            loss_utt = loss_fn(utt_pred, device_batch.utt_label)

            word_scores = device_batch.word_label[:, :, :3]
            word_mask = word_scores >= 0
            word_pred = torch.cat([w1, w2, w3], dim=2)
            word_pred_masked = word_pred * word_mask
            word_target_masked = word_scores * word_mask
            loss_word = loss_fn(word_pred_masked, word_target_masked)
            num_word_targets = word_mask.shape[0] * word_mask.shape[1] * word_mask.shape[2]
            loss_word = loss_word * num_word_targets / word_mask.sum().clamp(min=1)

            loss = (
                settings.loss_w_phn * loss_phn
                + settings.loss_w_utt * loss_utt
                + settings.loss_w_word * loss_word
            )

            if settings.use_conpco and loss_pco is not None:
                loss_oe, loss_center_clap = loss_pco(
                    phn_audio_feats,
                    phn_text_feats,
                    device_batch.phn_score,
                    device_batch.phn_id,
                )
                loss = loss + settings.loss_w_pco * loss_oe + settings.loss_w_clap * loss_center_clap

            if settings.use_mdd and mdd_logit is not None and loss_mdd_fn is not None:
                mdd_mask = device_batch.mdd_label >= 0
                mdd_logit_valid = mdd_logit.squeeze(-1)[mdd_mask]
                mdd_label_valid = device_batch.mdd_label[mdd_mask]
                loss_mdd = loss_mdd_fn(mdd_logit_valid, mdd_label_valid)
                loss = loss + settings.loss_w_mdd * loss_mdd

            if settings.use_mdd and diag_logit is not None and loss_diag_fn is not None:
                diag_for_loss = diag_logit
                if phnvar_qf is not None and phnvar_df is not None:
                    from p011.phnvar import perturb_diag_logits

                    diag_for_loss = perturb_diag_logits(
                        diag_for_loss,
                        phnvar_qf,
                        phnvar_df,
                        sigma=settings.phnvar_sigma,
                        alpha=settings.phnvar_alpha,
                        beta=settings.phnvar_beta,
                    )
                loss_diag = loss_diag_fn(
                    diag_for_loss.reshape(-1, diag_for_loss.shape[-1]),
                    device_batch.diag_label.long().reshape(-1),
                )
                loss = loss + settings.loss_w_diag * loss_diag

            loss_value = loss.item()
            (loss / grad_accum_steps).backward()

            is_last_microbatch = batch_index + 1 == len(train_loader)
            should_step = is_last_microbatch or (batch_index + 1) % grad_accum_steps == 0
            if should_step:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            running_loss += loss_value
            batch_bar.set_postfix(loss=f"{running_loss / (batch_index + 1):.4f}")

        tr_metrics = _evaluate(model, train_loader, device, settings)
        te_metrics = _evaluate(model, test_loader, device, settings)

        te_phn_mse = te_metrics["phn_mse"]
        te_phn_pcc = te_metrics["phn_pcc"]

        log_dict: dict[str, float | int] = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"train/{key}": value for key, value in tr_metrics.items()},
            **{f"test/{key}": value for key, value in te_metrics.items()},
        }
        wandb.log(log_dict, step=epoch)

        is_best = te_phn_mse < best_mse
        epoch_bar.set_postfix(
            pcc=f"{te_phn_pcc:.4f}",
            mse=f"{te_phn_mse:.4f}",
            best=f"{best_pcc:.4f}" if best_pcc > float("-inf") else "–",
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
    model: nn.Module,
    loader: DataLoader[Batch],
    device: torch.device,
    settings: Settings,
    mdd_threshold: float | None = None,
) -> dict[str, float]:
    """Run a full evaluation pass and return flat metrics."""
    model.eval()

    all_p: list[Tensor] = []
    all_phn_score: list[Tensor] = []
    all_u: list[list[Tensor]] = [[], [], [], [], []]
    all_utt_label: list[Tensor] = []
    all_w: list[Tensor] = []
    all_word_label: list[Tensor] = []
    all_mdd_logit: list[Tensor] = []
    all_mdd_label: list[Tensor] = []
    all_diag_logit: list[Tensor] = []
    all_diag_label: list[Tensor] = []
    all_phn_id: list[Tensor] = []

    for batch in loader:
        device_batch = _move_batch_to_device(batch, device)
        outputs = _forward_model(model, device_batch, settings)

        if settings.use_mdd:
            u1, u2, u3, u4, u5, p, w1, w2, w3, mdd_logit, diag_logit_out, *_ = outputs
            all_mdd_logit.append(mdd_logit.cpu())
            all_mdd_label.append(device_batch.mdd_label.cpu())
            all_diag_logit.append(diag_logit_out.cpu())
            all_diag_label.append(device_batch.diag_label.cpu())
            all_phn_id.append(device_batch.phn_id.cpu())
        else:
            u1, u2, u3, u4, u5, p, w1, w2, w3, *_ = outputs

        all_p.append(p.cpu())
        all_phn_score.append(device_batch.phn_score.cpu())
        for index, utt_output in enumerate([u1, u2, u3, u4, u5]):
            all_u[index].append(utt_output.cpu())
        all_utt_label.append(device_batch.utt_label.cpu())
        all_w.append(torch.cat([w1, w2, w3], dim=2).cpu())
        all_word_label.append(device_batch.word_label.cpu())

    cat_p = torch.cat(all_p)
    cat_phn_score = torch.cat(all_phn_score)
    cat_utt_pred = torch.cat([torch.cat(all_u[index]) for index in range(5)], dim=1)
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
        **{f"utt_mse_{name}": value for name, value in zip(utt_names, utt_mse, strict=True)},
        **{f"utt_pcc_{name}": value for name, value in zip(utt_names, utt_pcc, strict=True)},
        **{f"word_mse_{name}": value for name, value in zip(word_names, word_mse, strict=True)},
        **{f"word_pcc_{name}": value for name, value in zip(word_names, word_pcc, strict=True)},
    }

    if settings.use_mdd and all_mdd_logit:
        threshold = settings.mdd_eval_threshold if mdd_threshold is None else mdd_threshold
        mdd_result = eval_mdd(
            torch.cat(all_mdd_logit),
            torch.cat(all_mdd_label),
            threshold=threshold,
            diag_logit=torch.cat(all_diag_logit) if all_diag_logit else None,
            diag_label=torch.cat(all_diag_label) if all_diag_label else None,
            canon_phn_id=torch.cat(all_phn_id) if all_phn_id else None,
        )
        metrics.update({f"mdd_{key}": value for key, value in mdd_result.items()})

    model.train()
    return metrics
