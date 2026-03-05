#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.0",
#     "numpy>=1.26",
#     "wandb>=0.25",
#     "scipy>=1.14",
# ]
# ///
# NOTE: Official ConPCO pins torch==2.5.0+cu118, numpy==1.26.4.
# RTX 5070 (sm_120/Blackwell) requires torch>=2.6. For exact version
# matching, run on RunPod with an older GPU (L4/A5000) and pin via:
#   [tool.uv.sources]
#   torch = { index = "pytorch-cu118" }
#   [[tool.uv.index]]
#   name = "pytorch-cu118"
#   url = "https://download.pytorch.org/whl/cu118"
#   explicit = true
"""
Reproduce ConPCO (Yan & Chen, ICASSP 2025) results on SpeechOcean762.

Uses their precomputed features + their HierCB model + ConPCO loss.
Target: PCC ~0.743 on phone-level accuracy.

Data: references/ConPCO/data/seq_data_librispeech_v4/
Code: references/ConPCO/src/

Usage:
    cd /home/simon/github/peacock-asr
    uv run runs/reproduce_conpco.py --seed 42
    uv run runs/reproduce_conpco.py --seed 42 --no-conpco  # ablation: MSE only
"""

import os
import sys
import time
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import wandb

# Add ConPCO source to path
REPO_ROOT = Path(__file__).resolve().parent.parent
CONPCO_SRC = REPO_ROOT / "references" / "ConPCO" / "src"
sys.path.insert(0, str(CONPCO_SRC))

from models.conPCO_norm import ContrastivePhonemicOrdinalRegularizer
from models.gopt_ssl_3m_bfr_cat_utt_clap import HierCB


def get_args():
    parser = argparse.ArgumentParser(description="Reproduce ConPCO ICASSP 2025")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embed-dim", type=int, default=12)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--p-depth", type=int, default=1)
    parser.add_argument("--w-depth", type=int, default=1)
    parser.add_argument("--u-depth", type=int, default=1)
    parser.add_argument("--ssl-drop", type=float, default=0.1)
    # ConPCO hyperparams (from paper)
    parser.add_argument("--conpco", type=str, default="on", choices=["on", "off"],
                        help="Enable/disable ConPCO loss ('on' or 'off')")
    parser.add_argument("--pco-ld", type=float, default=5.0, help="Diverse term weight")
    parser.add_argument("--pco-lt", type=float, default=0.1, help="Tightness term weight")
    parser.add_argument("--pco-mg", type=float, default=1.0, help="Ordinal margin")
    parser.add_argument("--clap-t2a", type=float, default=0.1, help="CLAP text-to-audio weight")
    parser.add_argument("--loss-w-phn", type=float, default=1.0)
    parser.add_argument("--loss-w-pco", type=float, default=1.0)
    parser.add_argument("--loss-w-clap", type=float, default=1.0)
    parser.add_argument("--loss-w-word", type=float, default=1.0)
    parser.add_argument("--loss-w-utt", type=float, default=1.0)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--wandb-project", type=str, default="peacock-asr")
    parser.add_argument("--tag", type=str, default="v4", help="Version tag for wandb")
    return parser.parse_args()


class GoPDataset(Dataset):
    """ConPCO dataset loader — loads precomputed numpy features."""

    def __init__(self, split: str, data_dir: Path):
        prefix = "tr" if split == "train" else "te"
        d = data_dir

        # GOP features (84-dim)
        norm_mean, norm_std = 3.203, 4.045
        self.feat = torch.tensor(np.load(d / f"{prefix}_feat.npy"), dtype=torch.float)
        self.feat = self._norm_valid(self.feat, norm_mean, norm_std)

        # Additional features
        self.feat_energy = torch.tensor(np.load(d / f"{prefix}_energy_feat.npy"), dtype=torch.float)
        self.feat_dur = torch.tensor(np.load(d / f"{prefix}_dur_feat.npy"), dtype=torch.float)
        self.feat_ssl1 = torch.tensor(np.load(d / f"{prefix}_hubert_feat_v2.npy"), dtype=torch.float)
        self.feat_ssl2 = torch.tensor(np.load(d / f"{prefix}_w2v_300m_feat_v2.npy"), dtype=torch.float)
        self.feat_ssl3 = torch.tensor(np.load(d / f"{prefix}_wavlm_feat_v2.npy"), dtype=torch.float)

        # Labels
        self.phn_label = torch.tensor(np.load(d / f"{prefix}_label_phn.npy"), dtype=torch.float)
        self.utt_label = torch.tensor(np.load(d / f"{prefix}_label_utt.npy"), dtype=torch.float)
        self.word_label = torch.tensor(np.load(d / f"{prefix}_label_word.npy"), dtype=torch.float)
        self.word_id = torch.tensor(np.load(d / f"{prefix}_word_id.npy"), dtype=torch.float)

        # Normalize labels to 0-2 range (same as phone score range)
        self.utt_label = self.utt_label / 5
        self.word_label[:, :, 0:3] = self.word_label[:, :, 0:3] / 5

    @staticmethod
    def _norm_valid(feat, mean, std):
        norm = torch.zeros_like(feat)
        for i in range(feat.shape[0]):
            for j in range(feat.shape[1]):
                if feat[i, j, 0] != 0:
                    norm[i, j, :] = (feat[i, j, :] - mean) / std
                else:
                    break
        return norm

    def __len__(self):
        return self.feat.shape[0]

    def __getitem__(self, idx):
        return (
            self.feat[idx],
            self.feat_ssl1[idx],
            self.feat_ssl2[idx],
            self.feat_ssl3[idx],
            self.feat_energy[idx],
            self.feat_dur[idx],
            self.phn_label[idx, :, 1],  # phone accuracy score
            self.phn_label[idx, :, 0],  # phone ID
            self.utt_label[idx],
            self.word_label[idx],
            self.word_id[idx],
        )


def compute_pcc(pred, target):
    """Pearson correlation, ignoring padded tokens (target < 0)."""
    valid_pred, valid_target = [], []
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            if target[i, j] >= 0:
                valid_pred.append(pred[i, j].item())
                valid_target.append(target[i, j].item())
    if len(valid_pred) < 2:
        return 0.0
    return float(np.corrcoef(valid_pred, valid_target)[0, 1])


def compute_utt_pcc(pred, target):
    """Per-aspect utterance PCC. Returns dict."""
    aspects = ["accuracy", "completeness", "fluency", "prosodic", "total"]
    result = {}
    for i, name in enumerate(aspects):
        result[f"utt_{name}_pcc"] = float(np.corrcoef(pred[:, i], target[:, i])[0, 1])
    return result


def compute_word_pcc(pred, target, word_labels):
    """Word-level PCC with proper word boundary averaging."""
    word_id = word_labels[:, :, -1]
    target_scores = word_labels[:, :, 0:3]

    valid_pred, valid_target = [], []
    for i in range(target_scores.shape[0]):
        prev_w_id = 0
        start_id = 0
        for j in range(target_scores.shape[1]):
            cur_w_id = int(word_id[i, j])
            if cur_w_id != prev_w_id:
                valid_pred.append(np.mean(pred[i, start_id:j, :].numpy(), axis=0))
                valid_target.append(np.mean(target_scores[i, start_id:j, :].numpy(), axis=0))
                if cur_w_id == -1:
                    break
                prev_w_id = cur_w_id
                start_id = j

    valid_pred = np.array(valid_pred)
    valid_target = np.array(valid_target).round(2)

    aspects = ["accuracy", "stress", "total"]
    result = {}
    for i, name in enumerate(aspects):
        result[f"word_{name}_pcc"] = float(np.corrcoef(valid_pred[:, i], valid_target[:, i])[0, 1])
    return result


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_phn_pred, all_phn_target = [], []
    all_utt_pred, all_utt_target = [], []
    all_word_pred, all_word_labels = [], []

    for batch in loader:
        (feat, ssl1, ssl2, ssl3, eng, dur, phn_label, phns,
         utt_label, word_label, word_id) = batch

        feat = feat.to(device)
        ssl = torch.cat([ssl2.to(device), ssl1.to(device), ssl3.to(device)], dim=-1)
        eng, dur = eng.to(device), dur.to(device)
        phns = phns.to(device)
        word_id = word_id.to(device)
        word_label_dev = word_label.to(device)

        u1, u2, u3, u4, u5, p, w1, w2, w3, _, _ = model(
            feat, eng, dur, ssl, phns, word_label_dev[:, :, -1], word_id
        )

        p = p.squeeze(2).cpu()
        all_phn_pred.append(p)
        all_phn_target.append(phn_label)

        utt_pred = torch.cat([u1, u2, u3, u4, u5], dim=1).cpu()
        all_utt_pred.append(utt_pred)
        all_utt_target.append(utt_label)

        word_pred = torch.cat([w1, w2, w3], dim=2).cpu()
        all_word_pred.append(word_pred)
        all_word_labels.append(word_label)

    all_phn_pred = torch.cat(all_phn_pred)
    all_phn_target = torch.cat(all_phn_target)
    all_utt_pred = torch.cat(all_utt_pred)
    all_utt_target = torch.cat(all_utt_target)
    all_word_pred = torch.cat(all_word_pred)
    all_word_labels = torch.cat(all_word_labels)

    metrics = {}
    metrics["phn_pcc"] = compute_pcc(all_phn_pred, all_phn_target)
    metrics.update(compute_utt_pcc(all_utt_pred.numpy(), all_utt_target.numpy()))
    metrics.update(compute_word_pcc(all_word_pred, all_utt_target, all_word_labels))

    # MSE (valid tokens only)
    mask = all_phn_target >= 0
    mse_vals = ((all_phn_pred - all_phn_target) ** 2) * mask
    metrics["phn_mse"] = float(mse_vals.sum() / mask.sum())

    return metrics


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_dir = REPO_ROOT / "references" / "ConPCO" / "data" / "seq_data_librispeech_v4"

    # Datasets
    print("Loading train set...")
    tr_dataset = GoPDataset("train", data_dir)
    print("Loading test set...")
    te_dataset = GoPDataset("test", data_dir)

    tr_loader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True)
    te_loader = DataLoader(te_dataset, batch_size=2500, shuffle=False)

    # Model: HierCB (their full architecture)
    input_dim = 84 + 7 + 1  # GOP + energy + duration
    model = HierCB(
        embed_dim=args.embed_dim,
        num_heads=args.heads,
        p_depth=args.p_depth,
        w_depth=args.w_depth,
        u_depth=args.u_depth,
        ssl_drop=args.ssl_drop,
        input_dim=input_dim,
    )
    model = nn.DataParallel(model).to(device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e3
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e3
    print(f"Total params: {total_params:.1f}K, Trainable: {trainable_params:.1f}K")

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=5e-7, betas=(0.95, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)

    loss_fn = nn.MSELoss()
    use_conpco = args.conpco == "on"
    if use_conpco:
        loss_pco = ContrastivePhonemicOrdinalRegularizer(
            args.pco_ld, args.pco_lt, args.clap_t2a, args.pco_mg,
        )

    # wandb
    run_name = f"reproduce-conpco-{'ON' if use_conpco else 'OFF'}-seed{args.seed}"
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        group="reproduce-conpco-v4",
        config={
            "experiment": "reproduce_conpco",
            "version": args.tag,
            "track": "09",
            "seed": args.seed,
            "conpco_enabled": use_conpco,
            "architecture": "HierCB",
            "features": "Kaldi-GOP(84) + energy(7) + dur(1) + 3xSSL(3072)",
            "input_dim": input_dim,
            "embed_dim": args.embed_dim,
            "n_epochs": args.n_epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "pco_ld": args.pco_ld,
            "pco_lt": args.pco_lt,
            "pco_mg": args.pco_mg,
            "clap_t2a": args.clap_t2a,
            "total_params_k": total_params,
        },
        tags=["track09", "reproduction", "conpco", args.tag],
    )

    best_phn_pcc = 0.0
    best_mse = 999.0
    best_epoch = 0
    best_metrics = {}  # metrics at best MSE epoch (official selection criterion)
    global_step = 0

    for epoch in range(args.n_epochs):
        model.train()
        epoch_losses = {"phn": 0, "utt": 0, "word": 0, "pco": 0, "clap": 0, "total": 0}
        n_batches = 0

        for batch in tr_loader:
            (feat, ssl1, ssl2, ssl3, eng, dur, phn_label, phns,
             utt_label, word_label, word_id) = batch

            feat = feat.to(device)
            ssl = torch.cat([ssl2.to(device), ssl1.to(device), ssl3.to(device)], dim=-1)
            eng, dur = eng.to(device), dur.to(device)
            phns = phns.to(device)
            phn_label = phn_label.to(device)
            utt_label = utt_label.to(device)
            word_label = word_label.to(device)
            word_id = word_id.to(device)

            # Warmup
            if global_step <= 100 and global_step % 5 == 0:
                warm_lr = (global_step / 100) * args.lr
                for pg in optimizer.param_groups:
                    pg["lr"] = warm_lr

            # Noise augmentation — always create tensor to match official RNG
            # trajectory (official#L125), even when noise=0 (adds zeros).
            noise = (torch.rand(
                [feat.shape[0], feat.shape[1], feat.shape[2]],
            ) - 1) * args.noise
            noise = noise.to(device, non_blocking=True)
            feat = feat + noise

            u1, u2, u3, u4, u5, p, w1, w2, w3, phn_audio, phn_text = model(
                feat, eng, dur, ssl, phns, word_label[:, :, -1], word_id
            )

            # Phone loss (mask padding)
            mask = phns >= 0
            p = p.squeeze(2) * mask
            phn_label_masked = phn_label * mask
            loss_phn = loss_fn(p, phn_label_masked) * (mask.numel() / mask.sum())

            # ConPCO loss
            loss_phn_pco = torch.tensor(0.0, device=device)
            loss_clap = torch.tensor(0.0, device=device)
            if use_conpco:
                loss_phn_pco, loss_clap = loss_pco(phn_audio, phn_text, phn_label_masked, phns)

            # Utterance loss
            utt_pred = torch.cat([u1, u2, u3, u4, u5], dim=1)
            loss_utt = loss_fn(utt_pred, utt_label)

            # Word loss (mask padding)
            word_scores = word_label[:, :, 0:3]
            wmask = word_scores >= 0
            word_pred = torch.cat([w1, w2, w3], dim=2) * wmask
            word_scores_masked = word_scores * wmask
            loss_word = loss_fn(word_pred, word_scores_masked) * (wmask.numel() / wmask.sum())

            # Total
            if use_conpco:
                loss = (args.loss_w_phn * loss_phn
                        + args.loss_w_utt * loss_utt
                        + args.loss_w_word * loss_word
                        + args.loss_w_pco * loss_phn_pco
                        + args.loss_w_clap * loss_clap)
            else:
                loss = (args.loss_w_phn * loss_phn
                        + args.loss_w_utt * loss_utt
                        + args.loss_w_word * loss_word)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses["phn"] += loss_phn.item()
            epoch_losses["utt"] += loss_utt.item()
            epoch_losses["word"] += loss_word.item()
            epoch_losses["pco"] += loss_phn_pco.item()
            epoch_losses["clap"] += loss_clap.item()
            epoch_losses["total"] += loss.item()
            n_batches += 1
            global_step += 1

        # Average losses
        for k in epoch_losses:
            epoch_losses[k] /= max(n_batches, 1)

        # Evaluate — match official: validate on BOTH train and test loaders
        # (official#L181). The train_loader iteration consumes DataLoader
        # shuffle RNG, which affects all subsequent random state.
        _ = evaluate(model, tr_loader, device)  # train eval (RNG alignment)
        metrics = evaluate(model, te_loader, device)

        # Track best by lowest test MSE (official#L199), save all metrics at that epoch
        if metrics["phn_mse"] < best_mse:
            best_mse = metrics["phn_mse"]
            best_epoch = epoch
            best_metrics = dict(metrics)
        if metrics["phn_pcc"] > best_phn_pcc:
            best_phn_pcc = metrics["phn_pcc"]

        # LR schedule — use best_mse (not current), matching paper's code
        if global_step > 100:
            scheduler.step(best_mse)

        lr = optimizer.param_groups[0]["lr"]

        # Log
        wandb.log({
            "epoch": epoch,
            "lr": lr,
            # Train losses
            "train/loss_total": epoch_losses["total"],
            "train/loss_phn": epoch_losses["phn"],
            "train/loss_utt": epoch_losses["utt"],
            "train/loss_word": epoch_losses["word"],
            "train/loss_pco": epoch_losses["pco"],
            "train/loss_clap": epoch_losses["clap"],
            # Test metrics
            "test/phn_pcc": metrics["phn_pcc"],
            "test/phn_mse": metrics["phn_mse"],
            **{f"test/{k}": v for k, v in metrics.items() if k.startswith("utt_")},
            **{f"test/{k}": v for k, v in metrics.items() if k.startswith("word_")},
            # Best tracking (by MSE, official selection criterion)
            "best/phn_pcc": best_metrics.get("phn_pcc", 0),
            "best/phn_mse": best_mse,
            "best/epoch": best_epoch,
            # Also track best PCC ever seen (may differ from best-MSE epoch)
            "best/phn_pcc_ever": best_phn_pcc,
        })

        print(
            f"Epoch {epoch:3d} | LR {lr:.6f} | "
            f"Phone PCC {metrics['phn_pcc']:.4f} | "
            f"Utt Total PCC {metrics.get('utt_total_pcc', 0):.4f} | "
            f"Best PCC {best_phn_pcc:.4f} (ep{best_epoch})"
        )

    wandb.finish()
    best_epoch_pcc = best_metrics.get("phn_pcc", 0)
    print(f"\nDone. Best MSE epoch: {best_epoch}")
    print(f"  PCC at best-MSE epoch: {best_epoch_pcc:.4f}  (this is what the paper reports)")
    print(f"  Best PCC ever seen:    {best_phn_pcc:.4f}")
    print(f"Target: ~0.743 (ConPCO paper, 5-seed mean)")


if __name__ == "__main__":
    args = get_args()

    # Seed everything
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    train(args)
