"""Pretraining loop for HierCB (MuFFIN §V.B, ref [41] HierTFR).

Three self-supervised objectives on SpeechOcean762 features:
  1. Masked Phoneme Prediction (MPP): predict masked canonical phone IDs
  2. Masked Word Prediction (MWP): predict masked lexical word IDs
  3. Utterance Comparative Labeling (UCL): predict which of two utterances has higher accuracy

Ported from HierTFR/src/traintest_hierTFR_v6_aspfix1_pretrain.py and
HierTFR/src/models/gopt_hierAtt_v61_aspfix_pre.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from p010.models.hiercb import (
    NUM_PHN_CLASSES,
    NUM_WORD_CLASSES,
    PHN_MASK_TOKEN,
    WORD_MASK_TOKEN,
    HierCB,
)
from p010.settings import Settings

# ── Masking ──────────────────────────────────────────────────────────────────

def mask_uniform(
    ids: torch.Tensor,
    mask_token: int,
    ignore_id: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace a uniform-random number of tokens with mask_token.

    Reimplemented from espnet.nets.pytorch_backend.maskctc.add_mask_token.mask_uniform.

    Args:
        ids:        [B, T] integer token IDs. Positions with ignore_id are padding.
        mask_token: Token ID to use as the mask replacement.
        ignore_id:  Padding token ID (excluded from masking).

    Returns:
        masked_in:  [B, T] input with some valid positions replaced by mask_token.
        targets:    [B, T] original IDs at masked positions, ignore_id elsewhere.
    """
    masked_in = ids.clone()
    targets = torch.full_like(ids, fill_value=ignore_id)

    for i in range(ids.shape[0]):
        valid = (ids[i] != ignore_id).nonzero(as_tuple=True)[0]
        n_valid = len(valid)
        if n_valid == 0:
            continue
        n_mask = int(np.random.randint(1, n_valid + 1))
        perm = np.random.choice(n_valid, n_mask, replace=False)
        mask_idx = valid[perm]
        masked_in[i, mask_idx] = mask_token
        targets[i, mask_idx] = ids[i, mask_idx]

    return masked_in, targets


# ── Pretrain Model ───────────────────────────────────────────────────────────

class HierCBPretrain(HierCB):
    """HierCB backbone + masked prediction heads for pretraining.

    Inherits all encoder layers from HierCB (same parameter names = weight transfer).
    Adds three prediction heads for MPP, MWP, UCL.
    """

    def __init__(self, **kwargs: object) -> None:
        kwargs["use_mdd"] = False  # no MDD during pretraining
        super().__init__(**kwargs)  # type: ignore[arg-type]

        d = self.embed_dim

        # Prediction heads
        self.phn_predi = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, NUM_PHN_CLASSES))
        self.word_predi = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, NUM_WORD_CLASSES))
        self.utt_predi = nn.Sequential(nn.LayerNorm(d * 2), nn.Linear(d * 2, 4))  # comparative pair → 4 classes

        # Losses (ignore_index=0 because IDs are +1 shifted, pad=0)
        self.phn_ce = nn.CrossEntropyLoss(ignore_index=0)
        self.word_ce = nn.CrossEntropyLoss(ignore_index=0)
        self.utt_ce = nn.CrossEntropyLoss(ignore_index=0)

    def forward(  # type: ignore[override]
        self,
        gop: torch.Tensor,
        energy: torch.Tensor,
        dur: torch.Tensor,
        ssl: torch.Tensor,
        phn: torch.Tensor,
        word_pos: torch.Tensor,
        word: torch.Tensor,
        utt_acc: torch.Tensor,  # [B] utterance accuracy scores for UCL
    ) -> dict[str, torch.Tensor]:
        """Forward pass with masking + prediction. Returns dict of losses and accuracies."""
        device = gop.device

        # ── Shared input projection ──────────────────────────────────────
        x = torch.cat([gop, self.ssl_drop(ssl), dur, energy], dim=-1)
        p_x = self.p_in_proj(x)
        w_x = self.w_in_proj(x)
        u_x = self.u_in_proj(x)

        # ── 1. Masked Phoneme Prediction ─────────────────────────────────
        # Shift IDs: phn_id -1(pad)→0, 0-38→1-39. Mask token = 41.
        phn_shifted = (phn.long() + 1).clamp(min=0)
        phn_masked, phn_targets = mask_uniform(phn_shifted, PHN_MASK_TOKEN, ignore_id=0)
        # +1 shift targets to match CE ignore_index=0
        phn_targets = phn_targets + 1
        phn_targets[phn_targets == 1] = 0  # original pad (was -1 → 0 → shifted to 1) back to ignore

        phn_one_hot = F.one_hot(phn_masked, num_classes=NUM_PHN_CLASSES).float()
        phn_embed = self.phn_proj(phn_one_hot)
        p_x = p_x + phn_embed + self.pos_embed

        p_tmp_feat = []
        for blk in self.phn_blocks:
            p_x = blk(p_x)
            p_tmp_feat.append(p_x)

        phn_logits = self.phn_predi(p_x)  # [B, 50, 42]
        phn_loss = self.phn_ce(phn_logits.reshape(-1, NUM_PHN_CLASSES), phn_targets.reshape(-1))
        phn_acc = _accuracy(phn_logits, phn_targets, ignore=0)

        # ── 2. Masked Word Prediction ────────────────────────────────────
        word_shifted = (word.long() + 1).clamp(min=0)
        word_masked, word_targets = mask_uniform(word_shifted, WORD_MASK_TOKEN, ignore_id=0)
        word_targets = word_targets + 1
        word_targets[word_targets == 1] = 0

        word_one_hot = F.one_hot(word_masked, num_classes=NUM_WORD_CLASSES).float()
        word_embed = self.word_proj(word_one_hot)

        phn2word_msk = self._make_word_pos_mask(word_pos)
        w_p_x = self.w_proj_cnn1(p_x.transpose(1, 2)).transpose(1, 2)
        w_x = self.w_proj_cnn2(w_x.transpose(1, 2)).transpose(1, 2)
        w_x_att = self.word_input_att(w_x, w_x, w_x, phn2word_msk)
        w_p_x_att = self.word_input_att1(w_p_x, w_p_x, w_p_x, phn2word_msk)
        x_word = self.w_in_cat_proj(torch.cat([w_x_att, w_p_x_att], dim=-1))
        x_word = x_word + word_embed + self.word_pos_embed((word_pos.int() + 1).clamp(min=0)) + p_x

        for blk in self.word_blocks:
            x_word = blk(x_word)

        word_logits = self.word_predi(x_word)  # [B, 50, 2607]
        word_loss = self.word_ce(word_logits.reshape(-1, NUM_WORD_CLASSES), word_targets.reshape(-1))
        word_acc = _accuracy(word_logits, word_targets, ignore=0)

        # ── 3. Utterance Comparative Labeling ────────────────────────────
        w1_proj = self.w_proj_ln1(x_word)
        w2_proj = self.w_proj_ln2(x_word)
        w3_proj = self.w_proj_ln3(x_word)
        u_w_feats = self.utt_feat_ext(w1_proj, w2_proj, w3_proj)
        u_p_feats = self.u_proj_cnn1(p_x.transpose(1, 2)).transpose(1, 2)
        u_w_feats = self.u_proj_cnn2(u_w_feats.transpose(1, 2)).transpose(1, 2)
        utt_feats = self.u_in_cat_proj(torch.cat([u_p_feats, u_w_feats, u_x], dim=-1)) + p_x

        for blk in self.utt_blocks:
            utt_feats = blk(utt_feats)

        # Pool all 5 aspects and sum (matching reference line 596)
        u_sum = (
            self.u1_att_pooling(utt_feats)
            + self.u2_att_pooling(utt_feats)
            + self.u3_att_pooling(utt_feats)
            + self.u4_att_pooling(utt_feats)
            + self.u5_att_pooling(utt_feats)
        )  # [B, 1, embed_dim] — squeeze to [B, embed_dim]
        u_sum = u_sum.squeeze(1)

        # Random pairing for comparative prediction
        B = utt_acc.shape[0]
        cp_idx = torch.from_numpy(np.random.choice(B, B)).to(device)
        cp_acc = utt_acc[cp_idx]

        # Labels: 1=other larger, 2=equal, 3=other smaller (0=ignore)
        cp_label = torch.zeros(B, dtype=torch.long, device=device)
        cp_label[cp_acc > utt_acc] = 1
        cp_label[cp_acc == utt_acc] = 2
        cp_label[cp_acc < utt_acc] = 3

        cp_feats = torch.cat([u_sum[cp_idx], u_sum], dim=-1)  # [B, embed_dim*2]
        utt_logits = self.utt_predi(cp_feats)  # [B, 4]
        utt_loss = self.utt_ce(utt_logits, cp_label)
        utt_acc_metric = (utt_logits.argmax(dim=-1) == cp_label).float().mean()

        return {
            "phn_loss": phn_loss,
            "phn_acc": phn_acc,
            "word_loss": word_loss,
            "word_acc": word_acc,
            "utt_loss": utt_loss,
            "utt_acc": utt_acc_metric,
        }


def _accuracy(logits: torch.Tensor, targets: torch.Tensor, ignore: int = 0) -> torch.Tensor:
    """Compute classification accuracy, ignoring positions with target == ignore."""
    mask = targets != ignore
    if mask.sum() == 0:
        return logits.new_tensor(0.0)
    pred = logits.reshape(-1, logits.shape[-1]).argmax(dim=-1)
    correct = (pred == targets.reshape(-1)) & mask.reshape(-1)
    return correct.float().sum() / mask.float().sum()


# ── Pretrain Training Loop ───────────────────────────────────────────────────

def pretrain_one_config(
    settings: Settings,
    model: HierCBPretrain,
    train_loader: DataLoader,
    checkpoint_dir: Path,
) -> Path:
    """Run pretraining. Returns path to best checkpoint.

    LR schedule: MultiStepLR (halve every 5 epochs from 20 to 95), matching HierTFR.
    Best checkpoint: highest mean(phn_acc, word_acc, utt_acc) on training set.
    """
    device = torch.device(settings.device if torch.cuda.is_available() else "cpu")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best_pretrain.pth"

    model = model.to(device)

    wandb.init(
        project=settings.wandb_project,
        entity=settings.wandb_entity,
        name="pretrain",
        config=settings.model_dump(),
        reinit="finish_previous",
    )

    trainables = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainables, lr=settings.lr, weight_decay=5e-7, betas=(0.95, 0.999))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(range(20, 100, 5)), gamma=0.5
    )

    warm_up_steps = 100
    global_step = 0
    best_acc = 0.0

    epoch_bar = trange(settings.pretrain_epochs, desc="pretrain", unit="ep")
    for epoch in epoch_bar:
        model.train()
        batch_bar = tqdm(train_loader, desc=f"ep{epoch:03d}", leave=False, unit="batch")

        for batch in batch_bar:
            gop, ssl, energy, dur, phn_score, phn_id, utt_label, word_label, word_id, mdd_label, diag_label = (
                t.to(device, non_blocking=True) for t in batch
            )
            word_pos = word_label[:, :, 3]
            utt_acc = utt_label[:, 0]  # utterance accuracy (first aspect)

            # Warmup: discrete steps every 5 (matching reference)
            if global_step <= warm_up_steps and global_step % 5 == 0:
                lr_now = (global_step / warm_up_steps) * settings.lr
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_now

            results = model(gop, energy, dur, ssl, phn_id, word_pos, word_id, utt_acc)
            loss = (
                settings.loss_w_phn * results["phn_loss"]
                + settings.loss_w_word * results["word_loss"]
                + settings.loss_w_utt * results["utt_loss"]
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1

            batch_bar.set_postfix(loss=f"{loss.item():.4f}")

        # Step scheduler after warmup
        if global_step > warm_up_steps:
            scheduler.step()

        # Epoch metrics (last batch — matching reference)
        acc_mean = (results["phn_acc"].item() + results["word_acc"].item() + results["utt_acc"].item()) / 3
        wandb.log({
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "pretrain/phn_loss": results["phn_loss"].item(),
            "pretrain/word_loss": results["word_loss"].item(),
            "pretrain/utt_loss": results["utt_loss"].item(),
            "pretrain/phn_acc": results["phn_acc"].item(),
            "pretrain/word_acc": results["word_acc"].item(),
            "pretrain/utt_acc": results["utt_acc"].item(),
            "pretrain/mean_acc": acc_mean,
        }, step=epoch)

        is_best = acc_mean > best_acc
        epoch_bar.set_postfix(
            acc=f"{acc_mean:.4f}",
            best=f"{best_acc:.4f}" if best_acc > 0 else "–",
            new="✓" if is_best else "",
        )

        if is_best:
            best_acc = acc_mean
            torch.save(model.state_dict(), best_path)

    wandb.finish()
    print(f"Best pretrain accuracy: {best_acc:.4f}, saved to {best_path}")
    return best_path
