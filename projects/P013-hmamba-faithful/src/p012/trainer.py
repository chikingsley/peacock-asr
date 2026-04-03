from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
import yaml
from jiwer import wer
from pydantic import ValidationError
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset

from .config import HMambaConfig
from .dataset import GoPDataset
from .losses import cross_entropy_lsm, decoupled_cross_entropy_lsm
from .models import HMamba
from .runtime import require_cuda_device
from .scheduler import TriStageLRScheduler

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None


def load_conf(config: str | os.PathLike[str]) -> dict:
    with open(config, encoding="utf-8") as f:
        try:
            parsed = HMambaConfig.model_validate(yaml.safe_load(f))
        except ValidationError as exc:
            raise ValueError(f"Invalid HMamba config at {config}") from exc
    return parsed.model_dump()


def load_phn_dict(path: str | os.PathLike[str]) -> dict[int, str]:
    with open(path, encoding="utf-8") as rf:
        try:
            phn_dict = json.load(rf)
            return {int(id_): phn for phn, id_ in phn_dict.items()}
        except json.JSONDecodeError:
            rf.seek(0)
            return {int(line.split()[1]): line.split()[0] for line in rf.readlines()}


def maybe_init_wandb(exp_dir: str) -> None:
    if wandb is not None:
        wandb.init(project="p012-hmamba-faithful", name=exp_dir)


def maybe_log(metrics: dict[str, float]) -> None:
    if wandb is not None:
        wandb.log(metrics)


def maybe_finish_wandb() -> None:
    if wandb is not None:
        wandb.finish()


def gen_result_header() -> list[str]:
    phn_header = ["epoch", "phone_train_mse", "phone_train_pcc", "phone_test_mse", "phone_test_pcc", "learning_rate"]
    utt_header_set = ["utt_train_mse", "utt_train_pcc", "utt_test_mse", "utt_test_pcc"]
    utt_header_score = ["accuracy", "completeness", "fluency", "prosodic", "total"]
    word_header_set = ["word_train_pcc", "word_test_pcc"]
    word_header_score = ["accuracy", "stress", "total"]
    stress_header_set = ["stress_train_f1", "stress_test_f1"]
    stress_header_score = ["macro", "micro"]
    mdd_header_set = ["mdd_train", "mdd_test"]
    mdd_header_score = ["precision", "recall", "f1"]

    header: list[str] = phn_header
    for dset in utt_header_set:
        header.extend(f"{dset}_{x}" for x in utt_header_score)
    for dset in word_header_set:
        header.extend(f"{dset}_{x}" for x in word_header_score)
    for dset in stress_header_set:
        header.extend(f"{dset}_{x}" for x in stress_header_score)
    header.extend(["per_train", "per_test"])
    for dset in mdd_header_set:
        header.extend(f"{dset}_{x}" for x in mdd_header_score)
    return header


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_pretrain_path(pretrain: str | None) -> Path | None:
    if not pretrain:
        return None
    path = Path(pretrain)
    if path.is_dir():
        return path / "models" / "best_audio_model.pth"
    return path


def load_checkpoint(model: nn.Module, checkpoint_path: str | os.PathLike[str], device: torch.device) -> None:
    state_dict = torch.load(checkpoint_path, map_location=device)
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    if any(key.startswith("_orig_mod.") for key in state_dict):
        state_dict = {key.removeprefix("_orig_mod."): value for key, value in state_dict.items()}
    unwrap_model(model).load_state_dict(state_dict, strict=False)


def save_checkpoint(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(unwrap_model(model).state_dict(), path)


def unwrap_model(model: nn.Module) -> nn.Module:
    if hasattr(model, "_orig_mod"):
        return cast(nn.Module, model._orig_mod)
    return model


def compile_top_level_model(model: nn.Module, args: argparse.Namespace) -> nn.Module:
    if not args.compile_model:
        return model

    compile_kwargs = {}
    if args.compile_backend:
        compile_kwargs["backend"] = args.compile_backend
    if args.compile_mode:
        compile_kwargs["mode"] = args.compile_mode
    print(f"Compiling top-level model with torch.compile({compile_kwargs})")
    return cast(nn.Module, torch.compile(model, **compile_kwargs))


def mdd_detection_metrics(
    logits: torch.Tensor,
    canophns: torch.Tensor,
    realphns: torch.Tensor,
    pad_mask: torch.Tensor,
) -> tuple[float, float, float]:
    pred = torch.argmax(logits, dim=-1)
    valid = pad_mask.bool()
    pred_mis = (pred != canophns) & valid
    true_mis = (realphns != canophns) & valid
    tp = (pred_mis & true_mis).sum().item()
    fp = (pred_mis & ~true_mis).sum().item()
    fn = (~pred_mis & true_mis).sum().item()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def train(audio_model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, args: argparse.Namespace) -> None:
    device = require_cuda_device("HMamba training")
    print(f"running on {device}")

    audio_model = audio_model.to(device)
    pretrain_path = resolve_pretrain_path(args.pretrain)
    if pretrain_path is not None:
        print(f"Loading pretrained model from {pretrain_path} ...")
        load_checkpoint(audio_model, pretrain_path, device)

    audio_model = compile_top_level_model(audio_model, args)
    trainables = []
    for name, param in audio_model.named_parameters():
        if not param.requires_grad:
            continue
        lr = args.lr * 0.045 if "utt_mlp" in name else args.lr
        trainables.append({"params": [param], "lr": lr})

    optimizer = torch.optim.Adam(trainables, lr=args.lr, weight_decay=5e-7, betas=(0.95, 0.999))
    scheduler = TriStageLRScheduler(
        optimizer,
        init_lr_scale=args.init_lr_scale,
        peak_lr=args.lr,
        final_lr=args.final_lr,
        phase_ratio=args.phase_ratio,
        total_steps=len(train_loader) * args.n_epochs,
    )
    loss_fn = nn.MSELoss()
    mis_stats = cast(GoPDataset, train_loader.dataset).mispronunciation_stats()

    best_phone_mse = float("inf")
    best_mdd_f1 = float("-inf")
    best_selected = float("-inf") if args.selection_metric == "mdd_f1" else float("inf")
    print("start training...")
    result = np.zeros([args.n_epochs, len(gen_result_header())])

    for epoch in range(1, args.n_epochs + 1):
        audio_model.train()
        for batch in train_loader:
            (
                audio_input,
                audio_input2,
                audio_input3,
                canophns,
                realphns,
                bies,
                phn_label,
                word_label,
                utt_label,
                _utt_id,
            ) = batch

            audio_input = audio_input.to(device, non_blocking=True)
            audio_input2 = [input2.to(device, non_blocking=True) for input2 in audio_input2]
            if isinstance(audio_input3, torch.Tensor):
                audio_input3 = audio_input3.to(device, non_blocking=True)
            phn_label = phn_label.to(device, non_blocking=True)
            utt_label = utt_label.to(device, non_blocking=True)
            word_label = word_label.to(device, non_blocking=True)
            canophns = canophns.to(device, non_blocking=True)
            realphns = realphns.to(device, non_blocking=True)
            bies = bies.to(device, non_blocking=True)
            pad_mask = (phn_label >= 0).to(device, non_blocking=True)

            if epoch == 1:
                print(f"[INFO] Params: {sum(p.numel() for p in audio_model.parameters()):,}")

            u1, u2, u3, u4, u5, p, w1, w2, w3, logits = audio_model(
                audio_input,
                audio_input2,
                audio_input3,
                canophns,
                bies,
                mask=pad_mask,
            )

            p = p.squeeze(2) * pad_mask
            loss_phn = loss_fn(p, phn_label * pad_mask)
            loss_phn = loss_phn * (pad_mask.shape[0] * pad_mask.shape[1]) / torch.sum(pad_mask)

            utt_preds = torch.cat((u1, u2, u3, u4, u5), dim=1)
            loss_utt = loss_fn(utt_preds, utt_label)

            word_scores = word_label[:, :, 0:3]
            word_mask = word_scores >= 0
            word_pred = torch.cat((w1, w2, w3), dim=2) * word_mask
            word_target = word_scores * word_mask
            loss_word = loss_fn(word_pred, word_target)
            loss_word = loss_word * (
                word_mask.shape[0] * word_mask.shape[1] * word_mask.shape[2]
            ) / torch.sum(word_mask)

            loss = args.loss_w_phn * loss_phn + args.loss_w_utt * loss_utt + args.loss_w_word * loss_word
            if args.loss_type == "xent":
                loss_xent = cross_entropy_lsm(logits, realphns.long(), lsm_prob=0.0, ignore_index=-1, training=True)
                loss = loss + args.loss_w_xent * loss_xent
            elif args.loss_type == "dexent":
                loss_xent = decoupled_cross_entropy_lsm(
                    logits,
                    realphns,
                    canophns,
                    a=args.loss_w_a,
                    ignore_index=-1,
                    training=True,
                    num_correct=mis_stats.num_correct,
                    num_mispronounced=mis_stats.num_mispronounced,
                )
                loss = loss + args.loss_w_xent * loss_xent
            else:
                raise ValueError("only xent and dexent are available.")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            maybe_log({"train/loss": float(loss.item())})

        print(f"Epoch-{epoch}, lr: {optimizer.param_groups[0]['lr']:.7f}")
        print("start validation")

        tr_metrics = validate(audio_model, train_loader, args)
        te_metrics = validate(audio_model, test_loader, args, valid=True)

        (
            tr_mse,
            tr_corr,
            tr_utt_mse,
            tr_utt_corr,
            _tr_word_mse,
            tr_word_corr,
            tr_word_stress,
            tr_per,
            tr_mdd,
        ) = tr_metrics
        (
            te_mse,
            te_corr,
            te_utt_mse,
            te_utt_corr,
            _te_word_mse,
            te_word_corr,
            te_word_stress,
            te_per,
            te_mdd,
            all_phn_target,
            valid_word_target,
            all_utt_target,
            _all_recog_target,
            all_phn,
            valid_word_pred,
            all_utt,
            _all_recog,
        ) = te_metrics

        selected_value = te_mdd[2] if args.selection_metric == "mdd_f1" else float(te_mse)
        improved_selected = (
            selected_value > best_selected if args.selection_metric == "mdd_f1" else selected_value < best_selected
        )

        if te_mse < best_phone_mse:
            best_phone_mse = float(te_mse)
            save_checkpoint(audio_model, Path(args.exp_dir) / "models" / "best_phone_mse_model.pth")

        if te_mdd[2] > best_mdd_f1:
            best_mdd_f1 = te_mdd[2]
            save_checkpoint(audio_model, Path(args.exp_dir) / "models" / "best_mdd_f1_model.pth")

        if improved_selected:
            best_selected = selected_value
            preds_dir = Path(args.exp_dir) / "preds"
            preds_dir.mkdir(parents=True, exist_ok=True)
            if not (preds_dir / "phn_target.npy").exists():
                np.save(preds_dir / "phn_target.npy", all_phn_target)
                np.save(preds_dir / "word_target.npy", valid_word_target)
                np.save(preds_dir / "utt_target.npy", all_utt_target)
            np.save(preds_dir / "phn_pred.npy", all_phn)
            np.save(preds_dir / "word_pred.npy", valid_word_pred)
            np.save(preds_dir / "utt_pred.npy", all_utt)
            save_checkpoint(audio_model, Path(args.exp_dir) / "models" / "best_audio_model.pth")

        result[epoch - 1, :6] = [epoch, tr_mse, tr_corr, te_mse, te_corr, optimizer.param_groups[0]["lr"]]
        result[epoch - 1, 6:26] = np.concatenate([tr_utt_mse, tr_utt_corr, te_utt_mse, te_utt_corr])
        result[epoch - 1, 26:32] = np.concatenate([tr_word_corr, te_word_corr])
        result[epoch - 1, 32:36] = np.concatenate([tr_word_stress, te_word_stress])
        result[epoch - 1, 36:38] = [tr_per, te_per]
        result[epoch - 1, 38:44] = [tr_mdd[0], tr_mdd[1], tr_mdd[2], te_mdd[0], te_mdd[1], te_mdd[2]]

        header = ",".join(gen_result_header())
        np.savetxt(Path(args.exp_dir) / "result.csv", result, delimiter=",", header=header, comments="")

        print(f"Phone: Test MSE: {float(te_mse):.3f}, CORR: {te_corr:.3f}")
        print(
            "Utterance:, "
            f"ACC: {te_utt_corr[0]:.3f}, "
            f"COM: {te_utt_corr[1]:.3f}, "
            f"FLU: {te_utt_corr[2]:.3f}, "
            f"PROC: {te_utt_corr[3]:.3f}, "
            f"Total: {te_utt_corr[4]:.3f}"
        )
        print(
            f"Word:, ACC: {te_word_corr[0]:.3f}, "
            f"Stress: {te_word_corr[1]:.3f}, "
            f"Total: {te_word_corr[2]:.3f}"
        )
        print(f"Phone error rate: {te_per:.3f}")
        print(f"MDD: Precision {te_mdd[0]:.3f}, Recall {te_mdd[1]:.3f}, F1 {te_mdd[2]:.3f}")
        print("-------------------validation finished-------------------")

        maybe_log(
            {
                "train/phone_pcc": tr_corr,
                "test/phone_pcc": te_corr,
                "train/utt_total_pcc": tr_utt_corr[4],
                "test/utt_total_pcc": te_utt_corr[4],
                "train/word_total_pcc": tr_word_corr[2],
                "test/word_total_pcc": te_word_corr[2],
                "train/per": tr_per,
                "test/per": te_per,
                "train/mdd_f1": tr_mdd[2],
                "test/mdd_f1": te_mdd[2],
            }
        )


def validate(
    audio_model: nn.Module,
    val_loader: DataLoader,
    args: argparse.Namespace,
    valid: bool = False,
) -> tuple[Any, ...]:
    device = require_cuda_device("HMamba validation")
    audio_model = audio_model.to(device)
    audio_model.eval()
    load_phn_dict(args.phn_dict)  # retained for compatibility with the original validation path

    all_phn, all_phn_target = [], []
    all_u1, all_u2, all_u3, all_u4, all_u5, all_utt_target = [], [], [], [], [], []
    all_w1, all_w2, all_w3, all_word_target = [], [], [], []
    all_recog, all_recog_target = [], []
    mdd_scores: list[tuple[float, float, float]] = []

    with torch.no_grad():
        for batch in val_loader:
            (
                audio_input,
                audio_input2,
                audio_input3,
                canophns,
                realphns,
                bies,
                phn_label,
                word_label,
                utt_label,
                _utt_id,
            ) = batch
            audio_input = audio_input.to(device)
            audio_input2 = [input2.to(device) for input2 in audio_input2]
            if isinstance(audio_input3, torch.Tensor):
                audio_input3 = audio_input3.to(device)
            canophns = canophns.to(device)
            realphns = realphns.to(device)
            bies = bies.to(device)
            pad_mask = (phn_label >= 0).to(device)

            u1, u2, u3, u4, u5, p, w1, w2, w3, logits = audio_model(
                audio_input,
                audio_input2,
                audio_input3,
                canophns,
                bies,
                mask=pad_mask,
            )

            mdd_scores.append(mdd_detection_metrics(logits, canophns, realphns, pad_mask))

            p = p.cpu().detach()
            u1, u2, u3, u4, u5 = (
                u1.cpu().detach(),
                u2.cpu().detach(),
                u3.cpu().detach(),
                u4.cpu().detach(),
                u5.cpu().detach(),
            )
            w1, w2, w3 = w1.cpu().detach(), w2.cpu().detach(), w3.cpu().detach()
            recogphns = torch.argmax(logits, dim=-1).cpu().detach().masked_select(pad_mask.cpu())
            realphns_cpu = realphns.cpu().detach().masked_select(pad_mask.cpu()).int()

            all_phn.append(p)
            all_phn_target.append(phn_label)
            all_u1.append(u1)
            all_u2.append(u2)
            all_u3.append(u3)
            all_u4.append(u4)
            all_u5.append(u5)
            all_utt_target.append(utt_label)
            all_w1.append(w1)
            all_w2.append(w2)
            all_w3.append(w3)
            all_word_target.append(word_label)
            all_recog.append(str(recogphns.tolist()))
            all_recog_target.append(str(realphns_cpu.tolist()))

    all_phn, all_phn_target = torch.cat(all_phn), torch.cat(all_phn_target)
    all_u1, all_u2, all_u3, all_u4, all_u5, all_utt_target = (
        torch.cat(all_u1),
        torch.cat(all_u2),
        torch.cat(all_u3),
        torch.cat(all_u4),
        torch.cat(all_u5),
        torch.cat(all_utt_target),
    )
    all_w1, all_w2, all_w3, all_word_target = (
        torch.cat(all_w1),
        torch.cat(all_w2),
        torch.cat(all_w3),
        torch.cat(all_word_target),
    )

    phn_mse, phn_corr = valid_phn(all_phn, all_phn_target)
    all_utt = torch.cat((all_u1, all_u2, all_u3, all_u4, all_u5), dim=1)
    utt_mse, utt_corr = valid_utt(all_utt, all_utt_target)
    all_word = torch.cat((all_w1, all_w2, all_w3), dim=2)
    word_mse, word_corr, word_stress, valid_word_pred, valid_word_target = valid_word(all_word, all_word_target)
    per = wer(all_recog_target, all_recog)
    mdd = tuple(float(np.mean([score[i] for score in mdd_scores])) for i in range(3))

    if valid:
        return (
            phn_mse,
            phn_corr,
            utt_mse,
            utt_corr,
            word_mse,
            word_corr,
            word_stress,
            per,
            mdd,
            all_phn_target,
            valid_word_target,
            all_utt_target,
            all_recog_target,
            all_phn,
            valid_word_pred,
            all_utt,
            all_recog,
        )
    return phn_mse, phn_corr, utt_mse, utt_corr, word_mse, word_corr, word_stress, per, mdd


def valid_phn(audio_output: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    valid_token_pred = []
    valid_token_target = []
    audio_output = audio_output.squeeze(2)
    for i in range(audio_output.shape[0]):
        for j in range(audio_output.shape[1]):
            if target[i, j] >= 0:
                valid_token_pred.append(audio_output[i, j])
                valid_token_target.append(target[i, j])
    valid_token_target_np = np.array(valid_token_target)
    valid_token_pred_np = np.array(valid_token_pred)
    valid_token_mse = np.mean((valid_token_target_np - valid_token_pred_np) ** 2)
    corr = np.corrcoef(valid_token_pred_np, valid_token_target_np)[0, 1]
    return float(valid_token_mse), float(corr)


def valid_utt(audio_output: torch.Tensor, target: torch.Tensor) -> tuple[list[float], list[float]]:
    mse = []
    corr = []
    for i in range(5):
        cur_mse = np.mean(((audio_output[:, i] - target[:, i]) ** 2).numpy())
        cur_corr = np.corrcoef(audio_output[:, i], target[:, i])[0, 1]
        mse.append(float(cur_mse))
        corr.append(float(cur_corr))
    return mse, corr


def valid_word(
    audio_output: torch.Tensor,
    target: torch.Tensor,
) -> tuple[list[float], list[float], list[float], np.ndarray, np.ndarray]:
    word_id = target[:, :, -1]
    target = target[:, :, 0:3]
    valid_token_pred = []
    valid_token_target = []

    for i in range(target.shape[0]):
        prev_w_id = 0
        start_id = 0
        for j in range(target.shape[1]):
            cur_w_id = word_id[i, j].int()
            if cur_w_id != prev_w_id:
                valid_token_pred.append(np.mean(audio_output[i, start_id:j, :].numpy(), axis=0))
                valid_token_target.append(np.mean(target[i, start_id:j, :].numpy(), axis=0))
                if cur_w_id == -1:
                    break
                prev_w_id = cur_w_id
                start_id = j

    valid_token_pred = np.array(valid_token_pred)
    valid_token_target = np.array(valid_token_target).round(2)

    mse_list = []
    corr_list = []
    for i in range(3):
        valid_token_mse = np.mean((valid_token_target[:, i] - valid_token_pred[:, i]) ** 2)
        corr = np.corrcoef(valid_token_pred[:, i], valid_token_target[:, i])[0, 1]
        mse_list.append(float(valid_token_mse))
        corr_list.append(float(corr))

    hyp = np.around(valid_token_pred[:, 1])
    ref = valid_token_target[:, 1]
    stress_list = [
        float(f1_score(ref, hyp, average="macro")),
        float(f1_score(ref, hyp, average="micro")),
    ]
    return mse_list, corr_list, stress_list, valid_token_pred, valid_token_target


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--lr", "--learning-rate", default=2e-3, type=float, metavar="LR", help="initial learning rate")
    parser.add_argument("--warmup-step", type=int, default=100, help="number of steps for warmup")
    parser.add_argument(
        "--phase-ratio",
        nargs=3,
        type=float,
        default=[0.4, 0.4, 0.2],
        help="Phase ratio used in the tri-stage scheduler.",
    )
    parser.add_argument("--init-lr-scale", type=float, default=1e-2)
    parser.add_argument("--final-lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=50, help="training batch size")
    parser.add_argument("--n-epochs", type=int, default=20, help="number of training epochs")
    parser.add_argument("--loss-w-phn", type=float, default=1)
    parser.add_argument("--loss-w-word", type=float, default=1)
    parser.add_argument("--loss-w-utt", type=float, default=1)
    parser.add_argument("--loss-type", type=str, default="dexent", choices=["xent", "dexent"])
    parser.add_argument("--loss-w-a", type=float, default=0.7)
    parser.add_argument("--loss-w-xent", type=float, default=0.003)
    parser.add_argument("--selection-metric", type=str, default="mdd_f1", choices=["phone_mse", "mdd_f1"])
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--model", type=str, default="HMamba")
    parser.add_argument("--model-conf", type=str, required=True)
    parser.add_argument("--am", type=str, default="librispeech")
    parser.add_argument("--gop-dir", type=str, required=True)
    parser.add_argument("--ssl-dir", type=str, default=None)
    parser.add_argument("--raw-dir", type=str, default=None)
    parser.add_argument("--exp-dir", type=str, required=True)
    parser.add_argument("--phn-dict", type=str, default="local/so762/vocab_merge.json")
    parser.add_argument(
        "--compile-model",
        action="store_true",
        help="Compile the top-level HMamba module with torch.compile.",
    )
    parser.add_argument("--compile-mode", type=str, default="default", help="torch.compile mode.")
    parser.add_argument("--compile-backend", type=str, default=None, help="Optional torch.compile backend override.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    device = require_cuda_device("HMamba training")
    maybe_init_wandb(args.exp_dir)

    print(f"I am process {os.getpid()}, running on {os.uname()[1]}: starting ({time.asctime()})")
    print(f"running on {device}")
    print(f"setting seed {args.seed}")
    set_seed(args.seed)
    print(f"now train with {args.am} acoustic models")

    conf = load_conf(args.model_conf)
    if args.model != "HMamba":
        raise ValueError(f"Invalid model {args.model}")

    audio_model = HMamba(**conf)
    print(f"resolved Mamba backend: {audio_model.resolved_mamba_backend}")
    tr_dataset = GoPDataset("train", data_dir=args.gop_dir, data_dir2=args.ssl_dir, data_dir3=args.raw_dir, am=args.am)
    te_dataset = GoPDataset("test", data_dir=args.gop_dir, data_dir2=args.ssl_dir, data_dir3=args.raw_dir, am=args.am)
    tr_dataloader = DataLoader(cast(Dataset[tuple[object, ...]], tr_dataset), batch_size=args.batch_size, shuffle=True)
    te_dataloader = DataLoader(
        cast(Dataset[tuple[object, ...]], te_dataset),
        batch_size=min(2500, len(te_dataset)),
        shuffle=False,
    )
    train(audio_model, tr_dataloader, te_dataloader, args)
    maybe_finish_wandb()


if __name__ == "__main__":
    main()
