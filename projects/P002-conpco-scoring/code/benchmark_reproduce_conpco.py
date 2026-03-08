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
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, cast

import reproduce_conpco as rc
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

if TYPE_CHECKING:
    from collections.abc import Callable

MIN_P95_SAMPLE_COUNT = 20

type PreconcatBatch = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]

type BenchmarkBatch = rc.GoPBatch | PreconcatBatch


class PreconcatGoPDataset(Dataset[PreconcatBatch]):
    def __init__(self, split: str, data_dir: Path) -> None:
        base = rc.GoPDataset(split, data_dir)
        self.feat = base.feat
        self.eng = base.feat_energy
        self.dur = base.feat_dur
        self.ssl = torch.cat([base.feat_ssl2, base.feat_ssl1, base.feat_ssl3], dim=-1)
        self.phn_label = base.phn_label[:, :, 1]
        self.phns = base.phn_label[:, :, 0]
        self.utt_label = base.utt_label
        self.word_label = base.word_label
        self.word_id = base.word_id

    def __len__(self) -> int:
        return self.feat.shape[0]

    def __getitem__(self, idx: object) -> PreconcatBatch:
        item = cast("int", idx)
        return (
            self.feat[item],
            self.ssl[item],
            self.eng[item],
            self.dur[item],
            self.phn_label[item],
            self.phns[item],
            self.utt_label[item],
            self.word_label[item],
            self.word_id[item],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Micro-benchmark the P002 ConPCO reproduction path.",
    )
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable DataLoader pin_memory",
    )
    parser.add_argument(
        "--preconcat-ssl",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Concatenate the three SSL tensors once during dataset build",
    )
    parser.add_argument(
        "--data-parallel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wrap the model in nn.DataParallel to mirror the reproduction path",
    )
    parser.add_argument(
        "--conpco",
        choices=["on", "off"],
        default="on",
        help="Include ConPCO loss terms in the benchmark step",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save the benchmark result as JSON",
    )
    return parser.parse_args()


def build_dataset(
    split: str,
    *,
    preconcat_ssl: bool,
    data_dir: Path,
) -> tuple[Dataset[object], float]:
    started = time.perf_counter()
    dataset = (
        PreconcatGoPDataset(split, data_dir)
        if preconcat_ssl
        else rc.GoPDataset(split, data_dir)
    )
    duration = time.perf_counter() - started
    return dataset, duration


def move_batch(
    batch: BenchmarkBatch,
    device: torch.device,
    *,
    preconcat_ssl: bool,
) -> dict[str, torch.Tensor]:
    if preconcat_ssl:
        feat, ssl, eng, dur, phn_label, phns, utt_label, word_label, word_id = batch
    else:
        (
            feat,
            ssl1,
            ssl2,
            ssl3,
            eng,
            dur,
            phn_label,
            phns,
            utt_label,
            word_label,
            word_id,
        ) = batch
        ssl = torch.cat([ssl2, ssl1, ssl3], dim=-1)

    return {
        "feat": feat.to(device, non_blocking=True),
        "ssl": ssl.to(device, non_blocking=True),
        "eng": eng.to(device, non_blocking=True),
        "dur": dur.to(device, non_blocking=True),
        "phn_label": phn_label.to(device, non_blocking=True),
        "phns": phns.to(device, non_blocking=True),
        "utt_label": utt_label.to(device, non_blocking=True),
        "word_label": word_label.to(device, non_blocking=True),
        "word_id": word_id.to(device, non_blocking=True),
    }


def build_model(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[
    nn.Module,
    torch.optim.Optimizer,
    torch.nn.MSELoss,
    Callable[..., tuple[torch.Tensor, torch.Tensor]] | None,
]:
    input_dim = 84 + 7 + 1
    model = rc.HierCB(
        embed_dim=12,
        num_heads=1,
        p_depth=1,
        w_depth=1,
        u_depth=1,
        ssl_drop=0.1,
        input_dim=input_dim,
    )
    if args.data_parallel and torch.cuda.device_count() > 0:
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3,
        weight_decay=5e-7,
        betas=(0.95, 0.999),
    )
    loss_fn = torch.nn.MSELoss()
    loss_pco: Callable[..., tuple[torch.Tensor, torch.Tensor]] | None = None
    if args.conpco == "on":
        loss_pco = rc.ContrastivePhonemicOrdinalRegularizer(5.0, 0.1, 0.1, 1.0)
    return model, optimizer, loss_fn, loss_pco


def run_steps(
    *,
    loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    model, optimizer, loss_fn, loss_pco = build_model(args, device)
    iterator = iter(loader)
    first_batch_started = time.perf_counter()
    first_batch = next(iterator)
    first_batch_s = time.perf_counter() - first_batch_started

    step_times: list[float] = []
    effective_steps = 0
    queued_batches = [first_batch, *list(loader)[: max(args.steps - 1, 0)]]

    for loaded_batch in queued_batches:
        batch = move_batch(loaded_batch, device, preconcat_ssl=args.preconcat_ssl)
        step_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)

        u1, u2, u3, u4, u5, p, w1, w2, w3, phn_audio, phn_text = model(
            batch["feat"],
            batch["eng"],
            batch["dur"],
            batch["ssl"],
            batch["phns"],
            batch["word_label"][:, :, -1],
            batch["word_id"],
        )

        mask = batch["phns"] >= 0
        p = p.squeeze(2) * mask
        phn_label_masked = batch["phn_label"] * mask
        loss_phn = loss_fn(p, phn_label_masked) * (mask.numel() / mask.sum())

        word_scores = batch["word_label"][:, :, 0:3]
        wmask = word_scores >= 0
        word_pred = torch.cat([w1, w2, w3], dim=2) * wmask
        word_scores_masked = word_scores * wmask
        loss_word = loss_fn(word_pred, word_scores_masked) * (
            wmask.numel() / wmask.sum()
        )

        utt_pred = torch.cat([u1, u2, u3, u4, u5], dim=1)
        loss_utt = loss_fn(utt_pred, batch["utt_label"])

        loss = loss_phn + loss_word + loss_utt
        if loss_pco is not None:
            loss_phn_pco, loss_clap = loss_pco(
                phn_audio,
                phn_text,
                phn_label_masked,
                batch["phns"],
            )
            loss = loss + loss_phn_pco + loss_clap

        loss.backward()
        optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_times.append(time.perf_counter() - step_started)
        effective_steps += 1
        if effective_steps >= args.steps:
            break

    return {
        "first_batch_s": first_batch_s,
        "mean_step_s": statistics.fmean(step_times),
        "p95_step_s": (
            max(step_times)
            if len(step_times) < MIN_P95_SAMPLE_COUNT
            else statistics.quantiles(step_times, n=MIN_P95_SAMPLE_COUNT)[-1]
        ),
        "steps": effective_steps,
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = rc.CONPCO_ROOT / "data" / "seq_data_librispeech_v4"

    dataset, dataset_load_s = build_dataset(
        args.split,
        preconcat_ssl=args.preconcat_ssl,
        data_dir=data_dir,
    )

    loader_started = time.perf_counter()
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.split == "train",
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    loader_build_s = time.perf_counter() - loader_started

    stats = run_steps(loader=loader, args=args, device=device)
    result = {
        "split": args.split,
        "device": str(device),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "preconcat_ssl": args.preconcat_ssl,
        "data_parallel": args.data_parallel,
        "conpco": args.conpco,
        "dataset_load_s": dataset_load_s,
        "loader_build_s": loader_build_s,
        **stats,
    }

    sys.stdout.write(f"{json.dumps(result, indent=2, sort_keys=True)}\n")
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
