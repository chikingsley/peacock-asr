#!/usr/bin/env python3
"""Run a real Citrinet P2-B fine-tune with NeMo."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from importlib import import_module
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-name",
        default="nvidia/stt_en_citrinet_256_ls",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=REPO_ROOT
        / "projects/P003-compact-backbones/code/citrinet/tokenizers/arpabet_41_wpe",
    )
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=REPO_ROOT
        / (
            "projects/P003-compact-backbones/experiments/citrinet/"
            "train_clean_100_full/manifests/train.jsonl"
        ),
    )
    parser.add_argument(
        "--eval-manifest",
        type=Path,
        default=REPO_ROOT
        / (
            "projects/P003-compact-backbones/experiments/citrinet/"
            "train_clean_100_full/manifests/eval.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT
        / (
            "projects/P003-compact-backbones/experiments/citrinet/checkpoints/"
            "citrinet_256_p2b_train_clean_100"
        ),
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--resume-from-checkpoint", type=Path, default=None)
    parser.add_argument("--save-every-n-steps", type=int, default=1000)
    parser.add_argument("--log-every-n-steps", type=int, default=10)
    parser.add_argument("--limit-train-batches", type=float, default=1.0)
    parser.add_argument("--limit-val-batches", type=float, default=1.0)
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser.add_argument("--precision", default="16-mixed")
    parser.add_argument("--wandb-project", default="peacock-asr-p003-citrinet")
    parser.add_argument("--wandb-entity", default="peacockery")
    parser.add_argument("--wandb-group", default="p003-citrinet-p2b")
    parser.add_argument("--wandb-mode", default="online")
    return parser.parse_args()


def _data_cfg(
    manifest: Path,
    batch_size: int,
    *,
    shuffle: bool,
    num_workers: int,
) -> dict[str, object]:
    return {
        "manifest_filepath": str(manifest.resolve()),
        "sample_rate": 16000,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "trim_silence": False,
        "use_start_end_token": False,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }


def _lightning_module() -> Any:
    return import_module("lightning.pytorch")


def _nemo_module() -> Any:
    return import_module("nemo.collections.asr")


def _build_logger(args: argparse.Namespace, output_dir: Path) -> Any | bool:
    if args.wandb_mode == "disabled":
        return False
    wandb_logger_cls = import_module("lightning.pytorch.loggers").WandbLogger
    os.environ.setdefault("WANDB_MODE", args.wandb_mode)
    os.environ.setdefault("WANDB_DIR", str((output_dir / "wandb").resolve()))
    return wandb_logger_cls(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.run_name,
        group=args.wandb_group,
        save_dir=str(output_dir.resolve()),
    )


def _trainer_kwargs(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    callbacks_module = import_module("lightning.pytorch.callbacks")
    lr_monitor_cls = callbacks_module.LearningRateMonitor
    model_checkpoint_cls = callbacks_module.ModelCheckpoint
    use_cuda = torch.cuda.is_available()
    checkpoint_dir = output_dir / "checkpoints"
    callbacks: list[object] = [
        model_checkpoint_cls(
            dirpath=str(checkpoint_dir),
            filename="step-{step:06d}",
            every_n_train_steps=args.save_every_n_steps,
            save_top_k=-1,
            save_last=True,
        )
    ]
    logger = _build_logger(args, output_dir)
    if logger:
        callbacks.append(lr_monitor_cls(logging_interval="step"))

    kwargs: dict[str, Any] = {
        "accelerator": "gpu" if use_cuda else "cpu",
        "devices": 1,
        "logger": logger,
        "enable_checkpointing": True,
        "callbacks": callbacks,
        "default_root_dir": str(output_dir),
        "num_sanity_val_steps": 0,
        "log_every_n_steps": args.log_every_n_steps,
        "accumulate_grad_batches": args.accumulate_grad_batches,
        "limit_train_batches": args.limit_train_batches,
        "limit_val_batches": args.limit_val_batches,
        "gradient_clip_val": args.gradient_clip_val,
    }
    if use_cuda:
        kwargs["precision"] = args.precision
    if args.max_steps > 0:
        kwargs["max_steps"] = args.max_steps
        kwargs["max_epochs"] = -1
    else:
        kwargs["max_epochs"] = args.max_epochs
    return kwargs


def _report_payload(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    model: Any,
    trainer: Any,
    nemo_path: Path,
) -> dict[str, Any]:
    return {
        "model_name": args.model_name,
        "tokenizer_dir": str(args.tokenizer_dir.resolve()),
        "train_manifest": str(args.train_manifest.resolve()),
        "eval_manifest": str(args.eval_manifest.resolve()),
        "output_dir": str(output_dir.resolve()),
        "run_name": args.run_name,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "accumulate_grad_batches": args.accumulate_grad_batches,
        "num_workers": args.num_workers,
        "lr": args.lr,
        "warmup_steps": args.warmup_steps,
        "max_epochs": args.max_epochs,
        "max_steps": args.max_steps,
        "resume_from_checkpoint": (
            str(args.resume_from_checkpoint.resolve())
            if args.resume_from_checkpoint is not None
            else None
        ),
        "precision": args.precision if torch.cuda.is_available() else "32-true",
        "global_step": int(trainer.global_step),
        "decoder_num_classes": len(model.decoder.vocabulary),
        "tokenizer_vocab_size": int(model.tokenizer.vocab_size),
        "nemo_artifact": str(nemo_path.resolve()),
        "finished_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume_from_checkpoint is not None:
        args.resume_from_checkpoint = args.resume_from_checkpoint.resolve()
    if args.run_name is None:
        args.run_name = output_dir.name

    pl = _lightning_module()
    nemo_asr = _nemo_module()
    open_dict = import_module("omegaconf").open_dict

    pl.seed_everything(args.seed, workers=True)

    model = nemo_asr.models.ASRModel.from_pretrained(args.model_name)
    model.change_vocabulary(
        new_tokenizer_dir=str(args.tokenizer_dir.resolve()),
        new_tokenizer_type="wpe",
    )

    with open_dict(model.cfg.optim):
        model.cfg.optim.lr = args.lr
        if "sched" in model.cfg.optim and model.cfg.optim.sched is not None:
            model.cfg.optim.sched.warmup_steps = args.warmup_steps

    train_cfg = _data_cfg(
        args.train_manifest,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    eval_cfg = _data_cfg(
        args.eval_manifest,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    model.setup_training_data(train_cfg)
    model.setup_validation_data(eval_cfg)

    trainer = pl.Trainer(**_trainer_kwargs(args, output_dir))
    trainer.fit(
        model,
        ckpt_path=(
            str(args.resume_from_checkpoint)
            if args.resume_from_checkpoint is not None
            else None
        ),
    )

    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    nemo_path = artifacts_dir / f"{args.run_name}.nemo"
    model.save_to(str(nemo_path))

    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(
            _report_payload(
                args=args,
                output_dir=output_dir,
                model=model,
                trainer=trainer,
                nemo_path=nemo_path,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    sys.stdout.write(f"{report_path}\n")


if __name__ == "__main__":
    main()
