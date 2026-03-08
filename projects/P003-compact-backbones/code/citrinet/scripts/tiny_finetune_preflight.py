#!/usr/bin/env python3
"""Run a tiny direct-adaptation fine-tune preflight for Citrinet P2-B."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightning.pytorch as pl
import nemo.collections.asr as nemo_asr
import torch
from omegaconf import open_dict

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
        default=REPO_ROOT / Path(
            "projects/P003-compact-backbones/code/citrinet/tokenizers/arpabet_41_wpe"
        ),
    )
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=REPO_ROOT / Path(
            "projects/P003-compact-backbones/experiments/citrinet/preflight/manifests/train.jsonl"
        ),
    )
    parser.add_argument(
        "--eval-manifest",
        type=Path,
        default=REPO_ROOT / Path(
            "projects/P003-compact-backbones/experiments/citrinet/preflight/manifests/eval.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / Path(
            "projects/P003-compact-backbones/experiments/citrinet/preflight/tiny_finetune"
        ),
    )
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    return parser.parse_args()


def _data_cfg(manifest: Path, batch_size: int, *, shuffle: bool) -> dict[str, object]:
    return {
        "manifest_filepath": str(manifest.resolve()),
        "sample_rate": 16000,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "trim_silence": False,
        "use_start_end_token": False,
        "num_workers": 2,
        "pin_memory": torch.cuda.is_available(),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = nemo_asr.models.ASRModel.from_pretrained(args.model_name)
    model.change_vocabulary(
        new_tokenizer_dir=str(args.tokenizer_dir.resolve()),
        new_tokenizer_type="wpe",
    )

    with open_dict(model.cfg.optim):
        model.cfg.optim.lr = args.lr
        if "sched" in model.cfg.optim and model.cfg.optim.sched is not None:
            model.cfg.optim.sched.warmup_steps = 1

    train_cfg = _data_cfg(args.train_manifest, args.batch_size, shuffle=True)
    eval_cfg = _data_cfg(args.eval_manifest, args.batch_size, shuffle=False)
    model.setup_training_data(train_cfg)
    model.setup_validation_data(
        eval_cfg
    )

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_steps=args.max_steps,
        logger=False,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        limit_val_batches=1,
        log_every_n_steps=1,
        default_root_dir=str(args.output_dir),
    )
    trainer.fit(model)

    report = {
        "model_name": args.model_name,
        "tokenizer_dir": str(args.tokenizer_dir.resolve()),
        "train_manifest": str(args.train_manifest.resolve()),
        "eval_manifest": str(args.eval_manifest.resolve()),
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "decoder_num_classes": len(model.decoder.vocabulary),
        "tokenizer_vocab_size": int(model.tokenizer.vocab_size),
        "global_step": int(trainer.global_step),
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sys.stdout.write(f"{report_path}\n")


if __name__ == "__main__":
    main()
