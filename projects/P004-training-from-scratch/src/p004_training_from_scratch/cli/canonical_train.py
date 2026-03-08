from __future__ import annotations

import argparse
import time
from pathlib import Path

from p004_training_from_scratch.canonical.trainer import (
    DEFAULT_OUTPUT_ROOT,
    FROZEN_STABLE_TRAIN_CONFIG,
    build_stable_train_config,
    run_canonical_train,
)
from p004_training_from_scratch.cli._common import dump_json


def build_parser() -> argparse.ArgumentParser:
    config = FROZEN_STABLE_TRAIN_CONFIG
    parser = argparse.ArgumentParser(
        description=(
            "Run the promoted stable canonical trainer with frozen production "
            "defaults for the current Blackwell-compatible lane."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--train-manifest", type=Path, default=config.train_manifest)
    parser.add_argument("--dev-manifest", type=Path, default=config.dev_manifest)
    parser.add_argument("--tokens", type=Path, default=config.tokens_path)
    parser.add_argument("--train-limit", type=int, default=config.train_limit)
    parser.add_argument("--dev-limit", type=int, default=config.dev_limit)
    parser.add_argument("--epochs", type=int, default=config.epochs)
    parser.add_argument("--batch-size", type=int, default=config.batch_size)
    parser.add_argument(
        "--model-type",
        choices=("tiny", "conformer_like", "conformer"),
        default=config.model_type,
    )
    parser.add_argument(
        "--attention-backend",
        choices=("mha", "flex_auto", "flex_triton", "flex_flash"),
        default=config.attention_backend,
    )
    parser.add_argument("--hidden-dim", type=int, default=config.hidden_dim)
    parser.add_argument("--encoder-layers", type=int, default=config.encoder_layers)
    parser.add_argument("--attention-heads", type=int, default=config.attention_heads)
    parser.add_argument(
        "--conv-kernel-size",
        type=int,
        default=config.conv_kernel_size,
    )
    parser.add_argument("--dropout", type=float, default=config.dropout)
    parser.add_argument("--learning-rate", type=float, default=config.learning_rate)
    parser.add_argument(
        "--loss-compute-dtype",
        choices=("model", "float32"),
        default=config.loss_compute_dtype,
        help="Compute CTC log-probs/loss in model dtype or force float32.",
    )
    parser.add_argument("--seed", type=int, default=config.seed)
    parser.add_argument("--num-workers", type=int, default=config.num_workers)
    parser.add_argument(
        "--bucket-size-multiplier",
        type=int,
        default=config.bucket_size_multiplier,
    )
    parser.add_argument(
        "--disable-pin-memory",
        action="store_true",
        help="Disable DataLoader pin_memory for debugging or constrained hosts.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=config.resume_from,
        help=(
            "Resume from a prior canonical checkpoint. `--epochs` is treated as "
            "the total target epoch count, not the number of extra epochs."
        ),
    )
    parser.add_argument(
        "--enable-compile",
        action="store_true",
        default=config.enable_compile,
        help=(
            "Enable torch.compile. The frozen stable production baseline keeps "
            "this off."
        ),
    )
    parser.add_argument(
        "--disallow-online-trackers",
        action="store_true",
        help="Force local/offline tracker mode even if W&B credentials are present.",
    )
    parser.add_argument("--disable-wandb", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    run_id = args.run_id or time.strftime("canonical_train_%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / run_id)
    config = build_stable_train_config(
        train_manifest=args.train_manifest,
        dev_manifest=args.dev_manifest,
        tokens_path=args.tokens,
        train_limit=args.train_limit,
        dev_limit=args.dev_limit,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_type=args.model_type,
        attention_backend=args.attention_backend,
        hidden_dim=args.hidden_dim,
        encoder_layers=args.encoder_layers,
        attention_heads=args.attention_heads,
        conv_kernel_size=args.conv_kernel_size,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        loss_compute_dtype=args.loss_compute_dtype,
        seed=args.seed,
        resume_from=args.resume_from,
        enable_compile=args.enable_compile,
        num_workers=args.num_workers,
        bucket_size_multiplier=args.bucket_size_multiplier,
        pin_memory=not args.disable_pin_memory,
        allow_online_trackers=not args.disallow_online_trackers,
        with_wandb=not args.disable_wandb,
    )
    result = run_canonical_train(output_dir=output_dir, config=config)
    dump_json(result)
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
