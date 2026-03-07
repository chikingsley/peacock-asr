from __future__ import annotations

import argparse
import time
from pathlib import Path

from p004_training_from_scratch.canonical.train_smoke import (
    DEFAULT_DEV_MANIFEST,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_TOKENS_PATH,
    DEFAULT_TRAIN_MANIFEST,
    run_canonical_train_smoke,
)
from p004_training_from_scratch.cli._common import dump_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the local canonical-lane CTC smoke on real phone targets.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument("--dev-manifest", type=Path, default=DEFAULT_DEV_MANIFEST)
    parser.add_argument("--tokens", type=Path, default=DEFAULT_TOKENS_PATH)
    parser.add_argument("--train-limit", type=int, default=12)
    parser.add_argument("--dev-limit", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument(
        "--model-type",
        choices=("tiny", "conformer_like"),
        default="tiny",
    )
    parser.add_argument(
        "--attention-backend",
        choices=("mha", "flex_auto", "flex_triton", "flex_flash"),
        default="mha",
    )
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--encoder-layers", type=int, default=4)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--conv-kernel-size", type=int, default=15)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--loss-compute-dtype",
        choices=("model", "float32"),
        default="model",
        help="Compute CTC log-probs/loss in model dtype or force float32.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help=(
            "Resume from a prior canonical checkpoint. `--epochs` is treated as "
            "the total target epoch count, not the number of extra epochs."
        ),
    )
    parser.add_argument(
        "--disable-compile",
        action="store_true",
        help="Run the canonical trainer without torch.compile.",
    )
    parser.add_argument("--allow-online-trackers", action="store_true")
    parser.add_argument("--disable-wandb", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    run_id = args.run_id or time.strftime("canonical_local_smoke_%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / run_id)
    result = run_canonical_train_smoke(
        output_dir=output_dir,
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
        enable_compile=not args.disable_compile,
        allow_online_trackers=args.allow_online_trackers,
        with_wandb=not args.disable_wandb,
    )
    dump_json(result)
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
