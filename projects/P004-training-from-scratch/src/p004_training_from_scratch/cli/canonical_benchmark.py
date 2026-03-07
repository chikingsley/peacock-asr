from __future__ import annotations

import argparse
import time
from pathlib import Path

from p004_training_from_scratch.canonical.benchmark import (
    DEFAULT_OUTPUT_ROOT,
    run_canonical_benchmark,
)
from p004_training_from_scratch.canonical.train_smoke import (
    DEFAULT_DEV_MANIFEST,
    DEFAULT_TOKENS_PATH,
    DEFAULT_TRAIN_MANIFEST,
)
from p004_training_from_scratch.cli._common import dump_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the stable canonical lane with compile disabled and enabled."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument("--dev-manifest", type=Path, default=DEFAULT_DEV_MANIFEST)
    parser.add_argument("--tokens", type=Path, default=DEFAULT_TOKENS_PATH)
    parser.add_argument("--train-limit", type=int, default=24)
    parser.add_argument("--dev-limit", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--model-type",
        choices=("tiny", "conformer_like"),
        default="conformer_like",
    )
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--encoder-layers", type=int, default=3)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--conv-kernel-size", type=int, default=15)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sdpa-batch-size", type=int, default=4)
    parser.add_argument("--sdpa-seq-len", type=int, default=1024)
    parser.add_argument("--sdpa-warmup-iters", type=int, default=5)
    parser.add_argument("--sdpa-timed-iters", type=int, default=20)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    output_dir = args.output_dir or (
        DEFAULT_OUTPUT_ROOT / time.strftime("c2_2_compile_sdpa_%Y%m%d_%H%M%S")
    )
    payload = run_canonical_benchmark(
        output_dir=output_dir,
        train_manifest=args.train_manifest,
        dev_manifest=args.dev_manifest,
        tokens_path=args.tokens,
        train_limit=args.train_limit,
        dev_limit=args.dev_limit,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_type=args.model_type,
        hidden_dim=args.hidden_dim,
        encoder_layers=args.encoder_layers,
        attention_heads=args.attention_heads,
        conv_kernel_size=args.conv_kernel_size,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        seed=args.seed,
        sdpa_batch_size=args.sdpa_batch_size,
        sdpa_seq_len=args.sdpa_seq_len,
        sdpa_warmup_iters=args.sdpa_warmup_iters,
        sdpa_timed_iters=args.sdpa_timed_iters,
    )
    dump_json(payload)
    return 0 if payload["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
