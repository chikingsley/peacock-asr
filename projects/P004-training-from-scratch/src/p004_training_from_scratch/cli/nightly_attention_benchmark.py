from __future__ import annotations

import argparse
import time
from pathlib import Path

from p004_training_from_scratch.cli._common import dump_json
from p004_training_from_scratch.nightly.attention_benchmark import (
    DEFAULT_OUTPUT_ROOT,
    run_nightly_attention_benchmark,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the nightly C2.3 attention benchmark on the local GPU.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--timed-iters", type=int, default=1)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    run_id = args.run_id or time.strftime("c2_3_nightly_attention_%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / run_id)
    result = run_nightly_attention_benchmark(
        output_dir=output_dir,
        batch_size=args.batch_size,
        attention_heads=args.attention_heads,
        seq_len=args.seq_len,
        head_dim=args.head_dim,
        warmup_iters=args.warmup_iters,
        timed_iters=args.timed_iters,
    )
    dump_json(result)
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
