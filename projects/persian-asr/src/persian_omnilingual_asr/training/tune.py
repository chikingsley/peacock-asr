"""CLI: sweep batch budgets for an Omni config on this GPU to maximize throughput.

Entry point ``persian-omni-tune``. Thin wrapper over :mod:`omni_finetune_core.tune`: points
the HF/fairseq2 caches at this project (like ``train.py``) so trials reuse the downloaded
model, runs each ``(max_audio_len, max_num_elements)`` candidate as an isolated subprocess,
and prints a table + recommendation. Needs the GPU free (don't run alongside a training job).

  # at the 30s cap the current 1B run uses, find the largest budget that fits:
  persian-omni-tune --config-file <1b-config.yaml> \
      --max-audio-len 480_000 --max-num-elements 960_000 1_440_000 1_920_000

  # try the full 40s inference cap too (more clip coverage, more memory):
  persian-omni-tune --config-file <cfg> \
      --max-audio-len 480_000 640_000 --max-num-elements 640_000 960_000 1_280_000
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

from omni_finetune_core.train import configure_environment
from omni_finetune_core.tune import format_report, sweep

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = ROOT / "runs" / "_tune"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Find the max batch budget that fits this GPU.")
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument(
        "--max-num-elements", type=int, nargs="+", required=True, help="batch-budget values to try"
    )
    parser.add_argument(
        "--max-audio-len",
        type=int,
        nargs="+",
        default=[480_000],
        help="clip-length cap(s) in samples; 480000=30s, 640000=40s (Omni inference cap)",
    )
    parser.add_argument("--steps", type=int, default=8, help="train steps per trial")
    parser.add_argument("--mem-ceiling", type=float, default=0.9, help="max peak reserved fraction")
    parser.add_argument("--timeout", type=float, default=600.0, help="per-trial seconds")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("recipe_args", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_environment(ROOT)

    extra = list(args.recipe_args)
    if extra and extra[0] == "--":
        extra = extra[1:]
    # Always run local single-GPU; let the user append more bare key=value overrides.
    overrides = ["common.cluster=none", *extra]

    # Candidate grid: a budget must hold at least one max-length clip, else it rounds to 0.
    candidates = [
        (mal, mne)
        for mal, mne in itertools.product(args.max_audio_len, args.max_num_elements)
        if mne >= mal
    ]

    results = sweep(
        args.config_file.resolve(),
        candidates,
        output_root=args.output_root,
        steps=args.steps,
        timeout_s=args.timeout,
        extra_overrides=overrides,
    )
    print(format_report(results, mem_ceiling=args.mem_ceiling))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
