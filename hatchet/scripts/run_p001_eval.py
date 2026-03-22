"""Run one P001 evaluation and emit a machine-readable summary line."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from p001_gop.scoring.runtime import cmd_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single P001 evaluation for Hatchet orchestration.",
    )
    parser.add_argument("--backend", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--score-variant", default="gop_sf")
    parser.add_argument("--score-alpha", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime_args = argparse.Namespace(
        backend=args.backend,
        feats=False,
        gopt=True,
        hmamba=False,
        device=args.device,
        limit=args.limit,
        no_cache=args.no_cache,
        workers=args.workers,
        seed=args.seed,
        score_variant=args.score_variant,
        score_alpha=args.score_alpha,
        verbose=args.verbose,
    )
    start = perf_counter()
    eval_name, eval_result = cmd_run(runtime_args)
    payload = {
        "eval_name": eval_name,
        "metrics": asdict(eval_result),
        "elapsed_s": perf_counter() - start,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"JSON_RESULT::{json.dumps(payload, sort_keys=True)}")  # noqa: T201


if __name__ == "__main__":
    main()
