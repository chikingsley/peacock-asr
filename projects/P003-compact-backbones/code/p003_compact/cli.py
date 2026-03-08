"""CLI entry points for P003 evaluation."""

from __future__ import annotations

import argparse
import logging
import sys

from p003_compact.scoring import cmd_prewarm_k2, cmd_run, cmd_sweep_alpha


def _register_run_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    run_p = subparsers.add_parser("run", help="Evaluate a phoneme backend")
    run_p.add_argument(
        "--backend",
        "-b",
        required=True,
        help="Backend: hf:<repo_id> or nemo:<path-or-repo>",
    )
    run_p.add_argument("--feats", action="store_true")
    run_p.add_argument(
        "--gopt",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    run_p.add_argument("--device", default=None)
    run_p.add_argument("--limit", "-l", type=int, default=0)
    run_p.add_argument("--no-cache", action="store_true")
    run_p.add_argument("--workers", "-w", type=int, default=0)
    run_p.add_argument("--seed", type=int, default=None)
    run_p.add_argument(
        "--score-variant",
        choices=["gop_sf", "logit_margin", "logit_combined"],
        default="gop_sf",
    )
    run_p.add_argument("--score-alpha", type=float, default=0.5)


def _register_sweep_alpha_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    sweep_p = subparsers.add_parser("sweep-alpha")
    sweep_p.add_argument("--backend", "-b", required=True)
    sweep_p.add_argument("--output-dir", default="artifacts/alpha-sweeps")
    sweep_p.add_argument("--source-alpha", type=float, default=0.5)
    sweep_p.add_argument("--alphas", default=None)
    sweep_p.add_argument("--alpha-start", type=float, default=0.0)
    sweep_p.add_argument("--alpha-stop", type=float, default=1.0)
    sweep_p.add_argument("--alpha-step", type=float, default=0.05)


def _register_prewarm_k2_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    prewarm_p = subparsers.add_parser(
        "prewarm-k2",
        help="Populate prepared-input and k2 topology caches",
    )
    prewarm_p.add_argument(
        "--backend",
        "-b",
        required=True,
        help="Backend: hf:<repo_id> or nemo:<path-or-repo>",
    )
    prewarm_p.add_argument(
        "--split",
        choices=["train", "test", "both"],
        default="both",
    )
    prewarm_p.add_argument("--limit", "-l", type=int, default=0)
    prewarm_p.add_argument("--device", default=None)


def main() -> None:
    try:
        from dotenv import load_dotenv  # noqa: PLC0415

        load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(
        prog="peacock-asr",
        description="Compact-backbone evaluation",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    _register_run_parser(subparsers)
    _register_sweep_alpha_parser(subparsers)
    _register_prewarm_k2_parser(subparsers)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    match args.command:
        case "run":
            cmd_run(args)
        case "sweep-alpha":
            cmd_sweep_alpha(args)
        case "prewarm-k2":
            cmd_prewarm_k2(args)
        case _:
            parser.print_help()
            sys.exit(1)


if __name__ == "__main__":
    main()
