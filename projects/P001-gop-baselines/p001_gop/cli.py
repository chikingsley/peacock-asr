"""CLI entry points for peacock-asr."""

from __future__ import annotations

import argparse
import logging
import sys

from p001_gop.scoring import _get_run_mode, cmd_prewarm_k2, cmd_run, cmd_sweep_alpha


def _register_run_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    run_p = subparsers.add_parser("run", help="Run GOP-SF with a backend")
    run_p.add_argument(
        "--backend",
        "-b",
        required=True,
        help="Backend: original, xlsr-espeak, or zipa",
    )
    run_p.add_argument(
        "--feats",
        action="store_true",
        help="Extract full feature vectors (LPP+LPR) and evaluate with SVR",
    )
    run_p.add_argument(
        "--gopt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Train GOPT transformer on feature vectors and evaluate",
    )
    run_p.add_argument(
        "--hmamba",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Train the P001 HMamba-style scorer on feature vectors and evaluate",
    )
    run_p.add_argument(
        "--device",
        default=None,
        help="Compute device: auto, cpu, cuda (default: auto)",
    )
    run_p.add_argument(
        "--limit",
        "-l",
        type=int,
        default=0,
        help="Limit utterances per split (0 = all)",
    )
    run_p.add_argument(
        "--no-cache",
        action="store_true",
        help="Skip feature cache (always re-extract)",
    )
    run_p.add_argument(
        "--workers",
        "-w",
        type=int,
        default=0,
        help="Parallel workers for GOP computation (0 = auto, 1 = sequential)",
    )
    run_p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed (used by GOPT training path)",
    )
    run_p.add_argument(
        "--score-variant",
        choices=["gop_sf", "logit_margin", "logit_combined"],
        default="gop_sf",
        help=(
            "Scalar score variant. Non-default variants currently support "
            "scalar mode only."
        ),
    )
    run_p.add_argument(
        "--score-alpha",
        type=float,
        default=0.5,
        help="Mixture weight for logit_combined in [0, 1] (default: 0.5).",
    )


def _register_sweep_alpha_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    sweep_p = subparsers.add_parser(
        "sweep-alpha",
        help="Fast scalar alpha sweep from cached gop_sf/logit_margin scores",
    )
    sweep_p.add_argument(
        "--backend",
        "-b",
        required=True,
        help="Backend cache name to sweep (e.g., xlsr-espeak (wav2vec2-...))",
    )
    sweep_p.add_argument(
        "--output-dir",
        default="artifacts/alpha-sweeps",
        help=(
            "Directory where sweep run folders are written "
            "(default: artifacts/alpha-sweeps)"
        ),
    )
    sweep_p.add_argument(
        "--source-alpha",
        type=float,
        default=0.5,
        help=(
            "Alpha used when reading source gop_sf/logit_margin caches "
            "(default: 0.5)."
        ),
    )
    sweep_p.add_argument(
        "--alphas",
        default=None,
        help=(
            "Optional comma-separated alpha values, e.g. 0,0.1,0.2. "
            "If provided, start/stop/step are ignored."
        ),
    )
    sweep_p.add_argument(
        "--alpha-start",
        type=float,
        default=0.0,
        help="Grid start alpha in [0, 1] when --alphas is not provided.",
    )
    sweep_p.add_argument(
        "--alpha-stop",
        type=float,
        default=1.0,
        help="Grid stop alpha in [0, 1] when --alphas is not provided.",
    )
    sweep_p.add_argument(
        "--alpha-step",
        type=float,
        default=0.05,
        help="Grid step for alpha when --alphas is not provided.",
    )


def _register_prewarm_k2_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    prewarm_p = subparsers.add_parser(
        "prewarm-k2",
        help="Populate the k2 topology cache for a backend",
    )
    prewarm_p.add_argument(
        "--backend",
        "-b",
        required=True,
        help="Backend: original, xlsr-espeak, or zipa",
    )
    prewarm_p.add_argument(
        "--split",
        choices=["train", "test", "both"],
        default="both",
    )
    prewarm_p.add_argument(
        "--limit",
        "-l",
        type=int,
        default=0,
        help="Limit utterances per split (0 = all)",
    )
    prewarm_p.add_argument(
        "--device",
        default=None,
        help="Compute device: auto, cpu, cuda (default: auto)",
    )


def main() -> None:
    from p001_gop.commands.batch import (  # noqa: PLC0415
        cmd_batch,
        register_batch_parser,
    )

    try:
        from dotenv import load_dotenv  # noqa: PLC0415

        load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(
        prog="peacock-asr",
        description="Pronunciation assessment benchmark",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    subparsers = parser.add_subparsers(dest="command", required=True)

    _register_run_parser(subparsers)
    register_batch_parser(subparsers)
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
        case "batch":
            cmd_batch(args, run_command=cmd_run, mode_name_fn=_get_run_mode)
        case "sweep-alpha":
            cmd_sweep_alpha(args)
        case "prewarm-k2":
            cmd_prewarm_k2(args)
        case _:
            parser.print_help()
            sys.exit(1)


if __name__ == "__main__":
    main()
