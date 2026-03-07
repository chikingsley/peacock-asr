from __future__ import annotations

import argparse
from pathlib import Path

from p004_training_from_scratch.canonical import (
    DEFAULT_MACHINE_OUTPUT,
    DEFAULT_OUTPUT,
    DEFAULT_SMOKE_AUDIO_LIST,
    run_canonical_preflight,
)

from ._common import dump_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the local canonical-lane preflight on the current machine."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Where to write the preflight JSON (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--machine-output",
        type=Path,
        default=DEFAULT_MACHINE_OUTPUT,
        help=(
            f"Where to write the machine manifest (default: {DEFAULT_MACHINE_OUTPUT})"
        ),
    )
    parser.add_argument(
        "--audio-list",
        type=Path,
        default=DEFAULT_SMOKE_AUDIO_LIST,
        help=(
            "Audio file list for the real-data smoke batch "
            f"(default: {DEFAULT_SMOKE_AUDIO_LIST})"
        ),
    )
    parser.add_argument(
        "--allow-online-trackers",
        action="store_true",
        help="Allow W&B online mode if credentials are present.",
    )
    parser.add_argument(
        "--without-wandb",
        action="store_true",
        help="Skip the W&B smoke check entirely.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    payload = run_canonical_preflight(
        output=args.output,
        machine_output=args.machine_output,
        audio_list_path=args.audio_list,
        allow_online_trackers=args.allow_online_trackers,
        with_wandb=not args.without_wandb,
    )
    dump_json(payload)
    return 0 if payload["overall_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
