from __future__ import annotations

import argparse
from pathlib import Path

from p004_training_from_scratch.cli._common import dump_json
from p004_training_from_scratch.machine_manifest import (
    DEFAULT_OUTPUT,
    capture_machine_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture a local machine manifest for P004."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Where to write the machine manifest (default: {DEFAULT_OUTPUT})",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    dump_json(capture_machine_manifest(output=args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
