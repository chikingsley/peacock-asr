from __future__ import annotations

import argparse
from pathlib import Path

from p004_training_from_scratch.cli._common import dump_json
from p004_training_from_scratch.reference_checks import (
    DEFAULT_OUTPUT,
    run_reference_checks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the P004 reference-lane preflight checks and write JSON."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Where to write the JSON summary (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dump_json(run_reference_checks(output=args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
