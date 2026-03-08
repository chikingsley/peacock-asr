from __future__ import annotations

import argparse

from citrinet_vast import VastClient

from ._common import dump_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Show Vast volumes available to the P003 Citrinet account."
    )
    parser.add_argument(
        "--type",
        default="all",
        help="Volume type filter passed to Vast show_volumes().",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    client = VastClient.from_env()
    volumes = client.show_volumes(volume_type=args.type)
    dump_json([volume.to_dict() for volume in volumes])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
