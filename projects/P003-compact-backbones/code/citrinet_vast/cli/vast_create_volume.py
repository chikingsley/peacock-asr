from __future__ import annotations

import argparse

from citrinet_vast import VastClient

from ._common import dump_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a reusable Vast local volume from an existing instance."
    )
    parser.add_argument("--instance-id", type=int, required=True)
    parser.add_argument("--size-gb", type=float, default=150.0)
    parser.add_argument("--name", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    client = VastClient.from_env()
    result = client.create_volume(
        instance_id=args.instance_id,
        size_gb=args.size_gb,
        name=args.name,
    )
    dump_json(result.to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
