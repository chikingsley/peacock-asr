from __future__ import annotations

import argparse

from citrinet_vast import VastClient

from ._common import dump_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Destroy a Vast instance for P003 Citrinet using the typed SDK wrapper."
    )
    parser.add_argument(
        "--instance-id",
        type=int,
        required=True,
        help="Vast instance ID to destroy.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    client = VastClient.from_env()
    result = client.destroy_instance(instance_id=args.instance_id)
    dump_json(result.to_dict())
    return 0 if result.destroyed else 1


if __name__ == "__main__":
    raise SystemExit(main())
