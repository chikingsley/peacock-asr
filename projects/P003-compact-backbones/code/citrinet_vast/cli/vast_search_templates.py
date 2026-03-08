from __future__ import annotations

import argparse

from citrinet_vast import VastClient

from ._common import dump_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search Vast templates for P003 Citrinet using the typed SDK wrapper."
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Optional Vast template search query.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    client = VastClient.from_env()
    templates = client.search_templates(query=args.query)
    dump_json([template.to_dict() for template in templates])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
