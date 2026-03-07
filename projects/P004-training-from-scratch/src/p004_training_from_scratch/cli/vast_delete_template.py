from __future__ import annotations

import argparse

from p004_training_from_scratch.cli._common import dump_json
from p004_training_from_scratch.vast import VastClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Delete a Vast template and remove it from the local registry."
    )
    parser.add_argument("--name", help="Template name.")
    parser.add_argument("--template-id", type=int, help="Vast template ID.")
    parser.add_argument("--hash-id", help="Vast template hash ID.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.name is None and args.template_id is None and args.hash_id is None:
        msg = "Pass at least one of --name, --template-id, or --hash-id."
        raise SystemExit(msg)
    client = VastClient.from_env()
    result = client.delete_template(
        name=args.name,
        template_id=args.template_id,
        hash_id=args.hash_id,
    )
    dump_json(result.to_dict())


if __name__ == "__main__":
    main()
