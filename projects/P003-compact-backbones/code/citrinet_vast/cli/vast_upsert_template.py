from __future__ import annotations

import argparse

from citrinet_vast import TemplateSpec, VastClient

from ._common import dump_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create or update a Vast template for P003 Citrinet using the typed SDK wrapper."
        )
    )
    parser.add_argument("--name", required=True, help="Unique Vast template name.")
    parser.add_argument(
        "--image",
        required=True,
        help="Container image for the template.",
    )
    parser.add_argument(
        "--disk-gb",
        type=float,
        default=100.0,
        help="Recommended disk size for the template.",
    )
    parser.add_argument(
        "--description",
        type=str,
        default=None,
        help="Template description.",
    )
    parser.add_argument("--href", type=str, default=None, help="Optional href field.")
    parser.add_argument("--repo", type=str, default=None, help="Optional repo field.")
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="Raw Vast Docker options string.",
    )
    parser.add_argument(
        "--onstart-cmd",
        type=str,
        default=None,
        help="Instance startup command.",
    )
    parser.add_argument(
        "--search-params",
        type=str,
        default=None,
        help="Vast offer search string stored on the template.",
    )
    parser.add_argument(
        "--image-tag",
        type=str,
        default=None,
        help="Optional image tag if not embedded in --image.",
    )
    parser.add_argument("--login", type=str, default=None, help="Registry login.")
    parser.add_argument(
        "--jupyter",
        action="store_true",
        help="Create a Jupyter template instead of SSH.",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Create a public template.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    client = VastClient.from_env()
    result = client.upsert_template(
        TemplateSpec(
            name=args.name,
            image=args.image,
            disk_gb=args.disk_gb,
            description=args.description,
            href=args.href,
            repo=args.repo,
            env=args.env,
            onstart_cmd=args.onstart_cmd,
            search_params=args.search_params,
            image_tag=args.image_tag,
            login=args.login,
            ssh=not args.jupyter,
            direct=True,
            jupyter=args.jupyter,
            public=args.public,
        )
    )
    dump_json(result.to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
