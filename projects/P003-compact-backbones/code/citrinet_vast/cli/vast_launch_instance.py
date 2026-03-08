from __future__ import annotations

import argparse

from citrinet_vast import LaunchInstanceSpec, VastClient

from ._common import dump_json, tupled


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch a Vast instance for P003 Citrinet using the typed SDK wrapper."
    )
    parser.add_argument("--gpu-name", required=True, help="GPU model name.")
    parser.add_argument(
        "--image", required=True, help="Container image to launch on Vast."
    )
    parser.add_argument(
        "--num-gpus", type=int, default=1, help="Number of GPUs to request."
    )
    parser.add_argument(
        "--disk-gb", type=float, default=150.0, help="Requested instance disk size."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="How many offers Vast should consider before launching the top one.",
    )
    parser.add_argument(
        "--order",
        type=str,
        default="score-",
        help="Vast order expression used during the launch offer search.",
    )
    parser.add_argument("--label", type=str, default=None, help="Instance label.")
    parser.add_argument(
        "--region",
        type=str,
        default=None,
        help="Optional Vast region or country filter.",
    )
    parser.add_argument(
        "--template-hash",
        type=str,
        default=None,
        help="Optional Vast template hash to apply during launch.",
    )
    parser.add_argument(
        "--query-clause",
        action="append",
        default=[],
        help="Additional Vast query clause, repeated as needed.",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="Environment variable in KEY=VALUE form, repeated as needed.",
    )
    parser.add_argument(
        "--onstart-cmd",
        type=str,
        default=None,
        help="Command string to run on instance startup.",
    )
    parser.add_argument(
        "--login", type=str, default=None, help="Optional registry login string."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Pass Vast's force flag during launch.",
    )
    parser.add_argument(
        "--cancel-unavailable",
        action="store_true",
        help="Cancel the launch instead of waiting on unavailable offers.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    client = VastClient.from_env()
    result = client.launch_instance(
        LaunchInstanceSpec(
            gpu_name=args.gpu_name,
            image=args.image,
            num_gpus=args.num_gpus,
            disk_gb=args.disk_gb,
            limit=args.limit,
            order=args.order,
            label=args.label,
            region=args.region,
            template_hash=args.template_hash,
            onstart_cmd=args.onstart_cmd,
            env=tupled(args.env),
            query_clauses=tupled(args.query_clause),
            login=args.login,
            force=args.force,
            cancel_unavailable=args.cancel_unavailable,
        )
    )
    dump_json(result.to_dict())
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
