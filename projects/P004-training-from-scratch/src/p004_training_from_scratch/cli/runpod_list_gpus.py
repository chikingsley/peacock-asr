from __future__ import annotations

import argparse

from p004_training_from_scratch.runpod import GpuSearchSpec, RunpodClient

from ._common import dump_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="List RunPod GPU types for P004 using the typed SDK wrapper."
    )
    parser.add_argument(
        "--priced",
        action="store_true",
        help="Enrich GPU rows with pricing via runpod.get_gpu().",
    )
    parser.add_argument(
        "--gpu-quantity",
        type=int,
        default=1,
        help="GPU quantity used for lowest-price lookups.",
    )
    parser.add_argument(
        "--min-memory-gb",
        type=int,
        default=None,
        help="Optional minimum VRAM filter.",
    )
    parser.add_argument(
        "--max-memory-gb",
        type=int,
        default=None,
        help="Optional maximum VRAM filter.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    client = RunpodClient.from_env()
    gpus = client.list_gpu_types(
        GpuSearchSpec(
            include_pricing=args.priced,
            gpu_quantity=args.gpu_quantity,
            min_memory_gb=args.min_memory_gb,
            max_memory_gb=args.max_memory_gb,
        )
    )
    dump_json([gpu.to_dict() for gpu in gpus])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
