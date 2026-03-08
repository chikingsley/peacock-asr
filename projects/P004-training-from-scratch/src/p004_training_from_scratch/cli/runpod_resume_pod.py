from __future__ import annotations

import argparse

from p004_training_from_scratch.runpod import RunpodClient

from ._common import dump_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resume a RunPod pod for P004 using the typed SDK wrapper."
    )
    parser.add_argument("--pod-id", required=True, help="RunPod pod id.")
    parser.add_argument(
        "--gpu-count",
        type=int,
        default=1,
        help="GPU count to request when resuming the pod.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    client = RunpodClient.from_env()
    result = client.resume_pod(args.pod_id, gpu_count=args.gpu_count)
    dump_json(result.to_dict())
    return 0 if result.pod_id is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
