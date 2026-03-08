from __future__ import annotations

import argparse
import json

from p004_training_from_scratch.runpod import PodCreateSpec, RunpodClient

from ._common import dump_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a RunPod pod for P004 using the typed SDK wrapper."
    )
    parser.add_argument("--name", required=True, help="Pod name.")
    parser.add_argument("--gpu-id", default=None, help="RunPod GPU type id.")
    parser.add_argument(
        "--image-name",
        default="",
        help="Container image name. Required unless --template-id is used.",
    )
    parser.add_argument(
        "--cloud-type",
        default="SECURE",
        choices=["ALL", "COMMUNITY", "SECURE"],
        help="RunPod cloud type.",
    )
    parser.add_argument(
        "--gpu-count", type=int, default=1, help="GPU count for the pod."
    )
    parser.add_argument(
        "--volume-in-gb", type=int, default=0, help="Volume size in GB."
    )
    parser.add_argument(
        "--container-disk-in-gb",
        type=int,
        default=None,
        help="Container disk size in GB.",
    )
    parser.add_argument(
        "--volume-mount-path",
        default="/workspace",
        help="Volume mount path inside the pod.",
    )
    parser.add_argument(
        "--template-id",
        default=None,
        help="RunPod template id. Recommended for standard PyTorch pods.",
    )
    parser.add_argument(
        "--env-json",
        default=None,
        help="Environment variables as a JSON object string.",
    )
    parser.add_argument(
        "--ports",
        default=None,
        help="Comma-separated port mappings, e.g. 22/tcp,8888/http.",
    )
    parser.add_argument(
        "--docker-args",
        default="",
        help="Optional docker args string.",
    )
    parser.add_argument(
        "--network-volume-id",
        default=None,
        help="Optional network volume id.",
    )
    parser.add_argument(
        "--support-public-ip",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Request public IP support.",
    )
    parser.add_argument(
        "--start-ssh",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable SSH startup.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    env = None
    if args.env_json is not None:
        parsed_env = json.loads(args.env_json)
        if not isinstance(parsed_env, dict):
            raise SystemExit("--env-json must decode to a JSON object.")
        env = {str(key): str(value) for key, value in parsed_env.items()}

    client = RunpodClient.from_env()
    result = client.create_pod(
        PodCreateSpec(
            name=args.name,
            gpu_type_id=args.gpu_id,
            image_name=args.image_name,
            cloud_type=args.cloud_type,
            support_public_ip=args.support_public_ip,
            start_ssh=args.start_ssh,
            gpu_count=args.gpu_count,
            volume_in_gb=args.volume_in_gb,
            container_disk_in_gb=args.container_disk_in_gb,
            docker_args=args.docker_args,
            ports=args.ports,
            volume_mount_path=args.volume_mount_path,
            env=env,
            template_id=args.template_id,
            network_volume_id=args.network_volume_id,
        )
    )
    dump_json(result.to_dict())
    return 0 if result.pod_id is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
