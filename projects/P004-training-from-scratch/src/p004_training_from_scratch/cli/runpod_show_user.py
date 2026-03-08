from __future__ import annotations

import argparse

from p004_training_from_scratch.runpod import RunpodClient

from ._common import dump_json


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description="Show RunPod user identity for P004 using the typed SDK wrapper."
    )


def main() -> int:
    _ = build_parser().parse_args()
    client = RunpodClient.from_env()
    dump_json(client.get_user().to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
