from __future__ import annotations

from citrinet_vast import VastClient

from ._common import dump_json


def main() -> int:
    client = VastClient.from_env()
    instances = client.show_instances()
    dump_json([instance.to_dict() for instance in instances])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
