from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any


def dump_json(data: Any) -> None:
    print(json.dumps(data, indent=2, sort_keys=True))


def tupled(items: Sequence[str] | None) -> tuple[str, ...]:
    if not items:
        return ()
    return tuple(items)
