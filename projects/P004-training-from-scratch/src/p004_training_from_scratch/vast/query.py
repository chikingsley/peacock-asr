from __future__ import annotations

import json
from collections.abc import Iterable

from .models import OfferSearchSpec


def _quote(value: str) -> str:
    return json.dumps(value)


def normalize_query_clauses(clauses: Iterable[str]) -> tuple[str, ...]:
    normalized = tuple(clause.strip() for clause in clauses if clause.strip())
    return normalized


def build_offer_query(spec: OfferSearchSpec) -> str | None:
    clauses = []
    if spec.gpu_name:
        clauses.append(f"gpu_name={_quote(spec.gpu_name)}")
    clauses.append(f"num_gpus={spec.num_gpus}")
    clauses.extend(normalize_query_clauses(spec.query_clauses))
    if not clauses:
        return None
    return " ".join(clauses)


def build_env_flags(env_pairs: Iterable[str]) -> str | None:
    normalized = []
    for pair in env_pairs:
        stripped = pair.strip()
        if not stripped:
            continue
        if "=" not in stripped:
            msg = f"Invalid env pair: {pair!r}. Expected KEY=VALUE."
            raise ValueError(msg)
        normalized.append(f"-e {stripped}")
    if not normalized:
        return None
    return " ".join(normalized)
