from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class FeatureErrorSummary:
    feature_distance: float
    reference_count: int

    @property
    def pfer(self) -> float:
        if self.reference_count == 0:
            return 0.0 if self.feature_distance == 0 else 1.0
        return self.feature_distance / self.reference_count

    def as_dict(self) -> dict[str, float]:
        return {
            "PFER": round(self.pfer, 4),
            "feature_distance": round(self.feature_distance, 4),
        }


def feature_edit_summary(
    reference: Sequence[str],
    hypothesis: Sequence[str],
) -> FeatureErrorSummary:
    return FeatureErrorSummary(
        feature_distance=feature_edit_distance(reference, hypothesis),
        reference_count=len(reference),
    )


def feature_edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> float:
    ref_len = len(reference)
    hyp_len = len(hypothesis)
    cost = [[0.0] * (hyp_len + 1) for _ in range(ref_len + 1)]

    for i in range(1, ref_len + 1):
        cost[i][0] = float(i)
    for j in range(1, hyp_len + 1):
        cost[0][j] = float(j)

    for i in range(1, ref_len + 1):
        for j in range(1, hyp_len + 1):
            substitution = cost[i - 1][j - 1] + phone_feature_cost(
                reference[i - 1],
                hypothesis[j - 1],
            )
            deletion = cost[i - 1][j] + 1.0
            insertion = cost[i][j - 1] + 1.0
            cost[i][j] = min(substitution, deletion, insertion)

    return cost[ref_len][hyp_len]


def alignment_feature_distance(
    reference: Sequence[str],
    hypothesis: Sequence[str],
    ops: Sequence[Any],
) -> float:
    total = 0.0
    for op in ops:
        if op.op == "match":
            continue
        if op.op == "substitution" and op.ref_index is not None and op.hyp_index is not None:
            total += phone_feature_cost(reference[op.ref_index], hypothesis[op.hyp_index])
        else:
            total += 1.0
    return total


def phone_feature_cost(reference: str, hypothesis: str) -> float:
    if reference == hypothesis:
        return 0.0
    ref_vector = _feature_vector(reference)
    hyp_vector = _feature_vector(hypothesis)
    if ref_vector is None or hyp_vector is None:
        return 1.0
    differences = sum(
        1
        for ref_value, hyp_value in zip(ref_vector, hyp_vector, strict=True)
        if ref_value != hyp_value
    )
    return differences / len(ref_vector)


@lru_cache(maxsize=2048)
def _feature_vector(phone: str) -> tuple[int, ...] | None:
    table = _feature_table()
    if table is None:
        return None
    try:
        segments = table.word_fts(phone)
    except Exception:  # noqa: BLE001 - panphon may raise varied errors on unparseable phones; treat any as "no vector"
        return None
    if len(segments) != 1:
        return None
    return tuple(int(value) for value in segments[0].numeric())


@lru_cache(maxsize=1)
def _feature_table() -> Any | None:
    try:
        import panphon
    except ImportError:
        return None
    return panphon.FeatureTable()
