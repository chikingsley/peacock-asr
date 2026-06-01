from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Sequence

AlignmentOpKind = Literal["match", "substitution", "insertion", "deletion"]


@dataclass(frozen=True)
class AlignmentOp:
    op: AlignmentOpKind
    ref_index: int | None
    hyp_index: int | None


@dataclass(frozen=True)
class AlignmentSummary:
    matches: int
    substitutions: int
    deletions: int
    insertions: int
    reference_count: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def per(self) -> float:
        if self.reference_count == 0:
            return 0.0 if self.errors == 0 else 1.0
        return self.errors / self.reference_count

    def as_dict(self) -> dict[str, int | float]:
        return {
            "PER": round(self.per, 4),
            "errors": self.errors,
            "matches": self.matches,
            "substitutions": self.substitutions,
            "deletions": self.deletions,
            "insertions": self.insertions,
            "reference_count": self.reference_count,
        }


def needleman_wunsch(reference: Sequence[str], hypothesis: Sequence[str]) -> list[AlignmentOp]:
    ref_len = len(reference)
    hyp_len = len(hypothesis)
    cost = [[0] * (hyp_len + 1) for _ in range(ref_len + 1)]
    back: list[list[AlignmentOpKind]] = [
        ["match"] * (hyp_len + 1) for _ in range(ref_len + 1)
    ]

    for i in range(1, ref_len + 1):
        cost[i][0] = i
        back[i][0] = "deletion"
    for j in range(1, hyp_len + 1):
        cost[0][j] = j
        back[0][j] = "insertion"

    for i in range(1, ref_len + 1):
        for j in range(1, hyp_len + 1):
            diag = cost[i - 1][j - 1] + (0 if reference[i - 1] == hypothesis[j - 1] else 1)
            deletion = cost[i - 1][j] + 1
            insertion = cost[i][j - 1] + 1

            if diag <= deletion and diag <= insertion:
                cost[i][j] = diag
                back[i][j] = "match" if reference[i - 1] == hypothesis[j - 1] else "substitution"
            elif deletion <= insertion:
                cost[i][j] = deletion
                back[i][j] = "deletion"
            else:
                cost[i][j] = insertion
                back[i][j] = "insertion"

    ops: list[AlignmentOp] = []
    i = ref_len
    j = hyp_len
    while i > 0 or j > 0:
        op = back[i][j]
        if op in {"match", "substitution"}:
            ops.append(AlignmentOp(op=op, ref_index=i - 1, hyp_index=j - 1))
            i -= 1
            j -= 1
        elif op == "deletion":
            ops.append(AlignmentOp(op=op, ref_index=i - 1, hyp_index=None))
            i -= 1
        else:
            ops.append(AlignmentOp(op=op, ref_index=None, hyp_index=j - 1))
            j -= 1

    ops.reverse()
    return ops


def summarize(ops: Sequence[AlignmentOp], reference_count: int) -> AlignmentSummary:
    return AlignmentSummary(
        matches=sum(1 for op in ops if op.op == "match"),
        substitutions=sum(1 for op in ops if op.op == "substitution"),
        deletions=sum(1 for op in ops if op.op == "deletion"),
        insertions=sum(1 for op in ops if op.op == "insertion"),
        reference_count=reference_count,
    )


def alignment_rows(
    reference: Sequence[str],
    hypothesis: Sequence[str],
    ops: Sequence[AlignmentOp],
) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for pos, op in enumerate(ops):
        rows.append(
            {
                "pos": pos,
                "op": op.op,
                "target": "" if op.ref_index is None else reference[op.ref_index],
                "recognized": "" if op.hyp_index is None else hypothesis[op.hyp_index],
            }
        )
    return rows


def summarize_by_word(
    ops: Sequence[AlignmentOp],
    word_spans: Sequence[tuple[int, int]],
) -> list[AlignmentSummary]:
    buckets: list[list[AlignmentOp]] = [[] for _ in word_spans]
    last_ref_index = 0
    for op in ops:
        if op.ref_index is not None:
            last_ref_index = op.ref_index
            word_index = _word_for_ref_index(word_spans, op.ref_index)
        else:
            word_index = _word_for_ref_index(word_spans, last_ref_index)
        if word_index is not None:
            buckets[word_index].append(op)

    summaries: list[AlignmentSummary] = []
    for bucket, (start, end) in zip(buckets, word_spans, strict=True):
        summaries.append(summarize(bucket, end - start))
    return summaries


def _word_for_ref_index(word_spans: Sequence[tuple[int, int]], ref_index: int) -> int | None:
    if not word_spans:
        return None
    for index, (start, end) in enumerate(word_spans):
        if start <= ref_index < end:
            return index
    if ref_index < word_spans[0][0]:
        return 0
    return len(word_spans) - 1
