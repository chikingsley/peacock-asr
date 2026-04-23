"""Needleman-Wunsch alignment over arbitrary sequences.

Used for two purposes in the free-speaking pipeline (Yan et al. 2025, App. D):

1. Word-level alignment of ASR transcribed words against reference words.
2. Phone-level alignment of G2P-converted phones against canonical phones.

We pick Needleman-Wunsch (global alignment) over plain Levenshtein because we
need the *operation sequence* (match / substitution / insertion / deletion)
for score assignment, not just the edit distance. Both yield the same minimal
cost here — NW with a unit substitution/indel cost is equivalent to
Levenshtein — but NW naturally exposes the operation trace via backtracking.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

AlignmentOpKind = Literal["match", "substitution", "insertion", "deletion"]


@dataclass(frozen=True)
class AlignmentOp:
    """A single step in an alignment.

    Attributes:
        op: The operation kind — ``match``, ``substitution``, ``insertion``,
            or ``deletion``.
        ref_index: Index into ``reference``; ``None`` when the operation is an
            insertion (a hypothesis unit with no reference counterpart).
        hyp_index: Index into ``hypothesis``; ``None`` when the operation is a
            deletion (a reference unit with no hypothesis counterpart).
    """

    op: AlignmentOpKind
    ref_index: int | None
    hyp_index: int | None


def needleman_wunsch(
    reference: Sequence[str],
    hypothesis: Sequence[str],
) -> list[AlignmentOp]:
    """Compute a minimum-cost alignment between ``reference`` and ``hypothesis``.

    Scoring: match costs 0, mismatch/insert/delete each cost 1. The returned
    list of :class:`AlignmentOp` reads left-to-right along the alignment, so
    emitting hypothesis tokens in the order they appear is equivalent to
    iterating the ops in order and skipping ``deletion`` entries.

    Edge cases:

    * Empty reference → every hypothesis token becomes an ``insertion``.
    * Empty hypothesis → every reference token becomes a ``deletion``.
    * Both empty → empty result.

    The implementation is pure Python with ``O(R * H)`` time and memory. For
    the free-speaking pipeline the longest sequence is ~20 words or ~50 phones,
    so this is fast enough without numpy.
    """

    ref_len = len(reference)
    hyp_len = len(hypothesis)

    # Cost table and operation back-pointers.
    cost: list[list[int]] = [[0] * (hyp_len + 1) for _ in range(ref_len + 1)]
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
            match = reference[i - 1] == hypothesis[j - 1]
            diag_cost = cost[i - 1][j - 1] + (0 if match else 1)
            del_cost = cost[i - 1][j] + 1
            ins_cost = cost[i][j - 1] + 1

            # Prefer diagonal on ties so matches/substitutions dominate shifts.
            if diag_cost <= del_cost and diag_cost <= ins_cost:
                cost[i][j] = diag_cost
                back[i][j] = "match" if match else "substitution"
            elif del_cost <= ins_cost:
                cost[i][j] = del_cost
                back[i][j] = "deletion"
            else:
                cost[i][j] = ins_cost
                back[i][j] = "insertion"

    ops: list[AlignmentOp] = []
    i = ref_len
    j = hyp_len
    while i > 0 or j > 0:
        op = back[i][j]
        if op in ("match", "substitution"):
            ops.append(AlignmentOp(op=op, ref_index=i - 1, hyp_index=j - 1))
            i -= 1
            j -= 1
        elif op == "deletion":
            ops.append(AlignmentOp(op=op, ref_index=i - 1, hyp_index=None))
            i -= 1
        else:  # insertion
            ops.append(AlignmentOp(op=op, ref_index=None, hyp_index=j - 1))
            j -= 1
    ops.reverse()
    return ops
