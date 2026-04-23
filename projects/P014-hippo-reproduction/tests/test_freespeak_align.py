"""Unit tests for :mod:`p014.freespeak.align`.

These cover the six canonical cases listed in the task:
(a) identical sequences -> all matches
(b) one insertion
(c) one deletion
(d) one substitution
(e) empty ref, 2 hyp words -> 2 insertions
(f) empty hyp, 2 ref words -> 2 deletions
"""

from p014.freespeak.align import AlignmentOp, needleman_wunsch


def _ops_of(ops: list[AlignmentOp]) -> list[str]:
    return [op.op for op in ops]


def test_align_identical_sequences_all_match() -> None:
    ref = ["HELLO", "WORLD"]
    hyp = ["HELLO", "WORLD"]
    ops = needleman_wunsch(ref, hyp)
    assert _ops_of(ops) == ["match", "match"]
    assert [op.ref_index for op in ops] == [0, 1]
    assert [op.hyp_index for op in ops] == [0, 1]


def test_align_one_insertion() -> None:
    ref = ["A", "B"]
    hyp = ["A", "X", "B"]
    ops = needleman_wunsch(ref, hyp)
    assert _ops_of(ops) == ["match", "insertion", "match"]
    inserted = next(op for op in ops if op.op == "insertion")
    assert inserted.ref_index is None
    assert inserted.hyp_index == 1


def test_align_one_deletion() -> None:
    ref = ["A", "B", "C"]
    hyp = ["A", "C"]
    ops = needleman_wunsch(ref, hyp)
    assert _ops_of(ops) == ["match", "deletion", "match"]
    deletion = next(op for op in ops if op.op == "deletion")
    assert deletion.ref_index == 1
    assert deletion.hyp_index is None


def test_align_one_substitution() -> None:
    ref = ["A", "B", "C"]
    hyp = ["A", "X", "C"]
    ops = needleman_wunsch(ref, hyp)
    assert _ops_of(ops) == ["match", "substitution", "match"]
    sub = next(op for op in ops if op.op == "substitution")
    assert sub.ref_index == 1
    assert sub.hyp_index == 1


def test_align_empty_ref_with_two_hyp_words_is_two_insertions() -> None:
    ops = needleman_wunsch([], ["A", "B"])
    assert _ops_of(ops) == ["insertion", "insertion"]
    assert [op.hyp_index for op in ops] == [0, 1]
    assert all(op.ref_index is None for op in ops)


def test_align_empty_hyp_with_two_ref_words_is_two_deletions() -> None:
    ops = needleman_wunsch(["A", "B"], [])
    assert _ops_of(ops) == ["deletion", "deletion"]
    assert [op.ref_index for op in ops] == [0, 1]
    assert all(op.hyp_index is None for op in ops)
