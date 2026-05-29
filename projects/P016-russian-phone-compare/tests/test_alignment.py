from p016_compare.alignment import needleman_wunsch, summarize, summarize_by_word


def test_per_counts_sub_del_ins() -> None:
    ops = needleman_wunsch(["a", "b", "c"], ["a", "x", "c", "d"])
    summary = summarize(ops, 3)
    assert summary.matches == 2
    assert summary.substitutions == 1
    assert summary.insertions == 1
    assert summary.deletions == 0
    assert summary.per == 2 / 3


def test_word_summaries_use_phone_spans() -> None:
    ops = needleman_wunsch(["a", "b", "c"], ["a", "x", "c"])
    summaries = summarize_by_word(ops, [(0, 2), (2, 3)])
    assert summaries[0].substitutions == 1
    assert summaries[0].reference_count == 2
    assert summaries[1].matches == 1
