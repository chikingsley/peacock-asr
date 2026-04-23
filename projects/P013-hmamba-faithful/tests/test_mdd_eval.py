from __future__ import annotations

from p012.mdd_eval import align_tokens, compute_mdd_metrics


def test_align_tokens_marks_deletion_and_substitution() -> None:
    aligned = align_tokens(["a", "b", "c"], ["a", "x"])
    assert aligned.ref == ["a", "b", "c"]
    assert len(aligned.hyp) == 3
    assert aligned.ops.count("C") == 1
    assert aligned.ops.count("S") == 1
    assert aligned.ops.count("D") == 1


def test_compute_mdd_metrics_counts_detected_error_with_wrong_diagnosis() -> None:
    metrics = compute_mdd_metrics(
        human_seq={"utt1": ["a", "x", "c"]},
        ref_seq={"utt1": ["a", "b", "c"]},
        hyp_seq={"utt1": ["a", "y", "c"]},
    )
    assert metrics["Precision"] == 1.0
    assert metrics["Recall"] == 1.0
    assert metrics["F1"] == 1.0
    assert metrics["Correct Diag"] == 0.0
    assert metrics["Error Diag"] == 1.0


def test_compute_mdd_metrics_counts_false_reject() -> None:
    metrics = compute_mdd_metrics(
        human_seq={"utt1": ["a", "b"]},
        ref_seq={"utt1": ["a", "b"]},
        hyp_seq={"utt1": ["a", "x"]},
    )
    assert metrics["Precision"] == 0.0
    assert metrics["Recall"] == 0.0
    assert metrics["F1"] == 0.0
    assert metrics["FR"] == 0.5
