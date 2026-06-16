"""Scoring lane: align canonical vs produced IPA and score by phone / feature edit distance."""

from __future__ import annotations

from capt.score.alignment import (
    AlignmentOp,
    AlignmentSummary,
    alignment_rows,
    needleman_wunsch,
    summarize,
    summarize_by_word,
)
from capt.score.features import (
    FeatureErrorSummary,
    alignment_feature_distance,
    feature_edit_distance,
    feature_edit_summary,
    phone_feature_cost,
)
from capt.score.phones import (
    normalize_phone_token,
    normalize_phone_tokens,
    split_phone_text,
)

__all__ = [
    "AlignmentOp",
    "AlignmentSummary",
    "FeatureErrorSummary",
    "alignment_feature_distance",
    "alignment_rows",
    "feature_edit_distance",
    "feature_edit_summary",
    "needleman_wunsch",
    "normalize_phone_token",
    "normalize_phone_tokens",
    "phone_feature_cost",
    "split_phone_text",
    "summarize",
    "summarize_by_word",
]
