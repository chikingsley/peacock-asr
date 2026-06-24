"""Benchmark primitives: the one scoring normalization and script detection."""

from __future__ import annotations

from typing import cast

import pytest

from omni_curator.audit.benchmark import dominant_script, normalize, score_pair


def test_normalize_folds_case_punctuation_whitespace():
    assert normalize("Салом,   ҷаҳон!") == "салом ҷаҳон"
    assert normalize("...") == ""
    assert normalize("♪") == ""


@pytest.mark.parametrize(
    ("text", "script"),
    [
        ("салом дӯстон", "Cyrl"),
        ("سلام دوستان", "Arab"),
        ("hello friends", "Latn"),
        ("გამარჯობა", "Geor"),
        ("12345 !!!", None),
    ],
)
def test_dominant_script(text, script):
    assert dominant_script(text) == script


def test_score_pair_identical_is_zero():
    detail = score_pair("салом ҷаҳон", "Салом, ҷаҳон!")  # normalization closes the gap
    assert detail["wer"] == 0.0
    assert detail["cer"] == 0.0


def test_score_pair_counts_substitutions():
    detail = score_pair("ман китоб хондам", "ман нома хондам")
    assert detail["wer"] == pytest.approx(1 / 3)
    words = cast("dict[str, int]", detail["words"])
    assert words["substitutions"] == 1
