from __future__ import annotations

from english_asr.evaluation import normalize_wer, pnc_surface, score_pnc_rows


def test_normalize_wer_ignores_case_and_punctuation() -> None:
    assert normalize_wer("No, we don't have the scroll.") == "no we don t have the scroll"
    assert normalize_wer("NO WE DON'T HAVE THE SCROLL") == "no we don t have the scroll"


def test_normalize_wer_collapses_unicode_punctuation_and_spacing() -> None:
    assert normalize_wer("  Hello—world…  ") == "hello world"


def test_pnc_surface_aligns_words_case_and_trailing_marks() -> None:
    surface = pnc_surface("\u201cHello, New York!\u201d")

    assert surface.words == ("hello", "new", "york")
    assert surface.capitalized == (True, True, True)
    assert surface.punctuation == (frozenset({","}), frozenset(), frozenset({"!"}))


def test_score_pnc_rows_reports_word_preservation_and_punctuation() -> None:
    report = score_pnc_rows(
        [
            {
                "lexical_text": "hello world",
                "reference_text": "Hello, world!",
                "prediction_text": "Hello, world.",
            },
            {
                "lexical_text": "how are you",
                "reference_text": "How are you?",
                "prediction_text": "How is everyone?",
            },
        ]
    )

    assert report["word_preservation"] == {
        "valid_rows": 1,
        "invalid_rows": 1,
        "rate": 0.5,
    }
    assert report["capitalization_accuracy"] == 1.0
    assert report["punctuation"]["per_mark"][","]["f1"] == 1.0
    assert report["punctuation"]["per_mark"]["!"]["recall"] == 0.0
    assert report["punctuation"]["per_mark"]["."]["precision"] == 0.0
