"""English ASR evaluation normalization.

Training preserves the embedded Parakeet tokenizer's case-and-punctuation surface. WER does not:
the same words must compare equal across casing and punctuation choices made by different
decoders.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Any

_WHITESPACE_RE = re.compile(r"\s+")
_APOSTROPHES = frozenset({"'", "\u2018", "\u2019", "\u02bc", "`"})
_PUNCTUATION_MARKS = (",", ".", "?", "!", ":", ";")


@dataclass(frozen=True)
class PncSurface:
    """Word identities plus punctuation-after-word and initial-capitalization labels."""

    words: tuple[str, ...]
    punctuation: tuple[frozenset[str], ...]
    capitalized: tuple[bool, ...]


def _is_word_character(character: str) -> bool:
    return character in _APOSTROPHES or unicodedata.category(character)[0] in {"L", "M", "N"}


def pnc_surface(text: str) -> PncSurface:
    """Extract a word-aligned English punctuation-and-capitalization surface."""
    text = unicodedata.normalize("NFKC", text)
    words: list[str] = []
    punctuation: list[frozenset[str]] = []
    capitalized: list[bool] = []
    index = 0
    while index < len(text):
        if not _is_word_character(text[index]):
            index += 1
            continue

        word_characters: list[str] = []
        while index < len(text) and _is_word_character(text[index]):
            character = text[index]
            word_characters.append("'" if character in _APOSTROPHES else character)
            index += 1
        word = "".join(word_characters).strip("'")
        if not word:
            continue

        following: set[str] = set()
        lookahead = index
        while lookahead < len(text) and not _is_word_character(text[lookahead]):
            character = text[lookahead]
            if character == "\u2026":
                following.add(".")
            elif character in _PUNCTUATION_MARKS:
                following.add(character)
            lookahead += 1

        first_cased = next((character for character in word if character.isalpha()), "")
        words.append(word.lower())
        punctuation.append(frozenset(following))
        capitalized.append(bool(first_cased and first_cased.isupper()))

    return PncSurface(tuple(words), tuple(punctuation), tuple(capitalized))


def score_pnc_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Score word-preserving PnC predictions against aligned references."""
    punctuation_counts: dict[str, Counter[str]] = {
        mark: Counter({"tp": 0, "fp": 0, "fn": 0}) for mark in _PUNCTUATION_MARKS
    }
    valid_rows = 0
    invalid_rows = 0
    capital_correct = 0
    capital_total = 0
    exact_rows = 0

    for row in rows:
        lexical = pnc_surface(row["lexical_text"])
        reference = pnc_surface(row["reference_text"])
        prediction = pnc_surface(row["prediction_text"])
        if lexical.words != reference.words:
            raise ValueError("reference text does not match its lexical source")
        if lexical.words != prediction.words:
            invalid_rows += 1
            continue
        valid_rows += 1
        capital_correct += sum(
            expected == actual
            for expected, actual in zip(reference.capitalized, prediction.capitalized, strict=True)
        )
        capital_total += len(reference.capitalized)
        row_exact = reference.capitalized == prediction.capitalized
        for mark in _PUNCTUATION_MARKS:
            for expected, actual in zip(reference.punctuation, prediction.punctuation, strict=True):
                expected_mark = mark in expected
                actual_mark = mark in actual
                if expected_mark and actual_mark:
                    punctuation_counts[mark]["tp"] += 1
                elif actual_mark:
                    punctuation_counts[mark]["fp"] += 1
                    row_exact = False
                elif expected_mark:
                    punctuation_counts[mark]["fn"] += 1
                    row_exact = False
        exact_rows += int(row_exact)

    punctuation_report: dict[str, dict[str, float | int]] = {}
    micro = Counter({"tp": 0, "fp": 0, "fn": 0})
    for mark, counts in punctuation_counts.items():
        micro.update(counts)
        precision_denominator = counts["tp"] + counts["fp"]
        recall_denominator = counts["tp"] + counts["fn"]
        precision = counts["tp"] / precision_denominator if precision_denominator else 0.0
        recall = counts["tp"] / recall_denominator if recall_denominator else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        punctuation_report[mark] = {
            **dict(counts),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    micro_precision_denominator = micro["tp"] + micro["fp"]
    micro_recall_denominator = micro["tp"] + micro["fn"]
    micro_precision = (
        micro["tp"] / micro_precision_denominator if micro_precision_denominator else 0.0
    )
    micro_recall = micro["tp"] / micro_recall_denominator if micro_recall_denominator else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )
    total_rows = valid_rows + invalid_rows
    return {
        "rows": total_rows,
        "word_preservation": {
            "valid_rows": valid_rows,
            "invalid_rows": invalid_rows,
            "rate": valid_rows / total_rows if total_rows else 0.0,
        },
        "capitalization_accuracy": (capital_correct / capital_total if capital_total else 0.0),
        "exact_pnc_row_rate": exact_rows / valid_rows if valid_rows else 0.0,
        "punctuation": {
            "per_mark": punctuation_report,
            "micro": {
                **dict(micro),
                "precision": micro_precision,
                "recall": micro_recall,
                "f1": micro_f1,
            },
        },
    }


def normalize_wer(text: str) -> str:
    """Return a conservative case-and-punctuation-neutral English WER surface."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = "".join(" " if unicodedata.category(char).startswith("P") else char for char in text)
    return _WHITESPACE_RE.sub(" ", text).strip()
