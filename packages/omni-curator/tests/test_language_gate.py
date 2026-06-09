"""Tajik language gate: exclusive letters decide first, vocabulary breaks shared-alphabet ties.

Locks in the fix for the ~1,563 valid Tajik clips the letter-only gate wrongly dropped
(commit 6259a355), and that э/ъ/ё are NOT treated as Russian-only (Tajik uses all three).
"""

from __future__ import annotations

import pytest

from omni_curator.process.language import keep_for_language

KEEP = [
    # Tajik-exclusive letters present — the original letter rule.
    "маҷлиси кории ҷумҳурӣ",
    # No exclusive letters either way — the vocabulary tiebreak (previously wrongly dropped).
    "Хушбахтона боз",
    "ман китоб хондам",
    "Казани, Сургута ва Иркутска ба Душанбе",
    "Се бак, се бак",  # се = Tajik 'three' (Russian: три)
    # э/ъ/ё are legitimate Tajik letters, not Russian markers (verbatim recovered clips).
    "Ба-ро. Баро-мад. Баромад. Феъл. Феъл. Биёед номи Ферузро дар тахтаи синф менависем",
    "Аз овоз ба овоз мегардонанд як бор, ё не? Хуб.",
]

DROP = [
    # Russian-exclusive letters (ы ц щ ь).
    "Это очень хорошо, мы согласны",
    "Сегодня в Москве был дождь",
    # No exclusive letters, but Russian function words dominate.
    "Президент России Владимир Путин считает",
    "Но я не знаю, кто он",
    # No signal at all -> conservative drop.
    "трамп байден",
]


@pytest.mark.parametrize("text", KEEP)
def test_tajik_is_kept(text):
    assert keep_for_language(text, "tgk_Cyrl"), f"valid Tajik dropped: {text!r}"


@pytest.mark.parametrize("text", DROP)
def test_non_tajik_is_dropped(text):
    assert not keep_for_language(text, "tgk_Cyrl"), f"non-Tajik kept: {text!r}"


def test_unregistered_language_keeps_everything():
    assert keep_for_language("whatever text", "kat_Geor")
