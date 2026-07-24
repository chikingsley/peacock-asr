"""Pinned per-language normalization behavior.

The normalizer is the single surface that training labels, eval references, and eval hypotheses
all pass through, so a silent change to it shifts every WER/CER number without touching a model
(this is exactly how the Persian ZWNJ surface regressed once before — a refactor swapped the
implementation and nothing caught it). These cases freeze the behavior in place: an input maps to
one expected output, and any drift fails loudly. Add a row when a language opts in or a rule
changes deliberately.

Invisible format characters are built with ``chr()`` on purpose — never written as raw glyphs.
"""

from __future__ import annotations

import unicodedata

import pytest

from omni_curator.process.normalize import NORMALIZERS, normalize

# Format characters (Cf: ZWNJ/ZWJ/bidi/BOM) must never survive normalization for any language —
# the Omni char tokenizer has no piece for them, so they would encode to <unk> ("⁇").
_FORMAT_CATEGORIES = {"Cf"}

ZWNJ = chr(0x200C)
ZWJ = chr(0x200D)
RLE = chr(0x202B)  # right-to-left embedding (bidi)
BOM = chr(0xFEFF)

# (input, expected) pairs per language code. Persian uses the NVIDIA fastconformer surface with
# ZWNJ -> space (the Farsi README policy): morphemes become separate
# word units, digits stay digits, Arabic letter variants fold to canonical Persian forms.
PINNED: dict[str, list[tuple[str, str]]] = {
    "eng_Latn": [
        ("Hello, world!", "Hello, world!"),
        ("I [disfluency] agree [noise].", "I agree."),
        ("It’s 2026—café & 10%.", "It's two thousand and twenty six cafe and ten percent."),
        ("“Quoted” co-operate@example.com", "Quoted co operate at example.com"),
        (
            "HELLO <COMMA> WORLD <PERIOD> REALLY <QUESTIONMARK> YES <EXCLAMATIONPOINT>",
            "HELLO, WORLD. REALLY? YES!",
        ),
    ],
    "fas_Arab": [
        (f"می{ZWNJ}خوام برم", "می خوام برم"),  # ZWNJ -> space, NOT glued (3 words)
        (f"کتاب{ZWNJ}ها را", "کتاب ها را"),
        (f"علاقه{ZWNJ}مند", "علاقه مند"),
        ("سال ۱۴۰۰ بود", "سال ۱۴۰۰ بود"),  # digits stay digits (the trained surface)
        ("کك ﻮ", "کک و"),  # REPLACEMENTS: ك -> ک, ﻮ -> و
        ("متن، با: نقطه.", "متن با نقطه"),  # punctuation discarded
        ("یکی|دو", "یکی دو"),  # transcript/segment separator -> word boundary
        ("hello world", ""),  # Latin letters -> discarded utterance (upstream SKIP rule)
    ],
}


@pytest.mark.parametrize(
    ("language", "text", "expected"),
    [(lang, text, expected) for lang, rows in PINNED.items() for text, expected in rows],
)
def test_pinned_output(language: str, text: str, expected: str) -> None:
    assert normalize(text, language) == expected


@pytest.mark.parametrize("language", sorted(NORMALIZERS))
def test_no_format_chars_survive(language: str) -> None:
    # ZWNJ + ZWJ + bidi + BOM interleaved with real letters; none may survive.
    dirty = f"a{ZWNJ}ب{ZWJ}ج{RLE}د{BOM}"
    out = normalize(dirty, language)
    assert not any(unicodedata.category(ch) in _FORMAT_CATEGORIES for ch in out)
