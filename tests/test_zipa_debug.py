"""Diagnostic tests for ZIPA backend mapping issues.

These tests don't need the ONNX model or GPU — they just check
whether our ARPABET->IPA mapping actually hits the ZIPA vocab.

ZIPA uses a 127-token CHARACTER-LEVEL vocab. Single-char IPA phones
(32/39) map fine. Multi-char diphthongs/affricates (7/39) have no
single token and are excluded from GOP scoring.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gopt_bench.backends.zipa import ARPABET_TO_IPA

# The 39 ARPABET phones used in SpeechOcean762 (no stress digits)
SPO762_PHONES = [
    "AA", "AE", "AH", "AO", "AW", "AY",
    "B", "CH", "D", "DH",
    "EH", "ER", "EY",
    "F", "G", "HH",
    "IH", "IY",
    "JH", "K", "L", "M", "N", "NG",
    "OW", "OY",
    "P", "R", "S", "SH",
    "T", "TH",
    "UH", "UW",
    "V", "W", "Y", "Z", "ZH",
]

# 7 phones that are multi-char IPA (no single token in ZIPA's char-level vocab)
EXPECTED_UNMAPPABLE = {"AW", "AY", "CH", "EY", "JH", "OW", "OY"}

# Load ZIPA vocab from the reference tokens.txt
TOKENS_PATH = (
    Path(__file__).parent.parent
    / "references" / "zipa" / "ipa_simplified" / "tokens.txt"
)


@pytest.fixture
def zipa_vocab() -> dict[str, int]:
    """Load ZIPA token->index mapping from tokens.txt."""
    if not TOKENS_PATH.exists():
        pytest.skip(f"tokens.txt not found at {TOKENS_PATH}")
    vocab: dict[str, int] = {}
    with TOKENS_PATH.open(encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                token = parts[0]
                idx = int(parts[1])
                vocab[token] = idx
    return vocab


def test_all_arpabet_phones_have_ipa_entry() -> None:
    """Every ARPABET phone should have an entry in the mapping dict."""
    for phone in SPO762_PHONES:
        assert phone in ARPABET_TO_IPA, f"{phone} missing from ARPABET_TO_IPA"


def test_single_char_phones_in_vocab(zipa_vocab: dict[str, int]) -> None:
    """Single-char IPA mappings (32 phones) should all be in the ZIPA vocab."""
    for phone in SPO762_PHONES:
        if phone in EXPECTED_UNMAPPABLE:
            continue
        ipa = ARPABET_TO_IPA[phone]
        assert len(ipa) == 1, f"{phone} -> '{ipa}' should be single char"
        assert ipa in zipa_vocab, (
            f"{phone} -> '{ipa}' (U+{ord(ipa):04X}) not in ZIPA vocab"
        )


def test_multi_char_phones_not_in_vocab(zipa_vocab: dict[str, int]) -> None:
    """Multi-char IPA (7 diphthongs/affricates) should NOT be in vocab."""
    for phone in EXPECTED_UNMAPPABLE:
        ipa = ARPABET_TO_IPA[phone]
        assert len(ipa) > 1, f"{phone} -> '{ipa}' expected multi-char"
        assert ipa not in zipa_vocab, (
            f"{phone} -> '{ipa}' unexpectedly found in vocab"
        )


def test_er_unicode_fix(zipa_vocab: dict[str, int]) -> None:
    """ER should map to U+025C, not U+025D."""
    ipa = ARPABET_TO_IPA["ER"]
    assert ipa == "\u025c", f"ER maps to U+{ord(ipa):04X}, expected U+025C"
    assert ipa in zipa_vocab, "U+025C should be in ZIPA vocab"


def test_g_unicode_fix(zipa_vocab: dict[str, int]) -> None:
    """G should map to plain ASCII 'g' (U+0067), not U+0261."""
    ipa = ARPABET_TO_IPA["G"]
    assert ipa == "g", f"G maps to U+{ord(ipa):04X}, expected U+0067"
    assert ipa in zipa_vocab, "ASCII 'g' should be in ZIPA vocab"


def test_coverage_count(zipa_vocab: dict[str, int]) -> None:
    """Exactly 32/39 phones should map successfully."""
    mapped = 0
    for phone in SPO762_PHONES:
        ipa = ARPABET_TO_IPA[phone]
        if ipa in zipa_vocab:
            mapped += 1
    assert mapped == 32, f"Expected 32 mapped phones, got {mapped}"
