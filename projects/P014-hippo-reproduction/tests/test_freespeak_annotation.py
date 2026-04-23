"""Tests for :mod:`p014.freespeak.annotation`.

Builds a synthetic :class:`ReadAloudAnnotation` + :class:`TranscriptionResult`
with one word substitution and one word insertion, then asserts the emitted
annotation follows the App. D score-assignment rules (insertion -> 0,
substitution -> inherited). ``grapheme_to_phones`` is monkeypatched so the
test does not require the g2p_en model.
"""

import pytest

from p014.config import FreeSpeakingAssignmentConfig
from p014.data import ReadAloudAnnotation
from p014.freespeak import annotation as annotation_module
from p014.freespeak.annotation import build_freespeak_annotations
from p014.freespeak.asr import TranscriptionResult


def test_build_freespeak_annotation_substitution_and_insertion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Reference annotation: two words, three phones each.
    #   word 0 "HELLO": phones H, EH, L  (scores 2.0, 1.0, 0.5)
    #   word 1 "WORLD": phones W, ER, L  (scores 1.5, 1.5, 0.0)
    reference = ReadAloudAnnotation(
        text="HELLO WORLD",
        words=("HELLO", "WORLD"),
        phone_tokens=("H", "EH", "L", "W", "ER", "L"),
        phone_to_word=(0, 0, 0, 1, 1, 1),
        phone_scores=(2.0, 1.0, 0.5, 1.5, 1.5, 0.0),
        word_scores=((2.0, 1.8, 1.6), (1.4, 1.2, 1.0)),
        utterance_scores=(1.5, 1.6, 1.7, 1.8, 1.9),
    )

    # ASR hypothesis: substitutes "HELLO" -> "HI", inserts "THERE" before
    # "WORLD". Word-level alignment -> [substitution(0), insertion, match(1)].
    transcript = TranscriptionResult(
        utterance_id="utt-0",
        text="HI THERE WORLD",
        words=("HI", "THERE", "WORLD"),
    )

    # Deterministic G2P: ASR words map to arbitrary phone sequences.
    phone_map = {
        "HI": ("HH", "AY"),
        "THERE": ("DH", "EH", "R"),
        "WORLD": ("W", "ER", "L"),
    }

    def fake_g2p(words: list[str]) -> list[tuple[str, ...]]:
        return [phone_map[word] for word in words]

    monkeypatch.setattr(annotation_module, "grapheme_to_phones", fake_g2p)

    out = build_freespeak_annotations(
        [reference], [transcript], FreeSpeakingAssignmentConfig()
    )
    assert len(out) == 1
    result = out[0]

    # NW tie-breaks to diagonal then deletion-before-insertion, so on the
    # two-vs-three word alignment the op sequence is
    #   insertion (HI, no ref), substitution (THERE<-HELLO), match (WORLD).
    # The substitution is the one that inherits HELLO's human score; HI is
    # the insertion and gets zeros.
    assert result.words == ("HI", "THERE", "WORLD")
    assert result.word_scores == (
        (0.0, 0.0, 0.0),  # HI -> insertion
        (2.0, 1.8, 1.6),  # THERE substitutes HELLO -> inherits its word score
        (1.4, 1.2, 1.0),  # WORLD matches -> inherits its word score
    )

    # Per-phone assertions:
    # HI (insertion) -> HH, AY both score 0.
    # THERE (substitution of HELLO with ref phones H, EH, L scored 2.0/1.0/0.5):
    #   NW-align (H, EH, L) vs (DH, EH, R) -> sub(H->DH), match(EH), sub(L->R)
    #   emitted: DH (inherits 2.0), EH (inherits 1.0), R (inherits 0.5).
    # WORLD (match) -> W, ER, L inherit 1.5, 1.5, 0.0.
    expected_tokens = (
        "HH",
        "AY",
        "DH",
        "EH",
        "R",
        "W",
        "ER",
        "L",
    )
    expected_scores = (0.0, 0.0, 2.0, 1.0, 0.5, 1.5, 1.5, 0.0)
    expected_phone_to_word = (0, 0, 1, 1, 1, 2, 2, 2)
    assert result.phone_tokens == expected_tokens
    assert result.phone_scores == pytest.approx(expected_scores)
    assert result.phone_to_word == expected_phone_to_word

    # Utterance targets inherit from the reference unchanged.
    assert result.utterance_scores == reference.utterance_scores
