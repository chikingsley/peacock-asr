import pytest

import capt.g2p as g2p_module
from capt.g2p import TargetG2P
from capt.normalization import normalize_phone_tokens, split_phone_text


def test_split_spaced_phone_text() -> None:
    assert split_phone_text("m ɪ s t ɚ") == ["m", "ɪ", "s", "t", "ə˞"]


def test_drops_ctc_blanks_and_stress() -> None:
    assert normalize_phone_tokens(["<pad>", "ˈa", "b", "|"]) == ["a", "b"]


def test_drops_zero_width_joiner() -> None:
    assert normalize_phone_tokens(["j", "\u200d", "a"]) == ["j", "a"]
    assert split_phone_text("j\u200da") == ["j", "a"]


def test_attaches_standalone_rhoticity_marker() -> None:
    assert normalize_phone_tokens(["ɜ", "˞"]) == ["ə˞"]


def test_expands_common_english_multisegment_tokens() -> None:
    assert normalize_phone_tokens(["dʒ", "eɪ", "tʃ"]) == ["d", "ʒ", "e", "ɪ", "t", "ʃ"]


def test_drops_combining_tilde_overlay_artifact() -> None:
    assert normalize_phone_tokens(["l", "\u0334", "e"]) == ["l", "e"]


def test_drops_dental_diacritic() -> None:
    assert normalize_phone_tokens(["t̪", "d̪", "z̪", "n̪"]) == ["t", "d", "z", "n"]


def test_drops_affricate_tie_bar_for_segment_comparison() -> None:
    assert split_phone_text("t͡ɕu") == ["t", "ɕ", "u"]


def test_expands_russian_affricate_tokens() -> None:
    assert normalize_phone_tokens(["ts", "tsː", "tɕ", "dʑ"]) == [
        "t",
        "s",
        "t",
        "s",
        "t",
        "ɕ",
        "d",
        "ʑ",
    ]


def test_normalizes_seen_phone_symbol_variants() -> None:
    assert normalize_phone_tokens(["ɝ", "ɜ˞", "ɜː", "ɚ"]) == ["ə˞", "ə˞", "ə˞", "ə˞"]
    assert normalize_phone_tokens(["ɫ", "ɑː", "uː", "ɲ"]) == ["l", "ɑ", "u", "nʲ"]
    assert normalize_phone_tokens(["oː", "v", "ə˞", "ʎ"]) == ["o", "ʊ", "v", "ə˞", "lʲ"]
    assert normalize_phone_tokens(["sʲː", "ɕː"]) == ["sʲ", "ɕ"]


def test_auto_zipa_uses_espeak_for_english_function_words(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        g2p_module,
        "_espeak_g2p",
        lambda words, voice: [["ð", "ə"], ["k", "w", "ɪ", "k"]],
    )

    result = TargetG2P("auto_zipa").from_words(["the", "quick"], "en_us")

    assert result.backend == "espeak-ng:en-us"
    assert result.phones_per_word_normalized[0] == ["ð", "ə"]


def test_russian_rewrites_latin_wifi_and_cardinal_digits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_words = []

    def fake_espeak_g2p(words: list[str], voice: str) -> list[list[str]]:
        seen_words.extend(words)
        return [[word] for word in words]

    monkeypatch.setattr(g2p_module, "_espeak_g2p", fake_espeak_g2p)

    result = TargetG2P("espeak").from_words(["wi-fi", "7"], "ru")

    assert seen_words == ["вай", "фай", "семь"]
    assert result.words == ["wi-fi", "7"]
    assert result.phones_per_word_raw == [["вай", "фай"], ["семь"]]


def test_russian_rewrites_roman_century_ordinals(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_words = []

    def fake_espeak_g2p(words: list[str], voice: str) -> list[list[str]]:
        seen_words.extend(words)
        return [[word] for word in words]

    monkeypatch.setattr(g2p_module, "_espeak_g2p", fake_espeak_g2p)

    result = TargetG2P("espeak").from_words(["xviii", "века"], "ru")

    assert seen_words == ["восемнадцатого", "века"]
    assert result.words == ["xviii", "века"]
    assert result.phones_per_word_raw == [["восемнадцатого"], ["века"]]


def test_russian_rewrites_dates_years_and_age_suffixes(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_words = []

    def fake_espeak_g2p(words: list[str], voice: str) -> list[list[str]]:
        seen_words.extend(words)
        return [[word] for word in words]

    monkeypatch.setattr(g2p_module, "_espeak_g2p", fake_espeak_g2p)

    TargetG2P("espeak").from_words(
        ["6", "октября", "1789", "года", "людовика", "xvi", "11-летнюю", "4-летнего"],
        "ru",
    )

    assert seen_words == [
        "шестого",
        "октября",
        "тысяча",
        "семьсот",
        "восемьдесят",
        "девятого",
        "года",
        "людовика",
        "шестнадцатого",
        "одиннадцати",
        "летнюю",
        "четырех",
        "летнего",
    ]
