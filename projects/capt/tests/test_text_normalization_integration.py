import re

import pytest

import capt.g2p.routing as g2p_module
from capt.g2p import TargetG2P
from capt.g2p.text_normalization import normalize_written_text


@pytest.mark.integration
@pytest.mark.parametrize(
    ("language", "text", "expected_fragments", "forbidden_pattern"),
    [
        (
            "en_us",
            "Out of 1,400 people grew by 8%.",
            ("one thousand four hundred", "eight percent"),
            r"[\d%,]",
        ),
        (
            "en_us",
            "The UN and U.S. discussed AI.",
            ("you en", "you ess", "ay eye"),
            r"\b(?:un|us|ai)\b",
        ),
        (
            "en_us",
            "The peak was 4,892 meters.",
            ("four thousand", "eight hundred", "ninety two", "meters"),
            r"[\d,]",
        ),
        (
            "ru",
            "Тем не менее, 80% наших товаров.",
            ("восемьдесят процентов",),
            r"[\d%]",
        ),
        (
            "ru",
            "4892 метра",
            ("четыре тысячи", "восемьсот", "девяносто два", "метра"),
            r"\d",
        ),
        (
            "ru",
            "В США ООН и АЭС использовали РЛС.",
            ("сэ шэ а", "о о эн", "а э эс", "эр эл эс"),
            r"\b(?:сша|оон|аэс|рлс)\b",
        ),
    ],
)
def test_real_nemo_normalizes_written_forms(
    language: str,
    text: str,
    expected_fragments: tuple[str, ...],
    forbidden_pattern: str,
) -> None:
    result = normalize_written_text(text, language)
    normalized = result.normalized_text.casefold()

    assert result.backend == f"nemo-text-processing:{language[:2]}"
    assert result.warnings == []
    assert all(fragment in normalized for fragment in expected_fragments)
    assert re.search(forbidden_pattern, normalized) is None


@pytest.mark.integration
def test_english_g2p_receives_real_nemo_words(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_words = []

    def fake_espeak_g2p(words: list[str], voice: str) -> list[list[str]]:
        seen_words.extend(words)
        return [[word] for word in words]

    monkeypatch.setattr(g2p_module, "_espeak_g2p", fake_espeak_g2p)

    result = TargetG2P("espeak").from_text("8% and U.S.", "en_us")

    assert result.words == ["eight", "percent", "and", "you", "ess"]
    assert seen_words == result.words
    assert result.phones_per_word_raw == [[word] for word in result.words]


@pytest.mark.integration
def test_russian_g2p_receives_real_nemo_words(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_words = []

    def fake_espeak_g2p(words: list[str], voice: str) -> list[list[str]]:
        seen_words.extend(words)
        return [[word] for word in words]

    monkeypatch.setattr(g2p_module, "_espeak_g2p", fake_espeak_g2p)

    result = TargetG2P("espeak").from_text("80% в США", "ru")

    assert result.words == ["восемьдесят", "процентов", "в", "сэ", "шэ", "а"]
    assert seen_words == result.words
    assert result.phones_per_word_raw == [[word] for word in result.words]
