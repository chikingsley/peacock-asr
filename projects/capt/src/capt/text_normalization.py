from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NEMO_CACHE_DIR = PROJECT_ROOT / ".nemo_text_processing"
NEMO_BACKEND = "nemo-text-processing"


@dataclass(frozen=True)
class WrittenTextNormalization:
    original_text: str
    normalized_text: str
    backend: str
    warnings: list[str]


EN_ACRONYM_RE = re.compile(r"\b(?:[A-Z]\.){2,}|\b[A-Z]{2,}\b")
RU_ACRONYM_RE = re.compile(r"\b[А-ЯЁ]{2,}\b")
ROMAN_NUMERAL_RE = re.compile(r"^[IVXLCDM]+$")
FOUR_DIGIT_NUMBER_RE = re.compile(r"\b([1-9])\d{3}\b")
NUMBER_THOUSAND_RE = re.compile(r"\b(\d+)\s+тысяч\b", re.IGNORECASE)

EN_LETTER_NAMES = {
    "A": "ay",
    "B": "bee",
    "C": "cee",
    "D": "dee",
    "E": "ee",
    "F": "eff",
    "G": "gee",
    "H": "aitch",
    "I": "eye",
    "J": "jay",
    "K": "kay",
    "L": "el",
    "M": "em",
    "N": "en",
    "O": "oh",
    "P": "pea",
    "Q": "cue",
    "R": "ar",
    "S": "ess",
    "T": "tee",
    "U": "you",
    "V": "vee",
    "W": "double you",
    "X": "ex",
    "Y": "why",
    "Z": "zee",
}

RU_ACRONYMS = {
    "АЭС": "а э эс",
    "ИИ": "и и",
    "ООН": "о о эн",
    "РЛС": "эр эл эс",
    "РС": "эр эс",
    "США": "сэ шэ а",
}

RU_LETTER_NAMES = {
    "А": "а",
    "Б": "бэ",
    "В": "вэ",
    "Г": "гэ",
    "Д": "дэ",
    "Е": "е",
    "Ё": "ё",
    "Ж": "жэ",
    "З": "зэ",
    "И": "и",
    "Й": "и краткое",
    "К": "ка",
    "Л": "эл",
    "М": "эм",
    "Н": "эн",
    "О": "о",
    "П": "пэ",
    "Р": "эр",
    "С": "эс",
    "Т": "тэ",
    "У": "у",
    "Ф": "эф",
    "Х": "ха",
    "Ц": "цэ",
    "Ч": "че",
    "Ш": "ша",
    "Щ": "ща",
    "Ъ": "твердый знак",
    "Ы": "ы",
    "Ь": "мягкий знак",
    "Э": "э",
    "Ю": "ю",
    "Я": "я",
}

RU_CARDINAL_QUANTITIES = {
    1: "одна",
    2: "две",
    3: "три",
    4: "четыре",
    5: "пять",
    6: "шесть",
    7: "семь",
    8: "восемь",
    9: "девять",
    10: "десять",
    11: "одиннадцать",
    12: "двенадцать",
    13: "тринадцать",
    14: "четырнадцать",
    15: "пятнадцать",
    16: "шестнадцать",
    17: "семнадцать",
    18: "восемнадцать",
    19: "девятнадцать",
    20: "двадцать",
    30: "тридцать",
    40: "сорок",
    50: "пятьдесят",
    60: "шестьдесят",
    70: "семьдесят",
    80: "восемьдесят",
    90: "девяносто",
    100: "сто",
    200: "двести",
    300: "триста",
    400: "четыреста",
    500: "пятьсот",
    600: "шестьсот",
    700: "семьсот",
    800: "восемьсот",
    900: "девятьсот",
}


@lru_cache(maxsize=4096)
def normalize_written_text(text: str, language: str) -> WrittenTextNormalization:
    nemo_lang = _nemo_language(language)
    prepared = _expand_acronyms(text, language)
    if nemo_lang is None:
        return WrittenTextNormalization(text, prepared, "identity", [])

    try:
        normalized = _nemo_normalize(prepared, nemo_lang)
    # Deliberate fallback to acronym-expanded text on any NeMo failure.
    except Exception as exc:  # pragma: no cover  # noqa: BLE001
        return WrittenTextNormalization(
            original_text=text,
            normalized_text=prepared,
            backend=f"{NEMO_BACKEND}:{nemo_lang}:failed",
            warnings=[f"NeMo text normalization failed; used acronym-expanded text: {exc}"],
        )

    return WrittenTextNormalization(
        original_text=text,
        normalized_text=_clean_normalized_text(normalized),
        backend=f"{NEMO_BACKEND}:{nemo_lang}",
        warnings=[],
    )


def _nemo_language(language: str) -> str | None:
    if language.startswith("ru"):
        return "ru"
    if language.startswith("en"):
        return "en"
    return None


def _expand_acronyms(text: str, language: str) -> str:
    if language.startswith("ru"):
        return RU_ACRONYM_RE.sub(_expand_russian_acronym_match, text)
    if language.startswith("en"):
        return EN_ACRONYM_RE.sub(_expand_english_acronym_match, text)
    return text


def _expand_english_acronym_match(match: re.Match[str]) -> str:
    token = match.group(0)
    letters = [char for char in token if char.isalpha()]
    if ROMAN_NUMERAL_RE.fullmatch("".join(letters)):
        return token
    return " ".join(EN_LETTER_NAMES.get(char, char.lower()) for char in letters)


def _expand_russian_acronym_match(match: re.Match[str]) -> str:
    token = match.group(0)
    if token in RU_ACRONYMS:
        return RU_ACRONYMS[token]
    return " ".join(RU_LETTER_NAMES.get(char, char.lower()) for char in token)


def _nemo_normalize(text: str, language: str) -> str:
    if language == "ru":
        candidates = _ru_nemo_normalizer().normalize(text, n_tagged=50)
        return _choose_russian_candidate(text, candidates)
    return _en_nemo_normalizer().normalize(text, verbose=False, punct_post_process=True)


@lru_cache(maxsize=1)
def _en_nemo_normalizer() -> Any:
    _quiet_nemo_logger()
    from nemo_text_processing.text_normalization.normalize import Normalizer

    _quiet_nemo_logger()
    NEMO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return Normalizer(input_case="cased", lang="en", cache_dir=str(NEMO_CACHE_DIR))


@lru_cache(maxsize=1)
def _ru_nemo_normalizer() -> Any:
    _quiet_nemo_logger()
    from nemo_text_processing.text_normalization.normalize_with_audio import NormalizerWithAudio

    _quiet_nemo_logger()
    NEMO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return NormalizerWithAudio(input_case="cased", lang="ru", cache_dir=str(NEMO_CACHE_DIR))


def _quiet_nemo_logger() -> None:
    logger = logging.getLogger("NeMo-text-processing")
    logger.setLevel(logging.ERROR)
    logger.propagate = False
    for handler in logger.handlers:
        handler.setLevel(logging.ERROR)


def _choose_russian_candidate(original_text: str, candidates: object) -> str:
    if isinstance(candidates, str):
        return candidates
    resolved = _candidate_strings(candidates)
    if not resolved:
        return original_text
    return min(resolved, key=lambda candidate: _russian_candidate_score(original_text, candidate))


def _candidate_strings(candidates: object) -> list[str]:
    if isinstance(candidates, Iterable) and not isinstance(candidates, str | bytes):
        return sorted(str(candidate) for candidate in candidates)
    return [str(candidates)]


def _russian_candidate_score(original_text: str, candidate: str) -> tuple[int, int, int, int, str]:
    lower = candidate.casefold()
    score = _symbol_penalty(candidate)
    if re.search(r"\d", candidate):
        score += 200
    score += _russian_percent_penalty(original_text, lower)
    score += _russian_thousands_penalty(original_text, lower)
    score += _russian_measure_penalty(original_text, lower)
    score += _russian_date_penalty(original_text, lower)
    return (
        score,
        len(candidate.split()),
        len(candidate),
        len(re.findall(r"\bпроцент\w*", lower)),
        candidate,
    )


def _symbol_penalty(candidate: str) -> int:
    return 200 if "%" in candidate else 0


def _russian_percent_penalty(original_text: str, candidate_lower: str) -> int:
    if "%" not in original_text:
        return 0
    return -20 if "процентов" in candidate_lower else 25


def _russian_thousands_penalty(original_text: str, candidate_lower: str) -> int:
    original_lower = original_text.casefold()
    score = 0
    for match in FOUR_DIGIT_NUMBER_RE.finditer(original_text):
        thousands_digit = int(match.group(1))
        if not re.search(r"\b(?:тысяча|тысячи|тысяч)\b", candidate_lower):
            score += 80
        if thousands_digit == 1 and re.search(r"\bтысяч\b", candidate_lower):
            score += 40
            if re.search(r"\bиз\s+(?:тысячи|одной тысячи)\b", candidate_lower):
                score -= 10
            if re.search(r"\bиз\s+(?:тысяча|одна тысяча)\b", candidate_lower):
                score += 10
        if thousands_digit in {2, 3, 4}:
            if re.search(r"\b\w+ тысяч\b", candidate_lower):
                score += 40
            if re.search(r"\b(?:две|три|четыре) тысяча\b", candidate_lower):
                score += 20
            if re.search(r"\b(?:две|три|четыре) тысячи\b", candidate_lower):
                score -= 10
    if re.search(r"\bиз\s+\d{4}\b", original_lower):
        if re.search(r"\bиз\s+(?:тысячи|одной тысячи)\b", candidate_lower):
            score -= 10
        if re.search(r"\bиз\s+(?:тысяча|одна тысяча)\b", candidate_lower):
            score += 10
    return score


def _russian_measure_penalty(original_text: str, candidate_lower: str) -> int:
    original_lower = original_text.casefold()
    score = 0
    if re.search(r"\b\d+\s+тысяч", original_lower) and "тысяч" not in candidate_lower:
        score += 10
    score += _russian_number_thousand_penalty(original_lower, candidate_lower)
    if re.search(r"\b\d+\s+метр", original_lower) and "метр" not in candidate_lower:
        score += 10
    if re.search(r"\b\d+\s+фут", original_lower) and "фут" not in candidate_lower:
        score += 10
    return score


def _russian_number_thousand_penalty(original_lower: str, candidate_lower: str) -> int:
    score = 0
    for match in NUMBER_THOUSAND_RE.finditer(original_lower):
        expected = _russian_quantity_cardinal(int(match.group(1)))
        if not expected:
            continue
        expected_phrase = f"{expected} тысяч"
        if expected_phrase in candidate_lower:
            score -= 15
        elif "тысяч" in candidate_lower:
            score += 25
    return score


def _russian_quantity_cardinal(number: int) -> str:
    if number in RU_CARDINAL_QUANTITIES:
        return RU_CARDINAL_QUANTITIES[number]
    hundreds, remainder = divmod(number, 100)
    parts: list[str] = []
    if hundreds:
        hundred = RU_CARDINAL_QUANTITIES.get(hundreds * 100)
        if not hundred:
            return ""
        parts.append(hundred)
    if remainder:
        if remainder in RU_CARDINAL_QUANTITIES:
            parts.append(RU_CARDINAL_QUANTITIES[remainder])
        else:
            tens, ones = divmod(remainder, 10)
            ten = RU_CARDINAL_QUANTITIES.get(tens * 10)
            one = RU_CARDINAL_QUANTITIES.get(ones)
            if not ten or not one:
                return ""
            parts.extend([ten, one])
    return " ".join(parts)


def _russian_date_penalty(original_text: str, candidate_lower: str) -> int:
    if not re.search(r"\b\d+\s+\w+бря\b", original_text.casefold()):
        return 0
    ordinal_words = ("первого", "второго", "третьего", "шестого")
    return 0 if any(word in candidate_lower for word in ordinal_words) else 10


def _clean_normalized_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
