from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from capt.normalization import normalize_phone_tokens, split_phone_text
from capt.text_normalization import WrittenTextNormalization, normalize_written_text

_WORD_RE = re.compile(r"[^\W_]+(?:[-'][^\W_]+)?", re.UNICODE)
_ROMAN_RE = re.compile(r"^[ivxlcdm]+$")
# Bounds for the supported integer/numeral range when spelling numbers as Russian words.
MAX_SPELLED_NUMBER = 9999
HUNDRED = 100
THOUSAND = 1000
TWO_THOUSAND = 2000
RU_ASR_WORD_REWRITES = {
    "wi-fi": ["вай", "фай"],
    "wifi": ["вай", "фай"],
}
RU_ONES = {
    1: "один",
    2: "два",
    3: "три",
    4: "четыре",
    5: "пять",
    6: "шесть",
    7: "семь",
    8: "восемь",
    9: "девять",
}
RU_THOUSAND_ONES = {
    1: "одна",
    2: "две",
    3: "три",
    4: "четыре",
    5: "пять",
    6: "шесть",
    7: "семь",
    8: "восемь",
    9: "девять",
}
RU_TEENS = {
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
}
RU_TENS = {
    20: "двадцать",
    30: "тридцать",
    40: "сорок",
    50: "пятьдесят",
    60: "шестьдесят",
    70: "семьдесят",
    80: "восемьдесят",
    90: "девяносто",
}
RU_HUNDREDS = {
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
RU_ORDINAL_GENITIVE = {
    1: "первого",
    2: "второго",
    3: "третьего",
    4: "четвертого",
    5: "пятого",
    6: "шестого",
    7: "седьмого",
    8: "восьмого",
    9: "девятого",
    10: "десятого",
    11: "одиннадцатого",
    12: "двенадцатого",
    13: "тринадцатого",
    14: "четырнадцатого",
    15: "пятнадцатого",
    16: "шестнадцатого",
    17: "семнадцатого",
    18: "восемнадцатого",
    19: "девятнадцатого",
    20: "двадцатого",
    30: "тридцатого",
    40: "сорокового",
    50: "пятидесятого",
    60: "шестидесятого",
    70: "семидесятого",
    80: "восьмидесятого",
    90: "девяностого",
}
ROMAN_VALUES = {
    "i": 1,
    "v": 5,
    "x": 10,
    "l": 50,
    "c": 100,
    "d": 500,
    "m": 1000,
}
RU_MONTH_GENITIVES = {
    "января",
    "февраля",
    "марта",
    "апреля",
    "мая",
    "июня",
    "июля",
    "августа",
    "сентября",
    "октября",
    "ноября",
    "декабря",
}
RU_COUNTING_GENITIVE = {
    1: "одного",
    2: "двух",
    3: "трех",
    4: "четырех",
    5: "пяти",
    6: "шести",
    7: "семи",
    8: "восьми",
    9: "девяти",
    10: "десяти",
    11: "одиннадцати",
    12: "двенадцати",
    13: "тринадцати",
    14: "четырнадцати",
    15: "пятнадцати",
    16: "шестнадцати",
    17: "семнадцати",
    18: "восемнадцати",
    19: "девятнадцати",
}


@dataclass(frozen=True)
class G2PResult:
    words: list[str]
    phones_per_word_raw: list[list[str]]
    phones_per_word_normalized: list[list[str]]
    backend: str
    warnings: list[str]
    input_text: str = ""
    normalized_text: str = ""
    text_normalization_backend: str = "identity"
    text_normalization_warnings: list[str] = field(default_factory=list)

    @property
    def flat_raw(self) -> list[str]:
        return [phone for phones in self.phones_per_word_raw for phone in phones]

    @property
    def flat_normalized(self) -> list[str]:
        return [phone for phones in self.phones_per_word_normalized for phone in phones]

    @property
    def word_spans(self) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        cursor = 0
        for phones in self.phones_per_word_normalized:
            start = cursor
            cursor += len(phones)
            spans.append((start, cursor))
        return spans


def words_from_text(text: str) -> list[str]:
    return [match.group(0).lower() for match in _WORD_RE.finditer(text)]


class TargetG2P:
    def __init__(self, preferred_backend: str = "auto") -> None:
        self.preferred_backend = preferred_backend

    def from_text(self, text: str, language: str) -> G2PResult:
        normalized = normalize_written_text(text, language)
        return self._from_words(
            words_from_text(normalized.normalized_text),
            language,
            text_normalization=normalized,
        )

    def from_words(self, words: list[str], language: str) -> G2PResult:
        return self._from_words(words, language)

    def _from_words(
        self,
        words: list[str],
        language: str,
        text_normalization: WrittenTextNormalization | None = None,
    ) -> G2PResult:
        if not words:
            return G2PResult(
                [],
                [],
                [],
                "none",
                ["ASR produced no parseable words."],
                **_text_norm_kwargs(text_normalization),
            )

        expanded_words, part_counts = _expand_words(words, language)
        warnings: list[str] = list(text_normalization.warnings) if text_normalization else []

        backend = self.preferred_backend
        if backend == "routed":
            backend = _routed_backend(language)
        if backend in {"auto", "paper"}:
            return _chain_g2p(
                words, expanded_words, part_counts, language, warnings, text_normalization
            )
        if backend in _SINGLE_BACKENDS:
            return _try_single(
                _SINGLE_BACKENDS[backend], backend, words, expanded_words, part_counts,
                language, warnings, text_normalization,
            )
        return _espeak_result(
            words, expanded_words, part_counts, language, warnings, text_normalization
        )


def _result(
    words: list[str],
    phones_per_word: list[list[str]],
    backend: str,
    warnings: list[str],
    text_normalization: WrittenTextNormalization | None = None,
) -> G2PResult:
    return G2PResult(
        words=words,
        phones_per_word_raw=phones_per_word,
        phones_per_word_normalized=[normalize_phone_tokens(phones) for phones in phones_per_word],
        backend=backend,
        warnings=warnings,
        **_text_norm_kwargs(text_normalization),
    )


def _text_norm_kwargs(
    text_normalization: WrittenTextNormalization | None,
) -> dict[str, str | list[str]]:
    if text_normalization is None:
        return {}
    return {
        "input_text": text_normalization.original_text,
        "normalized_text": text_normalization.normalized_text,
        "text_normalization_backend": text_normalization.backend,
        "text_normalization_warnings": text_normalization.warnings,
    }


def _expand_words(words: list[str], language: str) -> tuple[list[str], list[int]]:
    expanded: list[str] = []
    part_counts: list[int] = []
    for index, word in enumerate(words):
        previous_word = words[index - 1] if index else None
        next_word = words[index + 1] if index + 1 < len(words) else None
        parts = _spoken_word_parts(word, language, previous_word, next_word)
        expanded.extend(parts)
        part_counts.append(len(parts))
    return expanded, part_counts


def _spoken_word_parts(
    word: str,
    language: str,
    previous_word: str | None = None,
    next_word: str | None = None,
) -> list[str]:
    if language.startswith("ru"):
        normalized = re.sub(r"[-\u2010-\u2015]+", "-", word.casefold())
        if normalized in RU_ASR_WORD_REWRITES:
            return RU_ASR_WORD_REWRITES[normalized]
        numeric_suffix_parts = _russian_numeric_suffix_parts(normalized)
        if numeric_suffix_parts:
            return numeric_suffix_parts
        if normalized.isdecimal():
            next_normalized = next_word.casefold() if next_word else None
            if next_normalized in RU_MONTH_GENITIVES:
                ordinal_parts = _russian_ordinal_genitive_parts(int(normalized))
                if ordinal_parts:
                    return ordinal_parts
            if next_normalized == "года":
                year_parts = _russian_year_genitive_parts(int(normalized))
                if year_parts:
                    return year_parts
            numeric_parts = _russian_cardinal_parts(int(normalized))
            if numeric_parts:
                return numeric_parts
        roman_parts = _russian_roman_parts(normalized, previous_word, next_word)
        if roman_parts:
            return roman_parts

    parts = [part for part in re.split(r"[-\u2010-\u2015]+", word) if part]
    return parts or [word]


def _russian_numeric_suffix_parts(word: str) -> list[str]:
    if match := re.fullmatch(r"(\d+)-го", word):
        return _russian_year_genitive_parts(int(match.group(1))) or _russian_ordinal_genitive_parts(
            int(match.group(1))
        )
    if match := re.fullmatch(r"(\d+)-лет(.+)", word):
        number = int(match.group(1))
        suffix = match.group(2)
        genitive = _russian_counting_genitive_parts(number)
        if genitive:
            return [*genitive, f"лет{suffix}"]
    return []


def _russian_roman_parts(
    word: str,
    previous_word: str | None,
    next_word: str | None,
) -> list[str]:
    if not _ROMAN_RE.match(word):
        return []
    number = _roman_to_int(word)
    if number is None:
        return []
    previous_normalized = previous_word.casefold() if previous_word else None
    next_normalized = next_word.casefold() if next_word else None
    if next_normalized == "века" or previous_normalized == "людовика":
        ordinal = _russian_ordinal_genitive_parts(number)
        if ordinal:
            return ordinal
    return _russian_cardinal_parts(number)


def _roman_to_int(word: str) -> int | None:
    total = 0
    previous = 0
    for char in reversed(word):
        value = ROMAN_VALUES[char]
        if value < previous:
            total -= value
        else:
            total += value
            previous = value
    if total <= 0 or total > MAX_SPELLED_NUMBER:
        return None
    return total


def _russian_ordinal_genitive_parts(number: int) -> list[str]:
    if number in RU_ORDINAL_GENITIVE:
        return [RU_ORDINAL_GENITIVE[number]]
    tens, ones = divmod(number, 10)
    if tens and ones and tens * 10 in RU_TENS and ones in RU_ORDINAL_GENITIVE:
        return [RU_TENS[tens * 10], RU_ORDINAL_GENITIVE[ones]]
    return []


def _russian_year_genitive_parts(number: int) -> list[str]:
    if number < HUNDRED:
        return _russian_ordinal_genitive_parts(number)
    if number < THOUSAND:
        hundreds, remainder = divmod(number, HUNDRED)
        parts = [RU_HUNDREDS[hundreds * 100]] if hundreds else []
        ordinal = _russian_ordinal_genitive_parts(remainder)
        if ordinal:
            return [*parts, *ordinal]
        return []
    if THOUSAND <= number < TWO_THOUSAND:
        ordinal = _russian_year_genitive_parts(number - 1000)
        if ordinal:
            return ["тысяча", *ordinal]
    return []


def _russian_counting_genitive_parts(number: int) -> list[str]:
    if number in RU_COUNTING_GENITIVE:
        return [RU_COUNTING_GENITIVE[number]]
    tens, ones = divmod(number, 10)
    if tens and ones and tens * 10 in RU_TENS and ones in RU_COUNTING_GENITIVE:
        return [RU_TENS[tens * 10], RU_COUNTING_GENITIVE[ones]]
    return []


def _russian_cardinal_parts(number: int) -> list[str]:
    if number < 0 or number > MAX_SPELLED_NUMBER:
        return []
    if number == 0:
        return ["ноль"]

    parts: list[str] = []
    thousands, remainder = divmod(number, 1000)
    if thousands:
        if thousands in RU_THOUSAND_ONES:
            parts.append(RU_THOUSAND_ONES[thousands])
        else:
            return []
        if thousands == 1:
            parts.append("тысяча")
        elif thousands in {2, 3, 4}:
            parts.append("тысячи")
        else:
            parts.append("тысяч")
    parts.extend(_russian_under_thousand_parts(remainder))
    return parts


def _russian_under_thousand_parts(number: int) -> list[str]:
    parts: list[str] = []
    hundreds, remainder = divmod(number, 100)
    if hundreds:
        parts.append(RU_HUNDREDS[hundreds * 100])
    if remainder in RU_TEENS:
        parts.append(RU_TEENS[remainder])
        return parts
    tens, ones = divmod(remainder, 10)
    if tens:
        parts.append(RU_TENS[tens * 10])
    if ones:
        parts.append(RU_ONES[ones])
    return parts


def _merge_word_parts(
    phones_per_part: list[list[str]],
    part_counts: list[int],
) -> list[list[str]]:
    merged: list[list[str]] = []
    cursor = 0
    for count in part_counts:
        phones: list[str] = []
        for part_phones in phones_per_part[cursor : cursor + count]:
            phones.extend(part_phones)
        merged.append(phones)
        cursor += count
    return merged


def _espeak_g2p(words: list[str], voice: str) -> list[list[str]]:
    if shutil.which("espeak-ng") is None:
        raise RuntimeError("espeak-ng CLI not found; cannot run fallback G2P.")

    sentence_results = _espeak_g2p_sentence(words, voice)
    if sentence_results is not None:
        return sentence_results

    results: list[list[str]] = []
    for word in words:
        proc = subprocess.run(
            ["espeak-ng", "-q", "--ipa=3", "-v", voice, word],
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"espeak-ng failed for {word!r}: {proc.stderr.strip()}")
        results.append(split_phone_text(proc.stdout))
    return results


def _espeak_g2p_sentence(words: list[str], voice: str) -> list[list[str]] | None:
    proc = subprocess.run(
        ["espeak-ng", "-q", "--ipa=3", "-v", voice, " ".join(words)],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    phone_words = proc.stdout.split()
    if len(phone_words) != len(words):
        return None
    return [split_phone_text(phone_word) for phone_word in phone_words]


def _espeak_result(
    words: list[str],
    expanded_words: list[str],
    part_counts: list[int],
    language: str,
    warnings: list[str],
    text_normalization: WrittenTextNormalization | None,
) -> G2PResult:
    voice = _espeak_voice(language)
    phones = _merge_word_parts(_espeak_g2p(expanded_words, voice), part_counts)
    return _result(words, phones, f"espeak-ng:{voice}", warnings, text_normalization)


# FLEURS config / language code -> Epitran code (ISO 639-3 + script). Epitran's IPA convention
# matches ZIPA's training targets. Latin/Cyrillic are pure-Python; eng needs flite and jpn needs
# MeCab — a missing system dep raises and falls back to espeak (recorded in warnings).
# Keyed by the FLEURS config's first token (af_za -> "af"). Values verified against Epitran's
# installed map set. eng is omitted (Epitran English needs flite; CharsiuG2P covers it); CJK
# (cmn/ja/yue) need romanization first, so they route to per-script tools, not Epitran.
_EPITRAN_CODES = {
    "af": "afr-Latn", "am": "amh-Ethi", "ar": "ara-Arab", "az": "aze-Latn", "bn": "ben-Beng",
    "ca": "cat-Latn", "ceb": "ceb-Latn", "ckb": "ckb-Arab", "cs": "ces-Latn", "cy": "cym-Latn",
    "de": "deu-Latn", "es": "spa-Latn", "et": "est-Latn", "fa": "fas-Arab", "ff": "ful-Latn",
    "fi": "fin-Latn", "fil": "tgl-Latn", "fr": "fra-Latn", "ga": "gle-Latn", "gl": "glg-Latn",
    "ha": "hau-Latn", "hi": "hin-Deva", "hr": "hrv-Latn", "hu": "hun-Latn", "id": "ind-Latn",
    "it": "ita-Latn", "jv": "jav-Latn", "ka": "kat-Geor", "kk": "kaz-Cyrl", "km": "khm-Khmr",
    "kn": "kan-Knda", "ko": "kor-Hang", "ky": "kir-Cyrl", "lg": "lug-Latn", "lo": "lao-Laoo",
    "lt": "lit-Latn", "lv": "lav-Latn", "mi": "mri-Latn", "ml": "mal-Mlym", "mn": "mon-Cyrl-bab",
    "mr": "mar-Deva", "ms": "msa-Latn", "mt": "mlt-Latn", "my": "mya-Mymr", "nl": "nld-Latn",
    "ny": "nya-Latn", "oc": "oci-Latn", "om": "orm-Latn", "or": "ori-Orya", "pa": "pan-Guru",
    "pl": "pol-Latn", "ps": "pbu-Arab", "pt": "por-Latn", "ro": "ron-Latn", "ru": "rus-Cyrl",
    "sl": "slv-Latn", "sn": "sna-Latn", "so": "som-Latn", "sr": "srp-Cyrl", "sv": "swe-Latn",
    "sw": "swa-Latn", "ta": "tam-Taml", "te": "tel-Telu", "tg": "tgk-Cyrl", "th": "tha-Thai",
    "tr": "tur-Latn", "uk": "ukr-Cyrl", "ur": "urd-Arab", "uz": "uzb-Latn", "vi": "vie-Latn",
    "xh": "xho-Latn", "yo": "yor-Latn", "zu": "zul-Latn",
}


def _epitran_code(language: str) -> str:
    key = language.split("_", 1)[0].lower()
    if key not in _EPITRAN_CODES:
        raise RuntimeError(f"No Epitran code configured for {language!r}.")
    return _EPITRAN_CODES[key]


@lru_cache(maxsize=8)
def _epitran_instance(code: str) -> Any:
    import epitran

    return epitran.Epitran(code)


def _epitran_g2p(words: list[str], language: str) -> list[list[str]]:
    epi = _epitran_instance(_epitran_code(language))
    return [split_phone_text(epi.transliterate(word)) for word in words]


def _try_single(
    g2p_fn: Any,
    label: str,
    words: list[str],
    expanded_words: list[str],
    part_counts: list[int],
    language: str,
    warnings: list[str],
    text_normalization: WrittenTextNormalization | None,
) -> G2PResult:
    """Run one G2P backend; on any failure, record a warning and fall back to espeak-ng."""
    try:
        phones = _merge_word_parts(g2p_fn(expanded_words, language), part_counts)
        return _result(words, phones, label, warnings, text_normalization)
    except Exception as exc:  # noqa: BLE001 - any backend failure falls back to espeak
        warnings.append(f"{label} G2P failed for {language!r}: {exc}")
    try:
        return _espeak_result(words, expanded_words, part_counts, language, warnings,
                              text_normalization)
    except RuntimeError as exc:  # no espeak voice either (gap lang) — empty target, no crash
        warnings.append(f"no G2P backend available for {language!r}: {exc}")
        return _result(words, [[] for _ in words], "none", warnings, text_normalization)


def _chain_g2p(
    words: list[str],
    expanded_words: list[str],
    part_counts: list[int],
    language: str,
    warnings: list[str],
    text_normalization: WrittenTextNormalization | None,
) -> G2PResult:
    """ZIPA paper recipe: Epitran -> CharsiuG2P -> espeak-ng, normalized to ZIPA's inventory."""
    try:
        phones = _merge_word_parts(_epitran_g2p(expanded_words, language), part_counts)
        return _result(
            words, phones, f"epitran:{_epitran_code(language)}", warnings, text_normalization
        )
    except Exception as exc:  # noqa: BLE001 - chain to CharsiuG2P
        warnings.append(f"epitran G2P failed for {language!r}: {exc}")
    try:
        phones = _merge_word_parts(_charsiu_g2p(expanded_words, language), part_counts)
        return _result(
            words, phones, f"charsiu:{_charsiu_code(language)}", warnings, text_normalization
        )
    except Exception as exc:  # noqa: BLE001 - chain to espeak-ng
        warnings.append(f"charsiu G2P failed for {language!r}: {exc}")
    try:
        return _espeak_result(
            words, expanded_words, part_counts, language, warnings, text_normalization
        )
    except RuntimeError as exc:  # no working G2P (e.g. no espeak voice) — empty target, no crash
        warnings.append(f"no G2P backend available for {language!r}: {exc}")
        return _result(words, [[] for _ in words], "none", warnings, text_normalization)


# FLEURS / language code -> CharsiuG2P tag (the byT5 multilingual G2P, ZIPA's secondary source).
CHARSIU_MODEL_ID = "charsiu/g2p_multilingual_byT5_tiny_16_layers_100"
# Keyed by FLEURS first token -> CharsiuG2P tag (idiosyncratic: ger/dut/geo/cze/gre/bur/ice;
# regional variants chosen to match the FLEURS locale: spa-latin for es_419, por-bz for pt_br,
# vie-n, zho-s). Complementary to Epitran — Charsiu covers bg/el/da/is/hy/mk/sk/bs/be/gu etc.
_CHARSIU_CODES = {
    "af": "afr", "am": "amh", "ar": "ara", "az": "aze", "be": "bel", "bg": "bul", "bn": "ben",
    "bs": "bos", "ca": "cat", "cmn": "zho-s", "cs": "cze", "cy": "wel-nw", "da": "dan",
    "de": "ger", "el": "gre", "en": "eng-us", "es": "spa-latin", "et": "est", "fa": "fas",
    "fi": "fin", "fil": "tgl", "fr": "fra", "ga": "gle", "gl": "glg", "gu": "guj", "hi": "hin",
    "hr": "hbs-latn", "hu": "hun", "hy": "arm-e", "id": "ind", "is": "ice", "it": "ita",
    "ja": "jpn", "ka": "geo", "kk": "kaz", "km": "khm", "ko": "kor", "lb": "ltz", "lt": "lit",
    "mk": "mac", "mt": "mlt", "my": "bur", "nb": "nob", "nl": "dut", "or": "ori", "pl": "pol",
    "pt": "por-bz", "ro": "ron", "ru": "rus", "sd": "snd", "sk": "slo", "sl": "slv", "sr": "srp",
    "sv": "swe", "sw": "swa", "ta": "tam", "th": "tha", "tr": "tur", "uk": "ukr", "vi": "vie-n",
    "yue": "yue",
}


def _charsiu_code(language: str) -> str:
    key = language.split("_", 1)[0].lower()
    if key not in _CHARSIU_CODES:
        raise RuntimeError(f"No CharsiuG2P code configured for {language!r}.")
    return _CHARSIU_CODES[key]


@lru_cache(maxsize=1)
def _charsiu_model() -> tuple[Any, Any]:
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    tokenizer = AutoTokenizer.from_pretrained(CHARSIU_MODEL_ID)
    model = T5ForConditionalGeneration.from_pretrained(CHARSIU_MODEL_ID)
    model.eval()
    return tokenizer, model


def _charsiu_g2p(words: list[str], language: str) -> list[list[str]]:
    import torch

    code = _charsiu_code(language)
    tokenizer, model = _charsiu_model()
    prefixed = [f"<{code}>: {word}" for word in words]
    inputs = tokenizer(prefixed, padding=True, add_special_tokens=False, return_tensors="pt")
    with torch.no_grad():
        predictions = model.generate(**inputs, num_beams=1, max_length=64)
    decoded = tokenizer.batch_decode(predictions.tolist(), skip_special_tokens=True)
    return [split_phone_text(text) for text in decoded]


# FLEURS / language code -> trained Phonetisaurus FST. These ZIPA-distilled models cover the
# "gap" languages that have no Epitran/CharsiuG2P/espeak voice (so the paper chain returns empty).
# Stems are keyed by the full FLEURS config (ig_ng.fst); training lives in experiments/g2p_train.
G2P_MODELS_DIR = Path(__file__).parent / "g2p_models"


def _trained_model_path(language: str) -> Path:
    path = G2P_MODELS_DIR / f"{language}.fst"
    if not path.exists():
        raise RuntimeError(f"No trained G2P model for {language!r} at {path}.")
    return path


def _trained_g2p(words: list[str], language: str) -> list[list[str]]:
    import phonetisaurus

    model_path = _trained_model_path(language)
    predictions = dict(phonetisaurus.predict(words, model_path))
    # Phonetisaurus lowercases keys and may drop a word it cannot decode; key on the lowered word
    # and fall back to an empty pronunciation so the per-word alignment stays positional.
    return [split_phone_text(" ".join(predictions.get(word.lower(), []))) for word in words]


# Single-backend G2P dispatch: backend name -> (words, language) -> phones_per_word. Each runs
# through _try_single (espeak-ng fallback on failure). "trained" loads a ZIPA-distilled FST.
_SINGLE_BACKENDS = {
    "epitran": _epitran_g2p,
    "charsiu": _charsiu_g2p,
    "trained": _trained_g2p,
}


@lru_cache(maxsize=1)
def _routing_table() -> dict[str, str]:
    """Per-language best-G2P table produced by scripts/g2p_ablation.py (empty if not yet built)."""
    path = Path(__file__).parent / "g2p_routing.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _routed_backend(language: str) -> str:
    """Resolve the per-language best backend (full code, then bare lang); default = paper chain."""
    table = _routing_table()
    return table.get(language) or table.get(language.split("_", 1)[0]) or "paper"


def _espeak_voice(language: str) -> str:
    if language == "en_gb":
        return "en-gb"
    if language.startswith("en"):
        return "en-us"
    if language.startswith("ru"):
        return "ru"
    # FLEURS configs are "<lang>_<region>" (fr_fr, de_de, es_419); espeak-ng wants the bare
    # language ("fr", "de", "es"). Take the part before the first underscore; bare codes pass
    # through unchanged. This is what lets the espeak target lane run for any FLEURS language.
    return language.split("_", 1)[0]
