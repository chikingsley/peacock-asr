from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from p016_compare.normalization import normalize_phone_tokens, split_phone_text
from p016_compare.text_normalization import WrittenTextNormalization, normalize_written_text

_WORD_RE = re.compile(r"[^\W_]+(?:[-'][^\W_]+)?", re.UNICODE)
_ROMAN_RE = re.compile(r"^[ivxlcdm]+$")
CHARSIU_MODEL_ID = "charsiu/g2p_multilingual_byT5_tiny_16_layers_100"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_MFA_DIR = PROJECT_ROOT / ".mfa"
PROJECT_MFA_BIN = PROJECT_MFA_DIR / "env" / "bin" / "mfa"
PROJECT_MFA_ROOT = PROJECT_MFA_DIR / "root"
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
        if self.preferred_backend in {"auto_zipa", "charsiu"}:
            if self.preferred_backend == "auto_zipa" and language.startswith("en"):
                voice = _espeak_voice(language)
                phones = _merge_word_parts(_espeak_g2p(expanded_words, voice), part_counts)
                return _result(words, phones, f"espeak-ng:{voice}", warnings, text_normalization)
            try:
                phones = _merge_word_parts(_charsiu_g2p(expanded_words, language), part_counts)
                return _result(
                    words,
                    phones,
                    f"charsiu:{CHARSIU_MODEL_ID}",
                    warnings,
                    text_normalization,
                )
            except RuntimeError as exc:
                if self.preferred_backend == "charsiu":
                    raise
                warnings.append(str(exc))

        if self.preferred_backend == "espeak":
            voice = _espeak_voice(language)
            phones = _merge_word_parts(_espeak_g2p(expanded_words, voice), part_counts)
            return _result(words, phones, f"espeak-ng:{voice}", warnings, text_normalization)

        if language.startswith("ru") and self.preferred_backend in {"auto", "mfa"}:
            try:
                raw_phones, backend, mfa_warnings = _russian_mfa_g2p(expanded_words)
                warnings.extend(mfa_warnings)
                phones = _merge_word_parts(raw_phones, part_counts)
                return _result(words, phones, backend, warnings, text_normalization)
            except RuntimeError as exc:
                if self.preferred_backend == "mfa":
                    raise
                warnings.append(str(exc))

        voice = _espeak_voice(language)
        phones = _merge_word_parts(_espeak_g2p(expanded_words, voice), part_counts)
        backend = f"espeak-ng:{voice}"
        if self.preferred_backend == "auto_zipa":
            warnings.append("CharsiuG2P was unavailable; used espeak-ng fallback.")
        elif language.startswith("ru"):
            warnings.append("Russian MFA G2P was unavailable; used espeak-ng fallback.")
        return _result(words, phones, backend, warnings, text_normalization)


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
    if next_normalized == "века" or previous_normalized in {"людовика"}:
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
    if total <= 0 or total > 9999:
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
    if number < 100:
        return _russian_ordinal_genitive_parts(number)
    if number < 1000:
        hundreds, remainder = divmod(number, 100)
        parts = [RU_HUNDREDS[hundreds * 100]] if hundreds else []
        ordinal = _russian_ordinal_genitive_parts(remainder)
        if ordinal:
            return [*parts, *ordinal]
        return []
    if 1000 <= number < 2000:
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
    if number < 0 or number > 9999:
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


def _mfa_g2p(words: list[str], model_name: str) -> list[list[str]]:
    mfa_bin = _mfa_executable()
    if mfa_bin is None:
        raise RuntimeError("MFA CLI not found; cannot run Russian MFA G2P.")

    cmd = [
        mfa_bin,
        "g2p",
        "--quiet",
        "--no_use_mp",
        "--num_pronunciations",
        "1",
    ]
    with tempfile.TemporaryDirectory(prefix="p016_mfa_g2p_") as temp_dir:
        temp_path = Path(temp_dir)
        input_path = temp_path / "words.txt"
        output_path = temp_path / "phones.dict"
        input_path.write_text("\n".join(words) + "\n", encoding="utf-8")
        proc = subprocess.run(
            [
                *cmd,
                str(input_path),
                model_name,
                str(output_path),
            ],
            text=True,
            capture_output=True,
            check=False,
            env=_mfa_env(mfa_bin),
        )
        if proc.returncode != 0:
            raise RuntimeError(f"MFA G2P failed: {proc.stderr.strip() or proc.stdout.strip()}")
        if not output_path.exists():
            raise RuntimeError("MFA G2P did not write an output dictionary.")
        output = output_path.read_text(encoding="utf-8")

    pronunciations: dict[str, list[str]] = {}
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) < 2:
            continue
        pronunciations.setdefault(parts[0].lower(), parts[1:])

    missing = [word for word in words if word not in pronunciations]
    if missing:
        raise RuntimeError(f"MFA G2P returned no pronunciation for: {', '.join(missing[:8])}")
    return [pronunciations[word] for word in words]


def _russian_mfa_g2p(words: list[str]) -> tuple[list[list[str]], str, list[str]]:
    latin_positions = [index for index, word in enumerate(words) if _is_latin_word(word)]
    if not latin_positions:
        return _mfa_g2p(words, "russian_mfa"), "mfa:russian_mfa", []

    phones_by_index: list[list[str] | None] = [None] * len(words)
    cyrillic_pairs = [
        (index, word) for index, word in enumerate(words) if index not in latin_positions
    ]
    if cyrillic_pairs:
        cyrillic_phones = _mfa_g2p([word for _, word in cyrillic_pairs], "russian_mfa")
        for (index, _), phones in zip(cyrillic_pairs, cyrillic_phones, strict=True):
            phones_by_index[index] = phones

    latin_words = [words[index] for index in latin_positions]
    latin_phones = _espeak_g2p(latin_words, "en-us")
    for index, phones in zip(latin_positions, latin_phones, strict=True):
        phones_by_index[index] = phones

    return (
        [phones or [] for phones in phones_by_index],
        "mfa:russian_mfa+espeak-ng:en-us-latin",
        ["Latin-script words in Russian text used espeak-ng:en-us target G2P."],
    )


def _is_latin_word(word: str) -> bool:
    return bool(re.fullmatch(r"[a-z]+(?:'[a-z]+)?", word.casefold()))


def _mfa_executable() -> str | None:
    configured = os.getenv("MFA_BIN")
    if configured:
        configured_path = Path(configured).expanduser()
        if configured_path.exists():
            return str(configured_path)
    if PROJECT_MFA_BIN.exists():
        return str(PROJECT_MFA_BIN)
    discovered = shutil.which("mfa")
    if discovered:
        return discovered
    return None


def _mfa_env(mfa_bin: str) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("MFA_ROOT_DIR", str(PROJECT_MFA_ROOT))
    mfa_bin_dir = str(Path(mfa_bin).resolve().parent)
    env["PATH"] = f"{mfa_bin_dir}:{env.get('PATH', '')}"
    return env


def _charsiu_g2p(words: list[str], language: str) -> list[list[str]]:
    code = _charsiu_language_code(language)
    tokenizer, model = _charsiu_model()
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for CharsiuG2P.") from exc

    prefixed_words = [f"<{code}>: {word}" for word in words]
    inputs = tokenizer(
        prefixed_words,
        padding=True,
        add_special_tokens=False,
        return_tensors="pt",
    )
    with torch.no_grad():
        predictions = model.generate(**inputs, num_beams=1, max_length=64)
    phone_texts = tokenizer.batch_decode(predictions.tolist(), skip_special_tokens=True)
    return [split_phone_text(phone_text) for phone_text in phone_texts]


@lru_cache(maxsize=1)
def _charsiu_model() -> tuple[Any, Any]:
    try:
        from transformers import AutoTokenizer, T5ForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError("transformers is required for CharsiuG2P.") from exc

    tokenizer = AutoTokenizer.from_pretrained(CHARSIU_MODEL_ID)
    model = T5ForConditionalGeneration.from_pretrained(CHARSIU_MODEL_ID)
    model.eval()
    return tokenizer, model


def _charsiu_language_code(language: str) -> str:
    if language == "en_gb":
        return "eng-uk"
    if language.startswith("en"):
        return "eng-us"
    if language.startswith("ru"):
        return "rus"
    raise RuntimeError(f"CharsiuG2P does not have a configured language code for {language!r}.")


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


def _espeak_voice(language: str) -> str:
    if language == "en_gb":
        return "en-gb"
    if language.startswith("en"):
        return "en-us"
    if language.startswith("ru"):
        return "ru"
    return language
