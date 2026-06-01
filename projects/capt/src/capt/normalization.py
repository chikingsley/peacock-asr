from __future__ import annotations

import re
import unicodedata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

_BLANK_TOKENS = {
    "",
    "<blank>",
    "<blk>",
    "<ctc_blank>",
    "<pad>",
    "<s>",
    "</s>",
    "[pad]",
    "[unk]",
    "<unk>",
    "|",
    "▁",
}
_DROP_CHARS = {"ˈ", "ˌ", ".", "·", "‿", "\u032a", "\u0334", "\u035c", "\u0361"}
_ATTACH_TO_PREVIOUS = {"ː", "ˑ", "ʰ", "ʲ", "ʷ", "ˠ", "ˤ", "ʼ", "˞"}
_EXPAND_TOKENS = {
    "aɪ": ("a", "ɪ"),
    "aʊ": ("a", "ʊ"),
    "dʒ": ("d", "ʒ"),
    "dʑ": ("d", "ʑ"),
    "eɪ": ("e", "ɪ"),
    "oː": ("o", "ʊ"),
    "oʊ": ("o", "ʊ"),
    "ts": ("t", "s"),
    "tsː": ("t", "s"),
    "tɕ": ("t", "ɕ"),
    "tʃ": ("t", "ʃ"),
}
_CANONICAL_TOKENS = {
    "ɚ": "ə˞",
    "ɝ": "ə˞",
    "ɜ˞": "ə˞",
    "ɜː": "ə˞",
    "ɫ": "l",
    "ɑː": "ɑ",
    "ɔː": "ɔ",
    "iː": "i",
    "uː": "u",
    "sʲː": "sʲ",
    "ɕː": "ɕ",
    "ɲ": "nʲ",
    "ʎ": "lʲ",
}
_SPACE_RE = re.compile(r"\s+")

# ZIPA's broad-transcription token inventory (artifacts/.../tokens.txt): base IPA symbols + the
# 15 diacritics ZIPA keeps. The paper normalizes targets to this set (PHOIBLE-validate, unify
# Unicode, cap diacritics). We reproduce that: drop any char ZIPA cannot emit and cap to one
# diacritic, so the G2P target lives in the same space as the recognizer output.
_ZIPA_DIACRITICS = frozenset("ʰʲʷʼː˞ˠˤ" + "̴̥̩̪̺̃̚")
_ZIPA_BASE = frozenset(
    "abcdefghijklmnopqrstuvwxyz"
    "æçðøħŋœǀǁǂǃ"
    "ɐɑɒɓɔɕɖɗɘəɛɜɞɟɠɢɣɤɥɦɧɨɪɬɭɮɯɰɱɲɳɴɵɶɸɹɺɻɽɾʀʁʂʃʄʈʉʊʋʌʍʎʏʐʑʒʔʕʘʙʛʜʝʟʡʢ"
    "βθχᶑⱱ"
)


def split_phone_text(text: str) -> list[str]:
    cleaned = unicodedata.normalize("NFC", text.strip())
    if not cleaned:
        return []
    if _SPACE_RE.search(cleaned):
        return normalize_phone_tokens(cleaned.split())
    return normalize_phone_tokens(_split_ipa_clusters(cleaned))


def normalize_phone_tokens(tokens: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    for token in tokens:
        clean = normalize_phone_token(token)
        if not clean:
            continue
        if _attach_token(clean):
            if normalized:
                normalized[-1] = normalize_phone_token(f"{normalized[-1]}{clean}")
            continue
        normalized.extend(_expand_token(clean))
    return normalized


def normalize_phone_token(token: str) -> str:
    token = unicodedata.normalize("NFC", token.strip().lower())
    if token in _BLANK_TOKENS:
        return ""
    token = "".join(ch for ch in token if not _drop_char(ch))
    token = token.replace("ɡ", "g")
    token = token.strip()
    token = _CANONICAL_TOKENS.get(token, token)
    token = _restrict_to_zipa(token)
    if token in _BLANK_TOKENS:
        return ""
    return token


def _split_ipa_clusters(text: str) -> list[str]:
    clusters: list[str] = []
    current = ""
    for char in text:
        if char.isspace() or _drop_char(char):
            continue
        category = unicodedata.category(char)
        if not current:
            current = char
        elif category.startswith("M") or char in _ATTACH_TO_PREVIOUS:
            current += char
        else:
            clusters.append(current)
            current = char
    if current:
        clusters.append(current)
    return clusters


def _drop_char(char: str) -> bool:
    if char in _DROP_CHARS:
        return True
    return unicodedata.category(char).startswith("C")


def _attach_token(token: str) -> bool:
    return all(
        char in _ATTACH_TO_PREVIOUS or unicodedata.category(char).startswith("M")
        for char in token
    )


def _expand_token(token: str) -> list[str]:
    expanded = _EXPAND_TOKENS.get(token)
    if expanded is None:
        return [token]
    return list(expanded)


def _restrict_to_zipa(token: str) -> str:
    """Map a phone into ZIPA's broad inventory: keep only emittable chars, cap to 1 diacritic.

    Drops symbols ZIPA cannot output (e.g. the non-syllabic mark ̯) and extra diacritics, so a
    G2P target lands in the same space as the recognizer's output before alignment.
    """
    base = [ch for ch in token if ch in _ZIPA_BASE]
    diacritics = [ch for ch in token if ch in _ZIPA_DIACRITICS]
    if not base:
        # No base char: a standalone diacritic (ZIPA emits these as separate tokens) is kept so
        # the caller can attach it to the previous phone; anything else is unemittable -> drop.
        return "".join(diacritics[:1])
    return "".join(base) + "".join(diacritics[:1])
