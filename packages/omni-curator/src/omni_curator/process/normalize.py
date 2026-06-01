"""Pluggable per-language text normalization for ASR training labels.

A training label must be a clean, consistent, *spoken-form* string that the target model's
tokenizer fully covers. Normalization here does three things, in roughly this order:

- **shared base** (every language): ``unicodedata.normalize("NFKC", ...)``, strip stray
  control/format/symbol characters (ZWNJ, soft hyphen, ``♡``, ``™`` …), collapse whitespace;
- **number expansion**: replace clear digit runs with the language's spoken number words
  (Georgian via ``num2geotext``, Persian via ``num2words(lang="fa")``);
- **language-specific cleanup**: the official toolkit for that language
  (``anbani`` for Georgian script folding, ``hazm`` for Persian, ``tajiknlp`` for Tajik).

Dispatch is a registry ``{language_code: callable}``. An unknown language gets the shared base
only — it is *not* pretended to be fully normalized; opting a language in means adding it to the
registry. Each language's heavy library is imported lazily inside its handler so importing this
module stays cheap and a missing dependency only breaks the one language that needs it (with a
clear message), via the ``normalize`` optional-dependencies extra.
"""

from __future__ import annotations

import re
import unicodedata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

# --------------------------------------------------------------------------------------------
# Shared base (applied to every language, including unknown ones)
# --------------------------------------------------------------------------------------------

# Stray characters the Omni char tokenizer cannot represent and that never carry spoken content:
# Cc (control), Cf (format: ZWNJ/ZWJ/soft-hyphen), So (other symbols: ♡ ™ © ° №), Sk (modifier
# symbols). Real punctuation (Po/Pd/Pi/Pf/Ps/Pe: . , ! ? « » „ “ ” — … ( ) …) is *kept* — the
# tokenizer covers it and it is part of the legitimate label surface.
_STRIP_CATEGORIES = frozenset({"Cc", "Cf", "So", "Sk"})

# NFKC decomposes vulgar fractions (one-half, three-quarters, ...) into two digit runs joined by
# U+2044 FRACTION SLASH, which the Omni tokenizer cannot represent and which strands itself
# between the runs. Map it (and U+2215 DIVISION SLASH) to a space so each part expands as its own
# number rather than leaving a bare unknown slash.
_STRAY_TO_SPACE = frozenset({"\u2044", "\u2215"})

_WHITESPACE_RE = re.compile(r"\s+")


def _strip_stray(text: str) -> str:
    out: list[str] = []
    for ch in text:
        if ch in _STRAY_TO_SPACE:
            out.append(" ")
        elif unicodedata.category(ch) not in _STRIP_CATEGORIES:
            out.append(ch)
    return "".join(out)


def base_normalize(text: str) -> str:
    """NFKC, strip stray control/format/symbol chars, collapse whitespace. Always applied."""
    text = unicodedata.normalize("NFKC", text)
    text = _strip_stray(text)
    return _WHITESPACE_RE.sub(" ", text).strip()


# --------------------------------------------------------------------------------------------
# Conservative number-run detection
# --------------------------------------------------------------------------------------------

# A "clear" number run is one of:
#   - a plain integer:            \d+            (digits may be ASCII or Unicode, e.g. Persian)
#   - a thousands-grouped integer: \d{1,3}(,\d{3})+   (e.g. 5,000,000 ; 1,300)
# Decimal/version-like forms (3,7 ; 802.11) are deliberately *not* matched: the comma is
# ambiguous (thousands vs decimal) across these corpora, so we leave them to the language lib
# rather than guess. ``\d`` is Unicode-aware so Persian/Arabic-Indic digits are caught too.
_INT_RUN_RE = re.compile(r"\d{1,3}(?:,\d{3})+|\d+")


def _replace_number_runs(text: str, to_words: Callable[[int], str]) -> str:
    """Replace each clear integer run with its spoken-word form via ``to_words``.

    If ``to_words`` raises (e.g. ``num2geotext`` rejects >15 digits), the original run is left
    in place untouched — conservative by design.
    """

    def _sub(match: re.Match[str]) -> str:
        raw = match.group()
        digits = raw.replace(",", "")
        try:
            value = int(digits)
        except ValueError:
            return raw
        try:
            words = to_words(value)
        except Exception:  # noqa: BLE001 — third-party number libs raise assorted errors; keep raw
            return raw
        if not words:
            return raw
        return words

    return _INT_RUN_RE.sub(_sub, text)


# --------------------------------------------------------------------------------------------
# Georgian (kat_Geor): num2geotext + anbani Mtavruli→Mkhedruli folding
# --------------------------------------------------------------------------------------------


def normalize_georgian(text: str) -> str:
    """Normalize Georgian: fold Mtavruli capitals, expand integers to Georgian words.

    ``anbani`` folds Mtavruli (uppercase Georgian, U+1C90-U+1CBF) to Mkhedruli, the script the
    tokenizer covers; NFKC does *not* do this fold. ``num2geotext`` (v0.0.1, early) turns integer
    runs into Georgian words; it is wrapped defensively (it raises on >15 digits / negatives), so
    any run it cannot handle is left as-is.
    """
    from anbani.core import converter as _anbani_converter
    from num2geotext import int_num_to_geo_text

    text = base_normalize(text)
    # Fold Mtavruli capitals to Mkhedruli. convert() maps letter-by-letter and passes through
    # already-Mkhedruli letters, digits, spaces and punctuation unchanged.
    text = _anbani_converter.convert(text, "mtavruli", "mkhedruli")
    text = _replace_number_runs(text, int_num_to_geo_text)
    # number words / folding may have re-introduced double spaces.
    return _WHITESPACE_RE.sub(" ", text).strip()


# --------------------------------------------------------------------------------------------
# Persian (fas_Arab): hazm + num2words(lang="fa")
# --------------------------------------------------------------------------------------------

_HAZM_NORMALIZER = None


def _persian_to_words(value: int) -> str:
    from num2words import num2words

    return num2words(value, lang="fa")


def normalize_persian(text: str) -> str:
    """Normalize Persian: hazm orthographic normalization, then expand integers to Persian words.

    ``hazm`` fixes Persian orthography (correct Arabic/Persian letter forms, spacing, ZWNJ) but
    converts ASCII digits to Persian digits rather than words; we then replace the (now Persian
    or ASCII) digit runs with ``num2words(lang="fa")`` spoken words. ``int()``/``\\d`` are
    Unicode-aware, so Persian-Indic digits are handled.
    """
    global _HAZM_NORMALIZER  # noqa: PLW0603 — cache the expensive hazm Normalizer once
    import hazm

    if _HAZM_NORMALIZER is None:
        _HAZM_NORMALIZER = hazm.Normalizer()
    text = base_normalize(text)
    text = _HAZM_NORMALIZER.normalize(text)
    text = _replace_number_runs(text, _persian_to_words)
    return _WHITESPACE_RE.sub(" ", text).strip()


# --------------------------------------------------------------------------------------------
# Tajik (tgk_Cyrl): tajiknlp — mirrors projects/tajik-asr text_normalization.py
# --------------------------------------------------------------------------------------------

_TAJIK_CLEANER = None
_TAJIK_NORMALIZER = None

_TAJIK_PUNCT_RE = re.compile(r"[^\w\sЀ-ӿ]", re.UNICODE)
_TAJIK_CONTROL_RE = re.compile(r"[\u0000-\u001f\u007f-\u009f]")
_TAJIK_SOFT_HYPHEN = "\u00ad"


def normalize_tajik(text: str) -> str:
    """Normalize Tajik via ``tajiknlp``, mirroring the Tajik project's pipeline exactly.

    ``TextCleaner`` + ``TajikCyrillicNormalizer`` + NFKC, then the project's tokenizer-compat
    pass (lowercase, ``№``/soft-hyphen fixes, strip non-Cyrillic punctuation, collapse spaces).
    """
    global _TAJIK_CLEANER, _TAJIK_NORMALIZER  # noqa: PLW0603 — cache expensive tajiknlp objects
    from tajiknlp.components.cleaners.text_cleaner import TextCleaner
    from tajiknlp.components.normalizers.cyrillic import TajikCyrillicNormalizer
    from tajiknlp.schemas import Doc

    if _TAJIK_CLEANER is None:
        _TAJIK_CLEANER = TextCleaner()
    if _TAJIK_NORMALIZER is None:
        _TAJIK_NORMALIZER = TajikCyrillicNormalizer(lowercase=True)

    doc = Doc(text.split("\t", 1)[0].strip())
    doc = _TAJIK_CLEANER(doc)
    doc = _TAJIK_NORMALIZER(doc)
    normalized = unicodedata.normalize("NFKC", doc.metadata["normalized_text"])

    normalized = normalized.lower()
    normalized = normalized.replace(_TAJIK_SOFT_HYPHEN, " ")
    normalized = normalized.replace("№", " no ")
    normalized = normalized.replace("no", " no ")
    normalized = _TAJIK_CONTROL_RE.sub(" ", normalized)
    normalized = _TAJIK_PUNCT_RE.sub(" ", normalized)
    return _WHITESPACE_RE.sub(" ", normalized).strip()


# --------------------------------------------------------------------------------------------
# Registry + dispatch
# --------------------------------------------------------------------------------------------

#: Languages that have opted in to specialized normalization, keyed by FLORES-style code.
NORMALIZERS: dict[str, Callable[[str], str]] = {
    "kat_Geor": normalize_georgian,
    "fas_Arab": normalize_persian,
    "tgk_Cyrl": normalize_tajik,
}


def normalize(text: str, language: str) -> str:
    """Normalize ``text`` for ``language``.

    Dispatches to the language's specialized handler via :data:`NORMALIZERS`. An unknown language
    falls back to :func:`base_normalize` (NFKC + stray-char strip + whitespace) only — it is not
    treated as fully normalized; add the language to the registry to opt it in.
    """
    handler = NORMALIZERS.get(language)
    if handler is None:
        return base_normalize(text)
    return handler(text)
