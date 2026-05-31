from __future__ import annotations

import re
import unicodedata

from tajiknlp.components.cleaners.text_cleaner import TextCleaner
from tajiknlp.components.normalizers.cyrillic import TajikCyrillicNormalizer
from tajiknlp.schemas import Doc

PUNCT_RE = re.compile(r"[^\w\s\u0400-\u04ff]", re.UNICODE)
SPACE_RE = re.compile(r"\s+")
CONTROL_RE = re.compile(r"[\u0000-\u001f\u007f-\u009f]")
SOFT_HYPHEN = "\u00ad"

_CLEANER = TextCleaner()
_NORMALIZER = TajikCyrillicNormalizer(lowercase=True)


def strip_fleurs_token_trace(text: str) -> str:
    return text.split("\t", 1)[0].strip()


def tajiknlp_normalize(text: str) -> str:
    doc = Doc(strip_fleurs_token_trace(text))
    doc = _CLEANER(doc)
    doc = _NORMALIZER(doc)
    normalized = doc.metadata["normalized_text"]
    return unicodedata.normalize("NFKC", normalized)


def tokenizer_compat_normalize(text: str) -> str:
    text = text.lower()
    text = text.replace(SOFT_HYPHEN, " ")
    text = text.replace("№", " no ")
    text = text.replace("no", " no ")
    text = CONTROL_RE.sub(" ", text)
    text = PUNCT_RE.sub(" ", text)
    text = SPACE_RE.sub(" ", text)
    return text.strip()


def normalize_text(text: str) -> str:
    return tokenizer_compat_normalize(tajiknlp_normalize(text))
