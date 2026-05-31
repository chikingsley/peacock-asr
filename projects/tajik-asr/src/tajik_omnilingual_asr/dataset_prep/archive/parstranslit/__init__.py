"""Farsi (Perso-Arabic) -> Tajik (Cyrillic) transliteration.

Vendored from ParsTranslit (github.com/merchantrayyan/ParsTranslit, MIT,
EACL 2026 Findings) — a char-level CTranslate2 model. The model binaries live in
`ct2_fatg/` (gitignored; re-export from the upstream repo's `inference/`).

We pre-attach the separated Persian imperfective prefix می/نمی (commonly written
with a space in Farsi), which ParsTranslit otherwise mis-renders as a standalone
word `май` instead of the attached Tajik prefix `ме-`. The model/translator is
cached on first use (see `_parstranslit_upstream._get_translator`).
"""

from __future__ import annotations

import contextlib
import io
import re

from . import _parstranslit_upstream as _upstream

_MI_PREFIX = re.compile(r"(^|\s)(ن?می)\s+")


def attach_mi_prefix(text: str) -> str:
    """Join a space-separated می/نمی imperfective prefix to its verb (via ZWNJ)."""
    return _MI_PREFIX.sub("\\1\\2‌", text)


def fa_to_tajik(text: str) -> str:
    """Transliterate one Farsi (Perso-Arabic) string to Tajik Cyrillic."""
    with contextlib.redirect_stdout(io.StringIO()):  # upstream prints; suppress
        return _upstream.transliterate(attach_mi_prefix(text), "fatg")


def fa_to_tajik_batch(texts: list[str]) -> list[str]:
    return [fa_to_tajik(t) for t in texts]
