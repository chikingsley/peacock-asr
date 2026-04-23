"""Grapheme-to-phone conversion via g2pE (Yan et al. 2025, App. D).

Policy: we keep ARPAbet stress digits on the returned phones. Downstream —
both the phone vocabulary built in :mod:`p014.data` and the CTC-GOP extractor
in :mod:`p014.features.ctc_gop` — normalise phones via ``_normalise_phone``
(lower-case + strip trailing digits) when mapping to the CTC vocab. Keeping
stress digits in the annotation layer matches the SpeechOcean762 convention
(``AO1``, ``IY0``), so the free-speaking phone tokens live in the same symbol
space as the read-aloud ones.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from importlib import import_module
from typing import Any, cast

# g2p_en emits non-phone tokens for punctuation and spaces. We drop anything
# that is not a pure alphabetic or alpha+digit ARPAbet symbol.
_ARPABET_PATTERN = re.compile(r"^[A-Z]+[0-9]*$")


def grapheme_to_phones(words: Sequence[str]) -> list[tuple[str, ...]]:
    """Convert each word to its ARPAbet phone sequence.

    Args:
        words: An iterable of word strings. Capitalisation is ignored.

    Returns:
        One tuple of phone tokens per input word. An empty tuple is returned
        for words that produce no ARPAbet output (e.g. pure punctuation or
        completely unknown tokens).
    """

    g2p_module = cast(Any, import_module("g2p_en"))
    converter = g2p_module.G2p()

    results: list[tuple[str, ...]] = []
    for word in words:
        normalised = word.strip()
        if not normalised:
            results.append(())
            continue
        phones_any = converter(normalised)
        raw_phones = cast(list[str], phones_any)
        kept = tuple(token for token in raw_phones if _ARPABET_PATTERN.match(token))
        results.append(kept)
    return results
