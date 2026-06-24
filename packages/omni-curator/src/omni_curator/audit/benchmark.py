"""benchmark: one standardized way to score ASR — jiwer WER/CER.

Every dataset/model gets the same measurement so decisions are comparable: lowercase, strip
punctuation, collapse whitespace, then corpus-level word + character error rate.
"""

from __future__ import annotations

import re


def normalize(text: str) -> str:
    """The one scoring normalization: lowercase, punctuation -> space, collapse whitespace.

    Shared by :func:`wer_cer` and the store-level Scribe verification (``audit.verify``) so
    every comparison uses the same yardstick.
    """
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def wer_cer(reference: list[str], hypothesis: list[str]) -> tuple[float, float]:
    """Corpus WER and CER over aligned reference/hypothesis lists (normalized)."""
    import jiwer

    refs = [normalize(r) for r in reference]
    hyps = [normalize(h) for h in hypothesis]
    return (
        jiwer.process_words(refs, hyps).wer,
        jiwer.process_characters(refs, hyps).cer,
    )


def score_pair(reference: str, hypothesis: str) -> dict[str, object]:
    """Score one (reference, hypothesis) pair: WER, CER, and the full jiwer S/D/I/H breakdown.

    Both texts are run through :func:`normalize` first (the same yardstick as :func:`wer_cer`).
    Returns ``wer``/``cer`` plus per-edit counts from ``jiwer.process_words`` (word-level S/D/I/H)
    and ``jiwer.process_characters`` (character-level S/D/I/H), so a caller can persist the whole
    alignment, not just the headline rate.
    """
    import jiwer

    ref, hyp = normalize(reference), normalize(hypothesis)
    words = jiwer.process_words([ref], [hyp])
    chars = jiwer.process_characters([ref], [hyp])
    return {
        "wer": words.wer,
        "cer": chars.cer,
        "words": {
            "substitutions": words.substitutions,
            "deletions": words.deletions,
            "insertions": words.insertions,
            "hits": words.hits,
        },
        "chars": {
            "substitutions": chars.substitutions,
            "deletions": chars.deletions,
            "insertions": chars.insertions,
            "hits": chars.hits,
        },
    }


#: Unicode block per script code (FLORES-style codes, e.g. the ``Cyrl`` in ``tgk_Cyrl``).
_SCRIPT_RANGES: dict[str, tuple[str, str]] = {
    "Cyrl": ("\u0400", "\u04ff"),
    "Arab": ("\u0600", "\u06ff"),
    "Latn": ("a", "z"),  # counted over the lowercased text
    "Geor": ("\u10a0", "\u10ff"),
}


def dominant_script(text: str) -> str | None:
    """Return the script code most of ``text``'s letters belong to (``None``: nothing matched).

    Used by the Scribe verification to detect a hypothesis rendered in a different script than
    the stored label (e.g. Scribe writing Tajik speech in Perso-Arabic) — comparing across
    scripts makes WER meaningless, so such a hypothesis is transliterated before scoring.
    """
    lowered = text.lower()
    counts = {
        code: sum(lo <= ch <= hi for ch in lowered) for code, (lo, hi) in _SCRIPT_RANGES.items()
    }
    code, n = max(counts.items(), key=lambda kv: kv[1])
    return code if n > 0 else None


__all__ = ["dominant_script", "normalize", "score_pair", "wer_cer"]
