"""benchmark: one standardized way to score ASR — jiwer WER/CER.

Every dataset/model gets the same measurement so decisions are comparable: lowercase, strip
punctuation, collapse whitespace, then corpus-level word + character error rate.
"""

from __future__ import annotations

import re


def normalize(text: str) -> str:
    """The one scoring normalization: lowercase, punctuation -> space, collapse whitespace.

    Shared by :func:`wer_cer` and the store-level Scribe verification (``omni_curator.verify``) so
    every comparison uses the same yardstick.
    """
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


#: Backwards-compatible private alias for the normalizer.
_norm = normalize


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


__all__ = ["normalize", "score_pair", "wer_cer"]
