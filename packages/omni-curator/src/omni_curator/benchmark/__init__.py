"""benchmark: one standardized way to score ASR — jiwer WER/CER.

Every dataset/model gets the same measurement so decisions are comparable: lowercase, strip
punctuation, collapse whitespace, then corpus-level word + character error rate.
"""

from __future__ import annotations

import re


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def wer_cer(reference: list[str], hypothesis: list[str]) -> tuple[float, float]:
    """Corpus WER and CER over aligned reference/hypothesis lists (normalized)."""
    import jiwer

    refs = [_norm(r) for r in reference]
    hyps = [_norm(h) for h in hypothesis]
    return (
        jiwer.process_words(refs, hyps).wer,
        jiwer.process_characters(refs, hyps).cer,
    )


__all__ = ["wer_cer"]
