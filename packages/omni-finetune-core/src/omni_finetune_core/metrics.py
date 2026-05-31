"""Language-agnostic ASR scoring via jiwer.

Callers pass already-normalized reference/hypothesis strings (normalization is
language-specific and lives in the per-language projects). One ``jiwer.process_words``
pass yields WER/MER/WIL plus the substitution/deletion/insertion/hit breakdown — the
S/D/I split is the useful diagnostic for omni CTC, where a word-boundary merge shows up
as a deletion + substitution while CER barely moves.
"""

from __future__ import annotations

import jiwer
from pydantic import BaseModel, ConfigDict


class Measures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    wer: float
    cer: float
    mer: float
    wil: float
    substitutions: int
    deletions: int
    insertions: int
    hits: int

    @property
    def ref_words(self) -> int:
        return self.substitutions + self.deletions + self.hits


def compute_measures(references: list[str], hypotheses: list[str]) -> Measures:
    """Corpus-level metrics (percentages) over aligned reference/hypothesis lists."""
    if len(references) != len(hypotheses):
        msg = f"references ({len(references)}) and hypotheses ({len(hypotheses)}) differ in length"
        raise ValueError(msg)
    if not references:
        return Measures(
            wer=0.0, cer=0.0, mer=0.0, wil=0.0,
            substitutions=0, deletions=0, insertions=0, hits=0,
        )
    words = jiwer.process_words(references, hypotheses)
    chars = jiwer.process_characters(references, hypotheses)
    return Measures(
        wer=100.0 * words.wer,
        cer=100.0 * chars.cer,
        mer=100.0 * words.mer,
        wil=100.0 * words.wil,
        substitutions=words.substitutions,
        deletions=words.deletions,
        insertions=words.insertions,
        hits=words.hits,
    )
