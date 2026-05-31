"""Tokenizer coverage gate: count ``<unk>`` over a corpus for a char tokenizer.

The omni char tokenizer maps any out-of-vocab character to ``<unk>`` (which surfaces as
the ``⁇`` glyph). Before training/export, every text must encode with zero unknowns —
this is the gate that catches contamination (ZWNJ, emoji, stray scripts). The caller
loads its own fairseq2 ``Tokenizer`` (via its registered card); core just counts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from collections.abc import Iterable

    from fairseq2.data.tokenizers import Tokenizer


class AuditReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: int
    unk_rows: int
    unk_tokens: int

    @property
    def clean(self) -> bool:
        return self.unk_rows == 0


def audit_texts(texts: Iterable[str], tokenizer: Tokenizer) -> AuditReport:
    """Encode each text and tally rows/tokens that produced ``<unk>``."""
    unk_idx = tokenizer.vocab_info.unk_idx
    if unk_idx is None:
        msg = "tokenizer has no <unk> symbol; cannot audit coverage"
        raise ValueError(msg)
    encode = tokenizer.create_raw_encoder()
    rows = unk_rows = unk_tokens = 0
    for text in texts:
        rows += 1
        n = sum(1 for token_id in encode(text).tolist() if token_id == unk_idx)
        if n:
            unk_rows += 1
            unk_tokens += n
    return AuditReport(rows=rows, unk_rows=unk_rows, unk_tokens=unk_tokens)
