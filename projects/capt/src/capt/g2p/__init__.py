"""G2P target lane: text -> canonical IPA (per-language routed backends + ZIPA-distilled gaps)."""

from __future__ import annotations

from capt.g2p.routing import G2PResult, TargetG2P
from capt.g2p.text_normalization import WrittenTextNormalization, normalize_written_text

__all__ = ["G2PResult", "TargetG2P", "WrittenTextNormalization", "normalize_written_text"]
