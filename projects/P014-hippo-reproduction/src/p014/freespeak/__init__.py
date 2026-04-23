"""Simulated free-speaking pipeline (Yan et al. 2025 Appendix D + Section 2.4).

Public entry points:

* :func:`transcribe_split` — Whisper-large-v3 transcription of a dataset split.
* :func:`grapheme_to_phones` — g2pE-based grapheme-to-phone conversion.
* :func:`needleman_wunsch` — edit-distance alignment over arbitrary sequences.
* :func:`build_freespeak_annotations` — convert ASR/G2P outputs into
  :class:`~p014.data.ReadAloudAnnotation` records with paper-faithful score
  assignments (deletion → ignore, substitution → inherit ref score, insertion
  → zero).
"""

from __future__ import annotations

from p014.freespeak.align import AlignmentOp, needleman_wunsch
from p014.freespeak.annotation import build_freespeak_annotations
from p014.freespeak.asr import TranscriptionResult, transcribe_split
from p014.freespeak.g2p import grapheme_to_phones

__all__ = [
    "AlignmentOp",
    "TranscriptionResult",
    "build_freespeak_annotations",
    "grapheme_to_phones",
    "needleman_wunsch",
    "transcribe_split",
]
