"""Named ASR quality thresholds — grounded in NeMo Curator's audio quality guidance.

Recorded here so a data cutoff is never argued from memory again. Every WER tier below is taken
from NeMo Curator's audio quality-assessment docs (25.09), and the right tier depends on the
*recording type* of the corpus, not a single number for everything:

- **read / broadcast speech** (scripted Common Voice, FLEURS, audiobooks): clean, so a real label
  should track a fresh ASR pass closely — tight bounds.
- **conversational / spontaneous speech** (interviews, calls, drill audio): disfluent and noisier,
  so the same Scribe WER means something looser — wider bounds.

WER here is the store-level Scribe-verification score (``Sample.scribe_wer`` from
:mod:`omni_curator.verify`): the stored label scored against a fresh Scribe transcription. A tier
is a *max* WER — keep clips at or below it. ``Selection.max_scribe_wer`` in
:mod:`omni_curator.export` is where one of these gets applied.

Sources:
- WER filtering: https://docs.nvidia.com/nemo/curator/curate-audio/process-data/quality-assessment/wer-filtering
- Audio quality metrics: https://docs.nvidia.com/nemo/curator/about/concepts/audio/quality-metrics
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WerTiers:
    """Max-WER bounds for one recording type: ``excellent`` < ``good`` < ``acceptable``."""

    excellent: float
    good: float
    acceptable: float


#: Read / broadcast / scripted speech (Common Voice scripted, FLEURS, audiobooks).
#: NeMo Curator broadcast tiers: excellent 5%, good 15%, acceptable 25%.
BROADCAST = WerTiers(excellent=0.05, good=0.15, acceptable=0.25)

#: Conversational / spontaneous speech (interviews, calls, spontaneous corpora).
#: NeMo Curator conversational tiers: excellent 15%, good 35%, acceptable 60%.
CONVERSATIONAL = WerTiers(excellent=0.15, good=0.35, acceptable=0.60)

#: Coarser fallback keyed by the language's resource level when the recording type is unknown.
#: NeMo Curator: high-resource ≤20%, medium ≤30%, low ≤50%.
RESOURCE_MAX_WER = {"high": 0.20, "medium": 0.30, "low": 0.50}

#: NeMo Curator's documented "lenient" preset — remove only clearly broken clips, keep the rest.
LENIENT_MAX_WER = 0.50
LENIENT_MIN_DURATION_S = 0.3
LENIENT_MAX_DURATION_S = 60.0
LENIENT_MIN_WORDS = 1

#: OmniASR truncates input audio at 40 s, so a longer clip trains on a label whose tail has no
#: audio — never export above this regardless of WER tier (the one hard, model-imposed bound).
OMNI_MAX_DURATION_S = 40.0

__all__ = [
    "BROADCAST",
    "CONVERSATIONAL",
    "LENIENT_MAX_DURATION_S",
    "LENIENT_MAX_WER",
    "LENIENT_MIN_DURATION_S",
    "LENIENT_MIN_WORDS",
    "OMNI_MAX_DURATION_S",
    "RESOURCE_MAX_WER",
    "WerTiers",
]
