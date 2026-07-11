"""Named ASR quality thresholds — grounded in NeMo Curator's audio quality guidance.

Recorded here so a data cutoff is never argued from memory again. Every WER tier below is taken
from NeMo Curator's audio quality-assessment docs (25.09), and the right tier depends on the
*recording type* of the corpus, not a single number for everything:

- **read / broadcast speech** (scripted Common Voice, FLEURS, audiobooks): clean, so a real label
  should track a fresh ASR pass closely — tight bounds.
- **conversational / spontaneous speech** (interviews, calls, drill audio): disfluent and noisier,
  so the same Scribe WER means something looser — wider bounds.

WER here is the store-level Scribe-verification score (``Sample.scribe_wer`` from
:mod:`audit.verify`): the stored label scored against a fresh Scribe transcription. A tier
is a *max* WER — keep clips at or below it. ``Selection.max_scribe_wer`` in
:mod:`omni_curator.data.export` is where one of these gets applied.

Sources:
- WER filtering: https://docs.nvidia.com/nemo/curator/curate-audio/process-data/quality-assessment/wer-filtering
- Audio quality metrics: https://docs.nvidia.com/nemo/curator/about/concepts/audio/quality-metrics
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from omni_curator.audit.benchmark import normalize


@dataclass(frozen=True)
class WerTiers:
    """Max-WER bounds for one recording type: ``excellent`` < ``good`` < ``acceptable``."""

    excellent: float
    good: float
    acceptable: float


@dataclass(frozen=True)
class ASREdgeMismatch:
    """Character mismatch at the two transcript edges for one ASR/reference pair.

    This is the inspectable signal behind NeMo Speech Data Processor's
    ``DropASRErrorBeginningEnd`` idea. Insertions and deletions use the mismatching fragment's
    character length; substitutions use the absolute difference between the reference and ASR
    fragment lengths. A middle-only mismatch therefore reports zero at both edges.
    """

    beginning_operation: str | None
    beginning_error_chars: int
    beginning_reference: str
    beginning_hypothesis: str
    end_operation: str | None
    end_error_chars: int
    end_reference: str
    end_hypothesis: str


def _opcode_error_chars(
    opcode: tuple[str, int, int, int, int], reference: str, hypothesis: str
) -> tuple[str | None, int, str, str]:
    tag, ref_start, ref_end, hyp_start, hyp_end = opcode
    ref_fragment = reference[ref_start:ref_end]
    hyp_fragment = hypothesis[hyp_start:hyp_end]
    if tag == "equal":
        return None, 0, "", ""
    if tag == "delete":
        error_chars = len(ref_fragment)
    elif tag == "insert":
        error_chars = len(hyp_fragment)
    else:
        error_chars = abs(len(ref_fragment) - len(hyp_fragment))
    return tag, error_chars, ref_fragment, hyp_fragment


def asr_edge_mismatch(reference: str, hypothesis: str) -> ASREdgeMismatch:
    """Measure ASR/reference disagreement specifically at the beginning and end.

    Both strings first use the shared benchmark normalization so punctuation and casing cannot
    create a boundary alarm by themselves. The function only scores; choosing a rejection
    threshold remains an explicit corpus pilot decision.
    """
    normalized_reference = normalize(reference)
    normalized_hypothesis = normalize(hypothesis)
    opcodes = SequenceMatcher(
        None, normalized_reference, normalized_hypothesis, autojunk=False
    ).get_opcodes()
    beginning = _opcode_error_chars(opcodes[0], normalized_reference, normalized_hypothesis)
    end = _opcode_error_chars(opcodes[-1], normalized_reference, normalized_hypothesis)
    return ASREdgeMismatch(
        beginning_operation=beginning[0],
        beginning_error_chars=beginning[1],
        beginning_reference=beginning[2],
        beginning_hypothesis=beginning[3],
        end_operation=end[0],
        end_error_chars=end[1],
        end_reference=end[2],
        end_hypothesis=end[3],
    )


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

#: Bracketed/parenthesized descriptor segments; what remains after removing them (and every
#: non-letter) decides whether a label has any lexical content at all.
_DESCRIPTOR_SEGMENTS = re.compile(r"\[[^\]]*\]|\([^)]*\)")
_NON_LETTERS = re.compile(r"[\W\d_]+")


def is_descriptor_only(text: str) -> bool:
    """Return True when a label carries NO lexical content — only descriptors/symbols.

    Scribe labels non-speech with descriptors ('[outro jingle]', '[музыка]', '♪', '...'); a
    fresh Scribe pass produces the same descriptor, so such a label scores WER ~0 and a WER
    gate KEEPS it (~23k clips / 28 h of the tajik pool). There is nothing to train on —
    :class:`omni_curator.data.export.Selection` drops these at export.
    """
    without_segments = _DESCRIPTOR_SEGMENTS.sub(" ", text)
    return not _NON_LETTERS.sub("", without_segments)


__all__ = [
    "BROADCAST",
    "CONVERSATIONAL",
    "LENIENT_MAX_DURATION_S",
    "LENIENT_MAX_WER",
    "LENIENT_MIN_DURATION_S",
    "LENIENT_MIN_WORDS",
    "OMNI_MAX_DURATION_S",
    "RESOURCE_MAX_WER",
    "ASREdgeMismatch",
    "WerTiers",
    "asr_edge_mismatch",
    "is_descriptor_only",
]
