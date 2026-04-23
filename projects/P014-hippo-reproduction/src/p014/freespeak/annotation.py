"""Free-speaking annotation construction (Yan et al. 2025, App. D).

Given the ASR transcription, G2P output, and reference read-aloud annotation
for an utterance, produces a :class:`~p014.data.ReadAloudAnnotation` whose
phone/word tokens come from the ASR path while inheriting human scores from
the reference via Needleman-Wunsch alignment.

Score-assignment rules (App. D):

* ``deletion`` (ref unit with no ASR counterpart) → ignored. The unit simply
  does not appear in the new annotation, so no score is emitted.
* ``substitution`` / ``match`` → the ASR unit inherits the aligned reference
  unit's human score (word acc/stress/total or phone accuracy).
* ``insertion`` (ASR unit without a ref counterpart) → score 0.
"""

from __future__ import annotations

from p014.config import FreeSpeakingAssignmentConfig
from p014.data import ReadAloudAnnotation
from p014.freespeak.align import needleman_wunsch
from p014.freespeak.asr import TranscriptionResult
from p014.freespeak.g2p import grapheme_to_phones

_ZERO_WORD_SCORE: tuple[float, float, float] = (0.0, 0.0, 0.0)


def _canonical_phones_for_word(
    annotation: ReadAloudAnnotation, word_index: int
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    phones: list[str] = []
    scores: list[float] = []
    for phone, phone_word, phone_score in zip(
        annotation.phone_tokens,
        annotation.phone_to_word,
        annotation.phone_scores,
        strict=True,
    ):
        if phone_word == word_index:
            phones.append(phone)
            scores.append(phone_score)
    return tuple(phones), tuple(scores)


def _assign_word(
    reference: ReadAloudAnnotation,
    asr_word: str,
    asr_phones: tuple[str, ...],
    ref_word_index: int | None,
) -> tuple[tuple[float, float, float], list[str], list[float]]:
    """Resolve per-word score + per-phone (tokens, scores) for a single ASR word.

    ``ref_word_index = None`` signals an inserted ASR word → all zeros.
    Otherwise we align G2P phones against the canonical phones for the
    reference word and inherit per-phone scores per App. D.
    """

    if ref_word_index is None:
        # Insertion: no aligned reference → zero scores across the board.
        phone_scores = [0.0 for _ in asr_phones]
        return _ZERO_WORD_SCORE, list(asr_phones), phone_scores

    word_score = reference.word_scores[ref_word_index]
    ref_phones, ref_phone_scores = _canonical_phones_for_word(
        reference, ref_word_index
    )

    phone_ops = needleman_wunsch(ref_phones, asr_phones)
    emitted_tokens: list[str] = []
    emitted_scores: list[float] = []
    for op in phone_ops:
        if op.op == "deletion":
            # Ref phone with no ASR counterpart — dropped per App. D.
            continue
        if op.hyp_index is None:
            # Defensive: an emitted op must carry a hyp index.
            continue
        asr_phone = asr_phones[op.hyp_index]
        emitted_tokens.append(asr_phone)
        if op.op == "insertion":
            emitted_scores.append(0.0)
        else:  # match or substitution
            ref_phone_index = op.ref_index
            if ref_phone_index is None:
                emitted_scores.append(0.0)
            else:
                emitted_scores.append(ref_phone_scores[ref_phone_index])
    return word_score, emitted_tokens, emitted_scores


def _build_single(
    reference: ReadAloudAnnotation,
    transcript: TranscriptionResult,
    asr_phones: list[tuple[str, ...]],
) -> ReadAloudAnnotation:
    ref_words = list(reference.words)
    asr_words = list(transcript.words)

    word_ops = needleman_wunsch(ref_words, asr_words)
    emitted_words: list[str] = []
    emitted_word_scores: list[tuple[float, float, float]] = []
    emitted_phone_tokens: list[str] = []
    emitted_phone_scores: list[float] = []
    emitted_phone_to_word: list[int] = []

    current_word_index = 0
    for op in word_ops:
        if op.op == "deletion":
            # Reference word not produced by the learner — no ASR unit to score.
            continue
        if op.hyp_index is None:
            continue
        asr_word = asr_words[op.hyp_index]
        ref_word_index = op.ref_index if op.op != "insertion" else None

        word_score, phones_for_word, phone_scores_for_word = _assign_word(
            reference=reference,
            asr_word=asr_word,
            asr_phones=asr_phones[op.hyp_index],
            ref_word_index=ref_word_index,
        )
        emitted_words.append(asr_word)
        emitted_word_scores.append(word_score)
        for phone_token, phone_score in zip(
            phones_for_word, phone_scores_for_word, strict=True
        ):
            emitted_phone_tokens.append(phone_token)
            emitted_phone_scores.append(phone_score)
            emitted_phone_to_word.append(current_word_index)
        current_word_index += 1

    if not emitted_words:
        # Pathological case — learner produced no parseable speech. We still
        # need a non-empty annotation so downstream collation doesn't break;
        # use a single empty-word placeholder scored at zero so the example
        # contributes no gradient signal beyond the utterance head.
        emitted_words = [""]
        emitted_word_scores = [_ZERO_WORD_SCORE]

    return ReadAloudAnnotation(
        text=transcript.text,
        words=tuple(emitted_words),
        phone_tokens=tuple(emitted_phone_tokens),
        phone_to_word=tuple(emitted_phone_to_word),
        phone_scores=tuple(emitted_phone_scores),
        word_scores=tuple(emitted_word_scores),
        utterance_scores=reference.utterance_scores,
    )


def build_freespeak_annotations(
    reference_annotations: list[ReadAloudAnnotation],
    transcripts: list[TranscriptionResult],
    assignment: FreeSpeakingAssignmentConfig,
) -> list[ReadAloudAnnotation]:
    """Convert parallel (reference, transcript) pairs into free-speaking records.

    ``assignment`` is validated against the paper-faithful rules
    (ignore/aligned-segment/zero). Any deviation raises ``ValueError`` so that
    mis-configured YAMLs fail fast.
    """

    if assignment.deletion.value != "ignore":
        raise ValueError(
            f"Paper-faithful pipeline requires deletion=ignore, got {assignment.deletion}"
        )
    if assignment.substitution.value != "aligned_segment":
        raise ValueError(
            "Paper-faithful pipeline requires substitution=aligned_segment, "
            f"got {assignment.substitution}"
        )
    if assignment.insertion.value != "zero":
        raise ValueError(
            f"Paper-faithful pipeline requires insertion=zero, got {assignment.insertion}"
        )

    if len(reference_annotations) != len(transcripts):
        raise ValueError(
            "reference_annotations and transcripts must have equal length; got "
            f"{len(reference_annotations)} vs {len(transcripts)}"
        )

    results: list[ReadAloudAnnotation] = []
    for reference, transcript in zip(reference_annotations, transcripts, strict=True):
        asr_phones_per_word = grapheme_to_phones(list(transcript.words))
        results.append(_build_single(reference, transcript, asr_phones_per_word))
    return results
