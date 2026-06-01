from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from p016_compare.alignment import (
    AlignmentOp,
    alignment_rows,
    needleman_wunsch,
    summarize,
)
from p016_compare.feature_metrics import (
    alignment_feature_distance,
    feature_edit_summary,
)
from p016_compare.g2p import G2PResult, TargetG2P
from p016_compare.recognizers import (
    PhoneRecognitionResult,
    ZipaOnnxRecognizer,
    safe_recognize,
)

if TYPE_CHECKING:
    from pathlib import Path

RecognizerName = Literal["zipa"]


@dataclass(frozen=True)
class LaneConfig:
    name: str
    recognizer: RecognizerName
    target_backend: str
    languages: tuple[str, ...] = ()

    def applies_to(self, language: str) -> bool:
        return not self.languages or any(language.startswith(prefix) for prefix in self.languages)


@dataclass(frozen=True)
class LaneResult:
    name: str
    recognition: PhoneRecognitionResult
    target: G2PResult
    sentence: dict[str, int | float]
    words: list[dict[str, str | int | float]]
    alignment: list[dict[str, str | int]]
    timing: dict[str, float | bool]


DEFAULT_LANE_CONFIGS = (LaneConfig("zipa", "zipa", "routed"),)


class PronunciationComparePipeline:
    def __init__(
        self,
        lane_configs: tuple[LaneConfig, ...] = DEFAULT_LANE_CONFIGS,
    ) -> None:
        self.lane_configs = lane_configs
        self.zipa = ZipaOnnxRecognizer()

    def recognize(
        self,
        audio_path: str | Path,
    ) -> dict[RecognizerName, PhoneRecognitionResult]:
        """Run each distinct recognizer once on the audio (independent of any target text)."""
        recognitions: dict[RecognizerName, PhoneRecognitionResult] = {}
        for config in self.lane_configs:
            if config.recognizer not in recognitions:
                recognitions[config.recognizer] = safe_recognize(
                    _recognizer_for(config.recognizer, self), audio_path
                )
        return recognitions

    def score_text(
        self,
        language: str,
        target_text: str,
        recognitions: dict[RecognizerName, PhoneRecognitionResult],
    ) -> list[LaneResult]:
        """Score one target text against precomputed recognitions, per applicable lane.

        Target phones come from G2P of ``target_text`` — the known reference (read-aloud) or
        the ASR hypothesis (free-form). Recognitions are reused across both modes because they
        depend only on the audio, so a recognizer runs once per sample, not once per mode.
        """
        lanes: list[LaneResult] = []
        for config in self.lane_configs:
            if not config.applies_to(language):
                continue
            target = TargetG2P(config.target_backend).from_text(target_text, language=language)
            lanes.append(_score_lane(config.name, target, recognitions[config.recognizer]))
        return lanes


def _recognizer_for(
    _name: RecognizerName,
    pipeline: PronunciationComparePipeline,
) -> ZipaOnnxRecognizer:
    return pipeline.zipa


def _score_lane(
    name: str,
    target: G2PResult,
    recognition: PhoneRecognitionResult,
) -> LaneResult:
    if recognition.error:
        return LaneResult(
            name=name,
            recognition=recognition,
            target=target,
            sentence={
                "PER": 1.0,
                "errors": len(target.flat_normalized),
                "reference_count": len(target.flat_normalized),
            },
            words=[],
            alignment=[],
            timing={},
        )

    ops = needleman_wunsch(target.flat_normalized, recognition.normalized_tokens)
    sentence = summarize(ops, len(target.flat_normalized)).as_dict()
    sentence.update(
        feature_edit_summary(target.flat_normalized, recognition.normalized_tokens).as_dict()
    )
    word_buckets = _ops_by_word(ops, target.word_spans)
    words: list[dict[str, str | int | float]] = []
    for word, raw_phones, phones, bucket in zip(
        target.words,
        target.phones_per_word_raw,
        target.phones_per_word_normalized,
        word_buckets,
        strict=True,
    ):
        recognized = _recognized_tokens_for_ops(recognition.normalized_tokens, bucket)
        word_summary = summarize(bucket, len(phones))
        feature_distance = alignment_feature_distance(
            target.flat_normalized,
            recognition.normalized_tokens,
            bucket,
        )
        pfer = feature_distance / len(phones) if phones else 0.0
        row = {
            "word": word,
            "target_phones": " ".join(phones),
            "target_phones_raw": " ".join(raw_phones),
            "recognized_phones": " ".join(recognized),
            "substitutions_detail": _substitution_detail(
                target.flat_normalized,
                recognition.normalized_tokens,
                bucket,
            ),
            "deletions_detail": _deletion_detail(target.flat_normalized, bucket),
            "insertions_detail": _insertion_detail(recognition.normalized_tokens, bucket),
            "PFER": round(pfer, 4),
            "feature_distance": round(feature_distance, 4),
        }
        row.update(word_summary.as_dict())
        words.append(row)

    return LaneResult(
        name=name,
        recognition=recognition,
        target=target,
        sentence=sentence,
        words=words,
        alignment=alignment_rows(target.flat_normalized, recognition.normalized_tokens, ops),
        timing={},
    )


def _ops_by_word(
    ops: list[AlignmentOp],
    word_spans: list[tuple[int, int]],
) -> list[list[AlignmentOp]]:
    buckets: list[list[AlignmentOp]] = [[] for _ in word_spans]
    last_ref_index = 0
    for op in ops:
        if op.ref_index is not None:
            last_ref_index = op.ref_index
            word_index = _word_for_ref_index(word_spans, op.ref_index)
        else:
            word_index = _word_for_ref_index(word_spans, last_ref_index)
        if word_index is not None:
            buckets[word_index].append(op)
    return buckets


def _word_for_ref_index(
    word_spans: list[tuple[int, int]],
    ref_index: int,
) -> int | None:
    if not word_spans:
        return None
    for index, (start, end) in enumerate(word_spans):
        if start <= ref_index < end:
            return index
    if ref_index < word_spans[0][0]:
        return 0
    return len(word_spans) - 1


def _recognized_tokens_for_ops(
    hypothesis: list[str],
    ops: list[AlignmentOp],
) -> list[str]:
    return [hypothesis[op.hyp_index] for op in ops if op.hyp_index is not None]


def _substitution_detail(
    reference: list[str],
    hypothesis: list[str],
    ops: list[AlignmentOp],
) -> str:
    parts = [
        f"{reference[op.ref_index]}->{hypothesis[op.hyp_index]}"
        for op in ops
        if op.op == "substitution" and op.ref_index is not None and op.hyp_index is not None
    ]
    return ", ".join(parts)


def _deletion_detail(reference: list[str], ops: list[AlignmentOp]) -> str:
    return ", ".join(
        reference[op.ref_index]
        for op in ops
        if op.op == "deletion" and op.ref_index is not None
    )


def _insertion_detail(hypothesis: list[str], ops: list[AlignmentOp]) -> str:
    return ", ".join(
        hypothesis[op.hyp_index]
        for op in ops
        if op.op == "insertion" and op.hyp_index is not None
    )
