from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal

from p016_compare.alignment import (
    AlignmentOp,
    alignment_rows,
    needleman_wunsch,
    summarize,
)
from p016_compare.asr import AsrResult, QwenAsrTranscriber
from p016_compare.audio import audio_duration_seconds
from p016_compare.feature_metrics import (
    alignment_feature_distance,
    feature_edit_summary,
)
from p016_compare.g2p import G2PResult, TargetG2P
from p016_compare.recognizers import (
    PhoneRecognitionResult,
    XlsrEspeakRecognizer,
    ZipaOnnxRecognizer,
    safe_recognize,
)

RecognizerName = Literal["zipa", "xlsr-espeak"]


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


@dataclass(frozen=True)
class PipelineResult:
    asr: AsrResult
    lanes: list[LaneResult]
    timing: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "asr": {
                "text": self.asr.text,
                "language": self.asr.language,
                "model_id": self.asr.model_id,
            },
            "targets": {
                lane.name: _target_payload(lane.target) for lane in self.lanes
            },
            "lanes": [
                {
                    "name": lane.name,
                    "recognizer": lane.recognition.name,
                    "model_id": lane.recognition.model_id,
                    "error": lane.recognition.error,
                    "target": _target_payload(lane.target),
                    "raw_text": lane.recognition.raw_text,
                    "raw_tokens": lane.recognition.raw_tokens,
                    "normalized_tokens": lane.recognition.normalized_tokens,
                    "sentence": lane.sentence,
                    "words": lane.words,
                    "alignment": lane.alignment,
                    "timing": lane.timing,
                }
                for lane in self.lanes
            ],
            "timing": self.timing,
        }


DEFAULT_LANE_CONFIGS = (
    LaneConfig("zipa", "zipa", "mfa"),
    LaneConfig("xlsr-espeak", "xlsr-espeak", "espeak"),
)
DIAGNOSTIC_LANE_CONFIGS = (
    LaneConfig("zipa-charsiu", "zipa", "charsiu", languages=("ru",)),
    LaneConfig("xlsr-mfa", "xlsr-espeak", "mfa", languages=("ru",)),
)


class PronunciationComparePipeline:
    def __init__(
        self,
        lane_configs: tuple[LaneConfig, ...] = DEFAULT_LANE_CONFIGS,
    ) -> None:
        self.asr = QwenAsrTranscriber()
        self.lane_configs = lane_configs
        self.zipa = ZipaOnnxRecognizer()
        self.xlsr = XlsrEspeakRecognizer()

    def analyze(
        self,
        audio_path: str | Path,
        language: str,
    ) -> PipelineResult:
        pipeline_start = perf_counter()
        audio_seconds = audio_duration_seconds(audio_path)
        asr_start = perf_counter()
        asr = self.asr.transcribe(str(audio_path), language=language)
        asr_seconds = _elapsed(asr_start)
        recognitions: dict[RecognizerName, PhoneRecognitionResult] = {}
        recognition_seconds: dict[RecognizerName, float] = {}
        lanes: list[LaneResult] = []
        for config in self.lane_configs:
            if not config.applies_to(language):
                continue
            lane_start = perf_counter()
            target_start = perf_counter()
            target = TargetG2P(config.target_backend).from_text(asr.text, language=language)
            target_seconds = _elapsed(target_start)
            recognition = recognitions.get(config.recognizer)
            recognizer_cached = recognition is not None
            recognizer_seconds = 0.0
            if recognition is None:
                recognizer_start = perf_counter()
                recognition = safe_recognize(_recognizer_for(config.recognizer, self), audio_path)
                recognizer_seconds = _elapsed(recognizer_start)
                recognitions[config.recognizer] = recognition
                recognition_seconds[config.recognizer] = recognizer_seconds
            score_start = perf_counter()
            lane = _score_lane(config.name, target, recognition)
            score_seconds = _elapsed(score_start)
            lanes.append(
                _with_lane_timing(
                    lane,
                    {
                        "target_g2p_seconds": target_seconds,
                        "recognizer_seconds": recognizer_seconds,
                        "recognizer_cached": recognizer_cached,
                        "score_seconds": score_seconds,
                        "total_seconds": _elapsed(lane_start),
                    },
                )
            )
        total_seconds = _elapsed(pipeline_start)
        return PipelineResult(
            asr=asr,
            lanes=lanes,
            timing={
                "audio_seconds": audio_seconds,
                "asr_seconds": asr_seconds,
                "recognizer_seconds": dict(recognition_seconds),
                "lane_seconds": {lane.name: lane.timing for lane in lanes},
                "total_seconds": total_seconds,
                "rtf": _rtf(total_seconds, audio_seconds),
            },
        )


def _target_payload(target: G2PResult) -> dict[str, object]:
    return {
        "backend": target.backend,
        "warnings": target.warnings,
        "input_text": target.input_text,
        "normalized_text": target.normalized_text,
        "text_normalization_backend": target.text_normalization_backend,
        "text_normalization_warnings": target.text_normalization_warnings,
        "words": target.words,
        "phones_raw": target.phones_per_word_raw,
        "phones_normalized": target.phones_per_word_normalized,
    }


def _recognizer_for(
    name: RecognizerName,
    pipeline: PronunciationComparePipeline,
) -> ZipaOnnxRecognizer | XlsrEspeakRecognizer:
    if name == "zipa":
        return pipeline.zipa
    return pipeline.xlsr


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


def _with_lane_timing(
    lane: LaneResult,
    timing: dict[str, float | bool],
) -> LaneResult:
    return LaneResult(
        name=lane.name,
        recognition=lane.recognition,
        target=lane.target,
        sentence=lane.sentence,
        words=lane.words,
        alignment=lane.alignment,
        timing=timing,
    )


def _elapsed(start: float) -> float:
    return round(perf_counter() - start, 6)


def _rtf(total_seconds: float, audio_seconds: float) -> float | None:
    if audio_seconds <= 0:
        return None
    return round(total_seconds / audio_seconds, 6)


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
    parts = []
    for op in ops:
        if op.op == "substitution" and op.ref_index is not None and op.hyp_index is not None:
            parts.append(f"{reference[op.ref_index]}->{hypothesis[op.hyp_index]}")
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
