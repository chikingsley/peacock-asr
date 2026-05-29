from pathlib import Path

import p016_compare.pipeline as pipeline_module
from p016_compare.asr import AsrResult
from p016_compare.g2p import G2PResult
from p016_compare.pipeline import (
    DEFAULT_LANE_CONFIGS,
    DIAGNOSTIC_LANE_CONFIGS,
    PronunciationComparePipeline,
)
from p016_compare.recognizers import PhoneRecognitionResult


class FakeAsr:
    def __init__(self, text: str) -> None:
        self.text = text

    def transcribe(self, audio_path: str, language: str | None = None) -> AsrResult:
        return AsrResult(text=self.text, language=language or "", model_id="fake-asr")


class FakeRecognizer:
    def __init__(self, name: str) -> None:
        self.name = name
        self.model_id = f"fake-{name}"
        self.calls = 0

    def recognize(self, audio_path: str | Path) -> PhoneRecognitionResult:
        self.calls += 1
        return PhoneRecognitionResult(
            name=self.name,
            model_id=self.model_id,
            raw_text="p",
            raw_tokens=["p"],
            normalized_tokens=["p"],
        )


class FakeTargetG2P:
    def __init__(self, preferred_backend: str) -> None:
        self.preferred_backend = preferred_backend

    def from_text(self, text: str, language: str) -> G2PResult:
        return G2PResult(
            words=[text],
            phones_per_word_raw=[["p"]],
            phones_per_word_normalized=[["p"]],
            backend=self.preferred_backend,
            warnings=[],
        )


def test_pipeline_can_add_diagnostic_lanes_without_rerunning_recognizers(monkeypatch) -> None:
    monkeypatch.setattr(pipeline_module, "TargetG2P", FakeTargetG2P)
    monkeypatch.setattr(pipeline_module, "audio_duration_seconds", lambda audio_path: 1.0)
    pipeline = PronunciationComparePipeline(DEFAULT_LANE_CONFIGS + DIAGNOSTIC_LANE_CONFIGS)
    pipeline.asr = FakeAsr("привет")
    pipeline.zipa = FakeRecognizer("zipa")
    pipeline.xlsr = FakeRecognizer("xlsr-espeak")

    result = pipeline.analyze("fake.wav", "ru")

    assert [lane.name for lane in result.lanes] == [
        "zipa",
        "xlsr-espeak",
        "zipa-charsiu",
        "xlsr-mfa",
    ]
    assert pipeline.zipa.calls == 1
    assert pipeline.xlsr.calls == 1
    assert set(result.as_dict()["targets"]) == {"zipa", "xlsr-espeak", "zipa-charsiu", "xlsr-mfa"}
    timing = result.as_dict()["timing"]
    assert timing["audio_seconds"] == 1.0
    assert timing["rtf"] is not None
    assert result.as_dict()["lanes"][2]["timing"]["recognizer_cached"] is True


def test_mfa_lanes_are_russian_only(monkeypatch) -> None:
    monkeypatch.setattr(pipeline_module, "TargetG2P", FakeTargetG2P)
    monkeypatch.setattr(pipeline_module, "audio_duration_seconds", lambda audio_path: 1.0)
    pipeline = PronunciationComparePipeline(DEFAULT_LANE_CONFIGS + DIAGNOSTIC_LANE_CONFIGS)
    pipeline.asr = FakeAsr("hello")
    pipeline.zipa = FakeRecognizer("zipa")
    pipeline.xlsr = FakeRecognizer("xlsr-espeak")

    result = pipeline.analyze("fake.wav", "en_us")

    assert [lane.name for lane in result.lanes] == ["zipa", "xlsr-espeak"]
