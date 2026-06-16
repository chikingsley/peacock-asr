from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import onnxruntime as ort
import torch

from capt.audio import load_audio_16k
from capt.recognize._vendor_zipa import ctc_greedy_decode, get_fbank_extractor, load_tokens
from capt.score.phones import split_phone_text

if TYPE_CHECKING:
    from lhotse.features.kaldi.extractors import Fbank


@dataclass(frozen=True)
class PhoneRecognitionResult:
    name: str
    model_id: str
    raw_text: str
    raw_tokens: list[str]
    normalized_tokens: list[str]
    error: str | None = None

    @classmethod
    def failed(cls, name: str, model_id: str, error: Exception | str) -> PhoneRecognitionResult:
        return cls(
            name=name,
            model_id=model_id,
            raw_text="",
            raw_tokens=[],
            normalized_tokens=[],
            error=str(error),
        )


class ZipaOnnxRecognizer:
    """Universal IPA phone recognizer (ZIPA large CR-CTC), run in-process via onnxruntime.

    The ONNX session, fbank extractor and token map are built lazily once and reused across clips.
    """

    name = "zipa"
    model_id = "anyspeech/zipa-large-crctc-ns-800k"

    def __init__(
        self,
        model_path: str | Path | None = None,
        tokens_path: str | Path | None = None,
        providers: list[str] | None = None,
    ) -> None:
        root = Path(__file__).resolve().parents[3]
        self.model_path = Path(model_path or os.getenv("ZIPA_ONNX", _default_zipa_model(root)))
        self.tokens_path = Path(
            tokens_path or os.getenv("ZIPA_TOKENS", self.model_path.parent / "tokens.txt")
        )
        # default to CPU so eval never contends with a GPU training run; override via providers=
        self.providers = providers or ["CPUExecutionProvider"]

    @cached_property
    def _session(self) -> ort.InferenceSession:
        if not self.model_path.exists():
            raise RuntimeError(f"ZIPA ONNX model not found: {self.model_path}")
        return ort.InferenceSession(str(self.model_path), providers=self.providers)

    @cached_property
    def _vocab(self) -> dict[int, str]:
        if not self.tokens_path.exists():
            raise RuntimeError(f"ZIPA tokens file not found: {self.tokens_path}")
        return load_tokens(self.tokens_path)

    @cached_property
    def _extractor(self) -> Fbank:
        return get_fbank_extractor()

    def recognize(self, audio_path: str | Path) -> PhoneRecognitionResult:
        audio = load_audio_16k(audio_path)
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
        feature = self._extractor.extract_batch([audio_tensor], sampling_rate=16000)[0].unsqueeze(0)
        feat_lens = np.array([feature.shape[1]], dtype=np.int64)
        log_probs = self._session.run(None, {"x": feature.numpy(), "x_lens": feat_lens})[0][0]
        phones = ctc_greedy_decode(log_probs, self._vocab)
        raw = " ".join(phones)
        return PhoneRecognitionResult(
            name=self.name,
            model_id=self.model_id,
            raw_text=raw,
            raw_tokens=raw.split(),
            normalized_tokens=split_phone_text(raw),
        )


def safe_recognize(
    recognizer: ZipaOnnxRecognizer,
    audio_path: str | Path,
) -> PhoneRecognitionResult:
    try:
        return recognizer.recognize(audio_path)
    except Exception as exc:  # noqa: BLE001 - intentional: surface any recognizer failure as a failed result
        return PhoneRecognitionResult.failed(recognizer.name, recognizer.model_id, exc)


def _default_zipa_model(root: Path) -> str:
    artifact_dir = root / "artifacts" / "zipa-large-crctc-ns-800k"
    fp32 = artifact_dir / "model.onnx"
    if fp32.exists():
        return str(fp32)
    return str(artifact_dir / "model.fp16.onnx")
