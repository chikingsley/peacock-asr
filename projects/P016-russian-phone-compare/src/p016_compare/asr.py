from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AsrResult:
    text: str
    language: str
    model_id: str


class QwenAsrTranscriber:
    def __init__(self, model_id: str = "Qwen/Qwen3-ASR-1.7B") -> None:
        self.model_id = model_id
        self._model: Any | None = None

    def transcribe(self, audio_path: str, language: str | None = None) -> AsrResult:
        model = self._load_model()
        result = model.transcribe(audio=audio_path, language=_qwen_language(language))[0]
        return AsrResult(
            text=str(getattr(result, "text", "")).strip(),
            language=str(getattr(result, "language", language or "")),
            model_id=self.model_id,
        )

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            import torch
            from qwen_asr import Qwen3ASRModel
        except ImportError as exc:
            raise RuntimeError(
                "Qwen ASR is not installed. Run `uv sync` in this project, then retry."
            ) from exc

        cuda = torch.cuda.is_available()
        self._model = Qwen3ASRModel.from_pretrained(
            self.model_id,
            dtype=torch.bfloat16 if cuda else torch.float32,
            device_map="cuda:0" if cuda else "cpu",
            max_inference_batch_size=1,
            max_new_tokens=256,
        )
        return self._model


def _qwen_language(language: str | None) -> str | None:
    if language is None:
        return None
    normalized = language.lower()
    if normalized.startswith("ru"):
        return "Russian"
    if normalized.startswith("en"):
        return "English"
    return None
