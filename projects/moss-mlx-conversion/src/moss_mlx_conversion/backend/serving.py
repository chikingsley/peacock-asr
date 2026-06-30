from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from moss_mlx_conversion.backend.moss_transcribe import MossTranscribeBackend, STTOutput


@dataclass(frozen=True)
class TranscriptionRequest:
    audio: str | Path | np.ndarray
    language: str = "English"
    max_new_tokens: int | None = None


class MossSerialAdapter:
    max_batch_size = 1

    def __init__(self, backend: MossTranscribeBackend) -> None:
        self.backend = backend

    def supports_batch(self, request: Any) -> bool:
        del request
        return False

    def batch_key(self, request: Any) -> tuple[str, str]:
        del request
        return ("moss-transcribe", str(self.backend.model_dir))

    def run(self, request: TranscriptionRequest) -> STTOutput:
        return self.backend.generate(
            request.audio,
            language=request.language,
            max_new_tokens=request.max_new_tokens,
        )
