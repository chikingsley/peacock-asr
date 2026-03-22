"""Original CTC-GOP backend using the reference checkpoint.

Loads the wav2vec2-based CTC model from CTC-based-GOP/is24/models/checkpoint-8000.
Reference: https://github.com/YuanGongND/gopt (Gong et al.)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
import torch

from p001_gop.phones import ARPABET_TO_IDX, ARPABET_VOCAB
from p001_gop.settings import settings

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Mapping


def _find_repo_root() -> Path | None:
    """Walk up from this file to find the project root (contains pyproject.toml)."""
    current = Path(__file__).resolve().parent
    for _ in range(10):
        if (current / "pyproject.toml").exists():
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None


class _CTCModelOutput(Protocol):
    logits: torch.Tensor


class _CTCModel(Protocol):
    def eval(self) -> None: ...

    def to(self, device: torch.device) -> _CTCModel: ...

    def __call__(self, input_values: torch.Tensor) -> _CTCModelOutput: ...


class _AudioProcessor(Protocol):
    def __call__(
        self,
        audio: np.ndarray,
        *,
        return_tensors: str,
        sampling_rate: int,
    ) -> Mapping[str, torch.Tensor]: ...


class OriginalBackend:
    """Uses the CTC-based-GOP checkpoint-8000 (wav2vec2-xlsr-53 fine-tuned on
    LibriSpeech with 39 ARPABET phones)."""

    def __init__(self) -> None:
        self._model: _CTCModel | None = None
        self._processor: _AudioProcessor | None = None
        self._device: torch.device = torch.device("cpu")

    @property
    def name(self) -> str:
        return "original (checkpoint-8000)"

    @property
    def vocab(self) -> list[str]:
        return ARPABET_VOCAB

    @property
    def blank_index(self) -> int:
        return 0

    def _resolve_paths(self) -> tuple[Path, Path]:
        model_path = settings.ctc_gop_model_path
        proc_path = settings.ctc_gop_processor_path

        if model_path is None or proc_path is None:
            repo_root = _find_repo_root()
            if repo_root is None:
                msg = (
                    "Cannot find project root. Set PEACOCK_ASR_CTC_GOP_MODEL_PATH "
                    "and PEACOCK_ASR_CTC_GOP_PROCESSOR_PATH."
                )
                raise FileNotFoundError(msg)
            models = (
                repo_root
                / "third_party"
                / "CTC-based-GOP"
                / "is24"
                / "models"
            )
            if model_path is None:
                model_path = models / "checkpoint-8000"
            if proc_path is None:
                proc_path = models / "processor_config_gop"

        return model_path, proc_path

    def load(self) -> None:
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor  # noqa: PLC0415

        model_path, proc_path = self._resolve_paths()

        if not model_path.exists():
            msg = (
                f"Model not found at {model_path}. "
                "Make sure the vendored CTC-based-GOP checkout exists under "
                "projects/P001-gop-baselines/third_party/CTC-based-GOP."
            )
            raise FileNotFoundError(msg)

        logger.info("Loading original CTC model from %s", model_path)
        processor = cast(
            "_AudioProcessor", Wav2Vec2Processor.from_pretrained(str(proc_path)),
        )
        model = cast(
            "_CTCModel", Wav2Vec2ForCTC.from_pretrained(str(model_path)),
        )
        self._device = settings.torch_device
        model.eval()
        model.to(self._device)
        self._processor = processor
        self._model = model

    def get_posteriors(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        model = self._model
        processor = self._processor
        if model is None or processor is None:
            msg = "Call load() before get_posteriors()"
            raise RuntimeError(msg)

        inputs = processor(audio, return_tensors="pt", sampling_rate=sample_rate)
        input_values = inputs.get("input_values")
        if input_values is None:
            msg = "Processor did not return input_values."
            raise RuntimeError(msg)
        with torch.no_grad():
            iv = input_values.to(self._device)
            logits = model(iv).logits.squeeze(0)
            posteriors = logits.softmax(dim=-1)

        return posteriors.cpu().numpy(force=True).astype(np.float64)

    def map_phone(self, arpabet_phone: str) -> list[int] | None:
        idx = ARPABET_TO_IDX.get(arpabet_phone)
        return [idx] if idx is not None else None
