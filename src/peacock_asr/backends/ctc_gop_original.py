"""Original CTC-GOP backend using the reference checkpoint.

Loads the wav2vec2-based CTC model from CTC-based-GOP/is24/models/checkpoint-8000.
Reference: https://github.com/YuanGongND/gopt (Gong et al.)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from peacock_asr.settings import settings

logger = logging.getLogger(__name__)


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


# The checkpoint-8000 vocab (39 ARPABET phones + pad)
ARPABET_VOCAB = [
    "<pad>",  # 0
    "AA",
    "AE",
    "AH",
    "AO",
    "AW",
    "AY",  # 1-6
    "B",
    "CH",  # 7-8
    "D",
    "DH",  # 9-10
    "EH",
    "ER",
    "EY",  # 11-13
    "F",
    "G",  # 14-15
    "HH",  # 16
    "IH",
    "IY",  # 17-18
    "JH",  # 19
    "K",
    "L",
    "M",
    "N",
    "NG",  # 20-24
    "OW",
    "OY",  # 25-26
    "P",
    "R",  # 27-28
    "S",
    "SH",  # 29-30
    "T",
    "TH",  # 31-32
    "UH",
    "UW",  # 33-34
    "V",
    "W",
    "Y",
    "Z",
    "ZH",  # 35-39
]

ARPABET_TO_IDX = {p: i for i, p in enumerate(ARPABET_VOCAB)}


class OriginalBackend:
    """Uses the CTC-based-GOP checkpoint-8000 (wav2vec2-xlsr-53 fine-tuned on
    LibriSpeech with 39 ARPABET phones)."""

    def __init__(self) -> None:
        self._model: torch.nn.Module | None = None
        self._processor: object | None = None
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
            models = repo_root / "references" / "CTC-based-GOP" / "is24" / "models"
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
                "Make sure CTC-based-GOP is cloned alongside peacock-asr."
            )
            raise FileNotFoundError(msg)

        logger.info("Loading original CTC model from %s", model_path)
        self._processor = Wav2Vec2Processor.from_pretrained(str(proc_path))
        self._model = Wav2Vec2ForCTC.from_pretrained(str(model_path))
        self._model.eval()
        self._device = settings.torch_device
        self._model = self._model.to(self._device)

    def get_posteriors(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if self._model is None or self._processor is None:
            msg = "Call load() before get_posteriors()"
            raise RuntimeError(msg)

        inputs = self._processor(audio, return_tensors="pt", sampling_rate=sample_rate)
        with torch.no_grad():
            iv = inputs.input_values.to(self._device)
            logits = self._model(iv).logits.squeeze(0)
            posteriors = logits.softmax(dim=-1)

        return posteriors.cpu().numpy(force=True).astype(np.float64)

    def map_phone(self, arpabet_phone: str) -> list[int] | None:
        idx = ARPABET_TO_IDX.get(arpabet_phone)
        return [idx] if idx is not None else None
