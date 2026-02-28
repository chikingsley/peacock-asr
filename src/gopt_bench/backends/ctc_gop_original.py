from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from gopt_bench.settings import settings

logger = logging.getLogger(__name__)

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
        repo_root = Path(__file__).parents[3]
        models = repo_root / "references" / "CTC-based-GOP" / "is24" / "models"
        default_model = models / "checkpoint-8000"
        default_proc = models / "processor_config_gop"

        model_path = settings.ctc_gop_model_path or default_model
        proc_path = settings.ctc_gop_processor_path or default_proc
        return model_path, proc_path

    def load(self) -> None:
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor  # noqa: PLC0415

        model_path, proc_path = self._resolve_paths()

        if not model_path.exists():
            msg = (
                f"Model not found at {model_path}. "
                "Make sure CTC-based-GOP is cloned alongside gopt-bench."
            )
            raise FileNotFoundError(msg)

        logger.info("Loading original CTC model from %s", model_path)
        self._processor = Wav2Vec2Processor.from_pretrained(str(proc_path))
        self._model = Wav2Vec2ForCTC.from_pretrained(str(model_path))
        self._model.eval()

    def get_posteriors(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if self._model is None or self._processor is None:
            msg = "Call load() before get_posteriors()"
            raise RuntimeError(msg)

        inputs = self._processor(audio, return_tensors="pt", sampling_rate=sample_rate)
        with torch.no_grad():
            logits = self._model(inputs.input_values).logits.squeeze(0)
            posteriors = logits.softmax(dim=-1)

        return posteriors.numpy(force=True).astype(np.float64)

    def map_phone(self, arpabet_phone: str) -> list[int] | None:
        idx = ARPABET_TO_IDX.get(arpabet_phone)
        return [idx] if idx is not None else None
