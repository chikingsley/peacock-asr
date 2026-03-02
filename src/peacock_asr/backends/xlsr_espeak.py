from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
import torch

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Mapping

MODEL_ID = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
MODEL_REVISION = "main"  # pin to specific commit after first successful run

# ARPABET to IPA mapping (CMU Dict conventions -> espeak IPA tokens)
# The xlsr-espeak model uses espeak-ng IPA symbols
ARPABET_TO_IPA: dict[str, str] = {
    "AA": "\u0251\u02d0",
    "AE": "\u00e6",
    "AH": "\u0259",
    "AO": "\u0254\u02d0",
    "AW": "a\u028a",
    "AY": "a\u026a",
    "B": "b",
    "CH": "t\u0361\u0283",
    "D": "d",
    "DH": "\u00f0",
    "EH": "\u025b",
    "ER": "\u025d\u02d0",
    "EY": "e\u026a",
    "F": "f",
    "G": "\u0261",
    "HH": "h",
    "IH": "\u026a",
    "IY": "i\u02d0",
    "JH": "d\u0361\u0292",
    "K": "k",
    "L": "l",
    "M": "m",
    "N": "n",
    "NG": "\u014b",
    "OW": "o\u028a",
    "OY": "\u0254\u026a",
    "P": "p",
    "R": "\u0279",
    "S": "s",
    "SH": "\u0283",
    "T": "t",
    "TH": "\u03b8",
    "UH": "\u028a",
    "UW": "u\u02d0",
    "V": "v",
    "W": "w",
    "Y": "j",
    "Z": "z",
    "ZH": "\u0292",
}


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


class XLSREspeakBackend:
    """wav2vec2-xlsr-53-espeak-cv-ft: 387 IPA phonemes, 20ms frames."""

    def __init__(self) -> None:
        self._model: _CTCModel | None = None
        self._processor: _AudioProcessor | None = None
        self._device: torch.device = torch.device("cpu")
        self._vocab_list: list[str] = []
        self._token_to_idx: dict[str, int] = {}

    @property
    def name(self) -> str:
        return "xlsr-espeak (wav2vec2-xlsr-53-espeak-cv-ft)"

    @property
    def vocab(self) -> list[str]:
        return self._vocab_list

    @property
    def blank_index(self) -> int:
        return self._token_to_idx.get("<pad>", 0)

    def load(self) -> None:
        from transformers import (  # noqa: PLC0415
            Wav2Vec2CTCTokenizer,
            Wav2Vec2ForCTC,
            Wav2Vec2Processor,
        )

        from peacock_asr.settings import settings  # noqa: PLC0415

        hf_cache_dir = settings.models_dir / "huggingface"
        hf_cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Loading %s (cache: %s)...", MODEL_ID, hf_cache_dir)
        processor = cast(
            "_AudioProcessor",
            Wav2Vec2Processor.from_pretrained(
                MODEL_ID, revision=MODEL_REVISION, cache_dir=str(hf_cache_dir),
            ),
        )
        model = cast(
            "_CTCModel",
            Wav2Vec2ForCTC.from_pretrained(
                MODEL_ID,
                revision=MODEL_REVISION,
                cache_dir=str(hf_cache_dir),
            ),
        )
        self._device = settings.torch_device
        model.eval()
        model.to(self._device)
        self._processor = processor
        self._model = model

        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, cache_dir=str(hf_cache_dir),
        )
        vocab = tokenizer.get_vocab()
        self._token_to_idx = vocab
        self._vocab_list = [""] * len(vocab)
        for token, idx in vocab.items():
            self._vocab_list[idx] = token

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
        ipa = ARPABET_TO_IPA.get(arpabet_phone)
        if ipa is None:
            return None
        idx = self._token_to_idx.get(ipa)
        if idx is not None:
            return [idx]
        # Try without length mark
        short = ipa.replace("\u02d0", "")
        idx = self._token_to_idx.get(short)
        return [idx] if idx is not None else None
