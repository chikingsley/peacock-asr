"""HF Hub CTC backend for Hugging Face CTC phoneme models.

Used for evaluating custom-trained phoneme models such as
`Peacockery/wav2vec2-base-phoneme-en`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

from p003_compact.settings import settings

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from transformers import PreTrainedModel


class HFCTCBackend:
    """Loads a HF CTC model from Hugging Face Hub and produces posteriors."""

    def __init__(self, repo_id: str = "Peacockery/wav2vec2-base-phoneme-en") -> None:
        self._repo_id = repo_id
        self._model: Any | None = None
        self._processor: Any | None = None
        self._device: torch.device = torch.device("cpu")
        self._vocab: list[str] = []
        self._phone_to_idx: dict[str, int] = {}
        self._blank_index: int = 0

    @property
    def name(self) -> str:
        return f"hf-ctc ({self._repo_id.split('/')[-1]})"

    @property
    def vocab(self) -> list[str]:
        return self._vocab

    @property
    def blank_index(self) -> int:
        return self._blank_index

    def load(self) -> None:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415
        from transformers import (  # noqa: PLC0415
            AutoFeatureExtractor,
            AutoModelForCTC,
        )

        logger.info("Loading HF CTC model: %s", self._repo_id)
        token = settings.hf_token

        # Load vocab from tokenizer/vocab.json
        vocab_path = hf_hub_download(
            self._repo_id,
            "tokenizer/vocab.json",
            token=token,
        )
        with Path(vocab_path).open() as f:
            vocab_dict: dict[str, int] = json.load(f)

        # Build ordered vocab list
        idx_to_phone = {idx: phone for phone, idx in vocab_dict.items()}
        max_idx = max(idx_to_phone.keys())
        self._vocab = [idx_to_phone.get(i, f"<unk_{i}>") for i in range(max_idx + 1)]

        # Blank = [PAD] token
        self._blank_index = vocab_dict.get("[PAD]", vocab_dict.get("<pad>", 0))

        # Phone mapping: ARPABET phone -> index (excluding special tokens)
        self._phone_to_idx = {
            phone: idx
            for phone, idx in vocab_dict.items()
            if not phone.startswith("[") and not phone.startswith("<")
        }

        # Load model and feature extractor
        model = cast(
            "PreTrainedModel",
            AutoModelForCTC.from_pretrained(
                self._repo_id,
                token=token,
            ),
        )
        processor = AutoFeatureExtractor.from_pretrained(
            self._repo_id,
            token=token,
        )

        self._device = settings.torch_device
        model.eval()
        model = cast("PreTrainedModel", torch.nn.Module.to(model, self._device))
        self._model = model
        self._processor = processor

        logger.info(
            "Loaded %s: %d phones, blank=%d, device=%s",
            self._repo_id, len(self._vocab), self._blank_index, self._device,
        )

    def unload(self) -> None:
        model = self._model
        if model is not None:
            try:
                torch.nn.Module.to(model, torch.device("cpu"))
            except RuntimeError:
                logger.warning("Failed to move HF CTC model to CPU during unload.")
        self._model = None
        self._processor = None
        self._device = torch.device("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _posterior_transport_dtype(self) -> type[np.float32] | type[np.float64]:
        dtype_name = settings.ctc_posterior_transport_dtype.lower()
        if dtype_name == "float32":
            return np.float32
        if dtype_name == "float64":
            return np.float64
        msg = (
            "ctc_posterior_transport_dtype must be 'float32' or 'float64', "
            f"got {settings.ctc_posterior_transport_dtype!r}"
        )
        raise ValueError(msg)

    def get_posteriors_batch(
        self,
        audios: Sequence[np.ndarray],
        sample_rates: Sequence[int],
    ) -> list[np.ndarray]:
        model = self._model
        processor = cast("Any", self._processor)
        if model is None or processor is None:
            self.load()
            model = self._model
            processor = cast("Any", self._processor)
        if model is None or processor is None:
            msg = "HF CTC backend failed to reload before get_posteriors_batch()."
            raise RuntimeError(msg)
        if not audios:
            return []
        unique_rates = set(sample_rates)
        if len(unique_rates) != 1:
            msg = f"Batch requires a single sample rate, got {sorted(unique_rates)}"
            raise ValueError(msg)

        inputs = processor(
            list(audios),
            return_tensors="pt",
            sampling_rate=sample_rates[0],
            padding=True,
            return_attention_mask=True,
        )
        model_inputs = {
            key: value.to(self._device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }
        with torch.no_grad():
            logits = model(**model_inputs).logits[:, :, :len(self._vocab)]
            posteriors = logits.softmax(dim=-1)

        attention_mask = model_inputs.get("attention_mask")
        length_fn = getattr(model, "_get_feat_extract_output_lengths", None)
        if isinstance(attention_mask, torch.Tensor) and callable(length_fn):
            input_lengths = attention_mask.sum(dim=-1)
            output_lengths = cast(
                "torch.Tensor",
                length_fn(input_lengths),
            )
            lengths = output_lengths.to("cpu").tolist()
        else:
            lengths = [posteriors.shape[1]] * len(audios)

        transport_dtype = self._posterior_transport_dtype()
        posteriors_np = posteriors.float().cpu().numpy(force=True)
        return [
            posteriors_np[index, : min(int(length), posteriors_np.shape[1]), :]
            .astype(transport_dtype, copy=False)
            for index, length in enumerate(lengths)
        ]

    def get_posteriors(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        return self.get_posteriors_batch([audio], [sample_rate])[0]

    def map_phone(self, arpabet_phone: str) -> list[int] | None:
        idx = self._phone_to_idx.get(arpabet_phone)
        return [idx] if idx is not None else None
