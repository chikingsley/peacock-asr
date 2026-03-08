"""NeMo CTC backend for Citrinet-style `.nemo` artifacts."""

from __future__ import annotations

import logging
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from p003_compact.settings import settings

logger = logging.getLogger(__name__)
EXPECTED_SAMPLE_RATE = 16000

if TYPE_CHECKING:
    from collections.abc import Sequence


class NemoCTCBackend:
    """Loads a NeMo CTC model and produces frame-level posteriors."""

    def __init__(
        self,
        model_ref: str,
    ) -> None:
        self._model_ref = model_ref
        self._model: Any | None = None
        self._device: torch.device = torch.device("cpu")
        self._vocab: list[str] = []
        self._phone_to_idx: dict[str, int] = {}
        self._blank_index: int = 0
        self._resolved_model_path: Path | None = None

    @property
    def name(self) -> str:
        slug = (
            self._resolved_model_path.stem
            if self._resolved_model_path
            else self._model_ref
        )
        return f"nemo-ctc ({Path(slug).name})"

    @property
    def vocab(self) -> list[str]:
        return self._vocab

    @property
    def blank_index(self) -> int:
        return self._blank_index

    def _resolve_model_path(self) -> Path:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415

        candidate = Path(self._model_ref).expanduser()
        if candidate.exists():
            return candidate.resolve()
        try:
            local_path = hf_hub_download(
                repo_id=self._model_ref,
                filename="model.nemo",
                token=settings.hf_token,
            )
        except Exception as exc:
            msg = (
                "Nemo backend expects either a local `.nemo` path or a Hugging Face "
                f"repo containing `model.nemo`; failed to resolve {self._model_ref!r}"
            )
            raise RuntimeError(msg) from exc
        return Path(local_path).resolve()

    def load(self) -> None:
        nemo_asr = import_module("nemo.collections.asr")

        model_path = self._resolve_model_path()
        logger.info("Loading NeMo CTC model: %s", model_path)
        model = nemo_asr.models.ASRModel.restore_from(
            str(model_path),
            map_location=settings.torch_device,
        )
        self._device = settings.torch_device
        model.eval()
        model = torch.nn.Module.to(model, self._device)
        self._model = model
        self._resolved_model_path = model_path

        decoder_vocab = list(model.decoder.vocabulary)
        self._blank_index = len(decoder_vocab)
        self._vocab = [*decoder_vocab, "<blank>"]
        self._phone_to_idx = {
            token: index
            for index, token in enumerate(decoder_vocab)
            if token.isalpha() and token.upper() == token
        }

        logger.info(
            "Loaded %s: %d decoder tokens + blank=%d on %s",
            model_path.name,
            len(decoder_vocab),
            self._blank_index,
            self._device,
        )

    def unload(self) -> None:
        model = self._model
        if model is not None:
            try:
                torch.nn.Module.to(model, torch.device("cpu"))
            except RuntimeError:
                logger.warning("Failed to move NeMo model to CPU during unload.")
        self._model = None
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
        if model is None:
            self.load()
            model = self._model
        if model is None:
            msg = "NeMo CTC backend failed to load before get_posteriors_batch()."
            raise RuntimeError(msg)
        if not audios:
            return []
        unique_rates = set(sample_rates)
        if len(unique_rates) != 1:
            msg = f"Batch requires a single sample rate, got {sorted(unique_rates)}"
            raise ValueError(msg)
        sample_rate = sample_rates[0]
        if sample_rate != EXPECTED_SAMPLE_RATE:
            msg = (
                "NeMo Citrinet backend expects "
                f"{EXPECTED_SAMPLE_RATE}Hz audio, got {sample_rate}."
            )
            raise ValueError(msg)

        tensors = []
        lengths = []
        for audio in audios:
            array = np.asarray(audio, dtype=np.float32)
            if array.ndim > 1:
                array = array.mean(axis=1)
            tensor = torch.from_numpy(array)
            tensors.append(tensor)
            lengths.append(tensor.shape[0])

        padded = torch.nn.utils.rnn.pad_sequence(
            tensors,
            batch_first=True,
            padding_value=0.0,
        ).to(self._device)
        input_lengths = torch.tensor(lengths, dtype=torch.long, device=self._device)

        with torch.no_grad():
            log_probs, encoded_lengths, _predictions = model.forward(
                input_signal=padded,
                input_signal_length=input_lengths,
            )
            posteriors = log_probs.exp()

        transport_dtype = self._posterior_transport_dtype()
        posteriors_np = posteriors.float().cpu().numpy(force=True)
        lengths_np = encoded_lengths.to("cpu").tolist()
        return [
            posteriors_np[index, : min(int(length), posteriors_np.shape[1]), :].astype(
                transport_dtype, copy=False
            )
            for index, length in enumerate(lengths_np)
        ]

    def get_posteriors(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        return self.get_posteriors_batch([audio], [sample_rate])[0]

    def map_phone(self, arpabet_phone: str) -> list[int] | None:
        idx = self._phone_to_idx.get(arpabet_phone)
        return [idx] if idx is not None else None
