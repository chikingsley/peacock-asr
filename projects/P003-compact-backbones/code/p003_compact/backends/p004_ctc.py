"""Local P004 Conformer-CTC backend for GOP/GOPT evaluation.

References
- P004 canonical Conformer implementation:
  /home/simon/github/peacock-asr/projects/P004-training-from-scratch/src/p004_training_from_scratch/canonical/conformer.py
- P004 canonical log-mel frontend:
  /home/simon/github/peacock-asr/projects/P004-training-from-scratch/src/p004_training_from_scratch/canonical/common.py
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

from p003_compact.settings import settings

logger = logging.getLogger(__name__)
EXPECTED_SAMPLE_RATE = 16_000
DEFAULT_NUM_MELS = 80
_SMALL_CHANNEL_COUNT = 8

if TYPE_CHECKING:
    from collections.abc import Sequence


class P004ConformerCTCBackend:
    """Load a trained P004 canonical Conformer run from a local artifact dir."""

    def __init__(self, run_ref: str) -> None:
        self._run_ref = run_ref
        self._bundle_dir: Path | None = None
        self._report: dict[str, Any] | None = None
        self._model: Any | None = None
        self._mel_transform: Any | None = None
        self._device: torch.device = torch.device("cpu")
        self._vocab: list[str] = []
        self._phone_to_idx: dict[str, int] = {}
        self._blank_index: int = 0

    @property
    def name(self) -> str:
        if self._bundle_dir is not None:
            return f"p004-ctc ({self._bundle_dir.name})"
        return f"p004-ctc ({Path(self._run_ref).name})"

    @property
    def vocab(self) -> list[str]:
        return self._vocab

    @property
    def blank_index(self) -> int:
        return self._blank_index

    def load(self) -> None:
        import torchaudio  # noqa: PLC0415

        bundle_dir = self._resolve_bundle_dir()
        report = self._load_report(bundle_dir)
        config = cast("dict[str, Any]", report["config"])
        model_type = str(config["model_type"])
        attention_backend = str(config["attention_backend"])
        if model_type != "conformer":
            msg = f"p004 backend only supports conformer runs, got {model_type!r}"
            raise ValueError(msg)
        if attention_backend != "mha":
            msg = (
                "p004 backend only supports stable MHA conformer checkpoints, "
                f"got {attention_backend!r}"
            )
            raise ValueError(msg)

        tokens_path = self._resolve_tokens_path(bundle_dir, report)
        token_table = self._load_token_table(tokens_path)
        idx_to_phone = {idx: token for token, idx in token_table.items()}
        max_idx = max(idx_to_phone)
        self._vocab = [
            idx_to_phone.get(index, f"<unk_{index}>")
            for index in range(max_idx + 1)
        ]
        self._blank_index = token_table["<eps>"]
        self._phone_to_idx = {
            phone_symbol: idx
            for phone_symbol, idx in token_table.items()
            if phone_symbol != "<eps>"
        }

        model = _build_conformer_canonical_ctc(
            torch=torch,
            input_dim=DEFAULT_NUM_MELS,
            hidden_dim=int(config["hidden_dim"]),
            vocab_size=len(self._vocab),
            encoder_layers=int(config["encoder_layers"]),
            attention_heads=int(config["attention_heads"]),
            conv_kernel_size=int(config["conv_kernel_size"]),
            dropout=float(config["dropout"]),
        )
        state_payload = torch.load(
            self._resolve_state_path(bundle_dir, report),
            map_location="cpu",
            weights_only=False,
        )
        state_dict = _extract_state_dict(state_payload)
        model.load_state_dict(state_dict, strict=True)

        self._device = settings.torch_device
        model.eval()
        model = torch.nn.Module.to(model, self._device)
        self._model = model
        self._bundle_dir = bundle_dir
        self._report = report
        self._mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=EXPECTED_SAMPLE_RATE,
            n_fft=400,
            win_length=400,
            hop_length=160,
            n_mels=DEFAULT_NUM_MELS,
            center=True,
        ).to(self._device)
        logger.info(
            "Loaded %s: params=%s vocab=%d blank=%d device=%s",
            self.name,
            report.get("model", {}).get("parameter_count", "unknown"),
            len(self._vocab),
            self._blank_index,
            self._device,
        )

    def unload(self) -> None:
        model = self._model
        if model is not None:
            try:
                torch.nn.Module.to(model, torch.device("cpu"))
            except RuntimeError:
                logger.warning("Failed to move P004 model to CPU during unload.")
        self._model = None
        self._mel_transform = None
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
        import torchaudio  # noqa: PLC0415

        model = self._model
        mel_transform = self._mel_transform
        if model is None or mel_transform is None:
            self.load()
            model = self._model
            mel_transform = self._mel_transform
        if model is None or mel_transform is None:
            msg = "P004 backend failed to load before get_posteriors_batch()."
            raise RuntimeError(msg)
        if not audios:
            return []
        unique_rates = set(sample_rates)
        if len(unique_rates) != 1:
            msg = f"Batch requires a single sample rate, got {sorted(unique_rates)}"
            raise ValueError(msg)

        feature_rows: list[torch.Tensor] = []
        input_lengths: list[int] = []
        for audio, sample_rate in zip(audios, sample_rates, strict=True):
            features = _load_log_mel_features(
                audio=audio,
                sample_rate=sample_rate,
                torch=torch,
                torchaudio=torchaudio,
                device=self._device,
                mel_transform=mel_transform,
            )
            feature_rows.append(features)
            input_lengths.append(int(features.shape[0]))

        padded = torch.nn.utils.rnn.pad_sequence(feature_rows, batch_first=True)
        length_tensor = torch.tensor(
            input_lengths,
            dtype=torch.long,
            device=self._device,
        )
        with torch.inference_mode():
            logits = model(padded, length_tensor)
            posteriors = logits.softmax(dim=-1)

        transport_dtype = self._posterior_transport_dtype()
        posteriors_np = posteriors.float().cpu().numpy(force=True)
        return [
            posteriors_np[index, : min(length, posteriors_np.shape[1]), :].astype(
                transport_dtype,
                copy=False,
            )
            for index, length in enumerate(input_lengths)
        ]

    def get_posteriors(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        return self.get_posteriors_batch([audio], [sample_rate])[0]

    def map_phone(self, arpabet_phone: str) -> list[int] | None:
        idx = self._phone_to_idx.get(arpabet_phone)
        return [idx] if idx is not None else None

    def _resolve_bundle_dir(self) -> Path:
        candidate = Path(self._run_ref).expanduser().resolve()
        if not candidate.exists():
            msg = f"p004 run artifact path does not exist: {candidate}"
            raise FileNotFoundError(msg)
        if candidate.is_dir():
            return candidate
        return candidate.parent

    def _load_report(self, bundle_dir: Path) -> dict[str, Any]:
        report_path = bundle_dir / "report.json"
        if not report_path.is_file():
            msg = f"p004 run artifact is missing report.json: {report_path}"
            raise FileNotFoundError(msg)
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            msg = f"invalid P004 report payload: {report_path}"
            raise TypeError(msg)
        return payload

    def _resolve_tokens_path(self, bundle_dir: Path, report: dict[str, Any]) -> Path:
        candidates: list[Path] = [bundle_dir / "tokens.txt"]
        tokens_ref = report.get("tokens_path")
        if isinstance(tokens_ref, str):
            candidates.append(Path(tokens_ref).expanduser())
        candidates.extend(
            ancestor / "experiments" / "data" / "lang_phone" / "tokens.txt"
            for ancestor in bundle_dir.parents
        )

        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        msg = f"unable to resolve tokens.txt for p004 run bundle: {bundle_dir}"
        raise FileNotFoundError(msg)

    def _resolve_state_path(self, bundle_dir: Path, report: dict[str, Any]) -> Path:
        candidates = [bundle_dir / "model_state.pt"]
        artifacts = report.get("artifacts", {})
        if isinstance(artifacts, dict):
            model_state_ref = artifacts.get("model_state_path")
            if isinstance(model_state_ref, str):
                candidates.append(Path(model_state_ref).expanduser())
            checkpoint_ref = artifacts.get("checkpoint_path")
            if isinstance(checkpoint_ref, str):
                candidates.append(Path(checkpoint_ref).expanduser())
        candidates.extend([bundle_dir / "checkpoint.pt"])

        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        msg = f"unable to resolve model weights for p004 run bundle: {bundle_dir}"
        raise FileNotFoundError(msg)

    @staticmethod
    def _load_token_table(tokens_path: Path) -> dict[str, int]:
        token_table: dict[str, int] = {}
        for line in tokens_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            token, token_id = stripped.rsplit(maxsplit=1)
            token_table[token] = int(token_id)
        if "<eps>" not in token_table:
            msg = f"blank token <eps> missing from token table: {tokens_path}"
            raise ValueError(msg)
        return token_table


def _extract_state_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict) and "model_state" in payload:
        payload = payload["model_state"]
    if not isinstance(payload, dict):
        msg = "unsupported p004 checkpoint payload; expected a state dict or checkpoint"
        raise TypeError(msg)
    return {
        str(key).removeprefix("_orig_mod."): value
        for key, value in payload.items()
    }


def _load_log_mel_features(
    *,
    audio: np.ndarray,
    sample_rate: int,
    torch: Any,
    torchaudio: Any,
    device: torch.device,
    mel_transform: Any,
) -> torch.Tensor:
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim > 1:
        channel_axis = (
            0
            if array.shape[0] <= _SMALL_CHANNEL_COUNT
            and array.shape[0] < array.shape[-1]
            else -1
        )
        array = array.mean(axis=channel_axis)
    waveform = torch.from_numpy(np.ascontiguousarray(array)).unsqueeze(0).to(device)
    if sample_rate != EXPECTED_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform,
            sample_rate,
            EXPECTED_SAMPLE_RATE,
        )
    mels = mel_transform(waveform).clamp_min(1e-5).log()
    return mels.transpose(1, 2).squeeze(0).contiguous()


def _build_conformer_canonical_ctc(
    *,
    torch: Any,
    input_dim: int,
    hidden_dim: int,
    vocab_size: int,
    encoder_layers: int,
    attention_heads: int,
    conv_kernel_size: int,
    dropout: float,
) -> Any:
    class SinusoidalPositionalEncoding(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dropout = torch.nn.Dropout(dropout)

        def forward(self, features: Any) -> Any:
            seq_len = features.size(1)
            position = torch.arange(
                seq_len,
                device=features.device,
                dtype=torch.float32,
            ).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(
                    0,
                    hidden_dim,
                    2,
                    device=features.device,
                    dtype=torch.float32,
                )
                * (-math.log(10000.0) / hidden_dim)
            )
            pos_encoding = torch.zeros(
                (seq_len, hidden_dim),
                device=features.device,
                dtype=features.dtype,
            )
            pos_encoding[:, 0::2] = torch.sin(position * div_term).to(features.dtype)
            pos_encoding[:, 1::2] = torch.cos(position * div_term).to(features.dtype)
            return self.dropout(features + pos_encoding.unsqueeze(0))

    class PositionwiseFeedForward(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            ffn_dim = hidden_dim * 4
            self.linear1 = torch.nn.Linear(hidden_dim, ffn_dim)
            self.dropout1 = torch.nn.Dropout(dropout)
            self.linear2 = torch.nn.Linear(ffn_dim, hidden_dim)
            self.dropout2 = torch.nn.Dropout(dropout)

        def forward(self, features: Any) -> Any:
            features = self.linear1(features)
            features = torch.nn.functional.silu(features)
            features = self.dropout1(features)
            features = self.linear2(features)
            return self.dropout2(features)

    class ConformerConvModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.pointwise_in = torch.nn.Conv1d(hidden_dim, hidden_dim * 2, 1)
            self.depthwise = torch.nn.Conv1d(
                hidden_dim,
                hidden_dim,
                conv_kernel_size,
                padding=conv_kernel_size // 2,
                groups=hidden_dim,
            )
            self.batch_norm = torch.nn.BatchNorm1d(hidden_dim)
            self.pointwise_out = torch.nn.Conv1d(hidden_dim, hidden_dim, 1)
            self.dropout = torch.nn.Dropout(dropout)

        def forward(
            self,
            features: Any,
            *,
            key_padding_mask: Any | None,
        ) -> Any:
            if key_padding_mask is not None:
                features = features.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
            hidden = self.pointwise_in(features.transpose(1, 2))
            hidden = torch.nn.functional.glu(hidden, dim=1)
            hidden = self.depthwise(hidden)
            hidden = self.batch_norm(hidden)
            hidden = torch.nn.functional.silu(hidden)
            hidden = self.pointwise_out(hidden).transpose(1, 2)
            hidden = self.dropout(hidden)
            if key_padding_mask is not None:
                hidden = hidden.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
            return hidden

    class ConformerBlock(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.ffn1_norm = torch.nn.LayerNorm(hidden_dim)
            self.ffn1 = PositionwiseFeedForward()
            self.attn_norm = torch.nn.LayerNorm(hidden_dim)
            self.attn = torch.nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=attention_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.attn_dropout = torch.nn.Dropout(dropout)
            self.conv_norm = torch.nn.LayerNorm(hidden_dim)
            self.conv = ConformerConvModule()
            self.ffn2_norm = torch.nn.LayerNorm(hidden_dim)
            self.ffn2 = PositionwiseFeedForward()
            self.final_norm = torch.nn.LayerNorm(hidden_dim)

        def forward(
            self,
            features: Any,
            *,
            key_padding_mask: Any | None,
        ) -> Any:
            hidden = features + 0.5 * self.ffn1(self.ffn1_norm(features))
            attn_input = self.attn_norm(hidden)
            attn_output, _ = self.attn(
                attn_input,
                attn_input,
                attn_input,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            hidden = hidden + self.attn_dropout(attn_output)
            hidden = hidden + self.conv(
                self.conv_norm(hidden),
                key_padding_mask=key_padding_mask,
            )
            hidden = hidden + 0.5 * self.ffn2(self.ffn2_norm(hidden))
            hidden = self.final_norm(hidden)
            if key_padding_mask is not None:
                hidden = hidden.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
            return hidden

    class ConformerCanonicalCtc(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_proj = torch.nn.Linear(input_dim, hidden_dim)
            self.input_dropout = torch.nn.Dropout(dropout)
            self.positional_encoding = SinusoidalPositionalEncoding()
            self.blocks = torch.nn.ModuleList(
                ConformerBlock() for _ in range(encoder_layers)
            )
            self.output_norm = torch.nn.LayerNorm(hidden_dim)
            self.output = torch.nn.Linear(hidden_dim, vocab_size)

        def forward(
            self,
            features: Any,
            input_lengths: Any | None = None,
        ) -> Any:
            key_padding_mask = _build_key_padding_mask(
                torch=torch,
                input_lengths=input_lengths,
                max_length=features.size(1),
            )
            hidden = self.input_proj(features)
            hidden = self.input_dropout(hidden)
            hidden = self.positional_encoding(hidden)
            if key_padding_mask is not None:
                hidden = hidden.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
            for block in self.blocks:
                hidden = block(hidden, key_padding_mask=key_padding_mask)
            logits = self.output(self.output_norm(hidden))
            if key_padding_mask is not None:
                logits = logits.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
            return logits

    return ConformerCanonicalCtc()


def _build_key_padding_mask(
    *,
    torch: Any,
    input_lengths: Any | None,
    max_length: int,
) -> Any | None:
    if input_lengths is None:
        return None
    positions = torch.arange(max_length, device=input_lengths.device)
    return positions.unsqueeze(0) >= input_lengths.unsqueeze(1)
