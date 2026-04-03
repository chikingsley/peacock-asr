"""Frame-level SSL interface modules for P011 controls and HConv."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn

from p011.frame_store import FRAME_RATE_HZ
from p011.models.hconv import HConv
from p011.models.hiercb import HierCB
from p011.settings import SSLInterfaceMode
from p011.ssl_features import SSL_FEATURE_DIM, SSL_MODEL_KEYS, SSLModelKey, ssl_feature_dim

type SSLFrameMap = Mapping[SSLModelKey, torch.Tensor]
type SSLFrameLengthMap = Mapping[SSLModelKey, torch.Tensor]


def _frame_counts_from_durations(durations: torch.Tensor, total_frames: int) -> list[int]:
    """Convert one utterance's phone durations to frame counts."""
    valid = durations[:, 0] > 0
    counts = torch.round(durations[valid, 0] * FRAME_RATE_HZ).to(dtype=torch.long).tolist()
    if counts:
        delta = total_frames - sum(counts)
        counts[-1] = max(0, counts[-1] + delta)
    return counts


def _pool_frames_to_phones(
    frame_features: torch.Tensor,
    durations: torch.Tensor,
    total_frames: int,
) -> torch.Tensor:
    """Mean-pool frame-level features into 50 phone slots for one utterance."""
    out = torch.zeros(
        durations.shape[0],
        frame_features.shape[-1],
        device=frame_features.device,
        dtype=frame_features.dtype,
    )
    start = 0
    for phone_idx, frame_count in enumerate(_frame_counts_from_durations(durations, total_frames)):
        end = min(start + frame_count, total_frames)
        if end > start:
            out[phone_idx] = frame_features[start:end].mean(dim=0)
        start = end
    return out


def _pool_batch_frames_to_phones(
    frame_features: torch.Tensor,
    durations: torch.Tensor,
    frame_lengths: torch.Tensor,
) -> torch.Tensor:
    """Mean-pool one model's frame features for a full batch."""
    pooled = []
    for batch_idx in range(frame_features.shape[0]):
        pooled.append(
            _pool_frames_to_phones(
                frame_features[batch_idx],
                durations[batch_idx],
                int(frame_lengths[batch_idx].item()),
            )
        )
    return torch.stack(pooled, dim=0)


def _mean_pool_batch_frames(
    frame_features: torch.Tensor,
    frame_lengths: torch.Tensor,
) -> torch.Tensor:
    """Average-pool frame features over time for utterance-level SSL residuals."""
    pooled = []
    for batch_idx in range(frame_features.shape[0]):
        total_frames = int(frame_lengths[batch_idx].item())
        if total_frames > 0:
            pooled.append(frame_features[batch_idx, :total_frames].mean(dim=0))
        else:
            pooled.append(torch.zeros(frame_features.shape[-1], device=frame_features.device, dtype=frame_features.dtype))
    return torch.stack(pooled, dim=0)


class FrameHConvInterface(nn.Module):
    """Apply per-model HConv on frame-level all-layer SSL features."""

    def __init__(
        self,
        ssl_output_dim: int | None = None,
        ssl_model_keys: Sequence[SSLModelKey] = SSL_MODEL_KEYS,
    ) -> None:
        super().__init__()
        self.ssl_model_keys = tuple(ssl_model_keys)
        per_model_output_dim = None
        if ssl_output_dim is not None:
            num_models = len(self.ssl_model_keys)
            if ssl_output_dim % num_models != 0:
                raise ValueError(
                    f"ssl_output_dim must be divisible by number of SSL models ({num_models}), "
                    f"got {ssl_output_dim}"
                )
            per_model_output_dim = ssl_output_dim // num_models
        self.hconvs = nn.ModuleDict(
            {
                model_key: HConv(num_layers=25, feat_dim=SSL_FEATURE_DIM, output_dim=per_model_output_dim)
                for model_key in self.ssl_model_keys
            }
        )
        self._output_dim = sum(module.output_dim for module in self.hconvs.values())

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, ssl_frames: SSLFrameMap) -> dict[SSLModelKey, torch.Tensor]:
        return {key: self.hconvs[key](ssl_frames[key]) for key in self.ssl_model_keys}


class FrameLastLayerInterface(nn.Module):
    """Take the final SSL layer from the frame store, then pool to phones downstream."""

    def __init__(
        self,
        ssl_output_dim: int | None = None,
        ssl_model_keys: Sequence[SSLModelKey] = SSL_MODEL_KEYS,
    ) -> None:
        super().__init__()
        self.ssl_model_keys = tuple(ssl_model_keys)
        self._output_dim = ssl_feature_dim(self.ssl_model_keys)
        if ssl_output_dim is not None and ssl_output_dim != self._output_dim:
            raise ValueError(
                "FrameLastLayerInterface preserves the raw last-layer width; "
                f"expected ssl_output_dim={self._output_dim}, got {ssl_output_dim}"
            )

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, ssl_frames: SSLFrameMap) -> dict[SSLModelKey, torch.Tensor]:
        return {
            key: ssl_frames[key][:, :, -1, :]
            for key in self.ssl_model_keys
        }


class FrameLevelInterfaceModel(nn.Module):
    """Pronunciation model with a frame-level SSL interface followed by phone pooling."""

    def __init__(
        self,
        ssl_interface: SSLInterfaceMode,
        ssl_output_dim: int | None = None,
        ssl_model_keys: Sequence[SSLModelKey] = SSL_MODEL_KEYS,
        *,
        embed_dim: int = 24,
        num_heads: int = 1,
        p_depth: int = 3,
        w_depth: int = 2,
        u_depth: int = 1,
        ssl_drop: float = 0.2,
        use_mdd: bool = False,
    ) -> None:
        super().__init__()
        self.ssl_model_keys = tuple(ssl_model_keys)
        if ssl_interface is SSLInterfaceMode.HCONV:
            self.ssl_interface = FrameHConvInterface(
                ssl_output_dim=ssl_output_dim,
                ssl_model_keys=self.ssl_model_keys,
            )
        elif ssl_interface is SSLInterfaceMode.LAST:
            self.ssl_interface = FrameLastLayerInterface(
                ssl_output_dim=ssl_output_dim,
                ssl_model_keys=self.ssl_model_keys,
            )
        else:
            raise NotImplementedError("P011 frame-level path only supports ssl_interface=last or hconv")
        self.downstream = HierCB(
            embed_dim=embed_dim,
            num_heads=num_heads,
            p_depth=p_depth,
            w_depth=w_depth,
            u_depth=u_depth,
            ssl_drop=ssl_drop,
            ssl_dim=self.ssl_interface.output_dim,
            use_mdd=use_mdd,
        )

    def forward(
        self,
        gop: torch.Tensor,
        energy: torch.Tensor,
        dur: torch.Tensor,
        ssl_frames: SSLFrameMap,
        phn: torch.Tensor,
        word_pos: torch.Tensor,
        word: torch.Tensor,
        frame_lengths: SSLFrameLengthMap | None = None,
    ) -> tuple[torch.Tensor, ...]:
        if frame_lengths is None:
            raise ValueError("frame_lengths are required for frame-level SSL pooling")

        frame_ssl = self.ssl_interface(ssl_frames)
        pooled = [
            _pool_batch_frames_to_phones(frame_ssl[key], dur, frame_lengths[key])
            for key in self.ssl_model_keys
        ]
        phone_ssl = torch.cat(pooled, dim=-1)
        utt_ssl_residual = torch.cat(
            [_mean_pool_batch_frames(frame_ssl[key], frame_lengths[key]) for key in self.ssl_model_keys],
            dim=-1,
        )
        return self.downstream(gop, energy, dur, phone_ssl, phn, word_pos, word, utt_ssl_residual)


# Alias kept so the rest of the training stack can stay simple.
AllLayerInterfaceModel = FrameLevelInterfaceModel


__all__ = [
    "AllLayerInterfaceModel",
    "FrameLastLayerInterface",
    "FrameHConvInterface",
    "FrameLevelInterfaceModel",
]
