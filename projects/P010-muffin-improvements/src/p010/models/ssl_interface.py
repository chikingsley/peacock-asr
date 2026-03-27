"""Phone-level SSL interface modules for Shih-style HConv/CHConv.

Operates on pre-extracted phone-level all-layer SSL features from the
``*_all_layers.npy`` files (shape ``[utterances, phones, layers, dim]``).
HConv/CHConv aggregate across the layer dimension directly at phone level.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn

from p010.models.hconv import CHConv, HConv
from p010.models.hiercb import HierCB
from p010.settings import SSLInterfaceMode
from p010.ssl_features import SSL_FEATURE_DIM, SSL_MODEL_KEYS, SSLModelKey

type SSLFrameMap = Mapping[SSLModelKey, torch.Tensor]
type SSLFrameLengthMap = Mapping[SSLModelKey, torch.Tensor]


class PhoneHConvInterface(nn.Module):
    """Apply per-model HConv on phone-level all-layer SSL features."""

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

    def forward(self, ssl_layers: SSLFrameMap) -> torch.Tensor:
        return torch.cat(
            [self.hconvs[key](ssl_layers[key]) for key in self.ssl_model_keys],
            dim=-1,
        )


class PhoneCHConvInterface(nn.Module):
    """Apply CHConv on phone-level all-layer SSL features."""

    def __init__(
        self,
        ssl_output_dim: int | None = None,
        ssl_model_keys: Sequence[SSLModelKey] = SSL_MODEL_KEYS,
    ) -> None:
        super().__init__()
        self.ssl_model_keys = tuple(ssl_model_keys)
        self.chconv = CHConv(
            num_layers=25,
            feat_dims=[SSL_FEATURE_DIM for _ in self.ssl_model_keys],
            output_dim=ssl_output_dim,
        )

    @property
    def output_dim(self) -> int:
        return self.chconv.output_dim

    def forward(self, ssl_layers: SSLFrameMap) -> torch.Tensor:
        return self.chconv([ssl_layers[key] for key in self.ssl_model_keys])


class AllLayerInterfaceModel(nn.Module):
    """Pronunciation model with phone-level HConv/CHConv over all-layer SSL."""

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
        selected_keys = tuple(ssl_model_keys)
        if ssl_interface is SSLInterfaceMode.HCONV:
            self.ssl_interface: PhoneHConvInterface | PhoneCHConvInterface = PhoneHConvInterface(
                ssl_output_dim=ssl_output_dim,
                ssl_model_keys=selected_keys,
            )
        elif ssl_interface is SSLInterfaceMode.CHCONV:
            self.ssl_interface = PhoneCHConvInterface(
                ssl_output_dim=ssl_output_dim,
                ssl_model_keys=selected_keys,
            )
        else:
            raise ValueError(f"Unsupported interface mode for AllLayerInterfaceModel: {ssl_interface}")

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
        phone_ssl = self.ssl_interface(ssl_frames)
        return self.downstream(gop, energy, dur, phone_ssl, phn, word_pos, word)
