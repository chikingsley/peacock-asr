"""HConv and CHConv — Hierarchical Convolution interfaces for SSL layer aggregation.

Ported faithfully from:
  - HConv:  Shih & Harwath 2024 (2406.12209), code: github.com/atosystem/SSL_Interface
  - CHConv: Shih et al. 2025 (2511.08389) — multi-model concatenation variant

HConv applies 1D convolutions over the **layer dimension** of SSL hidden states,
learning to aggregate information across all transformer layers into a single vector
per time step. Key parameters: kernel=5, stride=3, depth=floor(log_3(L)).

CHConv extends HConv to multiple SSL models by concatenating their hidden states
along the feature dimension before applying HConv.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class HConv(nn.Module):
    """Hierarchical Convolution over SSL layers (Shih & Harwath 2024).

    Input:  [B, T, L, D]  — batch, time (phones), layers, hidden_dim
    Output: [B, T, D_out] — batch, time, output_dim

    The conv operates over the layer dimension L, collapsing it to 1.
    Uses kernel=5, stride=3 with floor(log_3(L)) layers, matching the
    published SSL_Interface code exactly.

    Args:
        num_layers:  Number of upstream SSL layers (e.g., 25 for Large models).
        feat_dim:    Hidden dimension per layer (e.g., 1024 for Large models).
        kernel_size: Conv1d kernel size over layer axis. Default 5.
        stride:      Conv1d stride over layer axis. Default 3.
        normalize:   Apply LayerNorm to input hidden states. Default False.
    """

    def __init__(
        self,
        num_layers: int = 25,
        feat_dim: int = 1024,
        kernel_size: int = 5,
        stride: int = 3,
        normalize: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.feat_dim = feat_dim
        self.normalize = normalize

        # Number of conv layers: floor(log_stride(L))
        n_convs = math.floor(math.log(num_layers) / math.log(stride))

        self.convs = nn.Sequential()
        L = num_layers
        for i in range(n_convs):
            padding = kernel_size // 2
            L = int(math.floor((L + 2 * padding - kernel_size) / stride + 1))

            # Last conv: reduce channels so flatten gives output_dim
            out_channels = math.ceil(feat_dim // L) if i == n_convs - 1 else feat_dim

            self.convs.append(nn.Conv1d(feat_dim, out_channels, kernel_size, stride, padding))
            self.convs.append(nn.ReLU())

            # Update feat_dim for next layer's in_channels
            if i == n_convs - 1:
                self._output_dim = out_channels * L
            feat_dim = out_channels

        self._final_L = L

    @property
    def output_dim(self) -> int:
        """Output feature dimension after flattening."""
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, T, L, D] — all-layer hidden states (phone-averaged or frame-level).

        Returns:
            [B, T, output_dim] — layer-aggregated features.
        """
        B, T, L, D = x.shape
        assert self.num_layers == L, f"Expected {self.num_layers} layers, got {L}"

        if self.normalize:
            x = F.layer_norm(x, (D,))

        # Reshape: [B, T, L, D] → [B*T, D, L] for Conv1d over layer dim
        x = x.reshape(B * T, L, D).permute(0, 2, 1)  # [B*T, D, L]

        # Apply conv stack
        x = self.convs(x)  # [B*T, D', L']

        # Flatten remaining layer and channel dims
        x = x.reshape(B, T, -1)  # [B, T, D'*L'] = [B, T, output_dim]
        return x


class CHConv(nn.Module):
    """Concatenation + Hierarchical Convolution for multi-model fusion (Shih et al. 2025).

    Takes N SSL models' all-layer hidden states, concatenates along the feature
    dimension, then applies HConv. Output is projected back to a target dimension.

    Input:  list of N tensors, each [B, T, L, D_i]
    Output: [B, T, output_dim]

    Models must have the same L and T (or be interpolated to match beforehand).

    Args:
        num_layers:   Aligned number of layers across models.
        feat_dims:    List of hidden dimensions per model (e.g., [1024, 1024, 1024]).
        output_dim:   Desired output dimension. If None, uses the HConv natural output.
        kernel_size:  Conv1d kernel size. Default 5.
        stride:       Conv1d stride. Default 3.
        normalize:    Apply LayerNorm to input. Default False.
    """

    def __init__(
        self,
        num_layers: int = 25,
        feat_dims: list[int] | None = None,
        output_dim: int | None = None,
        kernel_size: int = 5,
        stride: int = 3,
        normalize: bool = False,
    ) -> None:
        super().__init__()
        if feat_dims is None:
            feat_dims = [1024, 1024, 1024]  # wav2vec2-large + HuBERT-large + WavLM-large
        self.feat_dims = feat_dims
        self.concat_dim = sum(feat_dims)

        # HConv over the concatenated features
        self.hconv = HConv(
            num_layers=num_layers,
            feat_dim=self.concat_dim,
            kernel_size=kernel_size,
            stride=stride,
            normalize=normalize,
        )

        # Optional projection back to target dimension
        hconv_out = self.hconv.output_dim
        if output_dim is not None and output_dim != hconv_out:
            self.proj = nn.Linear(hconv_out, output_dim)
            self._output_dim = output_dim
        else:
            self.proj = None
            self._output_dim = hconv_out

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, model_features: list[torch.Tensor]) -> torch.Tensor:
        """Forward pass.

        Args:
            model_features: List of N tensors, each [B, T, L, D_i].
                            All must have same B, T, L.

        Returns:
            [B, T, output_dim]
        """
        # Concatenate along feature dimension: [B, T, L, sum(D_i)]
        x = torch.cat(model_features, dim=-1)

        # Apply HConv
        x = self.hconv(x)  # [B, T, hconv_output_dim]

        # Optional projection
        if self.proj is not None:
            x = self.proj(x)

        return x
