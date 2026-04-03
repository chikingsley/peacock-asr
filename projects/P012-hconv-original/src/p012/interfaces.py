from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import torch
from torch import nn

from .vendor import ensure_third_party_on_path

ensure_third_party_on_path()

from SSL_Interface.configs import (  # noqa: E402
    HierarchicalConvInterfaceConfig,
    WeightedSumInterfaceConfig,
)
from SSL_Interface.interfaces import (  # noqa: E402
    HierarchicalConvInterface,
    WeightSumInterface,
)
if TYPE_CHECKING:
    from s3prl.nn.upstream import S3PRLUpstream  # noqa: E402


InterfaceName = Literal["hconv", "weighted_sum"]


def infer_hconv_output_dim(
    upstream_layer_num: int,
    upstream_feat_dim: int,
    conv_kernel_size: int = 5,
    conv_kernel_stride: int = 3,
) -> int:
    if upstream_layer_num <= 1:
        raise ValueError("HConv requires more than one selected upstream layer.")

    reduced_layers = upstream_layer_num
    num_convs = math.floor(math.log(upstream_layer_num, conv_kernel_stride))
    if num_convs < 1:
        raise ValueError(
            "HConv requires enough layers for the configured kernel stride. "
            f"Got upstream_layer_num={upstream_layer_num}, conv_kernel_stride={conv_kernel_stride}."
        )

    padding = math.floor(conv_kernel_size / 2)
    dilation = 1
    for _ in range(num_convs):
        reduced_layers = math.floor(
            (
                reduced_layers
                + (2 * padding)
                - dilation * (conv_kernel_size - 1)
                - 1
            )
            / conv_kernel_stride
            + 1
        )

    channel_dim = math.ceil(upstream_feat_dim // reduced_layers)
    return channel_dim * reduced_layers


class InterfaceFeaturizer(nn.Module):
    def __init__(
        self,
        upstream: "S3PRLUpstream",
        interface: InterfaceName = "hconv",
        layer_selections: list[int] | None = None,
        normalize: bool = False,
        conv_kernel_size: int = 5,
        conv_kernel_stride: int = 3,
        output_dim: int | None = None,
    ) -> None:
        super().__init__()
        assert len(set(upstream.hidden_sizes)) == 1
        assert len(set(upstream.downsample_rates)) == 1

        self.interface_name = interface
        self.layer_selections = (
            sorted(layer_selections)
            if layer_selections is not None
            else list(range(upstream.num_layers))
        )
        self._hidden_size = upstream.hidden_sizes[0]
        self._downsample_rate = upstream.downsample_rates[0]

        for layer_id in self.layer_selections:
            if layer_id < 0 or layer_id >= upstream.num_layers:
                raise ValueError(
                    f"Layer selection {layer_id} is out of bounds for upstream with {upstream.num_layers} layers."
                )

        selected_layers = len(self.layer_selections)
        if interface == "hconv":
            inferred_output_dim = infer_hconv_output_dim(
                upstream_layer_num=selected_layers,
                upstream_feat_dim=self._hidden_size,
                conv_kernel_size=conv_kernel_size,
                conv_kernel_stride=conv_kernel_stride,
            )
            self._output_size = output_dim or inferred_output_dim
            config = HierarchicalConvInterfaceConfig(
                upstream_feat_dim=self._hidden_size,
                upstream_layer_num=selected_layers,
                normalize=normalize,
                conv_kernel_size=conv_kernel_size,
                conv_kernel_stride=conv_kernel_stride,
                output_dim=self._output_size,
            )
            self.interface_module = HierarchicalConvInterface(config)
        elif interface == "weighted_sum":
            self._output_size = self._hidden_size
            config = WeightedSumInterfaceConfig(
                upstream_feat_dim=self._hidden_size,
                upstream_layer_num=selected_layers,
                normalize=normalize,
            )
            self.interface_module = WeightSumInterface(config)
        else:
            raise ValueError(f"Unsupported interface: {interface}")

    @property
    def output_size(self) -> int:
        return self._output_size

    @property
    def downsample_rate(self) -> int:
        return self._downsample_rate

    def forward(
        self,
        all_hs: list[torch.FloatTensor],
        all_lens: list[torch.LongTensor],
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        selected_hs = [all_hs[index] for index in self.layer_selections]
        selected_lens = [all_lens[index] for index in self.layer_selections]

        if len(selected_hs) == 1 and self.interface_name == "weighted_sum":
            return selected_hs[0], selected_lens[0]

        for lens in selected_lens[1:]:
            if not torch.equal(selected_lens[0], lens):
                raise ValueError("Selected upstream layers must share identical valid lengths.")

        stacked_hs = torch.stack(selected_hs, dim=0)
        reduced_hs = self.interface_module(stacked_hs)
        return reduced_hs, selected_lens[0]
