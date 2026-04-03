from __future__ import annotations

import torch
import torch.nn as nn
from mamba_ssm import Mamba


class BiMamba(nn.Module):
    """Bidirectional wrapper around the official Mamba block."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        *,
        average_output: bool = True,
    ) -> None:
        super().__init__()
        kwargs = {
            "d_model": d_model,
            "d_state": d_state,
            "d_conv": d_conv,
            "expand": expand,
        }
        self.forward_mixer = Mamba(**kwargs)
        self.backward_mixer = Mamba(**kwargs)
        self.average_output = average_output
        self.backend_name = "official-mamba"

    def forward(self, hidden_states: torch.Tensor, inference_params: object | None = None) -> torch.Tensor:
        if inference_params is not None:
            raise NotImplementedError("Cached inference is not wired through the bidirectional wrapper.")
        if not hidden_states.is_cuda:
            raise RuntimeError("Official mamba-ssm runtime requires CUDA. Move HMamba and inputs to a CUDA device.")

        forward_out = self.forward_mixer(hidden_states)
        backward_out = self.backward_mixer(hidden_states.flip(1)).flip(1)
        if self.average_output:
            return 0.5 * (forward_out + backward_out)
        return forward_out + backward_out
