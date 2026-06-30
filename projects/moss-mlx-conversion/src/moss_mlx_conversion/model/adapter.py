from __future__ import annotations

from moss_mlx_conversion.mlx_compat import mx, nn


class MossGatedMLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(input_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(input_size, hidden_size, bias=False)
        self.down_proj = nn.Linear(hidden_size, output_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))
