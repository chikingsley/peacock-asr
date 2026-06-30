from __future__ import annotations

from importlib import import_module
from typing import Any


class _MissingModule:
    Module = object

    def __getattr__(self, name: str) -> Any:
        raise ModuleNotFoundError(
            "MLX is required for this operation. Run this on Apple Silicon with "
            "`mlx` and `mlx-lm` installed."
        )


try:
    _mx: Any = import_module("mlx.core")
    _nn: Any = import_module("mlx.nn")
except ModuleNotFoundError:
    _mx = _MissingModule()
    _nn = _MissingModule()

mx: Any = _mx
nn: Any = _nn


def require_mlx() -> None:
    if isinstance(mx, _MissingModule) or isinstance(nn, _MissingModule):
        raise ModuleNotFoundError(
            "MLX runtime is not available in this environment. The Linux side can "
            "build reference data and converted safetensors, but generation parity "
            "requires Apple Silicon."
        )
