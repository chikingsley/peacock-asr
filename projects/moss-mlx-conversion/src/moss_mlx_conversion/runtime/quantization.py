from __future__ import annotations

from typing import Any

from moss_mlx_conversion.mlx_compat import nn

DEFAULT_QUANTIZATION_MODE = "affine"


def configured_quantization(config: dict[str, Any]) -> dict[str, Any] | None:
    quantization = config.get("quantization")
    if isinstance(quantization, dict):
        return quantization
    quantization_config = config.get("quantization_config")
    if isinstance(quantization_config, dict):
        return quantization_config
    return None


def applies_to_scope(path: str, scope: str) -> bool:
    if scope == "text-decoder":
        return path.startswith("model.layers.")
    if scope == "audio-adapter":
        return path.startswith("audio_adapter.")
    if scope == "audio-encoder":
        return path.startswith("audio_model.")
    if scope == "text-and-adapter":
        return path.startswith(("model.layers.", "audio_adapter."))
    if scope == "all":
        return True
    raise ValueError(f"Unsupported quantization scope: {scope}")


def make_scope_predicate(
    *,
    scope: str,
    group_size: int,
) -> Any:
    def predicate(path: str, module: Any) -> bool:
        if not hasattr(module, "to_quantized") or not hasattr(module, "weight"):
            return False
        if module.weight.shape[-1] % group_size != 0:
            return False
        return applies_to_scope(path, scope)

    return predicate


def apply_configured_quantization(
    *,
    model: Any,
    config: dict[str, Any],
    weights: dict[str, Any],
) -> None:
    quantization = configured_quantization(config)
    if quantization is None:
        return

    group_size = int(quantization.get("group_size", 64))
    bits = int(quantization["bits"])
    mode = str(quantization.get("mode", DEFAULT_QUANTIZATION_MODE))
    scope = str(quantization.get("scope", "all"))

    def class_predicate(path: str, module: Any) -> bool | dict[str, Any]:
        if not hasattr(module, "to_quantized") or not hasattr(module, "weight"):
            return False
        if module.weight.shape[-1] % group_size != 0:
            return False
        per_layer = quantization.get(path)
        if isinstance(per_layer, dict):
            return per_layer
        if f"{path}.scales" in weights:
            return applies_to_scope(path, scope)
        return False

    nn.quantize(
        model,
        group_size=group_size,
        bits=bits,
        mode=mode,
        class_predicate=class_predicate,
    )
