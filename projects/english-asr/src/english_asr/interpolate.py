"""Interpolate a Parakeet candidate back toward its exact base weights."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any


def _validate_alpha(alpha: float, *, name: str = "alpha") -> None:
    if not 0 <= alpha <= 1:
        raise ValueError(f"{name} must be between zero and one")


def parameter_component(name: str) -> str:
    """Return the interpolation component for one ASR state-dictionary key."""
    parts = name.split(".")
    while parts and parts[0] in {"model", "module"}:
        parts.pop(0)
    return "encoder" if parts and parts[0] == "encoder" else "non_encoder"


def interpolate_state(
    base_state: dict[str, Any], candidate_state: dict[str, Any], alpha: float
) -> list[str]:
    """Mutate floating candidate tensors to ``alpha*candidate + (1-alpha)*base``."""
    _validate_alpha(alpha)
    if candidate_state.keys() != base_state.keys():
        raise ValueError("base and candidate state dictionaries differ")

    import torch  # noqa: PLC0415

    nonfloating_differences: list[str] = []
    with torch.no_grad():
        for name, value in candidate_state.items():
            reference = base_state[name]
            if value.is_floating_point():
                value.mul_(alpha).add_(reference, alpha=1.0 - alpha)
            elif not torch.equal(value, reference):
                nonfloating_differences.append(name)
    return nonfloating_differences


def interpolate_state_by_component(
    base_state: dict[str, Any],
    candidate_state: dict[str, Any],
    *,
    encoder_alpha: float,
    non_encoder_alpha: float,
) -> list[str]:
    """Interpolate encoder and decoder-side tensors with independent candidate weights."""
    _validate_alpha(encoder_alpha, name="encoder_alpha")
    _validate_alpha(non_encoder_alpha, name="non_encoder_alpha")
    if candidate_state.keys() != base_state.keys():
        raise ValueError("base and candidate state dictionaries differ")

    import torch  # noqa: PLC0415

    nonfloating_differences: list[str] = []
    with torch.no_grad():
        for name, value in candidate_state.items():
            reference = base_state[name]
            if value.is_floating_point():
                alpha = (
                    encoder_alpha if parameter_component(name) == "encoder" else non_encoder_alpha
                )
                value.mul_(alpha).add_(reference, alpha=1.0 - alpha)
            elif not torch.equal(value, reference):
                nonfloating_differences.append(name)
    return nonfloating_differences


def interpolate_model(  # noqa: PLR0913
    *,
    base: Path,
    candidate: Path,
    output: Path,
    alpha: float | None = None,
    encoder_alpha: float | None = None,
    non_encoder_alpha: float | None = None,
) -> dict[str, Any]:
    """Restore two exact-architecture NeMo models on CPU and save one interpolated model."""
    if output.exists():
        raise FileExistsError(f"immutable interpolation output already exists: {output}")

    uses_global_alpha = alpha is not None
    uses_component_alphas = encoder_alpha is not None or non_encoder_alpha is not None
    if uses_global_alpha == uses_component_alphas:
        raise ValueError("set either alpha or both encoder_alpha and non_encoder_alpha")
    if uses_component_alphas and (encoder_alpha is None or non_encoder_alpha is None):
        raise ValueError("encoder_alpha and non_encoder_alpha must be set together")

    from nemo.collections.asr.models import ASRModel  # noqa: PLC0415

    base_model = ASRModel.restore_from(str(base), map_location="cpu")
    candidate_model = ASRModel.restore_from(str(candidate), map_location="cpu")
    if alpha is not None:
        differences = interpolate_state(
            base_model.state_dict(), candidate_model.state_dict(), alpha
        )
        interpolation = {"alpha": alpha}
    elif encoder_alpha is not None and non_encoder_alpha is not None:
        differences = interpolate_state_by_component(
            base_model.state_dict(),
            candidate_model.state_dict(),
            encoder_alpha=encoder_alpha,
            non_encoder_alpha=non_encoder_alpha,
        )
        interpolation = {
            "encoder_alpha": encoder_alpha,
            "non_encoder_alpha": non_encoder_alpha,
        }
    else:  # pragma: no cover - guarded before model restoration
        raise RuntimeError("unreachable interpolation mode")
    output.parent.mkdir(parents=True, exist_ok=True)
    candidate_model.save_to(str(output))
    return {
        "base": str(base),
        "candidate": str(candidate),
        "output": str(output),
        **interpolation,
        "bytes": output.stat().st_size,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "nonfloating_candidate_values": differences,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--encoder-alpha", type=float)
    parser.add_argument("--non-encoder-alpha", type=float)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = interpolate_model(
        base=args.base.expanduser().resolve(),
        candidate=args.candidate.expanduser().resolve(),
        output=args.output.expanduser().resolve(),
        alpha=args.alpha,
        encoder_alpha=args.encoder_alpha,
        non_encoder_alpha=args.non_encoder_alpha,
    )
    print(result, flush=True)
    return 0
