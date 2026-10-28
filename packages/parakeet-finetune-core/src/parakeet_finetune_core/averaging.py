"""Average explicitly selected, trusted local checkpoints with strict state validation."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from parakeet_finetune_core.provenance import fingerprint

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

MIN_CHECKPOINTS = 2


def average_states(states: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Average floating tensors; preserve integer buffers from the first (best) state."""
    import torch

    sums, dtypes, count = {}, {}, 0
    for state in states:
        if count and state.keys() != sums.keys():
            raise ValueError("Checkpoint state keys differ")
        for key, value in state.items():
            if count and (value.shape != sums[key].shape or value.dtype != dtypes[key]):
                raise ValueError(f"Checkpoint shape or dtype differs: {key}")
            if value.is_floating_point():
                if not torch.isfinite(value).all():
                    raise ValueError(f"Checkpoint contains nonfinite values: {key}")
                if count:
                    sums[key].add_(value.to(torch.float64))
                else:
                    sums[key] = value.to(torch.float64).clone()
            elif count == 0:
                sums[key] = value.clone()
            dtypes[key] = value.dtype
        count += 1
    if count < MIN_CHECKPOINTS:
        raise ValueError("Checkpoint averaging requires at least two distinct checkpoints")
    return {
        key: (value / count).to(dtypes[key]) if value.is_floating_point() else value
        for key, value in sums.items()
    }


def average_checkpoints(template: Path, checkpoints: list[Path], output: Path) -> Path:
    """Export an averaged model with the template's trained tokenizer and configuration."""
    import torch
    from nemo.collections.asr.models import ASRModel

    if len({path.resolve() for path in checkpoints}) != len(checkpoints):
        raise ValueError("Checkpoint paths must be distinct")

    def states() -> Iterable[dict[str, Any]]:
        for path in checkpoints:
            # These paths refer to this workflow's own Lightning training artifacts.
            yield torch.load(path, map_location="cpu", weights_only=False)["state_dict"]

    state = average_states(states())
    model = ASRModel.restore_from(str(template), map_location="cpu")
    model.load_state_dict(state, strict=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.nemo")
    model.save_to(str(temporary))
    temporary.replace(output)
    report = {
        "template": fingerprint(template),
        "checkpoints": [fingerprint(path) for path in checkpoints],
        "output": fingerprint(output),
        "integer_buffers": "preserved from first checkpoint",
    }
    output.with_suffix(".provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    return output
