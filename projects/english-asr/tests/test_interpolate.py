from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from english_asr.interpolate import (
    interpolate_model,
    interpolate_state,
    interpolate_state_by_component,
    parameter_component,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_interpolate_state_mutates_floats_and_reports_nonfloating_drift() -> None:
    base = {"weight": torch.tensor([0.0, 2.0]), "counter": torch.tensor(1)}
    candidate = {"weight": torch.tensor([2.0, 4.0]), "counter": torch.tensor(2)}

    differences = interpolate_state(base, candidate, 0.25)

    assert torch.equal(candidate["weight"], torch.tensor([0.5, 2.5]))
    assert differences == ["counter"]


def test_interpolate_state_rejects_invalid_alpha_and_keys() -> None:
    with pytest.raises(ValueError, match="between"):
        interpolate_state({"a": torch.tensor(0.0)}, {"a": torch.tensor(1.0)}, 1.1)
    with pytest.raises(ValueError, match="differ"):
        interpolate_state({"a": torch.tensor(0.0)}, {"b": torch.tensor(1.0)}, 0.5)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("encoder.layers.0.weight", "encoder"),
        ("model.encoder.pre_encode.weight", "encoder"),
        ("module.model.encoder.bias", "encoder"),
        ("decoder.prediction.weight", "non_encoder"),
        ("joint.joint_net.2.weight", "non_encoder"),
        ("ctc_decoder.decoder_layers.0.weight", "non_encoder"),
    ],
)
def test_parameter_component(name: str, expected: str) -> None:
    assert parameter_component(name) == expected


def test_interpolate_state_by_component_uses_independent_alphas() -> None:
    base = {
        "encoder.weight": torch.tensor([0.0, 2.0]),
        "decoder.weight": torch.tensor([0.0, 2.0]),
        "counter": torch.tensor(1),
    }
    candidate = {
        "encoder.weight": torch.tensor([2.0, 4.0]),
        "decoder.weight": torch.tensor([2.0, 4.0]),
        "counter": torch.tensor(2),
    }

    differences = interpolate_state_by_component(
        base,
        candidate,
        encoder_alpha=0.75,
        non_encoder_alpha=0.25,
    )

    assert torch.equal(candidate["encoder.weight"], torch.tensor([1.5, 3.5]))
    assert torch.equal(candidate["decoder.weight"], torch.tensor([0.5, 2.5]))
    assert differences == ["counter"]


@pytest.mark.parametrize(
    ("encoder_alpha", "non_encoder_alpha", "match"),
    [(-0.1, 0.5, "encoder_alpha"), (0.5, 1.1, "non_encoder_alpha")],
)
def test_interpolate_state_by_component_rejects_invalid_alphas(
    encoder_alpha: float, non_encoder_alpha: float, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        interpolate_state_by_component(
            {"encoder.weight": torch.tensor(0.0)},
            {"encoder.weight": torch.tensor(1.0)},
            encoder_alpha=encoder_alpha,
            non_encoder_alpha=non_encoder_alpha,
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({}, "either alpha"),
        ({"alpha": 0.5, "encoder_alpha": 0.5, "non_encoder_alpha": 0.5}, "either alpha"),
        ({"encoder_alpha": 0.5}, "set together"),
    ],
)
def test_interpolate_model_rejects_invalid_alpha_modes(
    tmp_path: Path, kwargs: dict[str, float], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        interpolate_model(
            base=tmp_path / "base.nemo",
            candidate=tmp_path / "candidate.nemo",
            output=tmp_path / "output.nemo",
            **kwargs,
        )
