from __future__ import annotations

from typing import Literal

import pytest

from p004_training_from_scratch.canonical.common import (
    CanonicalModelConfig,
    build_canonical_ctc_model,
)


@pytest.mark.parametrize(
    "model_type",
    [
        pytest.param("tiny"),
        pytest.param("conformer_like"),
        pytest.param("conformer"),
    ],
)
def test_build_canonical_ctc_model_returns_expected_shape(
    model_type: Literal["tiny", "conformer_like", "conformer"],
) -> None:
    torch = pytest.importorskip("torch")

    model = build_canonical_ctc_model(
        torch=torch,
        input_dim=80,
        vocab_size=32,
        config=CanonicalModelConfig(
            model_type=model_type,
            hidden_dim=64,
            encoder_layers=2,
            attention_heads=4,
            conv_kernel_size=15,
            dropout=0.1,
        ),
    )
    features = torch.randn(2, 48, 80)
    input_lengths = torch.tensor([48, 36], dtype=torch.long)

    logits = model(features, input_lengths)

    assert tuple(logits.shape) == (2, 48, 32)


def test_build_canonical_ctc_model_rejects_invalid_attention_shape() -> None:
    torch = pytest.importorskip("torch")

    with pytest.raises(ValueError, match="hidden_dim must be divisible"):
        build_canonical_ctc_model(
            torch=torch,
            input_dim=80,
            vocab_size=32,
            config=CanonicalModelConfig(
                model_type="conformer_like",
                hidden_dim=62,
                encoder_layers=2,
                attention_heads=4,
                conv_kernel_size=15,
                dropout=0.1,
            ),
        )


def test_tiny_model_rejects_flex_attention_backend() -> None:
    torch = pytest.importorskip("torch")

    with pytest.raises(ValueError, match="tiny model only supports"):
        build_canonical_ctc_model(
            torch=torch,
            input_dim=80,
            vocab_size=32,
            config=CanonicalModelConfig(
                model_type="tiny",
                attention_backend="flex_triton",
                hidden_dim=64,
                encoder_layers=2,
                attention_heads=4,
                conv_kernel_size=15,
                dropout=0.1,
            ),
        )


def test_conformer_model_rejects_flex_attention_backend() -> None:
    torch = pytest.importorskip("torch")

    with pytest.raises(ValueError, match="conformer model currently only supports"):
        build_canonical_ctc_model(
            torch=torch,
            input_dim=80,
            vocab_size=32,
            config=CanonicalModelConfig(
                model_type="conformer",
                attention_backend="flex_triton",
                hidden_dim=64,
                encoder_layers=2,
                attention_heads=4,
                conv_kernel_size=15,
                dropout=0.1,
            ),
        )
