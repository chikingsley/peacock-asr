from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import torch

from p003_compact.gop import compute_gop_scalar
from p003_compact.k2_scalar import compute_scalar_terms_k2_batch
from p003_compact.settings import settings

K2_AVAILABLE = importlib.util.find_spec("k2") is not None


@pytest.mark.skipif(
    not K2_AVAILABLE,
    reason="k2 is not installed in the project environment",
)
def test_k2_scalar_backend_matches_python() -> None:
    vocab_size = 5
    n_frames = 30
    posteriors = np.full((n_frames, vocab_size), 1.0 / vocab_size)
    phone_indices = [1, 2, 3]

    original_backend = settings.ctc_scalar_backend
    original_device = settings.ctc_scalar_device
    try:
        settings.ctc_scalar_backend = "python"
        baseline = compute_gop_scalar(posteriors, phone_indices, blank=0)
        settings.ctc_scalar_backend = "k2"
        settings.ctc_scalar_device = "cpu"
        candidate = compute_gop_scalar(posteriors, phone_indices, blank=0)
    finally:
        settings.ctc_scalar_backend = original_backend
        settings.ctc_scalar_device = original_device

    np.testing.assert_allclose(
        candidate.scores,
        baseline.scores,
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        candidate.occupancies,
        baseline.occupancies,
        rtol=1e-3,
        atol=1e-3,
    )


@pytest.mark.skipif(
    not K2_AVAILABLE,
    reason="k2 is not installed in the project environment",
)
def test_k2_scalar_batch_matches_per_utterance() -> None:
    vocab_size = 5
    batch = [
        (
            np.full((24, vocab_size), 1.0 / vocab_size),
            [1, 2, 3],
        ),
        (
            np.full((18, vocab_size), 1.0 / vocab_size),
            [2, 1],
        ),
    ]

    results = compute_scalar_terms_k2_batch(
        batch,
        blank=0,
        device=torch.device("cpu"),
    )

    for (posteriors, phone_indices), (ll_self, scores, occupancies) in zip(
        batch,
        results,
        strict=True,
    ):
        original_backend = settings.ctc_scalar_backend
        original_device = settings.ctc_scalar_device
        try:
            settings.ctc_scalar_backend = "k2"
            settings.ctc_scalar_device = "cpu"
            single = compute_gop_scalar(posteriors, phone_indices, blank=0)
        finally:
            settings.ctc_scalar_backend = original_backend
            settings.ctc_scalar_device = original_device

        np.testing.assert_allclose(ll_self, single.ll_self, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(scores, single.scores, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(
            occupancies,
            single.occupancies,
            rtol=1e-3,
            atol=1e-3,
        )
