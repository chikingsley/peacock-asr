"""Tests for scalar score variant transformations."""

from __future__ import annotations

import numpy as np
import pytest

from p001_gop.score_variants import apply_score_variant


def _example_posteriors() -> np.ndarray:
    return np.array(
        [
            [0.05, 0.80, 0.10, 0.05],
            [0.05, 0.70, 0.20, 0.05],
            [0.10, 0.10, 0.75, 0.05],
            [0.10, 0.20, 0.60, 0.10],
        ],
        dtype=np.float64,
    )


def test_gop_sf_passthrough():
    baseline = [-0.4, -0.6]
    result = apply_score_variant(
        variant="gop_sf",
        score_alpha=0.5,
        posteriors=_example_posteriors(),
        phone_indices=[1, 2],
        baseline_scores=baseline,
    )
    assert result == baseline


def test_logit_combined_interpolates_between_baseline_and_margin():
    posteriors = _example_posteriors()
    baseline = [-0.4, -0.6]

    margin = apply_score_variant(
        variant="logit_margin",
        score_alpha=0.5,
        posteriors=posteriors,
        phone_indices=[1, 2],
        baseline_scores=baseline,
    )
    alpha_zero = apply_score_variant(
        variant="logit_combined",
        score_alpha=0.0,
        posteriors=posteriors,
        phone_indices=[1, 2],
        baseline_scores=baseline,
    )
    alpha_one = apply_score_variant(
        variant="logit_combined",
        score_alpha=1.0,
        posteriors=posteriors,
        phone_indices=[1, 2],
        baseline_scores=baseline,
    )
    alpha_half = apply_score_variant(
        variant="logit_combined",
        score_alpha=0.5,
        posteriors=posteriors,
        phone_indices=[1, 2],
        baseline_scores=baseline,
    )

    np.testing.assert_allclose(alpha_zero, baseline, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(alpha_one, margin, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(
        alpha_half,
        0.5 * np.array(margin) + 0.5 * np.array(baseline),
        rtol=1e-6,
        atol=1e-8,
    )


def test_logit_combined_rejects_invalid_alpha():
    with pytest.raises(ValueError, match="score_alpha"):
        apply_score_variant(
            variant="logit_combined",
            score_alpha=1.5,
            posteriors=_example_posteriors(),
            phone_indices=[1, 2],
            baseline_scores=[-0.4, -0.6],
        )


def test_logit_margin_rejects_invalid_shape():
    with pytest.raises(ValueError, match="Expected posteriors shape"):
        apply_score_variant(
            variant="logit_margin",
            score_alpha=0.5,
            posteriors=np.array([0.1, 0.9]),
            phone_indices=[1],
            baseline_scores=[0.0],
        )


def test_logit_margin_rejects_out_of_range_phone_index():
    with pytest.raises(ValueError, match="out of range"):
        apply_score_variant(
            variant="logit_margin",
            score_alpha=0.5,
            posteriors=_example_posteriors(),
            phone_indices=[9],
            baseline_scores=[0.0],
        )
