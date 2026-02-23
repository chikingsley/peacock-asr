"""Tests for the GOP-SF computation."""

from __future__ import annotations

import numpy as np
import torch

from gopt_bench.gop import GOPResult, _ctc_forward, compute_gop


class TestCTCForward:
    def test_uniform_posteriors(self):
        """Uniform posteriors should give a finite log-likelihood."""
        vocab_size = 5
        n_frames = 20
        posteriors = np.full((n_frames, vocab_size), 1.0 / vocab_size)
        params = torch.from_numpy(posteriors).double().T  # [V, T]
        seq = torch.tensor([1, 2, 3], dtype=torch.int32)
        result = _ctc_forward(params, seq, blank=0)
        assert np.isfinite(result)
        assert result > 0  # negative log-likelihood should be positive

    def test_single_phone(self):
        """Single phone sequence should work."""
        vocab_size = 4
        n_frames = 10
        posteriors = np.full((n_frames, vocab_size), 0.1)
        posteriors[:, 1] = 0.7  # phone 1 dominates
        params = torch.from_numpy(posteriors).double().T
        seq = torch.tensor([1], dtype=torch.int32)
        result = _ctc_forward(params, seq, blank=0)
        assert np.isfinite(result)


class TestComputeGOP:
    def test_empty_sequence(self):
        """Empty phone sequence returns empty result."""
        posteriors = np.ones((10, 5)) / 5
        result = compute_gop(posteriors, phone_indices=[], blank=0)
        assert isinstance(result, GOPResult)
        assert result.scores == []
        assert result.occupancies == []

    def test_basic_gop(self):
        """Basic GOP computation returns one score per phone."""
        vocab_size = 5
        n_frames = 30
        posteriors = np.full((n_frames, vocab_size), 1.0 / vocab_size)
        phone_indices = [1, 2, 3]
        result = compute_gop(posteriors, phone_indices, blank=0)
        assert len(result.scores) == 3
        assert len(result.occupancies) == 3
        assert all(np.isfinite(s) for s in result.scores)
        assert all(np.isfinite(o) for o in result.occupancies)

    def test_dominant_phone_gets_better_score(self):
        """When posteriors strongly favor the canonical phone, GOP should be higher."""
        vocab_size = 5
        n_frames = 30

        # Good pronunciation: canonical phone has high posterior
        good_posteriors = np.full((n_frames, vocab_size), 0.05)
        good_posteriors[:, 1] = 0.8
        good_result = compute_gop(good_posteriors, [1], blank=0)

        # Bad pronunciation: canonical phone has low posterior
        bad_posteriors = np.full((n_frames, vocab_size), 0.2)
        bad_posteriors[:, 1] = 0.2
        bad_result = compute_gop(bad_posteriors, [1], blank=0)

        # Higher GOP = better pronunciation (less negative ratio)
        assert good_result.scores[0] > bad_result.scores[0]
