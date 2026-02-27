"""Tests for the GOP-SF computation."""

from __future__ import annotations

import numpy as np
import torch

from gopt_bench.evaluate import evaluate_gop_feats
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


class TestFeatureExtraction:
    def test_features_shape(self):
        """Feature extraction produces [N_phones, 1 + V] matrix."""
        vocab_size = 5
        n_frames = 30
        posteriors = np.full((n_frames, vocab_size), 1.0 / vocab_size)
        phone_indices = [1, 2, 3]
        result = compute_gop(posteriors, phone_indices, blank=0, extract_features=True)
        assert result.features is not None
        assert result.features.shape == (3, 1 + vocab_size)  # 3 phones, 6 dims

    def test_features_none_by_default(self):
        """Features are None when extract_features=False."""
        posteriors = np.full((20, 5), 0.2)
        result = compute_gop(posteriors, [1, 2], blank=0)
        assert result.features is None

    def test_lpp_same_across_phones(self):
        """LPP (col 0) should be the same for all phones in an utterance."""
        vocab_size = 5
        n_frames = 30
        posteriors = np.full((n_frames, vocab_size), 1.0 / vocab_size)
        result = compute_gop(posteriors, [1, 2, 3], blank=0, extract_features=True)
        assert result.features is not None
        # Column 0 is LPP — same CTC NLL for the whole utterance
        lpp_values = result.features[:, 0]
        np.testing.assert_allclose(lpp_values[0], lpp_values[1], rtol=1e-5)
        np.testing.assert_allclose(lpp_values[0], lpp_values[2], rtol=1e-5)

    def test_self_substitution_lpr_near_zero(self):
        """LPR for substituting phone with itself should be ~0."""
        vocab_size = 5
        n_frames = 30
        posteriors = np.full((n_frames, vocab_size), 1.0 / vocab_size)
        phone_indices = [1, 2, 3]
        result = compute_gop(posteriors, phone_indices, blank=0, extract_features=True)
        assert result.features is not None
        # For phone at position 0 (index=1), LPR for token 1 should be ~0
        # because substituting with the same phone doesn't change the sequence
        lpr_self = result.features[0, 1 + phone_indices[0]]
        assert abs(lpr_self) < 1e-3

    def test_features_finite(self):
        """All feature values should be finite."""
        vocab_size = 5
        n_frames = 30
        posteriors = np.full((n_frames, vocab_size), 0.1)
        posteriors[:, 1] = 0.6  # phone 1 dominates
        result = compute_gop(posteriors, [1, 2], blank=0, extract_features=True)
        assert result.features is not None
        assert np.all(np.isfinite(result.features))


class TestEvaluateGOPFeats:
    def test_svr_evaluation(self):
        """SVR evaluation runs and returns valid PCC."""
        rng = np.random.default_rng(42)
        feat_dim = 5
        train_data = []
        test_data = []
        for phone in ["AA", "IH", "S"]:
            for _ in range(50):
                label = rng.choice([0, 1, 2])
                # Features correlated with label
                feats = (rng.standard_normal(feat_dim) + label).tolist()
                train_data.append((phone, feats, float(label)))
            for _ in range(20):
                label = rng.choice([0, 1, 2])
                feats = (rng.standard_normal(feat_dim) + label).tolist()
                test_data.append((phone, feats, float(label)))

        result = evaluate_gop_feats(train_data, test_data)
        assert np.isfinite(result.pcc)
        assert result.n_phones > 0
