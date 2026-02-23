"""Tests for the evaluation module."""

from __future__ import annotations

import math

import numpy as np

from gopt_bench.evaluate import EvalResult, balanced_sampling, evaluate_gop


class TestBalancedSampling:
    def test_equal_classes(self):
        """Equal-sized classes should remain unchanged in count."""
        features = np.array([[1], [2], [3], [4]])
        labels = np.array([0, 0, 1, 1])
        _bf, bl = balanced_sampling(features, labels)
        assert (bl == 0).sum() == (bl == 1).sum()

    def test_imbalanced_upsamples_minority(self):
        """Minority class should be upsampled to match majority."""
        features = np.arange(10).reshape(-1, 1)
        labels = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
        _bf, bl = balanced_sampling(features, labels)
        assert (bl == 0).sum() == (bl == 1).sum()


class TestEvaluateGOP:
    def test_perfect_correlation(self):
        """When GOP perfectly predicts scores, PCC should be high."""
        train_data = []
        test_data = []
        rng = np.random.default_rng(42)

        for _i in range(200):
            label = rng.choice([0, 1, 2])
            gop = label * 1.0 + rng.normal(0, 0.1)
            train_data.append(("AA", gop, float(label)))

        for _i in range(100):
            label = rng.choice([0, 1, 2])
            gop = label * 1.0 + rng.normal(0, 0.1)
            test_data.append(("AA", gop, float(label)))

        result = evaluate_gop(train_data, test_data)
        assert isinstance(result, EvalResult)
        assert result.pcc > 0.5  # should be decent given the signal

    def test_empty_data_returns_nan(self):
        """Empty test data should return NaN PCC, not crash."""
        result = evaluate_gop([], [])
        assert isinstance(result, EvalResult)
        assert math.isnan(result.pcc)
        assert result.n_phones == 0

    def test_single_sample_returns_nan(self):
        """Single sample should return NaN PCC."""
        train = [("AA", 1.0, 1.0)]
        test = [("AA", 1.0, 1.0)]
        result = evaluate_gop(train, test)
        assert isinstance(result, EvalResult)
        assert math.isnan(result.pcc) or result.n_phones <= 1

    def test_unknown_phone_in_test_skipped(self):
        """Phones in test but not train should be skipped."""
        train = [("AA", float(i), float(i % 3)) for i in range(50)]
        test = [("ZZ", 1.0, 1.0)]  # never seen in train
        result = evaluate_gop(train, test)
        assert result.n_phones == 0
