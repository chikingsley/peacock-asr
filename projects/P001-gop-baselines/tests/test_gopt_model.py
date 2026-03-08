"""Tests for the P001 GOPT transformer model."""

from __future__ import annotations

import math

import torch

from p001_gop.gopt_model import (
    PHONE_TO_ID,
    SEQ_LEN,
    GoptDataset,
    GOPTModel,
    UtteranceFeats,
    train_and_evaluate_gopt,
)
from p001_gop.phones import ARPABET_VOCAB


class TestPhoneToID:
    def test_all_arpabet_phones_mapped(self):
        """All 39 ARPABET phones should have a mapping."""
        assert len(PHONE_TO_ID) == 39
        for phone in ARPABET_VOCAB[1:]:
            assert phone in PHONE_TO_ID

    def test_ids_are_contiguous(self):
        """IDs should be 0..38."""
        ids = sorted(PHONE_TO_ID.values())
        assert ids == list(range(39))


class TestGOPTModelForward:
    def test_output_shape(self):
        """Model should produce [B, SEQ_LEN, 1] output."""
        model = GOPTModel(input_dim=42, embed_dim=24)
        x = torch.randn(2, SEQ_LEN, 42)
        phn = torch.randint(0, 39, (2, SEQ_LEN))
        out = model(x, phn)
        assert out.shape == (2, SEQ_LEN, 1)

    def test_padding_handled(self):
        """Phone ID -1 (padding) should not crash forward."""
        model = GOPTModel(input_dim=42, embed_dim=24)
        x = torch.randn(1, SEQ_LEN, 42)
        phn = torch.full((1, SEQ_LEN), -1, dtype=torch.long)
        phn[0, :5] = torch.tensor([0, 1, 2, 3, 4])
        out = model(x, phn)
        assert out.shape == (1, SEQ_LEN, 1)
        assert torch.isfinite(out).all()

    def test_different_input_dim(self):
        """Model should work with different feature dimensions."""
        for dim in [10, 42, 84]:
            model = GOPTModel(input_dim=dim, embed_dim=16)
            x = torch.randn(1, SEQ_LEN, dim)
            phn = torch.randint(0, 39, (1, SEQ_LEN))
            out = model(x, phn)
            assert out.shape == (1, SEQ_LEN, 1)


class TestGoptDataset:
    def _make_utt(
        self, n_phones: int = 5, input_dim: int = 42,
    ) -> UtteranceFeats:
        phones = list(PHONE_TO_ID.keys())[:n_phones]
        feat_vecs = [[float(i)] * input_dim for i in range(n_phones)]
        scores = [1.0] * n_phones
        return UtteranceFeats(
            phones=phones, feat_vecs=feat_vecs, scores=scores,
        )

    def test_padding_to_seq_len(self):
        """Short utterances should be zero-padded to SEQ_LEN."""
        utt = self._make_utt(n_phones=3)
        ds = GoptDataset([utt], norm_mean=0.0, norm_std=1.0, input_dim=42)
        feats, phn_ids, scores = ds[0]
        assert feats.shape == (SEQ_LEN, 42)
        assert phn_ids.shape == (SEQ_LEN,)
        assert scores.shape == (SEQ_LEN,)
        # First 3 positions should have real phone IDs
        assert (phn_ids[:3] >= 0).all()
        # Rest should be padding (-1)
        assert (phn_ids[3:] == -1).all()
        # Padded scores should be -1
        assert (scores[3:] == -1.0).all()

    def test_normalization(self):
        """Features should be z-score normalized."""
        utt = self._make_utt(n_phones=2)
        # All feature values are 0.0 and 1.0 (from _make_utt)
        ds = GoptDataset(
            [utt], norm_mean=0.5, norm_std=0.5, input_dim=42,
        )
        feats, _, _ = ds[0]
        # Phone 0: all 0.0 → (0-0.5)/0.5 = -1.0
        assert torch.allclose(feats[0], torch.full((42,), -1.0))
        # Phone 1: all 1.0 → (1-0.5)/0.5 = 1.0
        assert torch.allclose(feats[1], torch.full((42,), 1.0))

    def test_len(self):
        """Dataset length should match number of utterances."""
        utts = [self._make_utt() for _ in range(7)]
        ds = GoptDataset(utts, norm_mean=0.0, norm_std=1.0, input_dim=42)
        assert len(ds) == 7


class TestTrainAndEvaluate:
    def test_smoke(self):
        """Train on synthetic data, verify valid EvalResult."""
        rng = torch.Generator().manual_seed(42)
        phones = list(PHONE_TO_ID.keys())[:10]
        utts = []
        for _ in range(20):
            n = 8
            chosen = [phones[i % len(phones)] for i in range(n)]
            scores = [float(torch.rand(1, generator=rng).item() * 2)]
            scores = scores * n
            feat_vecs = [
                torch.randn(42, generator=rng).tolist() for _ in range(n)
            ]
            utts.append(UtteranceFeats(
                phones=chosen, feat_vecs=feat_vecs, scores=scores,
            ))

        train_utts = utts[:15]
        test_utts = utts[15:]

        result = train_and_evaluate_gopt(
            train_utts, test_utts,
            input_dim=42,
            n_epochs=5,
            batch_size=4,
        )

        assert result.n_phones > 0
        assert not math.isnan(result.mse)
        assert result.mse >= 0

    def test_empty_test_returns_nan(self):
        """Empty test set should return NaN PCC, not crash."""
        phones = list(PHONE_TO_ID.keys())[:5]
        train_utts = [
            UtteranceFeats(
                phones=phones,
                feat_vecs=[[0.0] * 42] * 5,
                scores=[1.0] * 5,
            ),
        ]

        result = train_and_evaluate_gopt(
            train_utts, [],
            input_dim=42,
            n_epochs=2,
            batch_size=1,
        )

        assert math.isnan(result.pcc)
        assert result.n_phones == 0
