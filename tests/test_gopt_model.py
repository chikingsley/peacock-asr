"""Tests for the GOPT transformer model."""

from __future__ import annotations

import math

import torch

from peacock_asr.backends.ctc_gop_original import ARPABET_VOCAB
from peacock_asr.gopt_model import (
    PHONE_TO_ID,
    SEQ_LEN,
    GoptDataset,
    GOPTModel,
    UtteranceFeats,
    train_and_evaluate_gopt,
    train_and_evaluate_gopt_conpco,
)


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


class TestGOPTModelConPCO:
    """Tests for ConPCO projection heads and return_projections API."""

    def test_return_projections_tuple(self):
        """use_conpco=True + return_projections=True → 3-element tuple."""
        model = GOPTModel(input_dim=42, embed_dim=24, use_conpco=True)
        x = torch.randn(2, SEQ_LEN, 42)
        phn = torch.randint(0, 39, (2, SEQ_LEN))
        out = model(x, phn, return_projections=True)
        assert isinstance(out, tuple)
        assert len(out) == 3
        scores, audio_proj, text_proj = out
        assert scores.shape == (2, SEQ_LEN, 1)
        assert audio_proj.shape == (2, SEQ_LEN, 24)
        assert text_proj.shape == (2, SEQ_LEN, 24)

    def test_default_forward_single_tensor(self):
        """use_conpco=True but return_projections=False → single tensor."""
        model = GOPTModel(input_dim=42, embed_dim=24, use_conpco=True)
        x = torch.randn(2, SEQ_LEN, 42)
        phn = torch.randint(0, 39, (2, SEQ_LEN))
        out = model(x, phn)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (2, SEQ_LEN, 1)

    def test_no_conpco_backward_compatible(self):
        """use_conpco=False → single tensor, no projection heads."""
        model = GOPTModel(input_dim=42, embed_dim=24, use_conpco=False)
        assert not hasattr(model, "phn_audio_proj")
        assert not hasattr(model, "phn_text_proj")
        x = torch.randn(1, SEQ_LEN, 42)
        phn = torch.randint(0, 39, (1, SEQ_LEN))
        out = model(x, phn)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (1, SEQ_LEN, 1)

    def test_projection_param_count(self):
        """ConPCO adds 2 identical MLPs: each is Linear(24,24)+Linear(24,24)."""
        model_base = GOPTModel(input_dim=42, embed_dim=24, use_conpco=False)
        model_conpco = GOPTModel(input_dim=42, embed_dim=24, use_conpco=True)
        base_params = sum(p.numel() for p in model_base.parameters())
        conpco_params = sum(p.numel() for p in model_conpco.parameters())
        # Each MLP: Linear(24,24) = 24*24+24 = 600, twice = 1200 per head, 2 heads = 2400
        expected_extra = 2 * (24 * 24 + 24 + 24 * 24 + 24)  # 2 × (600+600) = 2400
        assert conpco_params - base_params == expected_extra

    def test_gradients_flow_through_projections(self):
        """Gradients should flow from projection outputs to model parameters."""
        model = GOPTModel(input_dim=42, embed_dim=24, use_conpco=True)
        x = torch.randn(2, SEQ_LEN, 42)
        phn = torch.randint(0, 39, (2, SEQ_LEN))
        scores, audio_proj, text_proj = model(x, phn, return_projections=True)
        loss = audio_proj.sum() + text_proj.sum() + scores.sum()
        loss.backward()
        assert model.phn_audio_proj[0].weight.grad is not None
        assert model.phn_text_proj[0].weight.grad is not None


class TestTrainAndEvaluateConPCO:
    """Smoke test for train_and_evaluate_gopt_conpco."""

    def _make_utts(self, n: int = 20) -> list:
        rng = torch.Generator().manual_seed(42)
        phones = list(PHONE_TO_ID.keys())[:10]
        utts = []
        for _ in range(n):
            n_ph = 8
            chosen = [phones[i % len(phones)] for i in range(n_ph)]
            score = float(torch.rand(1, generator=rng).item() * 2)
            scores = [score] * n_ph
            feat_vecs = [
                torch.randn(42, generator=rng).tolist() for _ in range(n_ph)
            ]
            utts.append(UtteranceFeats(
                phones=chosen, feat_vecs=feat_vecs, scores=scores,
            ))
        return utts

    def test_p1b_ordinal_entropy_only(self):
        """P1-B: MSE + Ordinal Entropy (no CLAP)."""
        utts = self._make_utts()
        result = train_and_evaluate_gopt_conpco(
            utts[:15], utts[15:],
            input_dim=42,
            n_epochs=3,
            batch_size=4,
            use_ordinal_entropy=True,
            use_clap=False,
        )
        assert result.n_phones > 0
        assert not math.isnan(result.mse)

    def test_p1c_full_conpco(self):
        """P1-C: MSE + OE + CLAP."""
        utts = self._make_utts()
        result = train_and_evaluate_gopt_conpco(
            utts[:15], utts[15:],
            input_dim=42,
            n_epochs=3,
            batch_size=4,
            use_ordinal_entropy=True,
            use_clap=True,
        )
        assert result.n_phones > 0
        assert not math.isnan(result.mse)

    def test_history_logged(self):
        """History should contain per-component losses."""
        utts = self._make_utts()
        history: list[dict] = []
        train_and_evaluate_gopt_conpco(
            utts[:15], utts[15:],
            input_dim=42,
            n_epochs=3,
            batch_size=4,
            use_ordinal_entropy=True,
            use_clap=True,
            history_out=history,
        )
        assert len(history) == 3
        assert "loss_mse" in history[0]
        assert "loss_oe" in history[0]
        assert "loss_clap" in history[0]
