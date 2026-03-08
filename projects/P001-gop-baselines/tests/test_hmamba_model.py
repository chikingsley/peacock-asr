"""Tests for the P001 HMamba scorer adaptation."""

from __future__ import annotations

import math

import torch

from p001_gop.gopt_model import PHONE_TO_ID, SEQ_LEN, UtteranceFeats
from p001_gop.hmamba_model import HMambaPhoneModel, train_and_evaluate_hmamba


class TestHMambaPhoneModelForward:
    def test_output_shape(self) -> None:
        model = HMambaPhoneModel(input_dim=42, embed_dim=24, depth=5)
        x = torch.randn(2, SEQ_LEN, 42)
        phn = torch.randint(0, 39, (2, SEQ_LEN))
        out = model(x, phn)
        assert out.shape == (2, SEQ_LEN, 1)

    def test_padding_handled(self) -> None:
        model = HMambaPhoneModel(input_dim=42, embed_dim=24, depth=5)
        x = torch.randn(1, SEQ_LEN, 42)
        phn = torch.full((1, SEQ_LEN), -1, dtype=torch.long)
        phn[0, :4] = torch.tensor([0, 1, 2, 3])
        out = model(x, phn)
        assert out.shape == (1, SEQ_LEN, 1)
        assert torch.isfinite(out).all()


class TestTrainAndEvaluateHMamba:
    def test_smoke(self) -> None:
        rng = torch.Generator().manual_seed(7)
        phones = list(PHONE_TO_ID.keys())[:10]
        utts: list[UtteranceFeats] = []
        for _ in range(20):
            n = 8
            chosen = [phones[i % len(phones)] for i in range(n)]
            feat_vecs = [
                torch.randn(42, generator=rng).tolist() for _ in range(n)
            ]
            scores = [float(torch.rand(1, generator=rng).item() * 2)] * n
            utts.append(
                UtteranceFeats(
                    phones=chosen,
                    feat_vecs=feat_vecs,
                    scores=scores,
                )
            )

        result = train_and_evaluate_hmamba(
            utts[:15],
            utts[15:],
            input_dim=42,
            n_epochs=3,
            batch_size=4,
        )

        assert result.n_phones > 0
        assert not math.isnan(result.mse)
        assert result.mse >= 0.0
