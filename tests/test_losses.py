"""Tests for ConPCO loss components."""

from __future__ import annotations

import torch

from peacock_asr.losses import CLAPContrastiveLoss, OrdinalEntropyLoss


class TestOrdinalEntropyLoss:
    """Tests for the diversity + tightness ordinal entropy loss."""

    def _make_batch(
        self, n_phones: int = 3, seq_len: int = 10, embed_dim: int = 8,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create a batch with n_phones unique phones, some score 2.0."""
        batch_size = 2
        features = torch.randn(batch_size, seq_len, embed_dim, requires_grad=True)
        phn_ids = torch.randint(0, n_phones, (batch_size, seq_len))
        scores = torch.full((batch_size, seq_len), 1.6)
        # Ensure at least some tokens per phone have score 2.0
        for p in range(n_phones):
            mask = phn_ids == p
            if mask.any():
                idx = mask.nonzero()[0]
                scores[idx[0], idx[1]] = 2.0
        return features, scores, phn_ids

    def test_output_is_scalar(self):
        loss_fn = OrdinalEntropyLoss()
        features, scores, phn_ids = self._make_batch()
        loss = loss_fn(features, scores, phn_ids)
        assert loss.ndim == 0

    def test_gradient_flows(self):
        loss_fn = OrdinalEntropyLoss()
        features, scores, phn_ids = self._make_batch()
        loss = loss_fn(features, scores, phn_ids)
        loss.backward()
        assert features.grad is not None
        assert features.grad.abs().sum() > 0

    def test_padding_excluded(self):
        """Tokens with phn_id == -1 should not affect the loss."""
        loss_fn = OrdinalEntropyLoss()
        features, scores, phn_ids = self._make_batch()

        loss_no_pad = loss_fn(features, scores, phn_ids)

        # Add padding tokens — should not change loss
        features_padded = torch.cat(
            [features, torch.randn(2, 5, 8)], dim=1,
        ).detach().requires_grad_(True)
        scores_padded = torch.cat(
            [scores, torch.full((2, 5), -1.0)], dim=1,
        )
        phn_ids_padded = torch.cat(
            [phn_ids, torch.full((2, 5), -1, dtype=torch.long)], dim=1,
        )
        loss_with_pad = loss_fn(features_padded, scores_padded, phn_ids_padded)

        assert torch.allclose(loss_no_pad, loss_with_pad, atol=1e-5)

    def test_all_padding_returns_zero(self):
        """All-padding batch should return 0 loss."""
        loss_fn = OrdinalEntropyLoss()
        features = torch.randn(2, 10, 8, requires_grad=True)
        scores = torch.full((2, 10), -1.0)
        phn_ids = torch.full((2, 10), -1, dtype=torch.long)
        loss = loss_fn(features, scores, phn_ids)
        assert loss.item() == 0.0
        # Should still be differentiable
        loss.backward()

    def test_single_phoneme_returns_zero(self):
        """Only 1 unique phoneme → no diversity pairs → return 0."""
        loss_fn = OrdinalEntropyLoss()
        features = torch.randn(2, 10, 8, requires_grad=True)
        scores = torch.full((2, 10), 2.0)
        phn_ids = torch.zeros(2, 10, dtype=torch.long)  # all same phone
        loss = loss_fn(features, scores, phn_ids)
        assert loss.item() == 0.0

    def test_no_score2_returns_zero(self):
        """No tokens with score 2.0 → return 0 (can't form ordinal structure)."""
        loss_fn = OrdinalEntropyLoss()
        features = torch.randn(2, 10, 8, requires_grad=True)
        scores = torch.full((2, 10), 1.0)  # no 2.0 scores
        phn_ids = torch.randint(0, 5, (2, 10))
        loss = loss_fn(features, scores, phn_ids)
        assert loss.item() == 0.0

    def test_diversity_increases_with_separation(self):
        """More separated centers → larger diversity → more negative loss component."""
        loss_fn = OrdinalEntropyLoss(lambda_d=1.0, lambda_t=0.0)  # diversity only

        scores = torch.full((2, 10), 2.0)
        phn_ids = torch.tensor([[0] * 5 + [1] * 5] * 2)

        # Similar centers: both phones point in ~same direction
        features_close = torch.zeros(2, 10, 8)
        features_close[:, :5, 0] = 1.0   # phone 0: [1, 0, 0.1, ...]
        features_close[:, :5, 2] = 0.1
        features_close[:, 5:, 0] = 1.0   # phone 1: [1, 0.1, 0, ...]
        features_close[:, 5:, 1] = 0.1
        features_close = features_close.requires_grad_(True)
        loss_close = loss_fn(features_close, scores, phn_ids)

        # Orthogonal centers: phone 0 = [1,0,...], phone 1 = [0,1,...]
        features_far = torch.zeros(2, 10, 8)
        features_far[:, :5, 0] = 1.0  # phone 0
        features_far[:, 5:, 1] = 1.0  # phone 1
        features_far = features_far.requires_grad_(True)
        loss_far = loss_fn(features_far, scores, phn_ids)

        # More separation → more diversity → more negative loss
        assert loss_far.item() < loss_close.item()


class TestCLAPContrastiveLoss:
    """Tests for the audio-text contrastive alignment loss."""

    def _make_batch(
        self, n_phones: int = 3, seq_len: int = 10, embed_dim: int = 8,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = 2
        audio = torch.randn(batch_size, seq_len, embed_dim, requires_grad=True)
        text = torch.randn(batch_size, seq_len, embed_dim, requires_grad=True)
        phn_ids = torch.randint(0, n_phones, (batch_size, seq_len))
        scores = torch.full((batch_size, seq_len), 1.6)
        for p in range(n_phones):
            mask = phn_ids == p
            if mask.any():
                idx = mask.nonzero()[0]
                scores[idx[0], idx[1]] = 2.0
        return audio, text, scores, phn_ids

    def test_output_is_scalar(self):
        loss_fn = CLAPContrastiveLoss()
        audio, text, scores, phn_ids = self._make_batch()
        loss = loss_fn(audio, text, scores, phn_ids)
        assert loss.ndim == 0

    def test_gradient_flows_both_inputs(self):
        loss_fn = CLAPContrastiveLoss()
        audio, text, scores, phn_ids = self._make_batch()
        loss = loss_fn(audio, text, scores, phn_ids)
        loss.backward()
        assert audio.grad is not None and audio.grad.abs().sum() > 0
        assert text.grad is not None and text.grad.abs().sum() > 0

    def test_padding_excluded(self):
        loss_fn = CLAPContrastiveLoss()
        audio, text, scores, phn_ids = self._make_batch()
        loss_clean = loss_fn(audio, text, scores, phn_ids)

        # Add padding
        pad = torch.randn(2, 5, 8)
        audio_p = torch.cat([audio.detach(), pad], dim=1).requires_grad_(True)
        text_p = torch.cat([text.detach(), pad], dim=1).requires_grad_(True)
        scores_p = torch.cat([scores, torch.full((2, 5), -1.0)], dim=1)
        phn_ids_p = torch.cat([phn_ids, torch.full((2, 5), -1, dtype=torch.long)], dim=1)
        loss_padded = loss_fn(audio_p, text_p, scores_p, phn_ids_p)

        assert torch.allclose(loss_clean, loss_padded, atol=1e-5)

    def test_all_padding_returns_zero(self):
        loss_fn = CLAPContrastiveLoss()
        audio = torch.randn(2, 10, 8, requires_grad=True)
        text = torch.randn(2, 10, 8, requires_grad=True)
        scores = torch.full((2, 10), -1.0)
        phn_ids = torch.full((2, 10), -1, dtype=torch.long)
        loss = loss_fn(audio, text, scores, phn_ids)
        assert loss.item() == 0.0

    def test_single_phoneme_returns_zero(self):
        loss_fn = CLAPContrastiveLoss()
        audio = torch.randn(2, 10, 8, requires_grad=True)
        text = torch.randn(2, 10, 8, requires_grad=True)
        scores = torch.full((2, 10), 2.0)
        phn_ids = torch.zeros(2, 10, dtype=torch.long)
        loss = loss_fn(audio, text, scores, phn_ids)
        assert loss.item() == 0.0

    def test_identical_centers_low_loss(self):
        """When audio==text, contrastive loss should be low (perfect alignment)."""
        loss_fn = CLAPContrastiveLoss()
        shared = torch.randn(2, 10, 8)
        scores = torch.full((2, 10), 2.0)
        phn_ids = torch.tensor([[0] * 5 + [1] * 5] * 2)

        loss_same = loss_fn(
            shared.clone().requires_grad_(True),
            shared.clone().requires_grad_(True),
            scores, phn_ids,
        )

        # Random audio vs text should have higher loss
        loss_random = loss_fn(
            torch.randn(2, 10, 8, requires_grad=True),
            torch.randn(2, 10, 8, requires_grad=True),
            scores, phn_ids,
        )

        assert loss_same.item() < loss_random.item()
