"""Tests for HierCB model — output shapes, parameter count, MDD path."""

from __future__ import annotations

import torch


def _make_batch(B: int = 2, T: int = 50, ssl_dim: int = 3072):
    """Return a synthetic batch of the right shapes.

    word_pos: within-utterance word position (small int, 0..N_words-1) — fits word_pos_embed vocab=50
    word_id:  lexical vocabulary ID (0..2606) — fits word_proj one-hot vocab=2607
    These are distinct tensors, matching the two-file layout (tr_label_word + tr_word_id).
    """
    rng = torch.Generator()
    rng.manual_seed(42)
    gop = torch.randn(B, T, 84, generator=rng)
    energy = torch.randn(B, T, 7, generator=rng)
    dur = torch.randn(B, T, 1, generator=rng)
    ssl = torch.randn(B, T, ssl_dim, generator=rng)
    phn_id = torch.randint(0, 39, (B, T))
    phn_id[:, T // 2:] = -1  # second half is padding

    # word_pos: small sequential index within utterance (0..N_words), -1 = pad
    word_pos = torch.zeros(B, T, dtype=torch.long)
    for b in range(B):
        for i in range(T // 2):
            word_pos[b, i] = i // 4  # 4 phones per word → positions 0..12
    word_pos[:, T // 2:] = -1

    # word_id: lexical vocab IDs — deliberately larger than word_pos to distinguish them
    word_id = torch.randint(0, 2606, (B, T))
    word_id[:, T // 2:] = -1

    return gop, energy, dur, ssl, phn_id, word_pos.float(), word_id.float()


def test_hiercb_output_shapes_no_mdd() -> None:
    from p011.models.hiercb import HierCB

    model = HierCB(embed_dim=24, use_mdd=False)
    model.eval()

    gop, energy, dur, ssl, phn_id, word_pos, word_id = _make_batch(B=2)
    with torch.no_grad():
        out = model(gop, energy, dur, ssl, phn_id, word_pos, word_id)

    assert len(out) == 11, f"Expected 11 outputs without MDD, got {len(out)}"
    u1, u2, u3, u4, u5, p, w1, w2, w3, phn_audio, phn_text = out

    # Utterance: [B, 1]
    for u in (u1, u2, u3, u4, u5):
        assert u.shape == (2, 1), f"Utterance output shape mismatch: {u.shape}"

    # Phone: [B, 50, 1]
    assert p.shape == (2, 50, 1), f"Phone output shape: {p.shape}"

    # Word: [B, 50, 1]
    for w in (w1, w2, w3):
        assert w.shape == (2, 50, 1), f"Word output shape: {w.shape}"

    # ConPCO features: [B, 50, embed_dim]
    assert phn_audio.shape == (2, 50, 24)
    assert phn_text.shape == (2, 50, 24)


def test_hiercb_output_shapes_with_mdd() -> None:
    from p011.models.hiercb import HierCB

    model = HierCB(embed_dim=24, use_mdd=True)
    model.eval()

    gop, energy, dur, ssl, phn_id, word_pos, word_id = _make_batch(B=2)
    with torch.no_grad():
        out = model(gop, energy, dur, ssl, phn_id, word_pos, word_id)

    assert len(out) == 13, f"Expected 13 outputs with MDD, got {len(out)}"
    *_, mdd_logit, diag_logit, phn_audio, phn_text = out
    assert mdd_logit.shape == (2, 50, 1), f"MDD logit shape: {mdd_logit.shape}"
    assert diag_logit.shape == (2, 50, 39), f"Diag logit shape: {diag_logit.shape}"


def test_hiercb_parameter_count() -> None:
    """HierCB (embed_dim=24) should have ~600K parameters."""
    from p011.models.hiercb import HierCB

    model = HierCB(embed_dim=24, p_depth=3, w_depth=2, u_depth=1)
    n = sum(p.numel() for p in model.parameters())
    # embed_dim=24, dead weights removed → ~448K params; allow ±15%
    assert 380_000 < n < 520_000, f"Parameter count out of range: {n:,}"


def test_hiercb_gradients_flow() -> None:
    """All trainable parameters must receive gradients after a forward+backward pass."""
    from p011.models.hiercb import HierCB

    model = HierCB(embed_dim=24, use_mdd=False)
    gop, energy, dur, ssl, phn_id, word_pos, word_id = _make_batch(B=2)

    out = model(gop, energy, dur, ssl, phn_id, word_pos, word_id)
    u1, u2, u3, u4, u5, p, w1, w2, w3, phn_audio, phn_text = out
    # Include all outputs so gradients reach every active branch
    loss = (
        p.sum() + w1.sum() + w2.sum() + w3.sum()
        + u1.sum() + u2.sum() + u3.sum() + u4.sum() + u5.sum()
        + phn_audio.sum() + phn_text.sum()
    )
    loss.backward()

    no_grad = [name for name, param in model.named_parameters() if param.requires_grad and param.grad is None]
    assert len(no_grad) == 0, f"Parameters with no gradient: {no_grad}"


def test_word_pos_mask_same_word() -> None:
    """_make_word_pos_mask: positions with same word_pos should be able to attend to each other."""
    from p011.models.hiercb import HierCB

    model = HierCB(embed_dim=24)
    # B=1, T=6: words [0,0,1,1,-1,-1]
    word_pos = torch.tensor([[0, 0, 1, 1, -1, -1]], dtype=torch.float)
    mask = model._make_word_pos_mask(word_pos)
    assert mask.shape == (1, 6, 6)

    # Same word positions should have 1s
    assert mask[0, 0, 1] == 1.0, "Positions in same word must be 1"
    assert mask[0, 2, 3] == 1.0, "Positions in same word must be 1"

    # Different word positions should have 0s
    assert mask[0, 0, 2] == 0.0, "Positions in different words must be 0"

    # Padding positions must always be 0
    assert mask[0, 4, 4] == 0.0, "Padding positions must have 0 mask"
    assert mask[0, 0, 4] == 0.0, "Attending to padding must be 0"


def test_hiercb_deterministic_no_grad() -> None:
    """Two identical forward passes in eval mode must give identical outputs."""
    from p011.models.hiercb import HierCB

    model = HierCB(embed_dim=24)
    model.eval()

    inputs = _make_batch(B=2)
    with torch.no_grad():
        out1 = model(*inputs)
        out2 = model(*inputs)

    for t1, t2 in zip(out1, out2, strict=True):
        assert torch.allclose(t1, t2), "Eval outputs must be deterministic"


def test_hiercb_supports_single_ssl_stream() -> None:
    from p011.models.hiercb import HierCB

    model = HierCB(embed_dim=24, ssl_dim=1024)
    model.eval()

    gop, energy, dur, ssl, phn_id, word_pos, word_id = _make_batch(B=2, ssl_dim=1024)
    with torch.no_grad():
        out = model(gop, energy, dur, ssl, phn_id, word_pos, word_id)

    assert len(out) == 11
    assert out[5].shape == (2, 50, 1)
