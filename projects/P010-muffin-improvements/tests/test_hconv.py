"""Tests for HConv and CHConv modules."""

from __future__ import annotations

import math

import torch


def test_hconv_output_shape_base() -> None:
    """HConv with L=13, D=768 (Base model) → output_dim=768."""
    from p010.models.hconv import HConv

    hconv = HConv(num_layers=13, feat_dim=768)
    x = torch.randn(2, 50, 13, 768)
    out = hconv(x)
    assert out.shape == (2, 50, hconv.output_dim), f"Got {out.shape}"
    assert hconv.output_dim == 768, f"Got output_dim={hconv.output_dim}"


def test_hconv_output_shape_large() -> None:
    """HConv with L=25, D=1024 (Large model) → output_dim=1024."""
    from p010.models.hconv import HConv

    hconv = HConv(num_layers=25, feat_dim=1024)
    x = torch.randn(2, 50, 25, 1024)
    out = hconv(x)
    assert out.shape == (2, 50, hconv.output_dim), f"Got {out.shape}"
    # For L=25, stride=3: 2 convs. L: 25→9→3. Last conv out_channels=ceil(1024//3)=341.
    # Output: 341*3=1023
    assert hconv.output_dim == 1023, f"Got output_dim={hconv.output_dim}"


def test_hconv_num_convs() -> None:
    """Number of conv layers matches floor(log_3(L))."""
    from p010.models.hconv import HConv

    for L in [13, 25, 49]:
        hconv = HConv(num_layers=L, feat_dim=256)
        expected = math.floor(math.log(L) / math.log(3))
        # Count Conv1d layers (each followed by ReLU, so convs = len(sequential) // 2)
        n_convs = sum(1 for m in hconv.convs if isinstance(m, torch.nn.Conv1d))
        assert n_convs == expected, f"L={L}: expected {expected} convs, got {n_convs}"


def test_hconv_gradients_flow() -> None:
    """Gradients flow through HConv."""
    from p010.models.hconv import HConv

    hconv = HConv(num_layers=25, feat_dim=64)
    x = torch.randn(2, 10, 25, 64, requires_grad=True)
    out = hconv(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0


def test_chconv_output_shape() -> None:
    """CHConv with 3 models (L=25, D=1024 each) → configurable output_dim."""
    from p010.models.hconv import CHConv

    chconv = CHConv(num_layers=25, feat_dims=[1024, 1024, 1024], output_dim=1024)
    feats = [torch.randn(2, 50, 25, 1024) for _ in range(3)]
    out = chconv(feats)
    assert out.shape == (2, 50, 1024), f"Got {out.shape}"


def test_chconv_without_projection() -> None:
    """CHConv without explicit output_dim uses HConv's natural output."""
    from p010.models.hconv import CHConv

    chconv = CHConv(num_layers=25, feat_dims=[1024, 1024, 1024])
    feats = [torch.randn(2, 50, 25, 1024) for _ in range(3)]
    out = chconv(feats)
    assert out.shape == (2, 50, chconv.output_dim), f"Got {out.shape}"
    assert chconv.hconv.proj is None  # no projection layer


def test_hconv_output_projection_to_target_dim() -> None:
    """HConv can project to an explicit output dim for interface compatibility."""
    from p010.models.hconv import HConv

    hconv = HConv(num_layers=25, feat_dim=1024, output_dim=2048)
    x = torch.randn(2, 50, 25, 1024)
    out = hconv(x)
    assert out.shape == (2, 50, 2048), f"Got {out.shape}"


def test_phone_hconv_interface_total_projection() -> None:
    """PhoneHConvInterface can be configured to hit a total SSL output size."""
    from p010.models.ssl_interface import PhoneHConvInterface

    interface = PhoneHConvInterface(ssl_output_dim=3072)
    assert interface.output_dim == 3072


def test_phone_hconv_interface_single_model_projection() -> None:
    """A one-model HConv interface can preserve the upstream width via projection."""
    from p010.models.ssl_interface import PhoneHConvInterface

    interface = PhoneHConvInterface(ssl_output_dim=1024, ssl_model_keys=("hubert",))
    assert interface.output_dim == 1024


def test_chconv_gradients_flow() -> None:
    """Gradients flow through CHConv to all input models."""
    from p010.models.hconv import CHConv

    chconv = CHConv(num_layers=13, feat_dims=[64, 64], output_dim=64)
    feats = [torch.randn(2, 10, 13, 64, requires_grad=True) for _ in range(2)]
    out = chconv(feats)
    loss = out.sum()
    loss.backward()
    for i, f in enumerate(feats):
        assert f.grad is not None, f"No gradient for model {i}"
        assert f.grad.abs().sum() > 0, f"Zero gradient for model {i}"


def test_hconv_normalize_option() -> None:
    """HConv with normalize=True applies LayerNorm."""
    from p010.models.hconv import HConv

    hconv = HConv(num_layers=13, feat_dim=64, normalize=True)
    x = torch.randn(2, 10, 13, 64) * 100  # large scale
    out = hconv(x)
    assert out.shape == (2, 10, hconv.output_dim)
