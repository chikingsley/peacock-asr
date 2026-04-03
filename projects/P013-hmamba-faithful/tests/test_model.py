from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from p012.models.hmamba import AttentionPooling, HMamba


def test_attention_pooling_matches_linear_score_attention() -> None:
    pool = AttentionPooling(in_dim=4, temperature=1.0)
    with torch.no_grad():
        pool.attention.weight.copy_(torch.tensor([[1.0, 1.0, 1.0, 1.0]]))

    x = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    score = torch.tensor([[[1.0, -2.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]]])
    mask = torch.tensor([[True, True]])

    out = pool(x, score, mask)
    logits = torch.tensor([[-1.0, -4.0]])
    weights = torch.softmax(logits, dim=-1)
    expected = weights[0, 0] * x[0, 0] + weights[0, 1] * x[0, 1]
    assert torch.allclose(out[0], expected, atol=1e-6)


def test_hmamba_requires_cuda_runtime() -> None:
    config = yaml.safe_load(Path("conf/so762/HMamba.yaml").read_text(encoding="utf-8"))
    model = HMamba(**config)
    batch_size, seq_len = 1, 4
    gop = torch.randn(batch_size, seq_len, config["gop_dim"])
    ssl = [torch.randn(batch_size, seq_len, dim) for dim in config["ssl_dim"]]
    raw = torch.randn(batch_size, seq_len, config["raw_dim"])
    canophn = torch.randint(0, config["vocab_size"], (batch_size, seq_len))
    bies = torch.randint(0, 6, (batch_size, seq_len))
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    with pytest.raises(RuntimeError, match="requires CUDA"):
        model(gop, ssl, raw, canophn, bies, mask)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="official mamba-ssm runtime requires CUDA")
def test_hmamba_forward_shapes_cuda() -> None:
    config = yaml.safe_load(Path("conf/so762/HMamba.yaml").read_text(encoding="utf-8"))
    device = torch.device("cuda")
    model = HMamba(**config).to(device)
    model.eval()

    batch_size, seq_len = 2, 8
    gop = torch.randn(batch_size, seq_len, config["gop_dim"], device=device)
    ssl = [torch.randn(batch_size, seq_len, dim, device=device) for dim in config["ssl_dim"]]
    raw = torch.randn(batch_size, seq_len, config["raw_dim"], device=device)
    canophn = torch.randint(0, config["vocab_size"], (batch_size, seq_len), device=device)
    bies = torch.randint(0, 6, (batch_size, seq_len), device=device)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

    with torch.no_grad():
        outputs = model(gop, ssl, raw, canophn, bies, mask)

    assert len(outputs) == 10
    u1, _u2, _u3, _u4, u5, p, _w1, w2, _w3, logits = outputs
    assert u1.shape == (batch_size, 1)
    assert u5.shape == (batch_size, 1)
    assert p.shape == (batch_size, seq_len, 1)
    assert w2.shape == (batch_size, seq_len, 1)
    assert logits.shape == (batch_size, seq_len, config["vocab_size"])


def test_hmamba_reports_resolved_backend() -> None:
    config = yaml.safe_load(Path("conf/so762/HMamba.yaml").read_text(encoding="utf-8"))
    model = HMamba(**config)
    assert model.resolved_mamba_backend == "official-mamba"
