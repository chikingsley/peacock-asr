"""Smoke test — one training epoch end-to-end.

Requires P011_FEATURES_DIR and W&B to be configured. Marks WANDB_MODE=offline
so it doesn't push to the cloud; creates a temp checkpoint dir.

Run with:
    uv run pytest tests/test_smoke.py -v -s
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch


@pytest.fixture(autouse=True)
def wandb_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force W&B to offline mode for tests so nothing is pushed."""
    monkeypatch.setenv("WANDB_MODE", "offline")


def test_train_one_epoch(features_dir: Path, tmp_path: Path) -> None:
    """Full train → eval loop for 1 epoch. Checks that PCC is a finite float."""
    from p011.data import make_loaders
    from p011.models.hiercb import HierCB
    from p011.settings import Settings
    from p011.trainer import train_one_config

    settings = Settings(  # features_dir filled by fixture
        features_dir=features_dir,
        seed=42,
        n_epochs=1,
        batch_size=8,
        grad_accum_steps=2,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    train_loader, test_loader = make_loaders(
        settings.features_dir,
        settings.batch_size,
        num_workers=0,
        ssl_model_keys=settings.ssl_models,
    )
    model = HierCB(embed_dim=settings.embed_dim, ssl_dim=settings.selected_ssl_dim, use_mdd=settings.use_mdd)

    pcc = train_one_config(
        settings, model, train_loader, test_loader,
        run_name="smoke_test",
        checkpoint_dir=tmp_path / "smoke_ckpt",
    )

    assert isinstance(pcc, float), f"Expected float PCC, got {type(pcc)}"
    assert -1.0 <= pcc <= 1.0, f"PCC out of range: {pcc}"
    assert pcc == pcc, "PCC is NaN"
