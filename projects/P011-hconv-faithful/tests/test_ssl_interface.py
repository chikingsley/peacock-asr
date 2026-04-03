"""Tests for frame-store SSL interface integration."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch


def test_frame_store_dataset_and_collate(synthetic_features_dir: Path) -> None:
    from p011.data import FrameBatch, FrameStoreDataset, collate_frame_samples, has_frame_store_data

    assert has_frame_store_data(synthetic_features_dir, "train")

    dataset = FrameStoreDataset("train", synthetic_features_dir)
    sample = dataset[0]
    assert sample.ssl_frames["hubert"].shape == (8, 25, 1024)
    assert sample.ssl_frames["w2v_300m"].shape == (8, 25, 1024)

    batch = collate_frame_samples([dataset[0], dataset[1]])
    assert isinstance(batch, FrameBatch)
    assert batch.gop.shape == (2, 50, 84)
    assert batch.ssl_frames["hubert"].shape == (2, 9, 25, 1024)
    assert torch.equal(batch.frame_lengths["hubert"], torch.tensor([8, 9]))


def test_frame_store_dataset_subset(synthetic_features_dir: Path) -> None:
    from p011.data import FrameStoreDataset, collate_frame_samples, has_frame_store_data

    assert has_frame_store_data(synthetic_features_dir, "train", ssl_model_keys=("hubert",))

    dataset = FrameStoreDataset("train", synthetic_features_dir, ssl_model_keys=("hubert",))
    sample = dataset[0]
    assert tuple(sample.ssl_frames.keys()) == ("hubert",)

    batch = collate_frame_samples([dataset[0], dataset[1]])
    assert tuple(batch.ssl_frames.keys()) == ("hubert",)
    assert batch.ssl_frames["hubert"].shape == (2, 9, 25, 1024)


def test_frame_level_hconv_model_shapes(synthetic_features_dir: Path) -> None:
    from p011.data import FrameStoreDataset, collate_frame_samples
    from p011.models.ssl_interface import FrameLevelInterfaceModel
    from p011.settings import SSLInterfaceMode

    dataset = FrameStoreDataset("train", synthetic_features_dir, ssl_model_keys=("hubert",))
    batch = collate_frame_samples([dataset[0], dataset[1]])
    word_pos = batch.word_label[:, :, 3]

    model = FrameLevelInterfaceModel(
        SSLInterfaceMode.HCONV,
        ssl_model_keys=("hubert",),
    )
    model.eval()
    with torch.no_grad():
        outputs = model(
            batch.gop,
            batch.energy,
            batch.dur,
            batch.ssl_frames,
            batch.phn_id,
            word_pos,
            batch.word_id,
            batch.frame_lengths,
        )
    assert len(outputs) == 11
    assert outputs[5].shape == (2, 50, 1)


def test_frame_level_last_model_shapes(synthetic_features_dir: Path) -> None:
    from p011.data import FrameStoreDataset, collate_frame_samples
    from p011.models.ssl_interface import FrameLevelInterfaceModel
    from p011.settings import SSLInterfaceMode

    dataset = FrameStoreDataset("train", synthetic_features_dir, ssl_model_keys=("hubert",))
    batch = collate_frame_samples([dataset[0], dataset[1]])
    word_pos = batch.word_label[:, :, 3]

    model = FrameLevelInterfaceModel(
        SSLInterfaceMode.LAST,
        ssl_model_keys=("hubert",),
    )
    model.eval()
    with torch.no_grad():
        outputs = model(
            batch.gop,
            batch.energy,
            batch.dur,
            batch.ssl_frames,
            batch.phn_id,
            word_pos,
            batch.word_id,
            batch.frame_lengths,
        )
    assert len(outputs) == 11
    assert outputs[5].shape == (2, 50, 1)


def test_chconv_is_not_enabled_for_p011() -> None:
    from p011.models.ssl_interface import FrameLevelInterfaceModel
    from p011.settings import SSLInterfaceMode

    with pytest.raises(NotImplementedError):
        FrameLevelInterfaceModel(SSLInterfaceMode.CHCONV)


def test_frame_last_rejects_projection_override() -> None:
    from p011.models.ssl_interface import FrameLevelInterfaceModel
    from p011.settings import SSLInterfaceMode

    with pytest.raises(ValueError):
        FrameLevelInterfaceModel(
            SSLInterfaceMode.LAST,
            ssl_model_keys=("hubert",),
            ssl_output_dim=2048,
        )


def test_frame_level_smoke_train_one_epoch(
    synthetic_features_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from p011.data import make_loaders
    from p011.models.ssl_interface import FrameLevelInterfaceModel
    from p011.settings import Settings, SSLInterfaceMode
    from p011.trainer import train_one_config

    monkeypatch.setenv("WANDB_MODE", "offline")

    settings = Settings(
        features_dir=synthetic_features_dir,
        ssl_interface=SSLInterfaceMode.HCONV,
        ssl_models="hubert",
        seed=7,
        n_epochs=1,
        batch_size=2,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    train_loader, test_loader = make_loaders(
        settings.features_dir,
        settings.batch_size,
        num_workers=0,
        ssl_interface=settings.ssl_interface,
        ssl_model_keys=settings.ssl_models,
    )
    model = FrameLevelInterfaceModel(
        ssl_interface=settings.ssl_interface,
        ssl_output_dim=settings.resolved_ssl_output_dim,
        ssl_model_keys=settings.ssl_models,
        embed_dim=settings.embed_dim,
        num_heads=settings.num_heads,
        p_depth=settings.p_depth,
        w_depth=settings.w_depth,
        u_depth=settings.u_depth,
        ssl_drop=settings.ssl_drop,
        use_mdd=settings.use_mdd,
    )

    pcc = train_one_config(
        settings,
        model,
        train_loader,
        test_loader,
        run_name="frame_level_smoke",
        checkpoint_dir=tmp_path / "frame_level_ckpt",
    )

    assert isinstance(pcc, float)
    assert -1.0 <= pcc <= 1.0
