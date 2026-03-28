"""Tests for phone-level all-layer SSL interface integration."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch


def test_all_layer_dataset_and_collate(synthetic_features_dir: Path) -> None:
    from p010.data import AllLayerDataset, FrameBatch, collate_frame_samples, has_all_layer_data

    assert has_all_layer_data(synthetic_features_dir, "train")

    dataset = AllLayerDataset("train", synthetic_features_dir)
    assert len(dataset) == 4

    sample = dataset[0]
    assert sample.ssl_frames["w2v_300m"].shape == (50, 25, 1024)
    assert sample.ssl_frames["hubert"].shape == (50, 25, 1024)
    assert sample.ssl_frames["wavlm"].shape == (50, 25, 1024)

    batch = collate_frame_samples([dataset[0], dataset[1]])
    assert isinstance(batch, FrameBatch)
    assert batch.gop.shape == (2, 50, 84)
    assert batch.ssl_frames["w2v_300m"].shape == (2, 50, 25, 1024)


def test_all_layer_dataset_subset(synthetic_features_dir: Path) -> None:
    from p010.data import AllLayerDataset, collate_frame_samples, has_all_layer_data

    assert has_all_layer_data(synthetic_features_dir, "train", ssl_model_keys=("hubert",))

    dataset = AllLayerDataset("train", synthetic_features_dir, ssl_model_keys=("hubert",))
    sample = dataset[0]
    assert tuple(sample.ssl_frames.keys()) == ("hubert",)

    batch = collate_frame_samples([dataset[0], dataset[1]])
    assert tuple(batch.ssl_frames.keys()) == ("hubert",)
    assert batch.ssl_frames["hubert"].shape == (2, 50, 25, 1024)


def test_all_layer_hconv_model_shapes(synthetic_features_dir: Path) -> None:
    from p010.data import AllLayerDataset, collate_frame_samples
    from p010.models.ssl_interface import AllLayerInterfaceModel
    from p010.settings import SSLInterfaceMode

    dataset = AllLayerDataset("train", synthetic_features_dir)
    batch = collate_frame_samples([dataset[0], dataset[1]])
    word_pos = batch.word_label[:, :, 3]

    model = AllLayerInterfaceModel(SSLInterfaceMode.HCONV, ssl_output_dim=3072)
    model.eval()
    with torch.no_grad():
        outputs = model(
            batch.gop, batch.energy, batch.dur, batch.ssl_frames,
            batch.phn_id, word_pos, batch.word_id,
        )
    assert len(outputs) == 11
    assert outputs[5].shape == (2, 50, 1)  # phone predictions


def test_all_layer_hconv_model_shapes_single_stream(synthetic_features_dir: Path) -> None:
    from p010.data import AllLayerDataset, collate_frame_samples
    from p010.models.ssl_interface import AllLayerInterfaceModel
    from p010.settings import SSLInterfaceMode

    dataset = AllLayerDataset("train", synthetic_features_dir, ssl_model_keys=("hubert",))
    batch = collate_frame_samples([dataset[0], dataset[1]])
    word_pos = batch.word_label[:, :, 3]

    model = AllLayerInterfaceModel(
        SSLInterfaceMode.HCONV,
        ssl_output_dim=1024,
        ssl_model_keys=("hubert",),
    )
    model.eval()
    with torch.no_grad():
        outputs = model(
            batch.gop, batch.energy, batch.dur, batch.ssl_frames,
            batch.phn_id, word_pos, batch.word_id,
        )
    assert len(outputs) == 11
    assert outputs[5].shape == (2, 50, 1)


def test_all_layer_chconv_model_shapes(synthetic_features_dir: Path) -> None:
    from p010.data import AllLayerDataset, collate_frame_samples
    from p010.models.ssl_interface import AllLayerInterfaceModel
    from p010.settings import SSLInterfaceMode

    dataset = AllLayerDataset("train", synthetic_features_dir)
    batch = collate_frame_samples([dataset[0], dataset[1]])
    word_pos = batch.word_label[:, :, 3]

    model = AllLayerInterfaceModel(SSLInterfaceMode.CHCONV, ssl_output_dim=3072)
    model.eval()
    with torch.no_grad():
        outputs = model(
            batch.gop, batch.energy, batch.dur, batch.ssl_frames,
            batch.phn_id, word_pos, batch.word_id,
        )
    assert len(outputs) == 11
    assert outputs[5].shape == (2, 50, 1)


def test_all_layer_smoke_train_one_epoch(
    synthetic_features_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from p010.data import make_loaders
    from p010.models.ssl_interface import AllLayerInterfaceModel
    from p010.settings import Settings, SSLInterfaceMode
    from p010.trainer import train_one_config

    monkeypatch.setenv("WANDB_MODE", "offline")

    settings = Settings(
        features_dir=synthetic_features_dir,
        ssl_interface=SSLInterfaceMode.HCONV,
        seed=7,
        n_epochs=1,
        batch_size=2,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    train_loader, test_loader, _ = make_loaders(
        settings.features_dir,
        settings.batch_size,
        num_workers=0,
        ssl_interface=settings.ssl_interface,
        ssl_model_keys=settings.ssl_models,
    )
    model = AllLayerInterfaceModel(
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
        run_name="all_layer_smoke",
        checkpoint_dir=tmp_path / "all_layer_ckpt",
    )

    assert isinstance(pcc, float)
    assert -1.0 <= pcc <= 1.0


def test_all_layer_smoke_train_one_epoch_single_stream_accum(
    synthetic_features_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from p010.data import make_loaders
    from p010.models.ssl_interface import AllLayerInterfaceModel
    from p010.settings import Settings, SSLInterfaceMode
    from p010.trainer import train_one_config

    monkeypatch.setenv("WANDB_MODE", "offline")

    settings = Settings(
        features_dir=synthetic_features_dir,
        ssl_interface=SSLInterfaceMode.HCONV,
        ssl_models="hubert",
        seed=9,
        n_epochs=1,
        batch_size=1,
        grad_accum_steps=2,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    train_loader, test_loader, _ = make_loaders(
        settings.features_dir,
        settings.batch_size,
        num_workers=0,
        ssl_interface=settings.ssl_interface,
        ssl_model_keys=settings.ssl_models,
    )
    model = AllLayerInterfaceModel(
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
        run_name="all_layer_smoke_single_stream",
        checkpoint_dir=tmp_path / "all_layer_single_stream_ckpt",
    )

    assert isinstance(pcc, float)
    assert -1.0 <= pcc <= 1.0
