"""Tests for data.py — run against real SpeechOcean762 features.

Requires P010_FEATURES_DIR to be set. Tests fail (not skip) if data is absent.
See conftest.py for the features_dir fixture.
"""

from __future__ import annotations

from pathlib import Path

import torch


def test_dataset_shapes(features_dir: Path) -> None:
    from p010.data import GoPDataset
    from p010.ssl_features import ssl_feature_dim

    ds = GoPDataset("train", features_dir)
    assert len(ds) > 0, "Training set must be non-empty"

    # 11-tuple: gop, ssl, energy, dur, phn_score, phn_id, utt_label, word_label, word_id, mdd_label, diag_label
    sample = ds[0]
    assert len(sample) == 11, f"Expected 11-tuple, got {len(sample)}"
    gop, ssl, energy, dur, phn_score, phn_id, utt_label, word_label, word_id, mdd_label, diag_label = sample

    assert gop.shape == (50, 84), f"GOP shape: {gop.shape}"
    assert ssl.shape == (50, ssl_feature_dim(ds.ssl_model_keys)), f"SSL shape: {ssl.shape}"
    assert energy.shape == (50, 7), f"Energy shape: {energy.shape}"
    assert dur.shape == (50, 1), f"Dur shape: {dur.shape}"
    assert phn_score.shape == (50,), f"phn_score shape: {phn_score.shape}"
    assert phn_id.shape == (50,), f"phn_id shape: {phn_id.shape}"
    assert utt_label.shape == (5,), f"utt_label shape: {utt_label.shape}"
    assert word_label.shape == (50, 4), f"word_label shape: {word_label.shape}"
    assert word_id.shape == (50,), f"word_id shape: {word_id.shape}"
    assert mdd_label.shape == (50,), f"mdd_label shape: {mdd_label.shape}"
    assert diag_label.shape == (50,), f"diag_label shape: {diag_label.shape}"
    assert diag_label.dtype == torch.long, f"diag_label dtype: {diag_label.dtype}"


def test_word_pos_and_word_id_are_distinct(features_dir: Path) -> None:
    """word_label[:,3] (word_pos) must be small ints; word_id must be larger vocab IDs.

    This is the key contract: word_pos indexes within the utterance (max ~N_words ≤ 50),
    word_id is the lexical vocabulary ID (0..2606). Confusing them causes IndexError in
    word_pos_embed (vocab size 50) when lexical IDs far exceed 49.
    """
    from p010.data import GoPDataset

    ds = GoPDataset("train", features_dir)

    valid_word_pos = ds.word_label[ds.word_label[:, :, 3] >= 0, 3]
    valid_word_id = ds.word_id[ds.word_id >= 0]

    assert float(valid_word_pos.max()) < 50, (
        f"word_pos must be < 50 (word_pos_embed vocab size), got max={valid_word_pos.max()}"
    )
    assert float(valid_word_id.max()) > 49, (
        f"word_id should contain large vocab IDs (>49), got max={valid_word_id.max()}"
    )


def test_valid_positions_normalized(features_dir: Path) -> None:
    """Padding positions (all-zero GOP row) must stay zero after normalization."""
    from p010.data import GoPDataset

    ds = GoPDataset("train", features_dir)
    gop = ds.gop  # [N, 50, 84]
    raw_zero_mask = (gop == 0).all(dim=-1)
    assert raw_zero_mask.any(), "Expected some padding positions in the dataset"


def test_utt_label_divided_by_five(features_dir: Path) -> None:
    """Utterance labels are divided by 5; real SpeechOcean762 scores are 0-10."""
    from p010.data import GoPDataset

    ds = GoPDataset("train", features_dir)
    assert float(ds.utt_label.min()) >= 0.0
    assert float(ds.utt_label.max()) <= 2.1, (
        f"utt_label max {ds.utt_label.max():.3f} > 2.1 — /5 normalization may be wrong"
    )


def test_word_scores_divided_by_five(features_dir: Path) -> None:
    """Word score columns 0-2 are divided by 5; column 3 (word_pos) is left as-is."""
    from p010.data import GoPDataset

    ds = GoPDataset("train", features_dir)
    valid_scores = ds.word_label[ds.word_label[:, :, 0] >= 0][:, :3]
    assert float(valid_scores.max()) <= 2.1, (
        f"Word scores max {valid_scores.max():.3f} > 2.1 — /5 normalization may be wrong"
    )
    valid_wp = ds.word_label[ds.word_label[:, :, 3] >= 0, 3]
    assert float(valid_wp.max()) < 50, "word_pos must be a small within-utterance index"


def test_ssl_concat_shape(features_dir: Path) -> None:
    """SSL concat: [wav2vec2 | HuBERT | WavLM] → [N, 50, 3072]."""
    from p010.data import GoPDataset
    from p010.ssl_features import SSL_MODEL_KEYS, ssl_feature_dim

    ds = GoPDataset("train", features_dir)
    assert ds.ssl.shape[-1] == ssl_feature_dim(SSL_MODEL_KEYS), (
        f"SSL last dim should be {ssl_feature_dim(SSL_MODEL_KEYS)}, got {ds.ssl.shape[-1]}"
    )


def test_ssl_subset_shape(features_dir: Path) -> None:
    """Selecting a single SSL stream should reduce the concatenated width accordingly."""
    from p010.data import GoPDataset

    ds = GoPDataset("train", features_dir, ssl_model_keys=("hubert",))
    assert ds.ssl.shape[-1] == 1024, f"Single-model SSL width should be 1024, got {ds.ssl.shape[-1]}"


def test_mdd_labels(features_dir: Path) -> None:
    """MDD: -1 for padding, 0 for correct (score >= 0.5), 1 for mispronounced (score < 0.5)."""
    from p010.data import GoPDataset

    ds = GoPDataset("train", features_dir)
    mdd = ds.mdd_label
    phn = ds.phn_score

    padding = phn < 0
    assert torch.all(mdd[padding] == -1), "Padding positions must have MDD label -1"

    valid = phn >= 0
    mispron = phn[valid] < 0.5
    assert torch.all(mdd[valid][mispron] == 1), "Score < 0.5 → MDD must be 1 (mispronounced)"
    assert torch.all(mdd[valid][~mispron] == 0), "Score >= 0.5 → MDD must be 0 (correct)"


def test_make_loaders(features_dir: Path) -> None:
    from p010.data import make_loaders
    from p010.ssl_features import SSL_MODEL_KEYS, ssl_feature_dim

    train_loader, test_loader = make_loaders(features_dir, batch_size=4, num_workers=0)
    batch = next(iter(train_loader))
    assert len(batch) == 11, f"Expected 11-tuple from DataLoader, got {len(batch)}"
    gop, ssl, *_ = batch
    assert gop.shape == (4, 50, 84), f"GOP batch shape: {gop.shape}"
    assert ssl.shape == (4, 50, ssl_feature_dim(SSL_MODEL_KEYS)), f"SSL batch shape: {ssl.shape}"


def test_make_loaders_with_ssl_subset(features_dir: Path) -> None:
    from p010.data import make_loaders

    train_loader, _ = make_loaders(features_dir, batch_size=4, num_workers=0, ssl_model_keys=("wavlm",))
    batch = next(iter(train_loader))
    _, ssl, *_ = batch
    assert ssl.shape == (4, 50, 1024), f"Subset SSL batch shape: {ssl.shape}"
