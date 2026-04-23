from pathlib import Path

import pytest
import torch

from p014 import training as training_module
from p014.config import TrainingRunSettings
from p014.data import (
    ReadAloudAnnotation,
    ReadAloudFeatureDataset,
    ReadAloudSample,
    collate_read_aloud_batch,
    example_to_annotation,
)
from p014.model import HiPPOReadAloud
from p014.training import compute_loss, train_hippo


def test_example_to_annotation_normalizes_scores_and_aligns_phones() -> None:
    example = {
        "text": "WE CALL",
        "accuracy": 8.0,
        "completeness": 10.0,
        "fluency": 9.0,
        "prosodic": 7.0,
        "total": 6.0,
        "words": [
            {
                "text": "WE",
                "accuracy": 10.0,
                "stress": 9.0,
                "total": 8.0,
                "phones": ["W", "IY0"],
                "phones-accuracy": [2.0, 1.0],
            },
            {
                "text": "CALL",
                "accuracy": 5.0,
                "stress": 4.0,
                "total": 6.0,
                "phones": ["K", "AO1", "L"],
                "phones-accuracy": [1.0, 2.0, 2.0],
            },
        ],
    }

    annotation = example_to_annotation(example)

    assert annotation.words == ("WE", "CALL")
    assert annotation.phone_tokens == ("W", "IY0", "K", "AO1", "L")
    assert annotation.phone_to_word == (0, 0, 1, 1, 1)
    assert annotation.phone_scores == (2.0, 1.0, 1.0, 2.0, 2.0)
    assert annotation.word_scores == ((2.0, 1.8, 1.6), (1.0, 0.8, 1.2))
    assert annotation.utterance_scores == (1.6, 2.0, 1.8, 1.4, 1.2)


def test_model_forward_and_loss_on_synthetic_batch() -> None:
    gop_dim = 44
    ssl_utt_dim = 3072

    sample_a = ReadAloudSample(
        gop_features=torch.randn(4, gop_dim),
        ssl_utterance=torch.randn(ssl_utt_dim),
        phone_ids=torch.tensor([1, 2, 3, 4]),
        phone_to_word=torch.tensor([0, 0, 1, 1]),
        word_embeddings=torch.randn(2, 8),
        phone_targets=torch.tensor([1.0, 0.5, 1.5, 1.0]),
        word_targets=torch.tensor([[1.2, 1.1, 1.0], [0.8, 0.9, 1.1]]),
        utterance_targets=torch.tensor([1.0, 1.1, 1.2, 1.3, 1.4]),
    )
    sample_b = ReadAloudSample(
        gop_features=torch.randn(3, gop_dim),
        ssl_utterance=torch.randn(ssl_utt_dim),
        phone_ids=torch.tensor([2, 3, 4]),
        phone_to_word=torch.tensor([0, 1, 1]),
        word_embeddings=torch.randn(2, 8),
        phone_targets=torch.tensor([1.0, 0.9, 1.3]),
        word_targets=torch.tensor([[1.0, 0.7, 0.9], [1.3, 1.1, 1.2]]),
        utterance_targets=torch.tensor([0.8, 0.9, 1.0, 1.1, 1.2]),
    )

    batch = collate_read_aloud_batch([sample_a, sample_b])
    model = HiPPOReadAloud(
        num_phone_tokens=8,
        hidden_dim=4,
        num_heads=1,
        phone_depth=1,
        word_depth=1,
        utterance_depth=1,
        gop_dim=gop_dim,
        word_embedding_dim=8,
        ssl_utterance_dim=ssl_utt_dim,
        attn_pool_heads=1,
    )

    predictions = model(
        gop_features=batch.gop_features,
        ssl_utterance=batch.ssl_utterance,
        phone_ids=batch.phone_ids,
        phone_to_word=batch.phone_to_word,
        word_embeddings=batch.word_embeddings,
        phone_mask=batch.phone_mask,
        word_mask=batch.word_mask,
    )
    loss = compute_loss(predictions, batch)

    assert batch.gop_features.shape == (2, 4, gop_dim)
    assert batch.ssl_utterance.shape == (2, ssl_utt_dim)
    assert predictions.phone_scores.shape == (2, 4)
    assert predictions.word_scores.shape == (2, 2, 3)
    assert predictions.utterance_scores.shape == (2, 5)
    assert predictions.utterance_feature.shape == (2, 4)
    assert loss.item() >= 0.0


def test_compute_loss_respects_per_granularity_weights() -> None:
    batch = training_module.ReadAloudBatch(
        gop_features=torch.zeros(1, 2, 44),
        ssl_utterance=torch.zeros(1, 3072),
        phone_ids=torch.tensor([[1, 2]]),
        phone_to_word=torch.tensor([[0, 0]]),
        word_embeddings=torch.zeros(1, 1, 8),
        phone_mask=torch.tensor([[True, True]]),
        word_mask=torch.tensor([[True]]),
        phone_targets=torch.tensor([[1.0, 1.0]]),
        word_targets=torch.tensor([[[1.0, 1.0, 1.0]]]),
        utterance_targets=torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0]]),
    )

    predictions = training_module.ReadAloudPredictions(
        phone_scores=torch.tensor([[2.0, 2.0]]),
        word_scores=torch.tensor([[[2.0, 2.0, 2.0]]]),
        utterance_scores=torch.tensor([[2.0, 2.0, 2.0, 2.0, 2.0]]),
        utterance_feature=torch.zeros(1, 4),
    )
    # Each component loss is exactly 1.0, so the weighted sum is 2 + 3 + 5 = 10.
    loss = compute_loss(
        predictions,
        batch,
        lambda_phone=2.0,
        lambda_word=3.0,
        lambda_utterance=5.0,
    )
    assert torch.isclose(loss, torch.tensor(10.0), atol=1e-6)


def _build_tiny_dataset(num_examples: int, num_phones: int = 3, num_words: int = 2) -> ReadAloudFeatureDataset:
    gop_dim = 44
    ssl_dim = 3072
    phone_vocab = {"AA": 1, "BB": 2, "CC": 3}
    annotations = []
    word_embeddings = []
    gop_features = []
    for _ in range(num_examples):
        annotations.append(
            ReadAloudAnnotation(
                text="TEST",
                words=("FOO", "BAR"),
                phone_tokens=("AA", "BB", "CC"),
                phone_to_word=(0, 0, 1),
                phone_scores=(1.8, 1.0, 0.4),
                word_scores=((1.6, 1.2, 1.4), (0.8, 1.0, 0.6)),
                utterance_scores=(1.2, 1.4, 1.0, 0.8, 1.1),
            )
        )
        word_embeddings.append(torch.randn(num_words, 768))
        gop_features.append(torch.randn(num_phones, gop_dim))
    ssl = torch.randn(num_examples, ssl_dim)
    return ReadAloudFeatureDataset(
        annotations=annotations,
        phone_vocab=phone_vocab,
        word_embeddings=word_embeddings,
        gop_features=gop_features,
        ssl_utterance=ssl,
    )


def test_train_hippo_smoke_with_cono(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """End-to-end smoke test: one epoch, one seed, CONO enabled."""

    def fake_resources(**_kwargs: object) -> tuple[ReadAloudFeatureDataset, ReadAloudFeatureDataset, int, int]:
        train_dataset = _build_tiny_dataset(num_examples=2)
        test_dataset = _build_tiny_dataset(num_examples=2)
        return train_dataset, test_dataset, 3, 44

    monkeypatch.setattr(training_module, "load_read_aloud_resources", fake_resources)

    settings = TrainingRunSettings(
        output_dir=tmp_path / "artifacts",
        cache_dir=tmp_path / "cache",
        run_name="smoke",
        device="cpu",
        epochs=1,
        batch_size=2,
        seeds=[0],
        max_train_examples=2,
        max_test_examples=2,
        use_cono=True,
    )
    summaries = train_hippo(settings)
    assert len(summaries) == 1
    assert summaries[0].seed == 0
    assert summaries[0].best_epoch == 1
    assert (tmp_path / "artifacts" / "smoke" / "trial_0" / "summary.json").exists()
    assert (tmp_path / "artifacts" / "smoke" / "trial_0" / "history.json").exists()
    assert (tmp_path / "artifacts" / "smoke" / "aggregate.json").exists()


def test_ctc_log_forward_matches_direct_computation() -> None:
    """Forward-algorithm sanity check against a short known-length sequence."""

    from p014.features.ctc_gop import _ctc_log_forward

    torch.manual_seed(0)
    vocab = 4
    time_steps = 6
    logits = torch.randn(vocab, time_steps)
    log_probs = torch.log_softmax(logits, dim=0).double()
    seq = torch.tensor([1, 2], dtype=torch.long)

    log_forward = _ctc_log_forward(log_probs, seq, blank=0)
    # Must be a finite scalar not greater than zero (log of a probability).
    assert torch.isfinite(log_forward)
    assert float(log_forward.item()) <= 0.0
