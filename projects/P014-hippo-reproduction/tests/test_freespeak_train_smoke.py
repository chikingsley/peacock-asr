"""End-to-end smoke test for the free-speaking + curriculum training path.

Monkey-patches both :func:`load_read_aloud_resources` and
:func:`load_freespeak_resources` to avoid loading SpeechOcean762 / Whisper and
runs one epoch of training on CPU with a tiny synthetic dataset.
"""

from pathlib import Path

import pytest
import torch

from p014 import training as training_module
from p014.config import TrainingRunSettings
from p014.data import ReadAloudAnnotation, ReadAloudFeatureDataset
from p014.training import train_hippo


def _build_tiny_dataset(
    num_examples: int, phone_vocab: dict[str, int]
) -> ReadAloudFeatureDataset:
    gop_dim = 44
    ssl_dim = 3072
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
        word_embeddings.append(torch.randn(2, 768))
        gop_features.append(torch.randn(3, gop_dim))
    ssl = torch.randn(num_examples, ssl_dim)
    return ReadAloudFeatureDataset(
        annotations=annotations,
        phone_vocab=phone_vocab,
        word_embeddings=word_embeddings,
        gop_features=gop_features,
        ssl_utterance=ssl,
    )


def test_train_hippo_freespeak_curriculum_smoke(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    shared_phone_vocab = {"AA": 1, "BB": 2, "CC": 3}

    def fake_read_resources(
        **_kwargs: object,
    ) -> tuple[ReadAloudFeatureDataset, ReadAloudFeatureDataset, int, int]:
        return (
            _build_tiny_dataset(num_examples=2, phone_vocab=shared_phone_vocab),
            _build_tiny_dataset(num_examples=2, phone_vocab=shared_phone_vocab),
            3,
            44,
        )

    def fake_free_resources(
        **_kwargs: object,
    ) -> tuple[ReadAloudFeatureDataset, ReadAloudFeatureDataset, int, int]:
        return (
            _build_tiny_dataset(num_examples=2, phone_vocab=shared_phone_vocab),
            _build_tiny_dataset(num_examples=2, phone_vocab=shared_phone_vocab),
            3,
            44,
        )

    monkeypatch.setattr(
        training_module, "load_read_aloud_resources", fake_read_resources
    )
    monkeypatch.setattr(
        training_module, "load_freespeak_resources", fake_free_resources
    )

    settings = TrainingRunSettings(
        output_dir=tmp_path / "artifacts",
        cache_dir=tmp_path / "cache",
        run_name="freespeak-smoke",
        device="cpu",
        epochs=1,
        batch_size=2,
        seeds=[0],
        max_train_examples=2,
        max_test_examples=2,
        scenario="free_speaking",
        use_curriculum=True,
        use_cono=True,
    )
    summaries = train_hippo(settings)
    assert len(summaries) == 1
    assert summaries[0].seed == 0
    assert summaries[0].best_epoch == 1
    trial_dir = tmp_path / "artifacts" / "freespeak-smoke" / "trial_0"
    assert (trial_dir / "summary.json").exists()
    assert (trial_dir / "history.json").exists()
    assert (tmp_path / "artifacts" / "freespeak-smoke" / "aggregate.json").exists()
