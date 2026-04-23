from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from p014.features.ctc_gop import (
    FeatureExtractionSettings,
    extract_gop_for_split,
    merge_gop_shards,
)


class _FakeDataset:
    def __init__(self, examples: list[dict[str, Any]]) -> None:
        self._examples = examples

    def cast_column(self, _name: str, _audio: object) -> "_FakeDataset":
        return self

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._examples[index]


class _FakeTokenizer:
    @classmethod
    def from_pretrained(cls, _model_id: str) -> "_FakeTokenizer":
        return cls()

    def get_vocab(self) -> dict[str, int]:
        return {"<pad>": 0, "aa": 1}


class _FakeProcessor:
    @classmethod
    def from_pretrained(cls, _model_id: str) -> "_FakeProcessor":
        return cls()


class _FakeModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(vocab_size=2)

    @classmethod
    def from_pretrained(cls, _model_id: str) -> "_FakeModel":
        return cls()

    def to(self, _device: torch.device) -> "_FakeModel":
        return self

    def eval(self) -> "_FakeModel":
        return self


def test_extract_gop_for_split_resumes_from_per_example_parts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    examples = [
        {
            "id": "utt-0",
            "audio": {"array": [0.0], "sampling_rate": 16_000},
            "words": [{"phones": ["AA1"]}],
        },
        {
            "id": "utt-1",
            "audio": {"array": [0.0], "sampling_rate": 16_000},
            "words": [{"phones": ["AA1", "AA1"]}],
        },
        {
            "id": "utt-2",
            "audio": {"array": [0.0], "sampling_rate": 16_000},
            "words": [{"phones": ["AA1"]}],
        },
    ]
    fake_dataset = _FakeDataset(examples)
    fake_transformers = SimpleNamespace(
        Wav2Vec2CTCTokenizer=_FakeTokenizer,
        Wav2Vec2Processor=_FakeProcessor,
        Wav2Vec2ForCTC=_FakeModel,
    )

    monkeypatch.setattr("p014.features.ctc_gop.load_dataset", lambda *_args, **_kwargs: fake_dataset)
    monkeypatch.setattr("p014.features.ctc_gop.import_module", lambda _name: fake_transformers)
    monkeypatch.setattr("p014.features.ctc_gop._load_audio", lambda _payload: torch.zeros(16))
    monkeypatch.setattr("p014.features.ctc_gop._resolve_model_source", lambda _cache_dir, model_id: model_id)

    call_state = {"count": 0}

    def fake_compute(
        waveform: torch.Tensor,
        canonical_ids: list[int],
        processor: object,
        model: object,
        device: torch.device,
        blank_id: int,
    ) -> torch.Tensor:
        del waveform, processor, model, device, blank_id
        current = call_state["count"]
        call_state["count"] += 1
        if current == 2:
            raise RuntimeError("synthetic interruption")
        return torch.full((len(canonical_ids), 3), float(current + 1))

    monkeypatch.setattr("p014.features.ctc_gop._compute_gop_for_utterance", fake_compute)

    settings = FeatureExtractionSettings(model_id="test-model", progress=False)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        extract_gop_for_split(
            split="train",
            cache_dir=tmp_path,
            dataset_id="fake-dataset",
            device=torch.device("cpu"),
            settings=settings,
        )

    parts_dir = tmp_path / "features" / "gop" / "train.parts"
    assert parts_dir.exists()
    assert (parts_dir / "000000.pt").exists()
    assert (parts_dir / "000001.pt").exists()
    assert not (parts_dir / "000002.pt").exists()

    def fake_compute_resume(
        waveform: torch.Tensor,
        canonical_ids: list[int],
        processor: object,
        model: object,
        device: torch.device,
        blank_id: int,
    ) -> torch.Tensor:
        del waveform, processor, model, device, blank_id
        return torch.full((len(canonical_ids), 3), 9.0)

    monkeypatch.setattr("p014.features.ctc_gop._compute_gop_for_utterance", fake_compute_resume)
    cache_path = extract_gop_for_split(
        split="train",
        cache_dir=tmp_path,
        dataset_id="fake-dataset",
        device=torch.device("cpu"),
        settings=settings,
    )

    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    assert payload["gop_dim"] == 3
    assert payload["utterance_ids"] == ["utt-0", "utt-1", "utt-2"]
    assert tuple(payload["features"][0].shape) == (1, 3)
    assert tuple(payload["features"][1].shape) == (2, 3)
    assert tuple(payload["features"][2].shape) == (1, 3)
    assert torch.allclose(payload["features"][0], torch.full((1, 3), 1.0))
    assert torch.allclose(payload["features"][1], torch.full((2, 3), 2.0))
    assert torch.allclose(payload["features"][2], torch.full((1, 3), 9.0))
    assert not parts_dir.exists()
    assert not (tmp_path / "features" / "gop" / "train.parts.json").exists()


def test_extract_gop_shards_merge_into_standard_full_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    examples = [
        {
            "id": "utt-0",
            "audio": {"array": [0.0], "sampling_rate": 16_000},
            "words": [{"phones": ["AA1"]}],
        },
        {
            "id": "utt-1",
            "audio": {"array": [1.0], "sampling_rate": 16_000},
            "words": [{"phones": ["AA1", "AA1"]}],
        },
        {
            "id": "utt-2",
            "audio": {"array": [2.0], "sampling_rate": 16_000},
            "words": [{"phones": ["AA1"]}],
        },
        {
            "id": "utt-3",
            "audio": {"array": [3.0], "sampling_rate": 16_000},
            "words": [{"phones": ["AA1", "AA1"]}],
        },
    ]
    fake_dataset = _FakeDataset(examples)
    fake_transformers = SimpleNamespace(
        Wav2Vec2CTCTokenizer=_FakeTokenizer,
        Wav2Vec2Processor=_FakeProcessor,
        Wav2Vec2ForCTC=_FakeModel,
    )

    monkeypatch.setattr("p014.features.ctc_gop.load_dataset", lambda *_args, **_kwargs: fake_dataset)
    monkeypatch.setattr("p014.features.ctc_gop.import_module", lambda _name: fake_transformers)
    monkeypatch.setattr(
        "p014.features.ctc_gop._load_audio",
        lambda payload: torch.as_tensor(payload["array"], dtype=torch.float32),
    )
    monkeypatch.setattr("p014.features.ctc_gop._resolve_model_source", lambda _cache_dir, model_id: model_id)

    def fake_compute(
        waveform: torch.Tensor,
        canonical_ids: list[int],
        processor: object,
        model: object,
        device: torch.device,
        blank_id: int,
    ) -> torch.Tensor:
        del processor, model, device, blank_id
        value = float(waveform[0].item() + 1.0)
        return torch.full((len(canonical_ids), 3), value)

    monkeypatch.setattr("p014.features.ctc_gop._compute_gop_for_utterance", fake_compute)

    settings = FeatureExtractionSettings(model_id="test-model", progress=False)
    shard0 = extract_gop_for_split(
        split="train",
        cache_dir=tmp_path,
        dataset_id="fake-dataset",
        device=torch.device("cpu"),
        settings=settings,
        num_shards=2,
        shard_index=0,
    )
    shard1 = extract_gop_for_split(
        split="train",
        cache_dir=tmp_path,
        dataset_id="fake-dataset",
        device=torch.device("cpu"),
        settings=settings,
        num_shards=2,
        shard_index=1,
    )

    assert shard0.name == "train__shard_000_of_002.pt"
    assert shard1.name == "train__shard_001_of_002.pt"

    merged = merge_gop_shards(
        split="train",
        cache_dir=tmp_path,
        num_shards=2,
        dataset_id="fake-dataset",
    )
    payload = torch.load(merged, map_location="cpu", weights_only=False)

    assert payload["dataset_indices"] == [0, 1, 2, 3]
    assert payload["utterance_ids"] == ["utt-0", "utt-1", "utt-2", "utt-3"]
    assert tuple(payload["features"][0].shape) == (1, 3)
    assert tuple(payload["features"][1].shape) == (2, 3)
    assert tuple(payload["features"][2].shape) == (1, 3)
    assert tuple(payload["features"][3].shape) == (2, 3)
    assert torch.allclose(payload["features"][0], torch.full((1, 3), 1.0))
    assert torch.allclose(payload["features"][1], torch.full((2, 3), 2.0))
    assert torch.allclose(payload["features"][2], torch.full((1, 3), 3.0))
    assert torch.allclose(payload["features"][3], torch.full((2, 3), 4.0))
