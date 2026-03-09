"""Lazy manifest-backed dataset helpers for the canonical trainer."""

from __future__ import annotations

import gzip
import importlib
import json
import math
import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from p004_training_from_scratch.canonical.common import (
    DEFAULT_NUM_MELS,
    load_log_mel_features,
)


@dataclass(frozen=True, slots=True)
class ManifestCut:
    cut_id: str
    audio_path: Path
    phones: tuple[str, ...]
    duration_seconds: float | None


@dataclass(frozen=True, slots=True)
class PreparedExample:
    cut_id: str
    features: Any
    target_ids: tuple[int, ...]
    duration_seconds: float | None


def load_phone_token_table(tokens_path: Path) -> dict[str, int]:
    if not tokens_path.is_file():
        msg = f"token table not found: {tokens_path}"
        raise FileNotFoundError(msg)

    token_table: dict[str, int] = {}
    for line in tokens_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        token, token_id = stripped.rsplit(maxsplit=1)
        token_table[token] = int(token_id)
    if "<eps>" not in token_table:
        msg = f"blank token <eps> missing from token table: {tokens_path}"
        raise ValueError(msg)
    return token_table


def read_manifest_cuts(
    manifest_path: Path,
    *,
    limit: int | None,
    token_table: dict[str, int] | None = None,
    audio_path_probe_count: int | None = 32,
) -> list[ManifestCut]:
    normalized_limit = _normalize_limit(limit)
    if not manifest_path.is_file():
        msg = f"manifest not found: {manifest_path}"
        raise FileNotFoundError(msg)

    if manifest_path.suffix == ".gz":
        with gzip.open(manifest_path, "rt", encoding="utf-8") as handle:
            cuts = _read_manifest_cut_lines(
                handle,
                limit=normalized_limit,
                token_table=token_table,
                audio_path_probe_count=audio_path_probe_count,
            )
    else:
        with manifest_path.open("rt", encoding="utf-8") as handle:
            cuts = _read_manifest_cut_lines(
                handle,
                limit=normalized_limit,
                token_table=token_table,
                audio_path_probe_count=audio_path_probe_count,
            )

    if not cuts:
        msg = f"no usable cuts found in manifest: {manifest_path}"
        raise ValueError(msg)
    return cuts


class CanonicalManifestDataset(Dataset[PreparedExample]):
    def __init__(
        self,
        *,
        cuts: list[ManifestCut],
        token_table: dict[str, int],
    ) -> None:
        self._cuts = cuts
        self._token_table = token_table

    def __len__(self) -> int:
        return len(self._cuts)

    def __getitem__(self, index: int) -> PreparedExample:
        torch = importlib.import_module("torch")
        torchaudio = importlib.import_module("torchaudio")
        cut = self._cuts[index]
        features = load_log_mel_features(
            path=cut.audio_path,
            torch=torch,
            torchaudio=torchaudio,
            device=None,
            n_mels=DEFAULT_NUM_MELS,
        )
        return PreparedExample(
            cut_id=cut.cut_id,
            features=features,
            target_ids=tuple(self._token_table[phone] for phone in cut.phones),
            duration_seconds=cut.duration_seconds,
        )


class PreparedExampleCollator:
    def __call__(self, examples: list[PreparedExample]) -> dict[str, Any]:
        torch = importlib.import_module("torch")
        return collate_prepared_examples(examples, torch=torch)


class DurationBucketBatchSampler:
    def __init__(
        self,
        *,
        cuts: list[ManifestCut],
        batch_size: int,
        shuffle: bool,
        seed: int,
        bucket_size_multiplier: int = 50,
        drop_last: bool = False,
    ) -> None:
        if batch_size <= 0:
            msg = "batch_size must be positive"
            raise ValueError(msg)
        if bucket_size_multiplier <= 0:
            msg = "bucket_size_multiplier must be positive"
            raise ValueError(msg)

        self._cuts = cuts
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._seed = seed
        self._bucket_size = max(batch_size, batch_size * bucket_size_multiplier)
        self._drop_last = drop_last
        self._epoch = 0

    def __len__(self) -> int:
        if self._drop_last:
            return len(self._cuts) // self._batch_size
        return math.ceil(len(self._cuts) / self._batch_size)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        indices = list(range(len(self._cuts)))
        rng = random.Random(self._seed + self._epoch)
        if self._shuffle:
            rng.shuffle(indices)

        batches: list[list[int]] = []
        for start in range(0, len(indices), self._bucket_size):
            bucket = indices[start : start + self._bucket_size]
            bucket.sort(key=lambda index: self._duration_key(self._cuts[index]))
            bucket_batches = [
                bucket[offset : offset + self._batch_size]
                for offset in range(0, len(bucket), self._batch_size)
            ]
            if self._drop_last and bucket_batches and (
                len(bucket_batches[-1]) < self._batch_size
            ):
                bucket_batches.pop()
            if self._shuffle:
                rng.shuffle(bucket_batches)
            batches.extend(bucket_batches)

        if self._shuffle:
            rng.shuffle(batches)
        yield from batches

    @staticmethod
    def _duration_key(cut: ManifestCut) -> float:
        duration = cut.duration_seconds
        if duration is None or duration <= 0.0:
            return 0.0
        return duration


def collate_prepared_examples(
    examples: list[PreparedExample],
    *,
    torch: Any,
) -> dict[str, Any]:
    feature_rows = [example.features for example in examples]
    input_lengths = torch.tensor(
        [int(feature.shape[0]) for feature in feature_rows],
        dtype=torch.long,
    )
    target_lengths = torch.tensor(
        [len(example.target_ids) for example in examples],
        dtype=torch.long,
    )
    pairs = zip(
        input_lengths.tolist(),
        target_lengths.tolist(),
        [example.cut_id for example in examples],
        strict=True,
    )
    for input_length, target_length, cut_id in pairs:
        if int(target_length) >= int(input_length):
            msg = (
                f"CTC target sequence is longer than its input sequence for {cut_id}: "
                f"target_length={target_length}, input_length={input_length}"
            )
            raise ValueError(msg)

    flat_targets = torch.tensor(
        [token for example in examples for token in example.target_ids],
        dtype=torch.long,
    )
    if flat_targets.numel() == 0:
        msg = "CTC targets cannot be empty"
        raise ValueError(msg)

    padded_features = torch.nn.utils.rnn.pad_sequence(feature_rows, batch_first=True)
    return {
        "cut_ids": [example.cut_id for example in examples],
        "features": padded_features,
        "input_lengths": input_lengths,
        "targets": flat_targets,
        "target_lengths": target_lengths,
    }


def _read_manifest_cut_lines(
    lines: Iterable[str],
    *,
    limit: int | None,
    token_table: dict[str, int] | None,
    audio_path_probe_count: int | None,
) -> list[ManifestCut]:
    cuts: list[ManifestCut] = []
    remaining_audio_path_probes = audio_path_probe_count
    for raw_line in lines:
        payload = json.loads(raw_line)
        phones = tuple(
            phone for phone in str(payload["supervisions"][0]["text"]).split() if phone
        )
        if not phones:
            continue

        source = Path(str(payload["recording"]["sources"][0]["source"]))
        should_probe_path = remaining_audio_path_probes is None or (
            remaining_audio_path_probes > 0
        )
        if should_probe_path:
            if not source.is_file():
                msg = f"manifest listed missing audio file: {source}"
                raise FileNotFoundError(msg)
            if remaining_audio_path_probes is not None:
                remaining_audio_path_probes -= 1

        if token_table is not None:
            unknown = sorted({phone for phone in phones if phone not in token_table})
            if unknown:
                msg = f"manifest contains unknown phones: {', '.join(unknown)}"
                raise ValueError(msg)

        cuts.append(
            ManifestCut(
                cut_id=str(payload["id"]),
                audio_path=source,
                phones=phones,
                duration_seconds=_coerce_optional_float(payload.get("duration")),
            )
        )
        if limit is not None and len(cuts) == limit:
            break
    return cuts


def _normalize_limit(limit: int | None) -> int | None:
    if limit is None or limit <= 0:
        return None
    return limit


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


__all__ = [
    "CanonicalManifestDataset",
    "DurationBucketBatchSampler",
    "ManifestCut",
    "PreparedExample",
    "PreparedExampleCollator",
    "collate_prepared_examples",
    "load_phone_token_table",
    "read_manifest_cuts",
]
