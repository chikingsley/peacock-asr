from __future__ import annotations

import io
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

AUDIO_CHANNEL_DIMENSIONS = 2


@dataclass(frozen=True)
class Example:
    row_index: int
    audio: np.ndarray
    sampling_rate: int
    reference: str

    @property
    def audio_seconds(self) -> float:
        return len(self.audio) / self.sampling_rate


def _raw_rows(path: Path) -> Iterator[tuple[bytes, str]]:
    parquet = pq.ParquetFile(path)
    columns = set(parquet.schema_arrow.names)
    if {"audio_bytes", "normalized_text"}.issubset(columns):
        audio_column, text_column = "audio_bytes", "normalized_text"
    elif {"audio", "transcription"}.issubset(columns):
        audio_column, text_column = "audio", "transcription"
    else:
        raise ValueError(f"unsupported benchmark schema in {path}: {sorted(columns)}")

    for batch in parquet.iter_batches(batch_size=128, columns=[audio_column, text_column]):
        for audio, text in zip(
            batch.column(0).to_pylist(), batch.column(1).to_pylist(), strict=True
        ):
            raw_audio = audio["bytes"] if isinstance(audio, dict) else audio
            yield raw_audio, str(text)


def examples(path: Path, *, offset: int = 0, limit: int = 0) -> Iterator[Example]:
    yielded = 0
    for row_index, (audio_bytes, reference) in enumerate(_raw_rows(path)):
        if row_index < offset:
            continue
        if limit and yielded >= limit:
            return
        audio, sampling_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
        if audio.ndim == AUDIO_CHANNEL_DIMENSIONS:
            audio = audio.mean(axis=1)
        yield Example(
            row_index=row_index,
            audio=np.asarray(audio, dtype=np.float32),
            sampling_rate=int(sampling_rate),
            reference=reference,
        )
        yielded += 1
