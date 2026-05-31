"""Omni-parquet schema + IO shared by the omni CTC fine-tune datasets.

The training data is Hive-partitioned ``version=0/corpus=<c>/split=<s>/language=<l>`` with
one parquet schema: ``text`` (string), ``audio_bytes`` (int8 list = encoded FLAC/WAV
bytes), ``audio_size`` (int64 = samples = duration*16000). The partition columns are path
-derived, not stored in-file. fairseq2's mixture reader globs the whole ``version=0`` tree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

SAMPLE_RATE = 16_000

OMNI_SCHEMA: pa.Schema = pa.schema(
    [
        ("text", pa.string()),
        ("audio_bytes", pa.list_(pa.int8())),
        ("audio_size", pa.int64()),
    ]
)


@dataclass(frozen=True, slots=True)
class OmniRow:
    text: str
    audio: npt.NDArray[np.int8]  # encoded audio bytes as int8 (FLAC/WAV)
    audio_size: int  # samples (= seconds * SAMPLE_RATE)
    corpus: str


def partition_dir(version_root: Path, corpus: str, split: str, language: str) -> Path:
    """``<version_root>/corpus=<corpus>/split=<split>/language=<language>``."""
    return version_root / f"corpus={corpus}" / f"split={split}" / f"language={language}"


def write_shard(
    texts: list[str],
    audio: list[npt.NDArray[np.int8]],
    sizes: list[int],
    out_dir: Path,
    shard_index: int,
    *,
    row_group_size: int = 100,
) -> Path:
    """Write one parquet shard (text/audio_bytes/audio_size) into ``out_dir``."""
    if not (len(texts) == len(audio) == len(sizes)):
        msg = f"mismatched column lengths: {len(texts)}/{len(audio)}/{len(sizes)}"
        raise ValueError(msg)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_arrays(
        [
            pa.array(texts, type=pa.string()),
            pa.array(audio, type=pa.list_(pa.int8())),
            pa.array(sizes, type=pa.int64()),
        ],
        schema=OMNI_SCHEMA,
    )
    path = out_dir / f"part-{shard_index:05d}.parquet"
    pq.write_table(table, path, row_group_size=row_group_size)
    return path


def iter_split(
    version_root: Path,
    split: str,
    *,
    max_duration: float | None = None,
) -> Iterator[OmniRow]:
    """Yield rows from every ``corpus=*/split=<split>/language=*`` partition.

    ``max_duration`` (seconds) drops longer clips — e.g. the omni inference pipeline caps
    audio at 40s, so pass ``max_duration=40`` when reading a split for evaluation.
    """
    matches = version_root.glob(f"corpus=*/split={split}/language=*/*.parquet")
    for path in sorted(matches):
        corpus = next(p.split("=", 1)[1] for p in path.parts if p.startswith("corpus="))
        table = pq.read_table(path, columns=["text", "audio_bytes", "audio_size"])
        texts = table.column("text").to_pylist()
        blobs = table.column("audio_bytes").to_pylist()
        sizes = table.column("audio_size").to_pylist()
        for text, blob, size in zip(texts, blobs, sizes, strict=True):
            if max_duration is not None and size / SAMPLE_RATE > max_duration:
                continue
            yield OmniRow(
                text=text,
                audio=np.asarray(blob, dtype=np.int8),
                audio_size=size,
                corpus=corpus,
            )
