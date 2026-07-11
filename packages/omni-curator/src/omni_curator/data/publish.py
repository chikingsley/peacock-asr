"""Publish a store subset as a Hugging Face audio dataset (the community-set exporter).

The training exporter (:mod:`omni_curator.data.export`) writes the gated omni-parquet a model
trains on. This one writes the OTHER artifact worth sharing: a store subset in the standard
HF audio layout (``audio`` struct column + metadata columns, sharded parquet under ``data/``),
with the verification scores INCLUDED so consumers pick their own quality threshold instead
of inheriting ours. Typical use: every language-gated YouTube clip, no WER gate.

Rows stream through with bounded memory (one shard's FLAC bytes at a time), so a 1,000-hour
corpus exports on a small machine. Upload the resulting folder with
``HfApi.upload_large_folder`` and a README whose YAML declares the ``audio`` feature.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq

from omni_curator.audit.quality import is_descriptor_only
from omni_curator.process.language import keep_for_language

if TYPE_CHECKING:
    from collections.abc import Iterator

    from omni_curator.data.sample import Sample
    from omni_curator.data.store import CuratorStore

#: The HF audio-dataset schema: ``audio`` is the Hub's struct encoding of an Audio feature.
HF_AUDIO_SCHEMA: pa.Schema = pa.schema(
    [
        ("audio", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
        ("text", pa.string()),
        ("channel", pa.string()),
        ("video_id", pa.string()),
        ("clip_id", pa.string()),
        ("duration", pa.float64()),
        ("scribe_wer", pa.float64()),
        ("scribe_cer", pa.float64()),
        ("citation", pa.string()),
    ]
)


@dataclass
class PublishStats:
    """What the export wrote, plus what it skipped and why."""

    rows: int = 0
    hours: float = 0.0
    shards: int = 0
    skipped: dict[str, int] = field(default_factory=dict)

    def skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1


def _select(
    store: CuratorStore, *, source_prefix: str, language: str, stats: PublishStats
) -> Iterator[Sample]:
    """Yield publishable rows: language-gated, junk-free, audio present. NO quality gate."""
    for sample in store.iter_samples():
        if not sample.source.startswith(source_prefix):
            continue
        if is_descriptor_only(sample.text):  # '[outro jingle]' — non-speech, nothing to filter ON
            stats.skip("descriptor_only")
            continue
        if not keep_for_language(sample.text, language):
            stats.skip("language_gate")
            continue
        if not Path(sample.audio_path).exists():
            stats.skip("missing_audio")
            continue
        yield sample


def _shard_table(samples: list[Sample], source_prefix: str) -> pa.Table:
    rows = {name: [] for name in HF_AUDIO_SCHEMA.names}
    for s in samples:
        path = Path(s.audio_path)
        rows["audio"].append({"bytes": path.read_bytes(), "path": f"{s.id}.flac"})
        rows["text"].append(s.text)
        rows["channel"].append(s.source.removeprefix(source_prefix))
        rows["video_id"].append(s.id.rsplit("_", 1)[0])
        rows["clip_id"].append(s.id)
        rows["duration"].append(s.duration)
        rows["scribe_wer"].append(s.scribe_wer)
        rows["scribe_cer"].append(s.scribe_cer)
        rows["citation"].append(s.citation or "")
    return pa.Table.from_pydict(rows, schema=HF_AUDIO_SCHEMA)


def export_hf_audio_dataset(
    store: CuratorStore,
    out_dir: Path,
    *,
    source_prefix: str = "youtube-",
    language: str,
    rows_per_shard: int = 2_000,
) -> PublishStats:
    """Write the HF audio dataset shards to ``out_dir/data/``; return what happened.

    Includes every ``source_prefix`` row that is in-language and carries real speech —
    quality scores ride along as columns, deliberately ungated. Refuses to write into a
    dir that already has shards (no silently mixed exports).
    """
    data_dir = out_dir / "data"
    if any(data_dir.glob("*.parquet")):
        msg = f"shards already present under {data_dir}"
        raise FileExistsError(msg)
    data_dir.mkdir(parents=True, exist_ok=True)

    stats = PublishStats()
    batch: list[Sample] = []
    shards: list[Path] = []

    def flush() -> None:
        if not batch:
            return
        shard = data_dir / f"train-{len(shards):05d}.parquet"
        pq.write_table(_shard_table(batch, source_prefix), shard)
        shards.append(shard)
        stats.rows += len(batch)
        stats.hours += sum(s.duration for s in batch) / 3600
        batch.clear()

    for sample in _select(store, source_prefix=source_prefix, language=language, stats=stats):
        batch.append(sample)
        if len(batch) >= rows_per_shard:
            flush()
    flush()

    # train-00012.parquet -> train-00012-of-00040.parquet (the Hub's sharding convention).
    total = len(shards)
    for i, shard in enumerate(shards):
        shard.rename(data_dir / f"train-{i:05d}-of-{total:05d}.parquet")
    stats.shards = total
    return stats
