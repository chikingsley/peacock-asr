"""Materialize an ablation: ``CuratorStore`` -> omni-parquet dataset (closes the loop).

The store is the master pool; a *dataset version* is a recipe over it, not a copy (see
``CURATING.md``). This module is where that recipe is applied: a :class:`Selection` filters the
pool (by source / split / language, with optional duration and per-source caps) and the surviving
:class:`~omni_curator.sample.Sample`s are written as the SAME omni-parquet layout that
``omni-finetune-core`` trains on.

Layout (Hive-partitioned, matching ``omni_finetune_core.parquet`` /
``tajik_omnilingual_asr.dataset_prep.omni_parquet``)::

    <output_dir>/
      version=<N>/
        corpus=<source>/split=<split>/language=<lang>/part-00000.parquet   # text/audio/size
      language_distribution_<N>.tsv                                         # corpus<TAB>lang<TAB>h
      export_summary.json                                                   # recipe provenance

Each row is ``text`` (the label), ``audio_bytes`` (the clip's 16 kHz **FLAC bytes** as an int8
list), and ``audio_size`` (samples = ``round(duration * 16000)``, matching how the tajik/core
exporters compute it). The partition columns are path-derived, not stored in-file — fairseq2's
mixture reader globs the whole ``version=N`` tree and weights each corpus by the TSV hours.

Call it once per ablation: ``export_dataset(store, root / "datasets" / "v0", version=0)`` for
"everything", then ``export_dataset(store, root / "datasets" / "v1", version=0,
selection=Selection(sources=["fleurs", "commonvoice"], max_duration_seconds=30))`` for a subset.
The ``version=N`` partition is a fairseq2 layout requirement; the *ablation* axis is the
``datasets/vN`` directory, exactly as the tajik exporter versions by artifact dir.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from omni_curator.sample import Sample
    from omni_curator.store.sqlite import CuratorStore

SAMPLE_RATE = 16_000

#: The exact schema ``omni-finetune-core`` trains on (``omni_finetune_core.parquet.OMNI_SCHEMA``).
OMNI_SCHEMA: pa.Schema = pa.schema(
    [
        ("text", pa.string()),
        ("audio_bytes", pa.list_(pa.int8())),
        ("audio_size", pa.int64()),
    ]
)


@dataclass
class Selection:
    """A recipe: which store rows go into this ablation.

    All filters are AND-ed. ``sources`` / ``splits`` / ``languages`` keep only the listed values
    (``None`` = keep all). ``max_duration_seconds`` drops longer clips (the omni pipeline caps
    audio at 40 s, so an eval ablation often sets 40). ``max_per_source`` caps how many clips each
    source contributes, in store order, so a big corpus can't swamp a small one before the mixture
    weighting even runs.
    """

    sources: list[str] | None = None
    splits: list[str] | None = None
    languages: list[str] | None = None
    max_duration_seconds: float | None = None
    max_per_source: int | None = None

    def keeps(self, sample: Sample) -> bool:
        """Whether ``sample`` passes the value/duration filters (per-source cap handled later)."""
        if self.sources is not None and sample.source not in self.sources:
            return False
        if self.splits is not None and sample.split not in self.splits:
            return False
        if self.languages is not None and sample.language not in self.languages:
            return False
        return not (
            self.max_duration_seconds is not None
            and sample.duration > self.max_duration_seconds
        )


@dataclass
class ExportStats:
    """What an export produced — counts + hours per split/corpus, for the summary and the caller."""

    version: int
    output_dir: str
    rows: int = 0
    rows_by_split: dict[str, int] = field(default_factory=dict)
    rows_by_corpus: dict[str, int] = field(default_factory=dict)
    hours_by_split: dict[str, float] = field(default_factory=dict)
    hours_by_corpus: dict[str, float] = field(default_factory=dict)
    skipped_missing_audio: int = 0


def _sanitize(value: str) -> str:
    """Make a partition value path-safe (Hive ``key=value`` dirs can't contain ``/`` or ``=``)."""
    return value.replace("/", "_").replace("=", "_").replace(" ", "_")


def partition_dir(version_root: Path, corpus: str, split: str, language: str) -> Path:
    """``<version_root>/corpus=<corpus>/split=<split>/language=<language>`` (values sanitized)."""
    return (
        version_root
        / f"corpus={_sanitize(corpus)}"
        / f"split={_sanitize(split)}"
        / f"language={_sanitize(language)}"
    )


def _flac_bytes(audio_path: Path) -> np.ndarray:
    """Read ``audio_path`` as 16 kHz mono and return its re-encoded FLAC bytes as an int8 array.

    Re-encoding (rather than slurping the file verbatim) guarantees the stored bytes decode to
    16 kHz mono regardless of the source clip's container/rate, matching what the trainer expects.
    """
    import soundfile as sf

    samples, rate = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if rate != SAMPLE_RATE:
        import librosa

        samples = librosa.resample(samples, orig_sr=rate, target_sr=SAMPLE_RATE)
    import io

    buffer = io.BytesIO()
    sf.write(buffer, samples, SAMPLE_RATE, format="FLAC")
    return np.frombuffer(buffer.getvalue(), dtype=np.int8)


def _select(store: CuratorStore, selection: Selection) -> Iterator[Sample]:
    """Yield the samples kept by ``selection``, applying the per-source cap in store order."""
    seen: dict[str, int] = defaultdict(int)
    for sample in store.iter_samples():
        if not selection.keeps(sample):
            continue
        if (
            selection.max_per_source is not None
            and seen[sample.source] >= selection.max_per_source
        ):
            continue
        seen[sample.source] += 1
        yield sample


def _write_partition(
    samples: list[Sample],
    out_dir: Path,
    *,
    row_group_size: int,
) -> tuple[int, float, int]:
    """Write one partition's samples to ``part-00000.parquet``; return (rows, hours, skipped)."""
    texts: list[str] = []
    audio: list[np.ndarray] = []
    sizes: list[int] = []
    skipped = 0
    for sample in samples:
        path = Path(sample.audio_path)
        if not path.exists():
            skipped += 1
            continue
        texts.append(sample.text)
        audio.append(_flac_bytes(path))
        sizes.append(max(1, round(sample.duration * SAMPLE_RATE)))
    if not texts:
        return 0, 0.0, skipped
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_arrays(
        [
            pa.array(texts, type=pa.string()),
            pa.array(audio, type=pa.list_(pa.int8())),
            pa.array(sizes, type=pa.int64()),
        ],
        schema=OMNI_SCHEMA,
    )
    pq.write_table(table, out_dir / "part-00000.parquet", row_group_size=row_group_size)
    hours = sum(sizes) / SAMPLE_RATE / 3600
    return len(texts), hours, skipped


def write_language_distribution(version_root: Path, out_path: Path) -> Path:
    """Write ``corpus<TAB>language<TAB>hours`` over ``version_root`` (the mixture summary).

    Matches ``omni_finetune_core.mixture.write_language_distribution``: hours are total audio per
    ``(corpus, language)`` across all splits, from the ``audio_size`` column.
    """
    samples: dict[tuple[str, str], int] = defaultdict(int)
    for path in sorted(version_root.glob("corpus=*/split=*/language=*/*.parquet")):
        parts = {p.split("=", 1)[0]: p.split("=", 1)[1] for p in path.parts if "=" in p}
        key = (parts["corpus"], parts["language"])
        sizes = pq.read_table(path, columns=["audio_size"]).column("audio_size").to_pylist()
        samples[key] += sum(sizes)
    lines = ["corpus\tlanguage\thours"]
    lines.extend(
        f"{corpus}\t{language}\t{samples[corpus, language] / SAMPLE_RATE / 3600:.8f}"
        for corpus, language in sorted(samples)
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def export_dataset(
    store: CuratorStore,
    output_dir: Path,
    *,
    version: int = 0,
    selection: Selection | None = None,
    row_group_size: int = 100,
) -> ExportStats:
    """Materialize the ``selection`` over ``store`` as omni-parquet under ``output_dir``.

    ``output_dir`` is the ablation dir (e.g. ``datasets/v0``); the parquet tree lives at
    ``output_dir/version=<version>`` and the mixture TSV at
    ``output_dir/language_distribution_<version>.tsv``. ``version`` is the in-file fairseq2
    partition axis (almost always 0); the *ablation* axis is ``output_dir`` itself. Returns the
    :class:`ExportStats` (also written to ``export_summary.json``).

    Re-exporting into an existing non-empty ``version=N`` tree raises — pick a fresh ablation dir
    or remove the old tree first, so an ablation is never silently mixed with a previous run.
    """
    selection = selection or Selection()
    version_root = output_dir / f"version={version}"
    existing = next(version_root.glob("corpus=*/split=*/language=*/*.parquet"), None)
    if existing is not None:
        msg = f"version=N tree already populated at {version_root} (found {existing.name})"
        raise FileExistsError(msg)

    grouped: dict[tuple[str, str, str], list[Sample]] = defaultdict(list)
    for sample in _select(store, selection):
        grouped[sample.source, sample.split, sample.language].append(sample)

    stats = ExportStats(version=version, output_dir=str(output_dir))
    by_split: dict[str, int] = defaultdict(int)
    by_corpus: dict[str, int] = defaultdict(int)
    hours_split: dict[str, float] = defaultdict(float)
    hours_corpus: dict[str, float] = defaultdict(float)
    for (corpus, split, language), samples in sorted(grouped.items()):
        out_dir = partition_dir(version_root, corpus, split, language)
        rows, hours, skipped = _write_partition(
            samples, out_dir, row_group_size=row_group_size
        )
        stats.rows += rows
        stats.skipped_missing_audio += skipped
        by_split[split] += rows
        by_corpus[corpus] += rows
        hours_split[split] += hours
        hours_corpus[corpus] += hours

    stats.rows_by_split = dict(sorted(by_split.items()))
    stats.rows_by_corpus = dict(sorted(by_corpus.items()))
    stats.hours_by_split = dict(sorted(hours_split.items()))
    stats.hours_by_corpus = dict(sorted(hours_corpus.items()))

    if stats.rows:
        write_language_distribution(
            version_root, output_dir / f"language_distribution_{version}.tsv"
        )
    _write_summary(output_dir, version, selection, stats)
    return stats


def _selection_dict(selection: Selection) -> dict[str, object]:
    return {
        "sources": selection.sources,
        "splits": selection.splits,
        "languages": selection.languages,
        "max_duration_seconds": selection.max_duration_seconds,
        "max_per_source": selection.max_per_source,
    }


def _write_summary(
    output_dir: Path, version: int, selection: Selection, stats: ExportStats
) -> Path:
    """Record the recipe + result so an ablation dir documents how it was built."""
    summary = {
        "version": version,
        "output_dir": str(output_dir),
        "selection": _selection_dict(selection),
        "rows": stats.rows,
        "rows_by_split": stats.rows_by_split,
        "rows_by_corpus": stats.rows_by_corpus,
        "hours_by_split": stats.hours_by_split,
        "hours_by_corpus": stats.hours_by_corpus,
        "skipped_missing_audio": stats.skipped_missing_audio,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "export_summary.json"
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def iter_partition_rows(
    version_root: Path, *, columns: Iterable[str] | None = None
) -> Iterator[dict[str, object]]:
    """Yield each row (as a dict) from every partition under ``version_root`` — for verification."""
    cols = list(columns) if columns is not None else ["text", "audio_bytes", "audio_size"]
    for path in sorted(version_root.glob("corpus=*/split=*/language=*/*.parquet")):
        table = pq.read_table(path, columns=cols)
        yield from table.to_pylist()
