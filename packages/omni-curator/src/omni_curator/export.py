"""Materialize an ablation: ``CuratorStore`` -> omni-parquet dataset (closes the loop).

The store is the master pool; a *dataset version* is a recipe over it, not a copy (see
``CURATING.md``). This module is where that recipe is applied: a :class:`Selection` filters the
pool (by source / split / language, with optional duration and per-source caps) and the surviving
:class:`~omni_curator.sample.Sample`s are written as the SAME omni-parquet layout that
``omni-finetune-core`` trains on.

Materializing runs a fixed pipeline per sample — **filter -> normalize -> quality-filter ->
coverage-gate -> write**:

- **normalize**: each kept label is run through :func:`omni_curator.process.normalize` for its
  language (spoken form, script folding, number expansion) before it is written. The stored text
  in the master pool is never mutated — normalization applies to the exported copy only.
- **quality-filter**: the normalized text + duration are checked against the optional
  char/word-per-second and min-duration bounds on :class:`Selection` (the misaligned-transcript
  catcher from the NVIDIA Georgian ASR recipe). Off by default.
- **coverage-gate**: an injected ``coverage_check`` counts tokenizer ``<unk>`` rows over the
  normalized texts; a non-zero count fails the export (or warns, in non-strict mode). The
  tokenizer/fairseq2 dependency lives in the *consuming* project and is passed in, so the curator
  itself never depends on it.

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
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from omni_curator.process import normalize as normalize_text
from omni_curator.process.language import keep_for_language
from omni_curator.quality import is_descriptor_only

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from omni_curator.sample import Sample
    from omni_curator.store import CuratorStore

SAMPLE_RATE = 16_000

#: The exact schema ``omni-finetune-core`` trains on (``omni_finetune_core.parquet.OMNI_SCHEMA``).
OMNI_SCHEMA: pa.Schema = pa.schema(
    [
        ("text", pa.string()),
        ("audio_bytes", pa.list_(pa.int8())),
        ("audio_size", pa.int64()),
    ]
)


#: Names of the per-sample quality-filter reasons, for stable :class:`ExportStats` keys.
QUALITY_REASONS = (
    "min_duration",
    "max_chars_per_second",
    "min_words_per_second",
    "max_words_per_second",
)


@dataclass
class Selection:
    """A recipe: which store rows go into this ablation.

    All filters are AND-ed. ``sources`` / ``splits`` / ``languages`` keep only the listed values
    (``None`` = keep all). ``max_duration_seconds`` drops longer clips (the omni pipeline caps
    audio at 40 s, so an eval ablation often sets 40). ``max_per_source`` caps how many clips each
    source contributes, in store order, so a big corpus can't swamp a small one before the mixture
    weighting even runs.

    The quality fields (all optional, ``None`` = off) catch **audio/text misalignment** — a clip
    whose transcript is too dense or too sparse for its duration is almost certainly mislabelled.
    They are computed on the NORMALIZED text + the sample duration, so :meth:`keeps_quality` runs
    after normalization, not here. Bounds mirror the NVIDIA Georgian ASR recipe (words/sec outside
    ``[0.3, 2.67]`` dropped; chars/sec capped around 18):

    - ``min_duration_seconds`` — drop clips shorter than this many seconds.
    - ``max_chars_per_second`` — drop if ``len(norm_text) / duration`` exceeds it (~18 typical).
    - ``min_words_per_second`` / ``max_words_per_second`` — drop if
      ``len(norm_text.split()) / duration`` falls outside ``[min, max]``.

    Non-positive durations are treated as failing every rate bound (cannot be a valid clip).

    ``max_scribe_wer`` / ``max_scribe_cer`` are the store-level Scribe-verification gate (see
    :mod:`omni_curator.verify`): drop a clip whose label disagrees too much with a fresh Scribe
    transcription. They read ``sample.scribe_wer`` / ``sample.scribe_cer``, so you must run
    :func:`omni_curator.verify.verify_store` first for them to bite. A clip with NO score
    (``scribe_wer is None``) is KEPT even when a bound is set — an un-scored clip is never silently
    dropped; run verification to score it, then re-export.
    """

    sources: list[str] | None = None
    splits: list[str] | None = None
    languages: list[str] | None = None
    max_duration_seconds: float | None = None
    max_per_source: int | None = None
    min_duration_seconds: float | None = None
    max_chars_per_second: float | None = None
    min_words_per_second: float | None = None
    max_words_per_second: float | None = None
    max_scribe_wer: float | None = None
    max_scribe_cer: float | None = None
    #: Drop clips whose text isn't the target language (via the per-language gate registry). True =
    #: keep only the target language (the usual training case); False = keep every language in the
    #: store (e.g. exporting the other-language segments of language-learning content).
    language_gate: bool = True
    #: Drop labels with no lexical content ('[outro jingle]', '♪', '...') — they verify at WER ~0
    #: (a fresh Scribe pass emits the same descriptor) so the WER gate alone would keep them.
    drop_descriptor_only: bool = True
    #: Splits the CURATION gates (Scribe WER/CER, language, descriptor) apply to. Quality gates
    #: select training data; held-out benchmark splits must stay complete and comparable — a
    #: censored test set scores against a different exam. Hard structural bounds (duration) and
    #: the value filters (sources/splits/languages) always apply.
    gated_splits: frozenset[str] = frozenset({"train"})

    def gates(self, sample: Sample) -> bool:
        """Whether the curation gates (WER/CER/language/descriptor) apply to ``sample``."""
        return sample.split in self.gated_splits

    def keeps(self, sample: Sample) -> bool:
        """Whether ``sample`` passes the value/duration/scribe filters (per-source cap later)."""
        if (
            (self.sources is not None and sample.source not in self.sources)
            or (self.splits is not None and sample.split not in self.splits)
            or (self.languages is not None and sample.language not in self.languages)
        ):
            return False
        if (
            self.max_duration_seconds is not None
            and sample.duration > self.max_duration_seconds
        ):
            return False
        if not self.gates(sample):
            return True  # held-out split: curation gates below don't apply
        if self.drop_descriptor_only and is_descriptor_only(sample.text):
            return False
        # Scribe-verification gate: only drops clips that HAVE a score over the bound; an un-scored
        # clip (scribe_wer/cer is None) is kept — run verify_store first for the gate to bite.
        if (
            self.max_scribe_wer is not None
            and sample.scribe_wer is not None
            and sample.scribe_wer > self.max_scribe_wer
        ):
            return False
        return not (
            self.max_scribe_cer is not None
            and sample.scribe_cer is not None
            and sample.scribe_cer > self.max_scribe_cer
        )

    def keeps_quality(self, norm_text: str, duration: float) -> str | None:
        """Return the failing quality-filter reason, or ``None`` if the sample passes.

        Computed on the already-normalized text + duration; catches audio/text misalignment.
        A reason is one of :data:`QUALITY_REASONS`. The bounds are checked in that order, and the
        first failure wins (so a too-short clip is reported as ``min_duration``). A non-positive
        duration makes every per-second rate undefined / catastrophic, so any active rate bound
        treats it as a failure (the rate is taken as ``+inf``).
        """
        chars_per_second = len(norm_text) / duration if duration > 0 else float("inf")
        words_per_second = len(norm_text.split()) / duration if duration > 0 else float("inf")
        # (reason, active?, failed?) in QUALITY_REASONS order; first active-and-failed wins.
        checks = (
            (
                "min_duration",
                self.min_duration_seconds is not None,
                duration < (self.min_duration_seconds or 0.0),
            ),
            (
                "max_chars_per_second",
                self.max_chars_per_second is not None,
                chars_per_second > (self.max_chars_per_second or 0.0),
            ),
            (
                "min_words_per_second",
                self.min_words_per_second is not None,
                words_per_second < (self.min_words_per_second or 0.0),
            ),
            (
                "max_words_per_second",
                self.max_words_per_second is not None,
                words_per_second > (self.max_words_per_second or 0.0),
            ),
        )
        return next(
            (reason for reason, active, failed in checks if active and failed), None
        )


@dataclass
class ExportStats:
    """What an export produced — counts + hours per split/corpus, for the summary and the caller.

    ``rows`` is the number written (kept). ``dropped_by_quality`` is the per-reason breakdown of
    clips the quality filter rejected (keys from :data:`QUALITY_REASONS`); ``dropped_quality_total``
    sums it. ``hours`` is the total exported audio (the sum of ``hours_by_corpus``). ``unk_rows``
    is what the injected ``coverage_check`` reported (``None`` if no check ran).
    """

    version: int
    output_dir: str
    rows: int = 0
    rows_by_split: dict[str, int] = field(default_factory=dict)
    rows_by_corpus: dict[str, int] = field(default_factory=dict)
    hours_by_split: dict[str, float] = field(default_factory=dict)
    hours_by_corpus: dict[str, float] = field(default_factory=dict)
    skipped_missing_audio: int = 0
    dropped_by_quality: dict[str, int] = field(default_factory=dict)
    unk_rows: int | None = None

    @property
    def dropped_quality_total(self) -> int:
        """Total clips the quality filter dropped, across all reasons."""
        return sum(self.dropped_by_quality.values())

    @property
    def hours(self) -> float:
        """Total exported audio in hours (sum over corpora)."""
        return sum(self.hours_by_corpus.values())


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


def _chunked(
    items: list[tuple[Sample, str]], n: int
) -> Iterator[list[tuple[Sample, str]]]:
    """Yield ``items`` in lists of at most ``n`` (the streaming batch boundary)."""
    batch: list[tuple[Sample, str]] = []
    for item in items:
        batch.append(item)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch


def _batch_to_table(batch: list[tuple[Sample, str]]) -> tuple[pa.Table, int, int]:
    """Decode one batch of ``(sample, normalized_text)`` into an OMNI_SCHEMA table.

    Reads + FLAC-encodes only this batch's audio, so the memory held is one batch, not the whole
    partition. Returns the table, the batch's total audio samples (for the hours tally), and the
    count skipped for missing audio. The NORMALIZED label is written; the stored text is untouched.
    """
    texts: list[str] = []
    audio: list[np.ndarray] = []
    sizes: list[int] = []
    skipped = 0
    for sample, norm_text in batch:
        path = Path(sample.audio_path)
        if not path.exists():
            skipped += 1
            continue
        texts.append(norm_text)
        audio.append(_flac_bytes(path))
        sizes.append(max(1, round(sample.duration * SAMPLE_RATE)))
    table = pa.Table.from_arrays(
        [
            pa.array(texts, type=pa.string()),
            pa.array(audio, type=pa.list_(pa.int8())),
            pa.array(sizes, type=pa.int64()),
        ],
        schema=OMNI_SCHEMA,
    )
    return table, sum(sizes), skipped


def _write_partition(
    samples: list[tuple[Sample, str]],
    out_dir: Path,
    *,
    row_group_size: int,
) -> tuple[int, float, int]:
    """Stream one partition to ``part-00000.parquet`` in batches; return (rows, hours, skipped).

    Decode -> write -> free, ``row_group_size`` rows at a time, via a :class:`pq.ParquetWriter`
    (each ``write_table`` appends a row group). Peak memory is one batch of audio, never the whole
    partition — buffering a full partition is what OOM'd the early v0 dry-run (145 h decodes to
    ~30+ GB). The writer is opened lazily so an all-missing-audio partition writes no file.
    """
    out_path = out_dir / "part-00000.parquet"
    writer: pq.ParquetWriter | None = None
    rows = skipped = total_samples = 0
    for batch in _chunked(samples, row_group_size):
        table, batch_samples, batch_skipped = _batch_to_table(batch)
        skipped += batch_skipped
        if table.num_rows == 0:
            continue
        if writer is None:
            out_dir.mkdir(parents=True, exist_ok=True)
            writer = pq.ParquetWriter(out_path, OMNI_SCHEMA)
        writer.write_table(table, row_group_size=row_group_size)
        rows += table.num_rows
        total_samples += batch_samples
    if writer is not None:
        writer.close()
    return rows, total_samples / SAMPLE_RATE / 3600, skipped


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


def _normalize_and_filter(
    store: CuratorStore,
    selection: Selection,
) -> tuple[dict[tuple[str, str, str], list[tuple[Sample, str]]], dict[str, int]]:
    """Run filter -> normalize -> quality-filter, grouped by ``(corpus, split, language)``.

    Returns the grouped ``(sample, normalized_text)`` survivors plus the per-reason count of clips
    the quality filter dropped.
    """
    grouped: dict[tuple[str, str, str], list[tuple[Sample, str]]] = defaultdict(list)
    dropped: dict[str, int] = defaultdict(int)
    for sample in _select(store, selection):
        norm_text = normalize_text(sample.text, sample.language)
        # Language gate (the ONLY content-language filter): drop clips whose text isn't the target
        # language — e.g. a Russian segment from a Tajik source. The store keeps every clip (Scribe
        # auto-detect transcribes each in its own language); this is where we select the target.
        if (
            selection.language_gate
            and selection.gates(sample)
            and not keep_for_language(norm_text, sample.language)
        ):
            dropped["language"] += 1
            continue
        reason = (
            selection.keeps_quality(norm_text, sample.duration)
            if selection.gates(sample)
            else None
        )
        if reason is not None:
            dropped[reason] += 1
            continue
        grouped[sample.source, sample.split, sample.language].append((sample, norm_text))
    return grouped, dict(dropped)


def _run_coverage_gate(
    grouped: dict[tuple[str, str, str], list[tuple[Sample, str]]],
    coverage_check: Callable[[list[str]], int],
    *,
    strict: bool,
) -> int:
    """Count tokenizer ``<unk>`` rows over the normalized texts; fail (or warn) if any.

    The exported dataset must be 100% tokenizer-covered, so a non-zero count raises in strict mode
    and warns otherwise. Returns the unknown-row count.
    """
    texts = [norm_text for samples in grouped.values() for _, norm_text in samples]
    unk_rows = coverage_check(texts)
    if unk_rows > 0:
        msg = (
            f"coverage gate failed: {unk_rows} of {len(texts)} normalized rows contain tokenizer "
            "<unk> — the exported dataset must be 100% tokenizer-covered"
        )
        if strict:
            raise ValueError(msg)
        warnings.warn(msg, stacklevel=2)
    return unk_rows


def export_dataset(
    store: CuratorStore,
    output_dir: Path,
    *,
    version: int = 0,
    selection: Selection | None = None,
    row_group_size: int = 100,
    coverage_check: Callable[[list[str]], int] | None = None,
    strict: bool = True,
) -> ExportStats:
    """Materialize the ``selection`` over ``store`` as omni-parquet under ``output_dir``.

    The pipeline per sample is **filter -> normalize -> quality-filter -> coverage-gate -> write**:
    each kept sample's label is normalized via :func:`omni_curator.process.normalize` for its
    language, the normalized text + duration are checked against the :class:`Selection` quality
    bounds, and the surviving normalized texts are run through the optional ``coverage_check``
    before anything is written. The stored text is never mutated — normalization applies to the
    exported copy only.

    ``coverage_check`` is an injected callable ``list[str] -> int`` returning the number of rows
    that contain a tokenizer ``<unk>``; if it returns ``> 0`` the export fails (``strict=True``,
    the default) or warns (``strict=False``). It is injected so the curator never depends on
    fairseq2 / the consuming project's tokenizer; pass ``None`` to skip the gate entirely.

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

    grouped, dropped_by_quality = _normalize_and_filter(store, selection)

    stats = ExportStats(
        version=version,
        output_dir=str(output_dir),
        dropped_by_quality=dropped_by_quality,
    )

    # Coverage gate runs on the normalized texts BEFORE any parquet is written, so a failure
    # leaves no half-materialized ablation tree behind.
    if coverage_check is not None:
        stats.unk_rows = _run_coverage_gate(grouped, coverage_check, strict=strict)

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
        "min_duration_seconds": selection.min_duration_seconds,
        "max_chars_per_second": selection.max_chars_per_second,
        "min_words_per_second": selection.min_words_per_second,
        "max_words_per_second": selection.max_words_per_second,
        "max_scribe_wer": selection.max_scribe_wer,
        "max_scribe_cer": selection.max_scribe_cer,
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
        "hours": stats.hours,
        "skipped_missing_audio": stats.skipped_missing_audio,
        "dropped_by_quality": stats.dropped_by_quality,
        "dropped_quality_total": stats.dropped_quality_total,
        "unk_rows": stats.unk_rows,
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
