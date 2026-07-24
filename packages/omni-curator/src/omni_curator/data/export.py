"""Materialize an ablation: ``CuratorStore`` -> omni-parquet dataset (closes the loop).

The store is the master pool; a *dataset version* is a recipe over it, not a copy. This module is
where that recipe is applied: a :class:`Selection` filters the pool (by source / split / language,
with optional duration and per-source caps) and the surviving
:class:`~omni_curator.data.sample.Sample`s are written as the SAME omni-parquet layout that
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

import hashlib
import json
import shutil
import tempfile
import warnings
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from omni_curator.audit.quality import is_descriptor_only
from omni_curator.data.provenance import (
    UNKNOWN_LICENSE,
    LicenseValue,
    normalize_license_registry,
)
from omni_curator.process import normalize as normalize_text
from omni_curator.process.language import keep_for_language

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping
    from typing import TextIO

    from omni_curator.data.sample import Sample
    from omni_curator.data.store import CuratorStore

SAMPLE_RATE = 16_000

YOUTUBE_CATEGORY_TAXONOMY = frozenset(
    {
        "news",
        "talk",
        "podcast",
        "interview",
        "documentary",
        "education",
        "language_learning",
        "audiobook",
        "literature",
        "religion",
        "comedy",
        "entertainment",
        "children",
        "vlog",
        "travel",
        "food",
        "music",
    }
)

YOUTUBE_CATEGORY_ALIASES = {
    "analysis": "talk",
    "audiobooks": "audiobook",
    "childrens": "children",
    "educational": "education",
    "kids": "children",
    "lecture": "education",
    "lectures": "education",
    "lifestyle": "vlog",
    "religious": "religion",
    "sermon": "religion",
    "sermons": "religion",
    "tv": "news",
}
_MIN_VIDEOS_FOR_DEV_SPLIT = 2
_MIN_VIDEOS_FOR_TEST_AND_DEV_SPLITS = 3

#: The exact schema ``omni-finetune-core`` trains on (``omni_finetune_core.parquet.OMNI_SCHEMA``).
OMNI_SCHEMA: pa.Schema = pa.schema(
    [
        ("text", pa.string()),
        ("audio_bytes", pa.list_(pa.int8())),
        ("audio_size", pa.int64()),
    ]
)

#: What the export actually writes: the training columns + per-source license provenance + row
#: metadata. The trainer reads only its three columns by name (``columns=[...]``), so the extras are
#: inert for training but let an export be queried/filtered by license and source category.
#: ``license`` = the precise identifier (for attribution / ShareAlike); ``commercial_use`` = the
#: derived filter key. Constant per partition (a partition is one ``corpus=<source>``, so one
#: license). ``metadata`` is the per-row ``Sample.meta`` JSON.
EXPORT_SCHEMA: pa.Schema = pa.schema(
    [
        *OMNI_SCHEMA,
        pa.field("license", pa.string()),
        pa.field("commercial_use", pa.bool_()),
        pa.field("metadata", pa.string()),
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
    :mod:`omni_curator.audit.verify`): drop a clip whose label disagrees too much with a Scribe
    re-transcription. They read ``sample.scribe_wer`` / ``sample.scribe_cer``, so you must run
    :func:`omni_curator.audit.verify.verify_store` first for them to bite. A clip with NO score
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
    #: Trusted sources/authorities whose labels bypass the heuristic language gate. Use this for
    #: gold corpora or source-authority policies where the gate is known to be lossier than the
    #: source contract. Other curation gates still apply.
    trusted_language_sources: frozenset[str] = frozenset()
    trusted_language_authorities: frozenset[str] = frozenset()
    #: Drop labels with no lexical content ('[outro jingle]', '♪', '...') — they verify at WER ~0
    #: (a fresh Scribe pass emits the same descriptor) so the WER gate alone would keep them.
    drop_descriptor_only: bool = True
    #: Splits the CURATION gates (Scribe WER/CER, language, descriptor) apply to. Quality gates
    #: select training data; held-out benchmark splits must stay complete and comparable — a
    #: censored test set scores against a different exam. Hard structural bounds (duration) and
    #: the value filters (sources/splits/languages) always apply.
    gated_splits: frozenset[str] = frozenset({"train"})
    #: Video ids (``clip_id`` minus the ``_NNNN`` segment suffix) to carve into a held-out TEST
    #: split — a leakage-safe, whole-video benchmark for create-pipeline data that has no natural
    #: dev/test. A held-out clip is still gated as the ``train`` row it is stored as; the PASSING
    #: ones are regrouped into ``split=test``, the failing ones drop (so no held-out video ever
    #: reaches train). Pass the frozen manifest's ids.
    heldout_test_videos: frozenset[str] = frozenset()
    #: Optional deterministic split policy for YouTube/create rows. Disabled by default so existing
    #: exports are stable; enable it to create category-stratified, whole-video dev/test splits.
    youtube_split_policy: YoutubeSplitPolicy | None = None

    def gates(self, sample: Sample) -> bool:
        """Whether the curation gates (WER/CER/language/descriptor) apply to ``sample``."""
        return sample.split in self.gated_splits or self.is_heldout(sample)

    def applies_language_gate(self, sample: Sample) -> bool:
        """Whether the heuristic text-language gate applies to this sample."""
        if not self.language_gate or not self.gates(sample):
            return False
        provenance = sample.provenance
        authority = provenance.authority if provenance is not None else None
        return (
            sample.source not in self.trusted_language_sources
            and authority not in self.trusted_language_authorities
        )

    def is_heldout(self, sample: Sample) -> bool:
        """Whether ``sample``'s video is held out (gated as train, then regrouped to test)."""
        return bool(self.heldout_test_videos) and _video_id(sample) in self.heldout_test_videos

    def keeps(self, sample: Sample) -> bool:
        """Whether ``sample`` passes the value/duration/scribe filters (per-source cap later)."""
        if (
            (self.sources is not None and sample.source not in self.sources)
            or (self.splits is not None and sample.split not in self.splits)
            or (self.languages is not None and sample.language not in self.languages)
        ):
            return False
        if self.max_duration_seconds is not None and sample.duration > self.max_duration_seconds:
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
        A reason names the bound that failed; the bounds are checked in a fixed order, and the
        first failure wins (so a too-short clip is reported as ``min_duration``). A non-positive
        duration makes every per-second rate undefined / catastrophic, so any active rate bound
        treats it as a failure (the rate is taken as ``+inf``).
        """
        chars_per_second = len(norm_text) / duration if duration > 0 else float("inf")
        words_per_second = len(norm_text.split()) / duration if duration > 0 else float("inf")
        # (reason, active?, failed?) in a fixed order; first active-and-failed wins.
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
        return next((reason for reason, active, failed in checks if active and failed), None)


@dataclass
class ExportStats:
    """What an export produced — counts + hours per split/corpus, for the summary and the caller.

    ``rows`` is the number written (kept). ``dropped_by_quality`` is the per-reason breakdown of
    clips the quality filter rejected (one key per failed bound); ``dropped_quality_total``
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


@dataclass(frozen=True, kw_only=True)
class YoutubeSplitPolicy:
    """Deterministic YouTube train/dev/test policy, stratified by source category.

    Assignment is whole-video: all clips whose ids share the same ``video_id`` stay in the same
    split. Within each category, videos are ordered by a stable hash of ``seed/category/video_id``
    and the first buckets become test/dev.
    """

    dev_ratio: float = 0.05
    test_ratio: float = 0.05
    seed: str = "youtube-v1"

    def __post_init__(self) -> None:
        if self.dev_ratio < 0 or self.test_ratio < 0:
            msg = "YouTube split ratios must be non-negative"
            raise ValueError(msg)
        if self.dev_ratio + self.test_ratio >= 1:
            msg = "YouTube dev_ratio + test_ratio must be < 1"
            raise ValueError(msg)


def normalize_youtube_category(value: object) -> str:
    """Normalize source metadata category into the 17-genre YouTube taxonomy."""
    if not isinstance(value, str):
        return "uncategorized"
    key = value.strip().lower().replace("-", "_").replace(" ", "_")
    if key in YOUTUBE_CATEGORY_TAXONOMY:
        return key
    return YOUTUBE_CATEGORY_ALIASES.get(key, "uncategorized")


def _flac_bytes(audio_path: Path) -> np.ndarray:
    """Return ``audio_path`` as canonical 16 kHz mono FLAC bytes in an int8 array.

    Canonical FLAC clips pass through verbatim. Other containers, sample rates, and channel layouts
    are decoded and re-encoded, so the stored bytes still always match what the trainer expects.
    """
    import soundfile as sf

    info = sf.info(str(audio_path))
    if info.format == "FLAC" and info.samplerate == SAMPLE_RATE and info.channels == 1:
        return np.frombuffer(audio_path.read_bytes(), dtype=np.int8)

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


def _video_id(sample: Sample) -> str:
    """The clip's source video id — its ``clip_id`` (``{video_id}_{seg:04d}``) minus the suffix."""
    return sample.id.rsplit("_", 1)[0]


def _select(store: CuratorStore, selection: Selection) -> Iterator[Sample]:
    """Yield the samples kept by ``selection``, applying the per-source cap in store order."""
    seen: dict[str, int] = defaultdict(int)
    for sample in store.iter_samples(sources=selection.sources, splits=selection.splits):
        if not selection.keeps(sample):
            continue
        if selection.max_per_source is not None and seen[sample.source] >= selection.max_per_source:
            continue
        seen[sample.source] += 1
        yield sample


def _stable_video_key(policy: YoutubeSplitPolicy, category: str, video_id: str) -> int:
    payload = f"{policy.seed}\0{category}\0{video_id}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _split_counts(total: int, policy: YoutubeSplitPolicy) -> tuple[int, int]:
    """Return ``(test_n, dev_n)`` for one category's video count."""
    test_n = min(total, round(total * policy.test_ratio))
    dev_n = min(total - test_n, round(total * policy.dev_ratio))
    if total >= _MIN_VIDEOS_FOR_TEST_AND_DEV_SPLITS and policy.test_ratio > 0 and test_n == 0:
        test_n = 1
    if total - test_n >= _MIN_VIDEOS_FOR_DEV_SPLIT and policy.dev_ratio > 0 and dev_n == 0:
        dev_n = 1
    return test_n, dev_n


def _youtube_video_splits(
    items: list[tuple[Sample, str, str]], policy: YoutubeSplitPolicy
) -> dict[str, str]:
    """Video id -> split, with each category assigned by stable hash order."""
    by_category: dict[str, set[str]] = defaultdict(set)
    for sample, _norm_text, base_split in items:
        if base_split != "train" or not sample.source.startswith("youtube-"):
            continue
        category = normalize_youtube_category(sample.meta.get("category"))
        by_category[category].add(_video_id(sample))

    out: dict[str, str] = {}
    for category, video_ids in by_category.items():
        ordered = sorted(
            video_ids, key=lambda video_id: _stable_video_key(policy, category, video_id)
        )
        test_n, dev_n = _split_counts(len(ordered), policy)
        for video_id in ordered[:test_n]:
            out[video_id] = "test"
        for video_id in ordered[test_n : test_n + dev_n]:
            out[video_id] = "dev"
        for video_id in ordered[test_n + dev_n :]:
            out[video_id] = "train"
    return out


def _group_filtered(
    items: list[tuple[Sample, str, str]], selection: Selection
) -> dict[tuple[str, str, str], list[tuple[Sample, str]]]:
    grouped: dict[tuple[str, str, str], list[tuple[Sample, str]]] = defaultdict(list)
    youtube_splits = (
        _youtube_video_splits(items, selection.youtube_split_policy)
        if selection.youtube_split_policy is not None
        else {}
    )
    for sample, norm_text, base_split in items:
        out_split = youtube_splits.get(_video_id(sample), base_split)
        grouped[sample.source, out_split, sample.language].append((sample, norm_text))
    return grouped


def _chunked(items: list[tuple[Sample, str]], n: int) -> Iterator[list[tuple[Sample, str]]]:
    """Yield ``items`` in lists of at most ``n`` (the streaming batch boundary)."""
    batch: list[tuple[Sample, str]] = []
    for item in items:
        batch.append(item)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch


def _batch_to_table(
    batch: list[tuple[Sample, str]], license_str: str, *, commercial: bool
) -> tuple[pa.Table, int, int]:
    """Decode one batch of ``(sample, normalized_text)`` into an EXPORT_SCHEMA table.

    Reads + FLAC-encodes only this batch's audio, so the memory held is one batch, not the whole
    partition. Returns the table, the batch's total audio samples (for the hours tally), and the
    count skipped for missing audio. The NORMALIZED label is written; the stored text is untouched.
    ``license_str``/``commercial`` are constant for the partition (one source) and written per row.
    """
    texts: list[str] = []
    audio: list[np.ndarray] = []
    sizes: list[int] = []
    metadata: list[str] = []
    skipped = 0
    for sample, norm_text in batch:
        path = Path(sample.audio_path)
        if not path.exists():
            skipped += 1
            continue
        texts.append(norm_text)
        audio.append(_flac_bytes(path))
        sizes.append(max(1, round(sample.duration * SAMPLE_RATE)))
        metadata.append(json.dumps(sample.meta, ensure_ascii=False, sort_keys=True))
    n = len(texts)
    table = pa.Table.from_arrays(
        [
            pa.array(texts, type=pa.string()),
            pa.array(audio, type=pa.list_(pa.int8())),
            pa.array(sizes, type=pa.int64()),
            pa.array([license_str] * n, type=pa.string()),
            pa.array([commercial] * n, type=pa.bool_()),
            pa.array(metadata, type=pa.string()),
        ],
        schema=EXPORT_SCHEMA,
    )
    return table, sum(sizes), skipped


def _write_partition(
    samples: list[tuple[Sample, str]],
    out_dir: Path,
    *,
    row_group_size: int,
    license_str: str = "unknown",
    commercial: bool = False,
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
        table, batch_samples, batch_skipped = _batch_to_table(
            batch, license_str, commercial=commercial
        )
        skipped += batch_skipped
        if table.num_rows == 0:
            continue
        if writer is None:
            out_dir.mkdir(parents=True, exist_ok=True)
            writer = pq.ParquetWriter(out_path, EXPORT_SCHEMA)
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


def write_weighted_distribution(true_tsv: Path, out_path: Path, weights: dict[str, float]) -> Path:
    """Write a SAMPLING-WEIGHT copy of the mixture TSV with per-corpus hour overrides.

    fairseq2's mixture reader weights each corpus by the TSV hours (tempered by
    ``beta_corpus``), so overriding a corpus's listed hours is the lever for re-balancing the
    sampled batches — e.g. lifting a small clean read-speech corpus (FLEURS, ~12 true hours)
    to a meaningful share against ~1,000 h of conversational audio. The TRUE hours stay in the
    untouched ``language_distribution_<N>.tsv`` and ``export_summary.json``; this file is a
    declared training-mixture recipe, not provenance. Raises on a weight for a corpus that is
    not in the export (catches typos and stale recipes).
    """
    lines = true_tsv.read_text(encoding="utf-8").splitlines()
    corpora = {line.split("\t", 1)[0] for line in lines[1:]}
    unknown = set(weights) - corpora
    if unknown:
        msg = f"mixture weights for corpora not in the export: {sorted(unknown)}"
        raise ValueError(msg)
    out = [lines[0]]
    for line in lines[1:]:
        corpus, language, hours = line.split("\t")
        if corpus in weights:
            hours = f"{weights[corpus]:.8f}"
        out.append(f"{corpus}\t{language}\t{hours}")
    out_path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return out_path


def _normalize_and_filter(
    store: CuratorStore,
    selection: Selection,
) -> tuple[dict[tuple[str, str, str], list[tuple[Sample, str]]], dict[str, int]]:
    """Run filter -> normalize -> quality-filter, grouped by ``(corpus, split, language)``.

    Returns the grouped ``(sample, normalized_text)`` survivors plus the per-reason count of clips
    the quality filter dropped.
    """
    kept: list[tuple[Sample, str, str]] = []
    dropped: dict[str, int] = defaultdict(int)
    for sample in _select(store, selection):
        norm_text = normalize_text(sample.text, sample.language)
        if not norm_text:
            dropped["empty_normalized_text"] += 1
            continue
        if (
            selection.gates(sample)
            and selection.drop_descriptor_only
            and is_descriptor_only(norm_text)
        ):
            dropped["descriptor_only_normalized"] += 1
            continue
        # Language gate (the ONLY content-language filter): drop clips whose text isn't the target
        # language — e.g. a Russian segment from a Tajik source. The store keeps every clip (Scribe
        # auto-detect transcribes each in its own language); this is where we select the target.
        if selection.applies_language_gate(sample) and not keep_for_language(
            norm_text, sample.language
        ):
            dropped["language"] += 1
            continue
        reason = (
            selection.keeps_quality(norm_text, sample.duration) if selection.gates(sample) else None
        )
        if reason is not None:
            dropped[reason] += 1
            continue
        # Held-out clips are stored (and gated) as train; the survivors regroup into the test
        # partition. A held-out video's failing clips were just dropped above — so the whole
        # video is either in test or nowhere, never train (no leakage).
        out_split = "test" if selection.is_heldout(sample) else sample.split
        kept.append((sample, norm_text, out_split))
    return _group_filtered(kept, selection), dict(dropped)


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
    mixture_weights: dict[str, float] | None = None,
    licenses: Mapping[str, LicenseValue] | None = None,
    commercial_only: bool = False,
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
    license_registry = normalize_license_registry(licenses)
    for (corpus, split, language), samples in sorted(grouped.items()):
        # License is per-source (one corpus = one license). Unknown -> non-commercial,
        # so an unregistered source is conservatively excluded from a --commercial-only export.
        license_info = license_registry.get(corpus, UNKNOWN_LICENSE)
        if commercial_only and not license_info.commercial_use:
            continue
        out_dir = partition_dir(version_root, corpus, split, language)
        rows, hours, skipped = _write_partition(
            samples,
            out_dir,
            row_group_size=row_group_size,
            license_str=license_info.id,
            commercial=license_info.commercial_use,
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
        true_tsv = write_language_distribution(
            version_root, output_dir / f"language_distribution_{version}.tsv"
        )
        if mixture_weights:
            # The declared training-mixture recipe lives next to the true-hours TSV and in the
            # summary — reproducible from the export args, never a hand-edited orphan file.
            write_weighted_distribution(
                true_tsv, output_dir / "language_distribution_weighted.tsv", mixture_weights
            )
    _write_summary(output_dir, version, selection, stats, mixture_weights=mixture_weights)
    return stats


def export_nemo_manifests(
    store: CuratorStore,
    output_dir: Path,
    *,
    selection: Selection | None = None,
    coverage_check: Callable[[list[str]], int] | None = None,
    strict: bool = True,
    licenses: Mapping[str, LicenseValue] | None = None,
    commercial_only: bool = False,
) -> ExportStats:
    """Export normalized NeMo manifests that reference curator-owned canonical audio.

    This is the fast local-training path. It applies the same selection, normalization, quality,
    language, tokenizer-coverage, license, and split policy as :func:`export_dataset`, but it does
    not embed audio in Parquet and does not copy audio into a second materialized tree. The
    resulting ``<split>.jsonl`` files point directly at the stable paths already recorded in the
    curator store.

    The output directory is immutable and published atomically. Missing audio is counted and
    skipped, matching the Parquet exporter. Use the Parquet exporter when a self-contained,
    portable fairseq2/Hugging Face artifact is actually required.
    """
    selection = selection or Selection()
    if output_dir.exists():
        raise FileExistsError(f"immutable NeMo manifest export already exists: {output_dir}")

    grouped, dropped_by_quality = _normalize_and_filter(store, selection)
    stats = ExportStats(
        version=0,
        output_dir=str(output_dir),
        dropped_by_quality=dropped_by_quality,
    )
    if coverage_check is not None:
        stats.unk_rows = _run_coverage_gate(grouped, coverage_check, strict=strict)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    by_split: dict[str, int] = defaultdict(int)
    by_corpus: dict[str, int] = defaultdict(int)
    hours_split: dict[str, float] = defaultdict(float)
    hours_corpus: dict[str, float] = defaultdict(float)
    license_registry = normalize_license_registry(licenses)
    try:
        with ExitStack() as stack:
            handles: dict[str, TextIO] = {}
            for (corpus, split, language), samples in sorted(grouped.items()):
                license_info = license_registry.get(corpus, UNKNOWN_LICENSE)
                if commercial_only and not license_info.commercial_use:
                    continue
                if split not in handles:
                    handles[split] = stack.enter_context(
                        (temp / f"{_sanitize(split)}.jsonl").open("w", encoding="utf-8")
                    )
                handle = handles[split]
                for sample, norm_text in samples:
                    audio_path = Path(sample.audio_path)
                    if not audio_path.is_file():
                        stats.skipped_missing_audio += 1
                        continue
                    duration = float(sample.duration)
                    record = {
                        "audio_filepath": str(audio_path.resolve()),
                        "text": norm_text,
                        "duration": round(duration, 6),
                        "corpus": corpus,
                        "language": language,
                        "sample_id": sample.id,
                        "speaker_id": sample.speaker_id,
                        "citation": sample.citation,
                        "license": license_info.id,
                        "commercial_use": license_info.commercial_use,
                        "metadata": sample.meta,
                    }
                    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                    stats.rows += 1
                    by_split[split] += 1
                    by_corpus[corpus] += 1
                    hours_split[split] += duration / 3600.0
                    hours_corpus[corpus] += duration / 3600.0

        stats.rows_by_split = dict(sorted(by_split.items()))
        stats.rows_by_corpus = dict(sorted(by_corpus.items()))
        stats.hours_by_split = dict(sorted(hours_split.items()))
        stats.hours_by_corpus = dict(sorted(hours_corpus.items()))
        summary = {
            "format": "nemo-manifest",
            "audio_mode": "reference",
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
        (temp / "export_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temp.replace(output_dir)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise
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
        "trusted_language_sources": sorted(selection.trusted_language_sources) or None,
        "trusted_language_authorities": sorted(selection.trusted_language_authorities) or None,
        "heldout_test_videos": len(selection.heldout_test_videos) or None,
        "youtube_split_policy": (
            {
                "dev_ratio": selection.youtube_split_policy.dev_ratio,
                "test_ratio": selection.youtube_split_policy.test_ratio,
                "seed": selection.youtube_split_policy.seed,
                "taxonomy": sorted(YOUTUBE_CATEGORY_TAXONOMY),
            }
            if selection.youtube_split_policy is not None
            else None
        ),
    }


def _write_summary(
    output_dir: Path,
    version: int,
    selection: Selection,
    stats: ExportStats,
    *,
    mixture_weights: dict[str, float] | None = None,
) -> Path:
    """Record the recipe + result so an ablation dir documents how it was built."""
    summary = {
        "version": version,
        "output_dir": str(output_dir),
        "selection": _selection_dict(selection),
        "mixture_weights": mixture_weights,
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
