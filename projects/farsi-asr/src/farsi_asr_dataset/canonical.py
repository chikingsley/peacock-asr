from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import shutil
import sys
import unicodedata
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
from tqdm import tqdm

from farsi_asr_dataset.text_normalization import maybe_normalize

csv.field_size_limit(sys.maxsize)

DEFAULT_SAMPLE_RATE = 44_100
MANA_TTS_ALLOWED_MATCH_QUALITIES = {"HIGH", "MIDDLE"}
MANA_TTS_BAD_TRANSCRIPTS = {"", "nan", "none", "null"}
MANA_TTS_MAX_CER = 0.15

CANONICAL_SCHEMA = pa.schema(
    [
        ("sample_id", pa.string()),
        ("source", pa.string()),
        ("source_config", pa.string()),
        ("original_split", pa.string()),
        ("project_split", pa.string()),
        ("split_origin", pa.string()),
        ("audio_bytes", pa.binary()),
        ("audio_format", pa.string()),
        ("text", pa.string()),
        ("normalized_text", pa.string()),
        ("duration_seconds", pa.float64()),
        ("sample_rate", pa.int32()),
        ("speaker_or_group_id", pa.string()),
        ("license", pa.string()),
        ("source_url", pa.string()),
        ("metadata_json", pa.string()),
    ]
)


@dataclass(frozen=True)
class CanonicalSample:
    sample_id: str
    source: str
    source_config: str
    original_split: str
    project_split: str
    split_origin: str
    audio_bytes: bytes
    audio_format: str
    text: str
    normalized_text: str
    duration_seconds: float
    sample_rate: int | None
    speaker_or_group_id: str
    license: str
    source_url: str
    metadata_json: str


@dataclass
class SplitSummary:
    rows: int = 0
    hours: float = 0.0


@dataclass
class DatasetSummary:
    source: str
    source_config: str
    split_origin: str
    license: str
    output_root: str
    rows_by_split: dict[str, int] = field(default_factory=dict)
    hours_by_split: dict[str, float] = field(default_factory=dict)
    rows: int = 0
    hours: float = 0.0


class SplitWriters:
    def __init__(self, output_root: Path, rows_per_file: int) -> None:
        self.output_root = output_root
        self.rows_per_file = rows_per_file
        self.buffers: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.file_indexes: Counter[str] = Counter()

    def write(self, sample: CanonicalSample) -> None:
        row = asdict(sample)
        self.buffers[sample.project_split].append(row)
        if len(self.buffers[sample.project_split]) >= self.rows_per_file:
            self.flush_split(sample.project_split)

    def flush_split(self, split: str) -> None:
        rows = self.buffers[split]
        if not rows:
            return
        split_root = self.output_root / split
        split_root.mkdir(parents=True, exist_ok=True)
        file_index = self.file_indexes[split]
        path = split_root / f"{split}-{file_index:05d}.parquet"
        table = pa.Table.from_pylist(rows, schema=CANONICAL_SCHEMA)
        pq.write_table(table, path, compression=None)
        self.file_indexes[split] += 1
        rows.clear()

    def close(self) -> None:
        for split in list(self.buffers):
            self.flush_split(split)


def common_voice_reader(handle: Any) -> csv.DictReader[str]:
    # Common Voice TSV sentence fields can contain unmatched literal quote marks.
    return csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)


def stable_bucket(value: str, modulo: int = 10_000) -> int:
    digest = hashlib.sha1(value.encode("utf-8"), usedforsecurity=False).hexdigest()
    return int(digest[:12], 16) % modulo


def split_train_dev(sample_id: str, dev_pct: float = 0.05) -> str:
    return "dev" if stable_bucket(sample_id) < int(dev_pct * 10_000) else "train"


def split_train_dev_test(sample_id: str, dev_pct: float = 0.05, test_pct: float = 0.05) -> str:
    bucket = stable_bucket(sample_id)
    if bucket < int(test_pct * 10_000):
        return "test"
    if bucket < int((test_pct + dev_pct) * 10_000):
        return "dev"
    return "train"


def audio_info(audio_bytes: bytes) -> tuple[float, int | None]:
    try:
        info = sf.info(io.BytesIO(audio_bytes))
    except sf.LibsndfileError:
        return 0.0, None
    return float(info.frames / info.samplerate), int(info.samplerate)


def audio_struct_bytes(audio: Any) -> bytes:
    if not isinstance(audio, dict):
        raise TypeError(f"expected audio struct dict, got {type(audio).__name__}")
    data = audio.get("bytes")
    if data is None:
        raise ValueError("audio struct does not contain bytes")
    return bytes(data)


def float_audio_wav_bytes(audio: Any, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def int8_list_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    return bytes((int(item) + 256) % 256 for item in value)


def metadata(**values: Any) -> str:
    return json.dumps(values, ensure_ascii=False, sort_keys=True)


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result):
        return None
    return result


def contains_latin_script(value: str) -> bool:
    return any("LATIN" in unicodedata.name(character, "") for character in value)


def write_dataset(
    samples: Any,
    *,
    source: str,
    source_config: str,
    split_origin: str,
    license_name: str,
    output_root: Path,
    rows_per_file: int,
) -> DatasetSummary:
    writers = SplitWriters(output_root, rows_per_file)
    summaries: dict[str, SplitSummary] = defaultdict(SplitSummary)
    progress = tqdm(samples, desc=f"canonical {source}", unit="row")
    try:
        for sample in progress:
            writers.write(sample)
            summary = summaries[sample.project_split]
            summary.rows += 1
            summary.hours += sample.duration_seconds / 3600
    finally:
        writers.close()
        progress.close()

    rows_by_split = {split: summary.rows for split, summary in sorted(summaries.items())}
    hours_by_split = {
        split: summary.hours for split, summary in sorted(summaries.items())
    }
    dataset_summary = DatasetSummary(
        source=source,
        source_config=source_config,
        split_origin=split_origin,
        license=license_name,
        output_root=str(output_root),
        rows_by_split=rows_by_split,
        hours_by_split=hours_by_split,
        rows=sum(rows_by_split.values()),
        hours=sum(hours_by_split.values()),
    )
    (output_root / "summary.json").write_text(
        json.dumps(asdict(dataset_summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return dataset_summary


def iter_parquet_rows(files: list[Path], batch_size: int = 512) -> Any:
    for file_path in files:
        parquet = pq.ParquetFile(file_path)
        for batch in parquet.iter_batches(batch_size=batch_size):
            for row in batch.to_pylist():
                yield file_path, row


def youtube_samples(raw_root: Path, project_splits: set[str] | None = None) -> Any:
    data_root = raw_root / "asr_farsi_youtube_pourmand1376/data"
    for original_split, project_split in [("train", "train"), ("val", "dev"), ("test", "test")]:
        if project_splits is not None and project_split not in project_splits:
            continue
        files = sorted(data_root.glob(f"{original_split}-*.parquet"))
        for file_path, row in iter_parquet_rows(files):
            audio_bytes = audio_struct_bytes(row["audio"])
            duration, sample_rate = audio_info(audio_bytes)
            video_id = str(row["video_id"])
            segment_id = str(row["segment_id"])
            text = str(row["transcription"])
            yield CanonicalSample(
                sample_id=f"youtube:{video_id}:{segment_id}",
                source="asr_farsi_youtube",
                source_config="pourmand1376/asr-farsi-youtube-chunked-10-seconds",
                original_split=original_split,
                project_split=project_split,
                split_origin="native",
                audio_bytes=audio_bytes,
                audio_format="source",
                text=text,
                normalized_text=maybe_normalize(text) or "",
                duration_seconds=duration,
                sample_rate=sample_rate,
                speaker_or_group_id=video_id,
                license="unknown",
                source_url=str(row.get("youtube_url") or ""),
                metadata_json=metadata(
                    parquet_path=str(file_path),
                    title=row.get("title"),
                    video_id=video_id,
                    segment_id=segment_id,
                ),
            )


def worldspeech_samples(raw_root: Path, project_splits: set[str] | None = None) -> Any:
    data_root = raw_root / "worldspeech_fa_ir/data/fa_ir"
    for original_split in ["train", "test"]:
        if project_splits == {"test"} and original_split != "test":
            continue
        files = sorted(data_root.glob(f"{original_split}-*.parquet"))
        for file_path, row in iter_parquet_rows(files):
            audio_bytes = audio_struct_bytes(row["audio"])
            measured_duration, sample_rate = audio_info(audio_bytes)
            duration = float(row.get("duration") or measured_duration)
            sample_id = f"worldspeech:{row.get('segment_id')}"
            project_split = (
                "test" if original_split == "test" else split_train_dev(sample_id)
            )
            if project_splits is not None and project_split not in project_splits:
                continue
            text = str(row["human_transcript"])
            yield CanonicalSample(
                sample_id=sample_id,
                source="worldspeech",
                source_config="disco-eth/WorldSpeech/fa_ir",
                original_split=original_split,
                project_split=project_split,
                split_origin="native" if project_split == "test" else "derived",
                audio_bytes=audio_bytes,
                audio_format="source",
                text=text,
                normalized_text=maybe_normalize(text) or "",
                duration_seconds=duration,
                sample_rate=sample_rate,
                speaker_or_group_id=str(
                    row.get("source_url") or row.get("session_date") or row.get("segment_id")
                ),
                license="cc-by-nc-4.0",
                source_url=str(row.get("source_url") or ""),
                metadata_json=metadata(
                    parquet_path=str(file_path),
                    asr_transcript=row.get("asr_transcript"),
                    cer=row.get("cer"),
                    snr=row.get("snr"),
                    dnsmos_sig=row.get("dnsmos_sig"),
                    dnsmos_bak=row.get("dnsmos_bak"),
                    dnsmos_ovr=row.get("dnsmos_ovr"),
                    dnsmos_p808=row.get("dnsmos_p808"),
                    source=row.get("source"),
                    session_date=row.get("session_date"),
                    segment_id=row.get("segment_id"),
                ),
            )


def omni_samples(
    raw_root: Path,
    *,
    dataset_dir: str,
    source: str,
    source_config: str,
    license_name: str,
    split_origin_by_split: dict[str, str],
    project_splits: set[str] | None = None,
) -> Any:
    base = raw_root / dataset_dir / "version=0"
    for split_dir in sorted(base.glob("corpus=*/split=*/language=fas_Arab")):
        original_split = split_dir.parent.name.removeprefix("split=")
        if project_splits is not None and original_split not in project_splits:
            continue
        files = sorted(split_dir.glob("*.parquet"))
        for row_number, (file_path, row) in enumerate(iter_parquet_rows(files)):
            audio_bytes = int8_list_bytes(row["audio_bytes"])
            duration = float(row["audio_size"]) / 16_000
            text = str(row["text"])
            row_id = f"{file_path.name}:{row.get('__index_level_0__', row_number)}"
            yield CanonicalSample(
                sample_id=f"{source}:{original_split}:{row_id}",
                source=source,
                source_config=source_config,
                original_split=original_split,
                project_split=original_split,
                split_origin=split_origin_by_split.get(original_split, "native"),
                audio_bytes=audio_bytes,
                audio_format="flac",
                text=text,
                normalized_text=maybe_normalize(text) or "",
                duration_seconds=duration,
                sample_rate=16_000,
                speaker_or_group_id=str(file_path),
                license=license_name,
                source_url="",
                metadata_json=metadata(parquet_path=str(file_path)),
            )


def load_cv25_durations(cv_root: Path) -> dict[str, float]:
    durations = {}
    with (cv_root / "clip_durations.tsv").open(encoding="utf-8", newline="") as handle:
        reader = common_voice_reader(handle)
        for row in reader:
            clip = row.get("clip")
            duration_ms = row.get("duration[ms]")
            if clip and duration_ms:
                durations[clip] = float(duration_ms) / 1000
    return durations


def cv25_official_paths(cv_root: Path) -> set[str]:
    paths = set()
    for split in ["train", "dev", "test"]:
        with (cv_root / f"{split}.tsv").open(encoding="utf-8", newline="") as handle:
            for row in common_voice_reader(handle):
                path = row.get("path")
                if path:
                    paths.add(Path(path).name)
    return paths


def cv25_samples(raw_root: Path, project_splits: set[str] | None = None) -> Any:
    cv_root = raw_root / "mozilla_data_collective/extracted/cv-corpus-25.0-2026-03-09/fa"
    durations = load_cv25_durations(cv_root)
    official_paths = cv25_official_paths(cv_root)
    split_map = {
        "train": "train",
        "dev": "dev",
        "test": "test",
        "validated": "train",
    }
    for original_split, project_split in split_map.items():
        if project_splits is not None and project_split not in project_splits:
            continue
        with (cv_root / f"{original_split}.tsv").open(encoding="utf-8", newline="") as handle:
            reader = common_voice_reader(handle)
            for row_index, row in enumerate(reader):
                path_value = row.get("path")
                if not path_value:
                    continue
                audio_name = Path(path_value).name
                if original_split == "validated" and audio_name in official_paths:
                    continue
                audio_path = cv_root / "clips" / audio_name
                if not audio_path.exists():
                    continue
                audio_bytes = audio_path.read_bytes()
                duration = durations.get(audio_name)
                if duration is None:
                    duration = audio_info(audio_bytes)[0]
                text = str(row.get("sentence") or "")
                yield CanonicalSample(
                    sample_id=f"common_voice_25_0:{original_split}:{audio_name}",
                    source="common_voice_25_0",
                    source_config="mozilla_data_collective/cv-corpus-25.0/fa",
                    original_split=original_split,
                    project_split=project_split,
                    split_origin="native" if original_split != "validated" else "native_remainder",
                    audio_bytes=audio_bytes,
                    audio_format=audio_path.suffix.removeprefix(".") or "mp3",
                    text=text,
                    normalized_text=maybe_normalize(text) or "",
                    duration_seconds=float(duration),
                    sample_rate=None,
                    speaker_or_group_id=str(row.get("client_id") or ""),
                    license="cc0-1.0",
                    source_url="",
                    metadata_json=metadata(
                        row_index=row_index,
                        original_audio_path=path_value,
                        sentence_id=row.get("sentence_id"),
                        sentence_domain=row.get("sentence_domain"),
                        up_votes=row.get("up_votes"),
                        down_votes=row.get("down_votes"),
                        age=row.get("age"),
                        gender=row.get("gender"),
                        accents=row.get("accents"),
                        variant=row.get("variant"),
                        locale=row.get("locale"),
                        segment=row.get("segment"),
                    ),
                )


def neyshekar_samples(raw_root: Path) -> Any:
    dataset_root = raw_root / "neyshekar_v3_asr_aligned"
    dataset_path = dataset_root / "dataset.json"
    audio_zip = dataset_root / "audio.zip"
    if not dataset_path.exists() or not audio_zip.exists():
        raise FileNotFoundError(
            "Neyshekar canonical builds require the repaired aligned source at "
            f"{dataset_root}. Download Peacockery/neyshekar-v3-asr-aligned first."
        )
    rows = json.loads(dataset_path.read_text(encoding="utf-8"))
    with zipfile.ZipFile(audio_zip) as archive:
        members = {Path(name).name: name for name in archive.namelist() if not name.endswith("/")}
        for row in rows:
            audio_name = str(row["audio"])
            member = members[audio_name]
            audio_bytes = archive.read(member)
            measured_duration, sample_rate = audio_info(audio_bytes)
            text = str(row["text"])
            sample_id = f"neyshekar:{row['id']}"
            yield CanonicalSample(
                sample_id=sample_id,
                source="neyshekar",
                source_config="Peacockery/neyshekar-v3-asr-aligned",
                original_split="none",
                project_split=split_train_dev_test(sample_id),
                split_origin="asr_aligned_derived",
                audio_bytes=audio_bytes,
                audio_format=Path(audio_name).suffix.removeprefix(".") or "wav",
                text=text,
                normalized_text=maybe_normalize(text) or "",
                duration_seconds=float(row.get("duration") or measured_duration),
                sample_rate=sample_rate,
                speaker_or_group_id=sample_id,
                license="cc0-1.0",
                source_url="https://huggingface.co/datasets/Peacockery/neyshekar-v3-asr-aligned",
                metadata_json=metadata(id=row["id"], audio=audio_name),
            )


def mana_tts_samples(raw_root: Path) -> Any:
    data_root = raw_root / "mana_tts/dataset"
    files = sorted(data_root.glob("dataset_part_*.parquet"))
    columns = [
        "file name",
        "transcript",
        "duration",
        "match quality",
        "hypothesis",
        "CER",
        "search type",
        "ASRs",
        "audio",
        "samplerate",
    ]
    for file_path in files:
        table = pq.read_table(file_path, columns=columns)
        values = table.drop(["audio"]).to_pydict()
        audio_column = table.column("audio").combine_chunks()
        offsets = audio_column.offsets.to_numpy(zero_copy_only=False)
        audio_values = audio_column.values.to_numpy(zero_copy_only=False)
        for row_index, file_name_value in enumerate(values["file name"]):
            file_name = str(file_name_value)
            sample_id = f"mana_tts:{file_name}"
            match_quality = str(values["match quality"][row_index])
            cer = finite_float(values["CER"][row_index])
            text = str(values["transcript"][row_index]).strip()
            normalized_text = maybe_normalize(text) or ""
            if match_quality not in MANA_TTS_ALLOWED_MATCH_QUALITIES:
                continue
            if text.casefold() in MANA_TTS_BAD_TRANSCRIPTS or not normalized_text:
                continue
            if cer is None or cer > MANA_TTS_MAX_CER:
                continue
            sample_rate = int(values["samplerate"][row_index] or DEFAULT_SAMPLE_RATE)
            audio_start = int(offsets[row_index])
            audio_end = int(offsets[row_index + 1])
            audio_bytes = float_audio_wav_bytes(audio_values[audio_start:audio_end], sample_rate)
            measured_duration, measured_sample_rate = audio_info(audio_bytes)
            source_duration = finite_float(values["duration"][row_index])
            duration_seconds = (
                measured_duration if measured_duration > 0 else (source_duration or 0.0)
            )
            if duration_seconds <= 0:
                continue
            yield CanonicalSample(
                sample_id=sample_id,
                source="mana_tts",
                source_config="MahtaFetrat/Mana-TTS",
                original_split="train",
                project_split=split_train_dev_test(sample_id),
                split_origin="derived",
                audio_bytes=audio_bytes,
                audio_format="wav",
                text=text,
                normalized_text=normalized_text,
                duration_seconds=duration_seconds,
                sample_rate=measured_sample_rate or sample_rate,
                speaker_or_group_id="nasl-e-mana",
                license="cc0-1.0",
                source_url="https://huggingface.co/datasets/MahtaFetrat/Mana-TTS",
                metadata_json=metadata(
                    parquet_path=str(file_path),
                    file_name=file_name,
                    match_quality=values["match quality"][row_index],
                    hypothesis=values["hypothesis"][row_index],
                    cer=values["CER"][row_index],
                    max_cer=MANA_TTS_MAX_CER,
                    source_duration_seconds=source_duration,
                    measured_duration_seconds=measured_duration,
                    search_type=values["search type"][row_index],
                    asrs=values["ASRs"][row_index],
                ),
            )


def filter_project_splits(samples: Any, project_splits: set[str] | None) -> Any:
    for sample in samples:
        if sample.duration_seconds <= 0 or not math.isfinite(sample.duration_seconds):
            continue
        if contains_latin_script(sample.text):
            continue
        if project_splits is None or sample.project_split in project_splits:
            yield sample


def build_all(
    data_root: Path,
    rows_per_file: int,
    *,
    datasets: set[str] | None = None,
    project_splits: set[str] | None = None,
) -> list[DatasetSummary]:
    raw_root = data_root / "raw"
    # Canonical datasets live flattened directly under data_root (no canonical/ wrapper).
    canonical_root = data_root
    tasks = [
        (
            "fleurs",
            omni_samples(
                raw_root,
                dataset_dir="fleurs_fa_ir_omni",
                source="fleurs",
                source_config="google/fleurs/fa_ir",
                license_name="cc-by-4.0",
                split_origin_by_split={"train": "native", "dev": "native", "test": "native"},
                project_splits=project_splits,
            ),
            "native",
            "cc-by-4.0",
            {"train", "dev", "test"},
        ),
        (
            "thomcles",
            omni_samples(
                raw_root,
                dataset_dir="thomcles_persian_omni",
                source="thomcles_persian_farsi_speech",
                source_config="Thomcles/Persian-Farsi-Speech",
                license_name="cc-by-4.0",
                split_origin_by_split={"train": "native", "dev": "derived"},
                project_splits=project_splits,
            ),
            "native_train_plus_derived_dev",
            "cc-by-4.0",
            {"train", "dev"},
        ),
        (
            "common_voice_25",
            cv25_samples(raw_root, project_splits),
            "native",
            "cc0-1.0",
            {"train", "dev", "test"},
        ),
        (
            "youtube",
            youtube_samples(raw_root, project_splits),
            "native",
            "unknown",
            {"train", "dev", "test"},
        ),
        (
            "worldspeech",
            worldspeech_samples(raw_root, project_splits),
            "native_test_plus_derived_dev",
            "cc-by-nc-4.0",
            {"train", "dev", "test"},
        ),
        (
            "neyshekar",
            neyshekar_samples(raw_root),
            "asr_aligned_derived",
            "cc0-1.0",
            {"train", "dev", "test"},
        ),
        ("mana_tts", mana_tts_samples(raw_root), "derived", "cc0-1.0", {"train", "dev", "test"}),
    ]
    summaries = []
    for name, samples, split_origin, license_name, available_splits in tasks:
        if datasets is not None and name not in datasets:
            continue
        if project_splits is not None and available_splits.isdisjoint(project_splits):
            continue
        output_root = canonical_root / name
        if output_root.exists():
            shutil.rmtree(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        summaries.append(
            write_dataset(
                filter_project_splits(samples, project_splits),
                source=name,
                source_config=name,
                split_origin=split_origin,
                license_name=license_name,
                output_root=output_root,
                rows_per_file=rows_per_file,
            )
        )
    index_summaries = [asdict(summary) for summary in summaries]
    index_path = canonical_root / "index.json"
    if datasets is not None and project_splits is None and index_path.exists():
        rebuilt_sources = {summary["source"] for summary in index_summaries}
        existing_summaries = json.loads(index_path.read_text(encoding="utf-8"))
        index_summaries.extend(
            summary for summary in existing_summaries if summary["source"] not in rebuilt_sources
        )
        index_summaries.sort(key=lambda summary: summary["source"])
    index_path.write_text(
        json.dumps(index_summaries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summaries
