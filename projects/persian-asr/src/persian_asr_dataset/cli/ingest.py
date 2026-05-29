from __future__ import annotations

import argparse
import csv
import io
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import soundfile as sf
from tqdm import tqdm

from persian_asr_dataset.ledger import (
    LedgerSample,
    connect_ledger,
    upsert_sample,
)
from persian_asr_dataset.paths import DATA_ROOT, DEFAULT_LEDGER
from persian_asr_dataset.vendor.nvidia_stt_fa_fastconformer_hybrid_large import maybe_normalize

csv.field_size_limit(sys.maxsize)


@dataclass(frozen=True)
class CommonVoiceFileInput:
    split: str
    split_row_index: int
    audio_path: Path
    duration_seconds: float | None
    ingest_version: str


def common_voice_reader(handle: Any) -> csv.DictReader[str]:
    # Common Voice TSV sentence fields can contain unmatched literal quote marks.
    return csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest Persian corpus rows into the ledger.")
    parser.add_argument(
        "--source",
        choices=(
            "common_voice_25_0",
            "fleurs_omni",
            "thomcles_omni",
        ),
        required=True,
    )
    parser.add_argument("--split", action="append", default=[])
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--ingest-version", default="v0")
    return parser


def decode_duration(encoded: bytes) -> tuple[float | None, int | None]:
    info = sf.info(io.BytesIO(encoded))
    return float(info.frames / info.samplerate), int(info.samplerate)


def duration_from_omni_audio_size(audio_size: int) -> float:
    return float(audio_size / 16_000)


def find_common_voice_tsv_root(source_root: Path) -> Path:
    if (source_root / "clips").is_dir() and any(source_root.glob("*.tsv")):
        return source_root
    matches = sorted(
        path for path in source_root.rglob("validated.tsv") if (path.parent / "clips").is_dir()
    )
    if not matches:
        raise FileNotFoundError(f"found no Common Voice TSV root under {source_root}")
    return matches[0].parent


def tsv_value(row: dict[str, str], key: str) -> str | None:
    value = row.get(key)
    return value or None


def load_clip_durations(cv_root: Path) -> dict[str, float]:
    duration_path = cv_root / "clip_durations.tsv"
    if not duration_path.exists():
        return {}
    durations: dict[str, float] = {}
    with duration_path.open(encoding="utf-8", newline="") as handle:
        reader = common_voice_reader(handle)
        for row in reader:
            clip = row.get("clip")
            duration_ms = row.get("duration[ms]")
            if clip and duration_ms:
                durations[clip] = float(duration_ms) / 1000
    return durations


def common_voice_file_sample(
    row: dict[str, str],
    sample_input: CommonVoiceFileInput,
) -> LedgerSample:
    sample_rate = None
    duration_seconds = sample_input.duration_seconds
    if sample_input.duration_seconds is None:
        duration_seconds, sample_rate = decode_duration(sample_input.audio_path.read_bytes())
    raw_text = row.get("sentence") or ""
    audio_name = sample_input.audio_path.name
    return LedgerSample(
        sample_id=f"common_voice_25_0:{sample_input.split}:{audio_name}",
        source="common_voice_25_0",
        source_split=sample_input.split,
        source_row_id=str(sample_input.split_row_index),
        raw_text=raw_text,
        normalized_text=maybe_normalize(raw_text) or "",
        duration_seconds=duration_seconds,
        sample_rate=sample_rate,
        audio_ref=str(sample_input.audio_path),
        storage_kind="common_voice_audio_file",
        metadata={
            "original_audio_path": str(Path(row.get("path") or audio_name)),
            "split_row_index": sample_input.split_row_index,
            "client_id": tsv_value(row, "client_id"),
            "up_votes": tsv_value(row, "up_votes"),
            "down_votes": tsv_value(row, "down_votes"),
            "age": tsv_value(row, "age"),
            "gender": tsv_value(row, "gender"),
            "accent": tsv_value(row, "accent"),
            "locale": tsv_value(row, "locale"),
            "segment": tsv_value(row, "segment"),
            "variant": tsv_value(row, "variant"),
        },
        ingest_version=sample_input.ingest_version,
    )


def ingest_common_voice_files(args: argparse.Namespace) -> int:
    splits = args.split or ["train", "dev", "test", "validated", "other", "invalidated"]
    source_root = args.source_root or DATA_ROOT / "raw/mozilla_data_collective/extracted"
    cv_root = find_common_voice_tsv_root(source_root)
    clip_durations = load_clip_durations(cv_root)
    connection = connect_ledger(args.ledger)
    count = 0
    progress = tqdm(desc="common_voice_25_0 ingest", unit="row")
    for split in splits:
        tsv_path = cv_root / f"{split}.tsv"
        if not tsv_path.exists():
            continue
        with tsv_path.open(encoding="utf-8", newline="") as handle:
            reader = common_voice_reader(handle)
            for split_row_index, row in enumerate(reader):
                path_value = row.get("path")
                if not path_value:
                    continue
                audio_path = cv_root / "clips" / Path(path_value).name
                if not audio_path.exists():
                    continue
                sample_input = CommonVoiceFileInput(
                    split=split,
                    split_row_index=split_row_index,
                    audio_path=audio_path,
                    duration_seconds=clip_durations.get(audio_path.name),
                    ingest_version=args.ingest_version,
                )
                sample = common_voice_file_sample(row, sample_input)
                upsert_sample(connection, sample)
                count += 1
                progress.update()
                if args.limit and count >= args.limit:
                    connection.commit()
                    progress.close()
                    return count
        connection.commit()
    progress.close()
    connection.commit()
    return count


def omni_dataset(source: str) -> tuple[str, str, list[str]]:
    if source == "fleurs_omni":
        return "fleurs", "fleurs_fa_ir_omni", ["train", "dev", "test"]
    return "thomcles_persian_farsi_speech", "thomcles_persian_omni", ["train", "dev"]


def ingest_omni(args: argparse.Namespace) -> int:
    source_name, dataset_dir, default_splits = omni_dataset(args.source)
    splits = args.split or default_splits
    base = DATA_ROOT / "raw" / dataset_dir / "version=0"
    connection = connect_ledger(args.ledger)
    count = 0
    progress = tqdm(desc=f"{source_name} ingest", unit="row")
    for split in splits:
        pattern = f"corpus=*/split={split}/language=fas_Arab/*.parquet"
        for parquet_path in sorted(base.glob(pattern)):
            parquet = pq.ParquetFile(parquet_path)
            row_index = 0
            for batch in parquet.iter_batches(batch_size=512, columns=["text", "audio_size"]):
                for row in batch.to_pylist():
                    raw_text = str(row.get("text") or "")
                    audio_size = int(row["audio_size"])
                    sample = LedgerSample(
                        sample_id=f"{source_name}:{split}:{parquet_path.name}:{row_index}",
                        source=source_name,
                        source_split=split,
                        source_row_id=f"{parquet_path.name}:{row_index}",
                        raw_text=raw_text,
                        normalized_text=maybe_normalize(raw_text) or "",
                        duration_seconds=duration_from_omni_audio_size(audio_size),
                        sample_rate=16_000,
                        audio_ref=f"{parquet_path}:{row_index}",
                        storage_kind="omni_parquet",
                        metadata={
                            "parquet_path": str(parquet_path),
                            "audio_size": audio_size,
                        },
                        ingest_version=args.ingest_version,
                    )
                    upsert_sample(connection, sample)
                    count += 1
                    row_index += 1
                    progress.update()
                    if args.limit and count >= args.limit:
                        connection.commit()
                        progress.close()
                        return count
            connection.commit()
    progress.close()
    connection.commit()
    return count


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.source == "common_voice_25_0":
        count = ingest_common_voice_files(args)
    else:
        count = ingest_omni(args)
    print(f"ingested {count} rows into {args.ledger}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
