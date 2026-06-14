from __future__ import annotations

import argparse
import io
import json
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import librosa
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from tqdm import tqdm

from farsi_asr_dataset.paths import DEFAULT_DATA_ROOT, configure_external_caches

if TYPE_CHECKING:
    from collections.abc import Iterator

NEMO_SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class ExportStats:
    split: str
    rows_seen: int
    rows_written: int
    rows_skipped_duration: int
    rows_skipped_text: int
    audio_failures: int
    hours_written: float
    manifest_path: str
    audio_root: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export canonical Persian ASR Parquet splits to NeMo JSONL manifests."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--run-id",
        default=datetime.now(UTC).strftime("canonical-nemo-%Y%m%dT%H%M%SZ"),
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--split", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-duration", type=float, default=0.1)
    parser.add_argument("--max-duration", type=float, default=20.0)
    parser.add_argument("--overwrite-audio", action="store_true")
    return parser


def canonical_root(data_root: Path) -> Path:
    return data_root / "canonical"


def split_root(data_root: Path, dataset: str, split: str) -> Path:
    return canonical_root(data_root) / dataset / split


def available_datasets(data_root: Path) -> list[str]:
    root = canonical_root(data_root)
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def iter_rows(data_root: Path, datasets: list[str], split: str) -> Iterator[dict[str, Any]]:
    for dataset in datasets:
        root = split_root(data_root, dataset, split)
        if not root.exists():
            continue
        for parquet_path in sorted(root.glob("*.parquet")):
            parquet = pq.ParquetFile(parquet_path)
            for batch in parquet.iter_batches(batch_size=512):
                for row in batch.to_pylist():
                    row["canonical_dataset"] = dataset
                    yield row


def decode_with_soundfile(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    waveform, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    return np.asarray(waveform, dtype=np.float32), int(sample_rate)


def decode_with_librosa(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    with tempfile.NamedTemporaryFile(suffix=".audio") as handle:
        handle.write(audio_bytes)
        handle.flush()
        waveform, sample_rate = librosa.load(handle.name, sr=None, mono=False)
    return np.asarray(waveform, dtype=np.float32), int(sample_rate)


def decode_audio(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    try:
        return decode_with_soundfile(audio_bytes)
    except sf.LibsndfileError:
        return decode_with_librosa(audio_bytes)


def to_16k_mono(audio_bytes: bytes) -> np.ndarray:
    waveform, sample_rate = decode_audio(audio_bytes)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if sample_rate != NEMO_SAMPLE_RATE:
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=NEMO_SAMPLE_RATE)
    return np.asarray(waveform, dtype=np.float32)


def audio_path(audio_root: Path, sample_id: str) -> Path:
    safe_id = sample_id.replace("/", "_").replace(":", "_")
    return audio_root / safe_id[:2] / f"{safe_id}.flac"


def export_split(
    args: argparse.Namespace,
    datasets: list[str],
    split: str,
    run_dir: Path,
) -> ExportStats:
    manifest_dir = run_dir / "manifests"
    audio_root = run_dir / "audio" / split
    manifest_dir.mkdir(parents=True, exist_ok=True)
    audio_root.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{split}.jsonl"
    rows_seen = 0
    rows_written = 0
    rows_skipped_duration = 0
    rows_skipped_text = 0
    audio_failures = 0
    hours_written = 0.0
    progress = tqdm(iter_rows(args.data_root, datasets, split), desc=f"export {split}", unit="utt")
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for row in progress:
            rows_seen += 1
            if args.limit and rows_written >= args.limit:
                break
            duration = float(row["duration_seconds"])
            if duration < args.min_duration or duration > args.max_duration:
                rows_skipped_duration += 1
                continue
            text = str(row.get("normalized_text") or row.get("text") or "").strip()
            if not text:
                rows_skipped_text += 1
                continue
            path = audio_path(audio_root, str(row["sample_id"]))
            try:
                if args.overwrite_audio or not path.exists():
                    waveform = to_16k_mono(bytes(row["audio_bytes"]))
                    path.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(path, waveform, NEMO_SAMPLE_RATE, format="FLAC")
            except (sf.LibsndfileError, RuntimeError, ValueError, OSError) as exc:
                audio_failures += 1
                progress.write(f"audio failure {row['sample_id']}: {type(exc).__name__}: {exc}")
                continue
            payload = {
                "audio_filepath": str(path.resolve()),
                "duration": duration,
                "text": text,
                "sample_id": row["sample_id"],
                "source": row["source"],
                "source_config": row["source_config"],
                "canonical_dataset": row["canonical_dataset"],
                "split": split,
            }
            manifest.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            rows_written += 1
            hours_written += duration / 3600
    progress.close()
    return ExportStats(
        split=split,
        rows_seen=rows_seen,
        rows_written=rows_written,
        rows_skipped_duration=rows_skipped_duration,
        rows_skipped_text=rows_skipped_text,
        audio_failures=audio_failures,
        hours_written=hours_written,
        manifest_path=str(manifest_path),
        audio_root=str(audio_root),
    )


def main(argv: list[str] | None = None) -> int:
    configure_external_caches()
    args = build_parser().parse_args(argv)
    splits = args.split or ["train", "dev"]
    output_root = args.output_root or args.data_root / "training" / "parakeet" / "nemo_manifests"
    run_dir = output_root / args.run_id
    datasets = args.dataset or available_datasets(args.data_root)
    stats = [export_split(args, datasets, split, run_dir) for split in splits]
    metadata = {
        "run_id": args.run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "canonical_root": str(canonical_root(args.data_root)),
        "datasets": datasets,
        "filters": {
            "splits": splits,
            "limit": args.limit,
            "min_duration": args.min_duration,
            "max_duration": args.max_duration,
        },
        "stats": [asdict(item) for item in stats],
    }
    metadata_path = run_dir / "export_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
