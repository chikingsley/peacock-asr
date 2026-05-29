from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from tqdm import tqdm

from persian_asr_dataset.dataset.ledger import DEFAULT_LEDGER, connect_ledger
from persian_asr_dataset.paths import DEFAULT_NEMO_RUNS, configure_external_caches
from persian_asr_dataset.vendor.nvidia_stt_fa_fastconformer_hybrid_large import maybe_normalize

NEMO_MODEL_CARD = "nvidia/stt_fa_fastconformer_hybrid_large"
NEMO_SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class ExportSample:
    sample_id: str
    source: str
    source_split: str
    text: str
    duration_seconds: float
    sample_rate: int | None
    audio_ref: str
    storage_kind: str
    audio_path: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export ledger rows to a NeMo Curator JSONL manifest with audio files."
    )
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--run-id", default=datetime.now(UTC).strftime("nemo-fa-%Y%m%dT%H%M%SZ"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_NEMO_RUNS)
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--split", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-duration", type=float, default=1.0)
    parser.add_argument("--max-duration", type=float, default=20.0)
    parser.add_argument("--language", default="fa")
    parser.add_argument("--overwrite-audio", action="store_true")
    return parser


def placeholders(values: list[str]) -> str:
    return ", ".join("?" for _ in values)


def fetch_samples(args: argparse.Namespace) -> tuple[list[ExportSample], dict[str, int], Path]:
    run_dir = args.output_root / args.run_id
    audio_dir = run_dir / "audio"
    clauses = ["duration_seconds >= ?", "duration_seconds <= ?"]
    params: list[Any] = [args.min_duration, args.max_duration]

    if args.source:
        clauses.append(f"source IN ({placeholders(args.source)})")
        params.extend(args.source)
    if args.split:
        clauses.append(f"source_split IN ({placeholders(args.split)})")
        params.extend(args.split)

    limit_clause = " LIMIT ?" if args.limit else ""
    if args.limit:
        params.append(args.limit)

    query = f"""
        SELECT sample_id, source, source_split, raw_text, duration_seconds,
               sample_rate, audio_ref, storage_kind
        FROM samples
        WHERE {" AND ".join(clauses)}
        ORDER BY source, source_split, sample_id
        {limit_clause}
    """

    connection = connect_ledger(args.ledger)
    counters = {"candidate_rows": 0, "skipped_text_normalization": 0}
    samples: list[ExportSample] = []
    for row in connection.execute(query, params):
        counters["candidate_rows"] += 1
        (
            sample_id,
            source,
            source_split,
            raw_text,
            duration,
            sample_rate,
            audio_ref,
            storage_kind,
        ) = row
        text = maybe_normalize(str(raw_text))
        if text is None:
            counters["skipped_text_normalization"] += 1
            continue
        samples.append(
            ExportSample(
                sample_id=str(sample_id),
                source=str(source),
                source_split=str(source_split),
                text=text,
                duration_seconds=float(duration),
                sample_rate=int(sample_rate) if sample_rate is not None else None,
                audio_ref=str(audio_ref),
                storage_kind=str(storage_kind),
                audio_path=audio_dir / hashed_audio_name(str(sample_id)),
            )
        )
    return samples, counters, run_dir


def hashed_audio_name(sample_id: str) -> Path:
    digest = hashlib.sha1(sample_id.encode("utf-8")).hexdigest()
    return Path(digest[:2]) / f"{digest}.flac"


def parse_audio_ref(sample: ExportSample) -> tuple[Path, int]:
    if sample.storage_kind == "common_voice_parquet":
        parquet_path, row_index, _audio_name = sample.audio_ref.rsplit(":", 2)
    else:
        parquet_path, row_index = sample.audio_ref.rsplit(":", 1)
    return Path(parquet_path), int(row_index)


def common_voice_audio_bytes(row: dict[str, Any]) -> bytes:
    audio = row["audio"]
    encoded = audio.get("bytes")
    if encoded is None:
        path = audio.get("path")
        if path is None:
            raise ValueError("Common Voice row has neither audio bytes nor audio path")
        encoded = Path(path).read_bytes()
    return bytes(encoded)


def omni_audio_bytes(row: dict[str, Any]) -> bytes:
    audio_values = row["audio_bytes"]
    return np.asarray(audio_values, dtype=np.int8).tobytes()


def decode_to_16k_mono(encoded: bytes) -> np.ndarray:
    waveform, sample_rate = sf.read(io.BytesIO(encoded), dtype="float32", always_2d=False)
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if sample_rate != NEMO_SAMPLE_RATE:
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=NEMO_SAMPLE_RATE)
    return np.asarray(waveform, dtype=np.float32)


def materialize_audio(samples: list[ExportSample], overwrite: bool) -> dict[str, str]:
    by_parquet: dict[Path, list[tuple[ExportSample, int]]] = defaultdict(list)
    for sample in samples:
        parquet_path, row_index = parse_audio_ref(sample)
        by_parquet[parquet_path].append((sample, row_index))

    failures: dict[str, str] = {}
    progress = tqdm(samples, desc="materialize audio", unit="utt")
    for parquet_path, grouped in by_parquet.items():
        storage_kind = grouped[0][0].storage_kind
        column = "audio" if storage_kind == "common_voice_parquet" else "audio_bytes"
        rows = pq.read_table(parquet_path, columns=[column]).to_pylist()
        for sample, row_index in grouped:
            try:
                if sample.audio_path.exists() and not overwrite:
                    progress.update()
                    continue
                if row_index >= len(rows):
                    raise IndexError(
                        f"row index {row_index} is outside {parquet_path.name} rows={len(rows)}"
                    )
                encoded = (
                    common_voice_audio_bytes(rows[row_index])
                    if storage_kind == "common_voice_parquet"
                    else omni_audio_bytes(rows[row_index])
                )
                waveform = decode_to_16k_mono(encoded)
                sample.audio_path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(sample.audio_path, waveform, NEMO_SAMPLE_RATE, format="FLAC")
            except Exception as exc:  # noqa: BLE001 - export should keep going and report failures.
                failures[sample.sample_id] = f"{type(exc).__name__}: {exc}"
            finally:
                progress.update()
    progress.close()
    return failures


def write_manifest(
    samples: list[ExportSample],
    run_dir: Path,
    audio_failures: dict[str, str],
    language: str,
) -> Path:
    manifest_dir = run_dir / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            if sample.sample_id in audio_failures:
                continue
            payload = {
                "audio_filepath": str(sample.audio_path.resolve()),
                "text": sample.text,
                "duration": sample.duration_seconds,
                "language": language,
                "sample_rate": NEMO_SAMPLE_RATE,
                "sample_id": sample.sample_id,
                "source": sample.source,
                "source_split": sample.source_split,
                "model_card": NEMO_MODEL_CARD,
            }
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    return manifest_path


def write_run_metadata(
    args: argparse.Namespace,
    run_dir: Path,
    manifest_path: Path,
    counters: dict[str, int],
    audio_failures: dict[str, str],
) -> Path:
    metadata_path = run_dir / "run_metadata.json"
    payload = {
        "run_id": args.run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "model_card": NEMO_MODEL_CARD,
        "ledger": str(args.ledger),
        "manifest_path": str(manifest_path),
        "filters": {
            "source": args.source,
            "split": args.split,
            "limit": args.limit,
            "min_duration": args.min_duration,
            "max_duration": args.max_duration,
            "language": args.language,
            "normalizer": "nvidia/stt_fa_fastconformer_hybrid_large model-card maybe_normalize",
        },
        "counts": {
            **counters,
            "audio_failures": len(audio_failures),
        },
        "audio_failures": audio_failures,
    }
    metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata_path


def main(argv: list[str] | None = None) -> int:
    configure_external_caches()
    args = build_parser().parse_args(argv)
    samples, counters, run_dir = fetch_samples(args)
    audio_failures = materialize_audio(samples, overwrite=args.overwrite_audio)
    manifest_path = write_manifest(samples, run_dir, audio_failures, args.language)
    exported_count = len(samples) - len(audio_failures)
    counters["exported_rows"] = exported_count
    metadata_path = write_run_metadata(
        args,
        run_dir,
        manifest_path,
        counters,
        audio_failures,
    )
    print(f"wrote {manifest_path}")
    print(f"wrote {metadata_path}")
    print(f"exported {exported_count} rows")
    if audio_failures:
        print(f"audio failures: {len(audio_failures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
