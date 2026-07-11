"""Materialize NeMo manifests + audio files from local Farsi parquet splits.

Local omni-export corpora (fleurs/neyshekar/worldspeech) carry `audio_bytes` +
`normalized_text`; the restored upstream YouTube shards carry an `audio` struct +
`transcription` whose text still needs the canonical Persian normalizer.

Usage:
    uv run --no-sync experiments/parakeet/make_manifests.py \
        --corpus fleurs --split train
    uv run --no-sync experiments/parakeet/make_manifests.py \
        --corpus youtube_hf --split train --shards data/youtube_hf/data/train-0000[0-3]*.parquet
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow.parquet as pq
import soundfile as sf

from farsi_asr import LANGUAGE, ROOT

if TYPE_CHECKING:
    from collections.abc import Iterator

MIN_DUR = 0.5
MAX_DUR = 30.0
TARGET_SR = 16000
LOCAL_CORPORA = ("fleurs", "neyshekar", "worldspeech")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--shards", type=Path, nargs="*", default=None)
    parser.add_argument("--audio-root", type=Path, default=ROOT / "data/parakeet/audio")
    parser.add_argument("--manifest-root", type=Path, default=ROOT / "data/parakeet/manifests")
    parser.add_argument("--limit", type=int, default=0)
    return parser


def iter_rows(paths: list[Path]) -> Iterator[tuple[str, bytes, str, bool]]:
    """Yield (sample_id, audio_bytes, text, needs_processing) per row.

    Omni-export rows are already 16 kHz mono FLAC with export-normalized text;
    upstream rows (audio struct) need both the transcode and the text normalizer.
    """
    for path in paths:
        parquet = pq.ParquetFile(path)
        names = set(parquet.schema_arrow.names)
        if {"audio_bytes", "normalized_text"}.issubset(names):
            columns = ["sample_id", "audio_bytes", "normalized_text"]
            for batch in parquet.iter_batches(batch_size=128, columns=columns):
                for sample_id, audio, text in zip(
                    batch.column("sample_id").to_pylist(),
                    batch.column("audio_bytes").to_pylist(),
                    batch.column("normalized_text").to_pylist(),
                    strict=True,
                ):
                    yield (sample_id, audio, text, False)
        elif {"audio", "transcription"}.issubset(names):
            stem = path.stem
            index = 0
            for batch in parquet.iter_batches(batch_size=128, columns=["audio", "transcription"]):
                for audio, text in zip(
                    batch.column("audio").to_pylist(),
                    batch.column("transcription").to_pylist(),
                    strict=True,
                ):
                    yield (f"{stem}_{index:06d}", audio["bytes"], text, True)
                    index += 1
        else:
            raise SystemExit(f"unsupported schema in {path}: {sorted(names)}")


def to_16k_flac_bytes(audio: bytes) -> bytes:
    """Decode any soundfile-readable bytes to 16 kHz mono FLAC bytes."""
    import numpy as np
    import soxr

    data, rate = sf.read(io.BytesIO(audio), dtype="float32")
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if rate != TARGET_SR:
        data = soxr.resample(data, rate, TARGET_SR)
    out = io.BytesIO()
    sf.write(out, data, TARGET_SR, format="FLAC")
    return out.getvalue()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.shards:
        paths = sorted(args.shards)
    elif args.corpus in LOCAL_CORPORA:
        paths = sorted((ROOT / "data" / args.corpus / args.split).glob("*.parquet"))
    else:
        raise SystemExit(f"--shards is required for corpus {args.corpus!r}")
    if not paths:
        raise SystemExit(f"no parquet inputs for {args.corpus}/{args.split}")

    audio_dir = args.audio_root / args.corpus / args.split
    audio_dir.mkdir(parents=True, exist_ok=True)
    args.manifest_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest_root / f"{args.corpus}_{args.split}.jsonl"

    kept = 0
    skips = {"empty-text": 0, "out-of-bounds": 0, "undecodable": 0}
    total_seconds = 0.0
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for sample_id, audio, text, needs_processing in iter_rows(paths):
            record = build_record(
                audio_dir / f"{sample_id}.flac", audio, text, needs_processing=needs_processing
            )
            if isinstance(record, str):
                skips[record] += 1
                continue
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1
            total_seconds += record["duration"]
            if args.limit and kept >= args.limit:
                break
            if kept % 2000 == 0:
                print(f"  {kept} rows ({total_seconds / 3600:.2f} h)", flush=True)

    skip_note = ", ".join(f"{count} {reason}" for reason, count in skips.items())
    print(
        f"{manifest_path}: kept {kept} rows / {total_seconds / 3600:.2f} h (skipped {skip_note})",
        flush=True,
    )
    return 0


def build_record(
    audio_path: Path, audio: bytes, text: str, *, needs_processing: bool
) -> dict | str:
    """Return a manifest record, or the skip-reason key for an unusable row."""
    from omni_curator.process import normalize

    final_text = normalize(text, LANGUAGE) if needs_processing else text
    if not final_text.strip():
        return "empty-text"
    final_audio = None
    try:
        if audio_path.exists():
            info = sf.info(str(audio_path))
        else:
            final_audio = to_16k_flac_bytes(audio) if needs_processing else audio
            info = sf.info(io.BytesIO(final_audio))
    except (sf.LibsndfileError, RuntimeError):
        return "undecodable"
    if not MIN_DUR <= info.duration <= MAX_DUR:
        return "out-of-bounds"
    if final_audio is not None:
        audio_path.write_bytes(final_audio)
    return {
        "audio_filepath": str(audio_path),
        "text": final_text,
        "duration": round(info.duration, 3),
    }


if __name__ == "__main__":
    raise SystemExit(main())
