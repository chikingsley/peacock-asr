"""Regenerate 16 kHz mono FLAC audio that training manifests reference.

The Scribe-job audio cache (``data/curation/scribe_jobs/.../audio/*.flac``) is a
derived 16 kHz copy of the canonical audio. If it is missing, this rebuilds the
exact files a manifest's ``audio_filepath`` entries point at, decoding the
canonical ``audio_bytes`` (any rate/format) and resampling to 16 kHz mono FLAC.

One pass over the canonical parquets; only the sample_ids a manifest needs are
written. Re-runnable and idempotent (skips files that already exist).

  uv run python -m farsi_asr_dataset.dataset_prep.regenerate_audio \
    --manifest <train.jsonl> --manifest <dev.jsonl> [--limit N]
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import librosa
import pyarrow.parquet as pq
import soundfile as sf

from farsi_asr_dataset.paths import DEFAULT_DATA_ROOT

TARGET_SR = 16_000


def load_targets(manifests: list[Path]) -> dict[str, Path]:
    """Map sample_id -> destination audio_filepath from the manifest(s)."""
    targets: dict[str, Path] = {}
    for manifest in manifests:
        with manifest.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                targets[str(row["sample_id"])] = Path(str(row["audio_filepath"]))
    return targets


def to_flac_16k(raw: bytes) -> bytes:
    data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    mono = data if data.ndim == 1 else data.mean(axis=1)
    if sr != TARGET_SR:
        mono = librosa.resample(mono.astype("float32"), orig_sr=sr, target_sr=TARGET_SR)
    buf = io.BytesIO()
    sf.write(buf, mono, TARGET_SR, format="FLAC")
    return buf.getvalue()


def regenerate(targets: dict[str, Path], data_root: Path, limit: int = 0) -> tuple[int, int]:
    written = skipped = 0
    for parquet_path in sorted(data_root.glob("*/*/*.parquet")):
        pf = pq.ParquetFile(parquet_path)
        for batch in pf.iter_batches(batch_size=512, columns=["sample_id", "audio_bytes"]):
            sample_ids = batch.column(0).to_pylist()
            audio = batch.column(1).to_pylist()
            for sid, ab in zip(sample_ids, audio, strict=True):
                dest = targets.get(str(sid))
                if dest is None:
                    continue
                if dest.exists():
                    skipped += 1
                    continue
                raw = bytes(ab) if not isinstance(ab, (bytes, bytearray)) else ab
                if not raw:
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(to_flac_16k(raw))
                written += 1
                if limit and written >= limit:
                    return written, skipped
    return written, skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Regenerate 16 kHz FLAC audio for a manifest.")
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    targets = load_targets(args.manifest)
    have = sum(1 for p in targets.values() if p.exists())
    print(f"manifest targets: {len(targets)} | already present: {have}")
    written, skipped = regenerate(targets, args.data_root, args.limit)
    print(f"written: {written} | skipped(existing): {skipped}")
    missing = [sid for sid, p in targets.items() if not p.exists()]
    print(f"still missing after pass: {len(missing)}")
    if missing[:3]:
        print("missing examples:", missing[:3])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
