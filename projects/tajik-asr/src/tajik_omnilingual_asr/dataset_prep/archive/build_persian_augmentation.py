"""Build a Persian->Tajik transliterated augmentation partition for the Tajik
omni-parquet (a `corpus=persian_translit_*` partition that sits alongside the
real Tajik corpora under the same `version=0` tree).

Pipeline per Persian (Farsi) utterance: transliterate the transcript to Tajik
Cyrillic (ParsTranslit, with the می/نمی-attach fix) -> Tajik-normalize ->
soft-filter by Tajik-vocabulary coverage and drop residual Perso-Arabic ->
resample the audio to 16 kHz mono FLAC -> write parquet rows matching the Tajik
omni schema (text, audio_bytes:int8, audio_size).

This is CONTROLLED augmentation (see project memory): `--hours` caps the slice,
`--min-coverage` drops Iran-specific/garbled rows. The output is meant to be
mixed a minority share with real Tajik and GATED on the real Tajik test split.

  uv run --no-sync python -m tajik_omnilingual_asr.dataset_prep.build_persian_augmentation \
    --hours 6 --datasets fleurs --output-root <v1>/omni_parquet/version=0
"""

from __future__ import annotations

import argparse
import io
import re
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

from tajik_omnilingual_asr.dataset_prep.archive.parstranslit import fa_to_tajik
from tajik_omnilingual_asr.dataset_prep.text_normalization import normalize_text

SAMPLE_RATE = 16_000
LANGUAGE = "tgk_Cyrl"
SCHEMA = pa.schema(
    [
        ("text", pa.string()),
        ("audio_bytes", pa.list_(pa.int8())),
        ("audio_size", pa.int64()),
    ]
)
_VOCAB_FILE = Path(__file__).resolve().parent / "parstranslit" / "tajik_vocab.txt"
_PUNCT = re.compile(r"[^\w\s]")
_ARABIC = re.compile(r"[؀-ۿ]")
_PERSIAN_ASR_DATA = Path(__file__).resolve().parents[5] / "persian-asr" / "data"


def load_vocab() -> set[str]:
    return set(_VOCAB_FILE.read_text(encoding="utf-8").split())


def coverage(text: str, vocab: set[str]) -> float:
    words = [w for w in _PUNCT.sub(" ", text.lower()).split() if w]
    return sum(w in vocab for w in words) / len(words) if words else 0.0


def to_flac_16k(raw: bytes) -> bytes:
    data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    mono = data if data.ndim == 1 else data.mean(axis=1)
    if sr != SAMPLE_RATE:
        mono = librosa.resample(mono.astype("float32"), orig_sr=sr, target_sr=SAMPLE_RATE)
    buf = io.BytesIO()
    sf.write(buf, mono, SAMPLE_RATE, format="FLAC")
    return buf.getvalue()


def iter_persian_rows(
    data_root: Path,
    datasets: list[str],
    splits: list[str],
    min_dur: float,
    max_dur: float,
    min_words: int,
):
    for dataset in datasets:
        for split in splits:
            for parquet_path in sorted((data_root / dataset / split).glob("*.parquet")):
                pf = pq.ParquetFile(parquet_path)
                cols = ["text", "audio_bytes", "duration_seconds"]
                for batch in pf.iter_batches(batch_size=256, columns=cols):
                    for row in batch.to_pylist():
                        dur = float(row.get("duration_seconds") or 0.0)
                        text = (row.get("text") or "").strip()
                        audio = row.get("audio_bytes")
                        if (
                            dur < min_dur
                            or dur > max_dur
                            or len(text.split()) < min_words
                            or not audio
                        ):
                            continue
                        yield text, bytes(audio), dur


def new_columns() -> dict[str, list[Any]]:
    return {"text": [], "audio_bytes": [], "audio_size": []}


def write_shard(
    columns: dict[str, list[Any]], out_dir: Path, shard_index: int, row_group_size: int
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_arrays(
        [
            pa.array(columns["text"], type=pa.string()),
            pa.array(columns["audio_bytes"], type=pa.list_(pa.int8())),
            pa.array(columns["audio_size"], type=pa.int64()),
        ],
        schema=SCHEMA,
    )
    pq.write_table(
        table, out_dir / f"part-{shard_index:05d}.parquet", row_group_size=row_group_size
    )
    for v in columns.values():
        v.clear()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build a Persian->Tajik transliterated augmentation partition."
    )
    p.add_argument("--persian-root", type=Path, default=_PERSIAN_ASR_DATA)
    p.add_argument("--datasets", default="fleurs", help="comma-separated canonical datasets")
    p.add_argument("--splits", default="train", help="comma-separated splits")
    p.add_argument("--hours", type=float, default=6.0, help="cap on kept audio hours (0 = no cap)")
    p.add_argument("--min-coverage", type=float, default=0.55)
    p.add_argument("--min-duration", type=float, default=1.0)
    p.add_argument("--max-duration", type=float, default=20.0)
    p.add_argument("--min-words", type=int, default=2)
    p.add_argument("--corpus", default="persian_translit_fleurs")
    p.add_argument("--split", default="train", help="target split label in the parquet partition")
    p.add_argument("--output-root", type=Path, required=True, help="omni_parquet/version=0 dir")
    p.add_argument("--rows-per-file", type=int, default=1000)
    p.add_argument("--row-group-size", type=int, default=100)
    p.add_argument("--limit", type=int, default=0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    vocab = load_vocab()
    out_dir = (
        args.output_root / f"corpus={args.corpus}" / f"split={args.split}" / f"language={LANGUAGE}"
    )
    columns = new_columns()
    shard = kept = seen = 0
    kept_sec = 0.0
    drop = {"empty": 0, "coverage": 0, "arabic": 0, "audio": 0}
    target_sec = args.hours * 3600

    for text, raw, dur in iter_persian_rows(
        args.persian_root,
        args.datasets.split(","),
        args.splits.split(","),
        args.min_duration,
        args.max_duration,
        args.min_words,
    ):
        if target_sec and kept_sec >= target_sec:
            break
        if args.limit and seen >= args.limit:
            break
        seen += 1
        tajik = normalize_text(fa_to_tajik(text))
        if not tajik:
            drop["empty"] += 1
            continue
        if _ARABIC.search(tajik):
            drop["arabic"] += 1
            continue
        if coverage(tajik, vocab) < args.min_coverage:
            drop["coverage"] += 1
            continue
        try:
            flac = to_flac_16k(raw)
        except Exception:
            drop["audio"] += 1
            continue
        columns["text"].append(tajik)
        columns["audio_bytes"].append(np.frombuffer(flac, dtype=np.int8))
        columns["audio_size"].append(round(dur * SAMPLE_RATE))
        kept += 1
        kept_sec += dur
        if len(columns["text"]) >= args.rows_per_file:
            write_shard(columns, out_dir, shard, args.row_group_size)
            shard += 1
    if columns["text"]:
        write_shard(columns, out_dir, shard, args.row_group_size)

    print(f"output: {out_dir}")
    print(f"seen={seen} kept={kept} hours={kept_sec / 3600:.2f}")
    print(f"dropped: {drop}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
