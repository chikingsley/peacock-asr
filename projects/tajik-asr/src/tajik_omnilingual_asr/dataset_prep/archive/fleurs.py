from __future__ import annotations

import argparse
import csv
import io
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from tajik_omnilingual_asr.dataset_prep.text_normalization import normalize_text

OMNI_SAMPLE_RATE = 16_000
SPLIT_MAP = {"train": "train", "dev": "dev", "test": "test"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare FLEURS Tajik for Omnilingual ASR mixture parquet recipes."
    )
    parser.add_argument("--dataset", default="google/fleurs")
    parser.add_argument("--config", default="tg_tj")
    parser.add_argument("--corpus", default="fleurs")
    parser.add_argument("--language", default="tgk_Cyrl")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("/home/simon/github/peacock-asr/projects/tajik-asr/data/raw/fleurs_tg_tj_raw"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/simon/github/peacock-asr/projects/tajik-asr/data/raw/fleurs_tg_tj_omni"),
    )
    parser.add_argument("--version", default="0")
    parser.add_argument("--rows-per-file", type=int, default=1000)
    parser.add_argument("--row-group-size", type=int, default=100)
    return parser


def download_raw(args: argparse.Namespace) -> None:
    repo = args.dataset
    cfg = args.config
    raw = args.raw_root
    for split in ("train", "dev", "test"):
        tsv_path = hf_hub_download(repo, f"data/{cfg}/{split}.tsv", repo_type="dataset")
        audio_path = hf_hub_download(repo, f"data/{cfg}/audio/{split}.tar.gz", repo_type="dataset")
        out_dir = raw / split
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(tsv_path, out_dir / "data.tsv")
        shutil.unpack_archive(audio_path, out_dir / "audio", format="gztar")
        nested = out_dir / "audio" / split
        if nested.is_dir():
            for f in nested.iterdir():
                f.rename(out_dir / "audio" / f.name)
            nested.rmdir()
        wav_count = len(list((out_dir / "audio").glob("*.wav")))
        print(f"{split}: {wav_count} wav files")


def process_split(args: argparse.Namespace, split: str) -> float:
    tsv_path = args.raw_root / split / "data.tsv"
    audio_dir = args.raw_root / split / "audio"
    out_dir = (
        args.output_root
        / f"version={args.version}"
        / f"corpus={args.corpus}"
        / f"split={split}"
        / f"language={args.language}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    shard: list[dict[str, Any]] = []
    shard_index = 0
    total_audio_size = 0
    total_rows = 0

    with open(tsv_path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # header
        rows = list(reader)

    progress = tqdm(rows, desc=f"{args.config} {split}", unit="utt")
    for row in progress:
        audio_file = row[1]
        text = normalize_text(row[3])

        waveform, sr = sf.read(str(audio_dir / audio_file))
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        if sr != OMNI_SAMPLE_RATE:
            waveform = librosa.resample(
                waveform.astype(np.float32),
                orig_sr=sr,
                target_sr=OMNI_SAMPLE_RATE,
            )
        waveform = waveform.astype(np.float32, copy=False)

        buf = io.BytesIO()
        sf.write(buf, waveform, OMNI_SAMPLE_RATE, format="FLAC")
        audio_bytes = np.frombuffer(buf.getvalue(), dtype=np.int8).tolist()
        audio_size = int(waveform.shape[0])

        shard.append({"text": text, "audio_bytes": audio_bytes, "audio_size": audio_size})
        total_audio_size += audio_size
        total_rows += 1

        if len(shard) >= args.rows_per_file:
            _write_shard(shard, out_dir, shard_index, args.row_group_size)
            shard.clear()
            shard_index += 1

    if shard:
        _write_shard(shard, out_dir, shard_index, args.row_group_size)

    hours = total_audio_size / OMNI_SAMPLE_RATE / 3600
    print(f"{split}: {total_rows} rows, {hours:.4f} hours")
    return hours


def _write_shard(
    rows: list[dict[str, Any]], out_dir: Path, shard_index: int, row_group_size: int
) -> None:
    table = pa.Table.from_pylist(
        rows,
        schema=pa.schema([
            ("text", pa.string()),
            ("audio_bytes", pa.list_(pa.int8())),
            ("audio_size", pa.int64()),
        ]),
    )
    pq.write_table(
        table,
        out_dir / f"part-{shard_index:05d}.parquet",
        row_group_size=row_group_size,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    download_raw(args)
    hours_by_key: dict[tuple[str, str], float] = defaultdict(float)
    for split in ("train", "dev", "test"):
        hours = process_split(args, split)
        hours_by_key[(args.corpus, args.language)] += hours
    stats_path = args.output_root / f"language_distribution_{args.version}.tsv"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8") as f:
        f.write("corpus\tlanguage\thours\n")
        for (corpus, language), hours in sorted(hours_by_key.items()):
            f.write(f"{corpus}\t{language}\t{hours:.8f}\n")
    print(f"wrote {stats_path}")
    print(f"dataset root: {args.output_root / f'version={args.version}'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
