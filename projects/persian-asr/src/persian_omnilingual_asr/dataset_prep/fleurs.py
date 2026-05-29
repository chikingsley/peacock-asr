from __future__ import annotations

import argparse
import io
from collections import defaultdict
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

from persian_asr_dataset.vendor.nvidia_stt_fa_fastconformer_hybrid_large import maybe_normalize

SPLIT_MAP = {"train": "train", "validation": "dev", "test": "test"}
OMNI_SAMPLE_RATE = 16_000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare FLEURS fa_ir for Omnilingual ASR mixture parquet recipes."
    )
    parser.add_argument("--dataset", default="google/fleurs")
    parser.add_argument("--config", default="fa_ir")
    parser.add_argument("--corpus", default="fleurs")
    parser.add_argument("--language", default="fas_Arab")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/simon/github/peacock-asr/projects/persian-asr/data/raw/fleurs_fa_ir_omni"),
    )
    parser.add_argument("--version", default="0")
    parser.add_argument("--rows-per-file", type=int, default=1000)
    parser.add_argument("--row-group-size", type=int, default=100)
    parser.add_argument("--limit-per-split", type=int, default=0)
    return parser


def audio_to_flac_int8(audio: dict[str, Any]) -> tuple[list[int], int]:
    waveform = np.asarray(audio["array"])
    sample_rate = int(audio["sampling_rate"])
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if sample_rate != OMNI_SAMPLE_RATE:
        waveform = librosa.resample(
            waveform.astype(np.float32), orig_sr=sample_rate, target_sr=OMNI_SAMPLE_RATE
        )
        sample_rate = OMNI_SAMPLE_RATE
    waveform = waveform.astype(np.float32, copy=False)
    buffer = io.BytesIO()
    sf.write(buffer, waveform, sample_rate, format="FLAC")
    encoded = np.frombuffer(buffer.getvalue(), dtype=np.int8)
    return encoded.tolist(), int(waveform.shape[0])


def text_for(example: dict[str, Any]) -> str:
    raw = example.get("raw_transcription") or example.get("transcription") or ""
    return maybe_normalize(str(raw)) or ""


def write_shard(
    rows: list[dict[str, Any]],
    out_dir: Path,
    shard_index: int,
    row_group_size: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(
        rows,
        schema=pa.schema(
            [
                ("text", pa.string()),
                ("audio_bytes", pa.list_(pa.int8())),
                ("audio_size", pa.int64()),
            ]
        ),
    )
    pq.write_table(
        table,
        out_dir / f"part-{shard_index:05d}.parquet",
        row_group_size=row_group_size,
    )


def prepare_split(args: argparse.Namespace, source_split: str, target_split: str) -> float:
    dataset = load_dataset(
        args.dataset,
        args.config,
        split=source_split,
        streaming=True,
        trust_remote_code=True,
    )
    out_dir = (
        args.output_root
        / f"version={args.version}"
        / f"corpus={args.corpus}"
        / f"split={target_split}"
        / f"language={args.language}"
    )
    shard: list[dict[str, Any]] = []
    shard_index = 0
    total_audio_size = 0
    total_rows = 0
    progress = tqdm(dataset, desc=f"{args.config} {source_split}->{target_split}", unit="utt")
    for example in progress:
        audio_bytes, audio_size = audio_to_flac_int8(example["audio"])
        shard.append(
            {
                "text": text_for(example),
                "audio_bytes": audio_bytes,
                "audio_size": audio_size,
            }
        )
        total_audio_size += audio_size
        total_rows += 1
        if len(shard) >= args.rows_per_file:
            write_shard(shard, out_dir, shard_index, args.row_group_size)
            shard.clear()
            shard_index += 1
        if args.limit_per_split and total_rows >= args.limit_per_split:
            break
    if shard:
        write_shard(shard, out_dir, shard_index, args.row_group_size)
    hours = total_audio_size / OMNI_SAMPLE_RATE / 3600
    print(f"{target_split}: {total_rows} rows, {hours:.4f} hours")
    return hours


def write_stats(args: argparse.Namespace, hours_by_key: dict[tuple[str, str], float]) -> Path:
    stats_path = args.output_root / f"language_distribution_{args.version}.tsv"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8") as handle:
        handle.write("corpus\tlanguage\thours\n")
        for (corpus, language), hours in sorted(hours_by_key.items()):
            handle.write(f"{corpus}\t{language}\t{hours:.8f}\n")
    print(f"wrote {stats_path}")
    return stats_path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    hours_by_key: dict[tuple[str, str], float] = defaultdict(float)
    for source_split, target_split in SPLIT_MAP.items():
        hours = prepare_split(args, source_split, target_split)
        hours_by_key[(args.corpus, args.language)] += hours
    write_stats(args, hours_by_key)
    print(f"dataset root: {args.output_root / f'version={args.version}'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
