from __future__ import annotations

import argparse
import io
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import HfApi, HfFileSystem
from tqdm import tqdm

from persian_asr_dataset.text_normalization import maybe_normalize

OMNI_SAMPLE_RATE = 16_000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare Thomcles/Persian-Farsi-Speech for Omnilingual ASR recipes."
    )
    parser.add_argument("--dataset", default="Thomcles/Persian-Farsi-Speech")
    parser.add_argument("--revision", default="refs/convert/parquet")
    parser.add_argument("--subset", default="default")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-column", default="sentence")
    parser.add_argument("--corpus", default="thomcles_persian_farsi_speech")
    parser.add_argument("--language", default="fas_Arab")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/simon/github/peacock-asr/projects/persian-asr/data/raw/thomcles_persian_omni"),
    )
    parser.add_argument("--version", default="0")
    parser.add_argument("--rows-per-file", type=int, default=200)
    parser.add_argument("--row-group-size", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dev-every", type=int, default=100)
    parser.add_argument("--min-mos-ovr", type=float, default=3.0)
    parser.add_argument("--max-audio-seconds", type=float, default=60.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-shards", type=int, default=0)
    return parser


def audio_to_flac_int8(audio: dict[str, Any]) -> tuple[list[int], int]:
    encoded = audio.get("bytes")
    if not encoded:
        path = audio.get("path")
        if path is None:
            raise ValueError("audio example has neither bytes nor path")
        encoded = Path(path).read_bytes()

    waveform, sample_rate = sf.read(io.BytesIO(encoded), dtype="float32", always_2d=False)
    waveform = np.asarray(waveform)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if sample_rate != OMNI_SAMPLE_RATE:
        raise ValueError(f"expected 16 kHz audio, got {sample_rate} Hz")

    buffer = io.BytesIO()
    sf.write(buffer, waveform.astype(np.float32, copy=False), sample_rate, format="FLAC")
    flac = np.frombuffer(buffer.getvalue(), dtype=np.int8)
    return flac.tolist(), int(waveform.shape[0])


def target_split(index: int, dev_every: int) -> str:
    if dev_every > 0 and index % dev_every == 0:
        return "dev"
    return "train"


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


def flush_split(
    split_name: str,
    shards: dict[str, list[dict[str, Any]]],
    shard_indices: dict[str, int],
    args: argparse.Namespace,
) -> None:
    rows = shards[split_name]
    if not rows:
        return

    out_dir = (
        args.output_root
        / f"version={args.version}"
        / f"corpus={args.corpus}"
        / f"split={split_name}"
        / f"language={args.language}"
    )
    write_shard(rows, out_dir, shard_indices[split_name], args.row_group_size)
    rows.clear()
    shard_indices[split_name] += 1


def write_stats(args: argparse.Namespace, hours: float) -> Path:
    stats_path = args.output_root / f"language_distribution_{args.version}.tsv"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8") as handle:
        handle.write("corpus\tlanguage\thours\n")
        handle.write(f"{args.corpus}\t{args.language}\t{hours:.8f}\n")
    print(f"wrote {stats_path}")
    return stats_path


def parquet_paths(args: argparse.Namespace) -> list[str]:
    api = HfApi()
    files = api.list_repo_files(args.dataset, repo_type="dataset", revision=args.revision)
    prefix = f"{args.subset}/{args.split}/"
    paths = sorted(path for path in files if path.startswith(prefix) and path.endswith(".parquet"))
    if args.max_shards:
        paths = paths[: args.max_shards]
    if not paths:
        raise ValueError(
            f"found no parquet shards for {args.dataset}@{args.revision} under {prefix}"
        )
    return paths


def remote_path(args: argparse.Namespace, parquet_path: str) -> str:
    return f"datasets/{args.dataset}@{args.revision}/{parquet_path}"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    filesystem = HfFileSystem()
    paths = parquet_paths(args)

    shards: dict[str, list[dict[str, Any]]] = {"train": [], "dev": []}
    shard_indices: dict[str, int] = defaultdict(int)
    split_rows: dict[str, int] = defaultdict(int)
    split_hours: dict[str, float] = defaultdict(float)
    source_index = 0
    kept_rows = 0

    progress = tqdm(desc=args.dataset, unit="utt")
    for path in paths:
        with filesystem.open(remote_path(args, path), "rb") as handle:
            parquet = pq.ParquetFile(handle)
            for batch in parquet.iter_batches(batch_size=args.batch_size):
                for example in batch.to_pylist():
                    mos_ovr = example.get("mos_ovr")
                    if mos_ovr is not None and float(mos_ovr) < args.min_mos_ovr:
                        source_index += 1
                        progress.update()
                        continue

                    audio_bytes, audio_size = audio_to_flac_int8(example["audio"])
                    seconds = audio_size / OMNI_SAMPLE_RATE
                    if seconds > args.max_audio_seconds:
                        source_index += 1
                        progress.update()
                        continue

                    split_name = target_split(source_index, args.dev_every)
                    shards[split_name].append(
                        {
                            "text": maybe_normalize(str(example.get(args.text_column) or ""))
                            or "",
                            "audio_bytes": audio_bytes,
                            "audio_size": audio_size,
                        }
                    )
                    split_rows[split_name] += 1
                    split_hours[split_name] += seconds / 3600
                    kept_rows += 1
                    source_index += 1
                    progress.update()

                    if len(shards[split_name]) >= args.rows_per_file:
                        flush_split(split_name, shards, shard_indices, args)

                    if args.limit and kept_rows >= args.limit:
                        break

                if args.limit and kept_rows >= args.limit:
                    break

        if args.limit and kept_rows >= args.limit:
            break

    progress.close()

    for split_name in shards:
        flush_split(split_name, shards, shard_indices, args)

    total_hours = sum(split_hours.values())
    write_stats(args, total_hours)
    for split_name in ("train", "dev"):
        print(f"{split_name}: {split_rows[split_name]} rows, {split_hours[split_name]:.4f} hours")
    print(f"dataset root: {args.output_root / f'version={args.version}'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
