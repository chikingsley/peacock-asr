from __future__ import annotations

import argparse
import io
import tarfile
from csv import DictReader
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
from tqdm import tqdm

from tajik_omnilingual_asr.dataset_prep.text_normalization import normalize_text

OMNI_SAMPLE_RATE = 16_000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare Common Voice Tajik for Omnilingual ASR mixture parquet recipes."
    )
    parser.add_argument(
        "--tarball",
        type=Path,
        required=True,
        help="Path to common-voice-scripted-speech-25-0-tajik-*.tar.gz",
    )
    parser.add_argument("--corpus", default="common_voice_25")
    parser.add_argument("--language", default="tgk_Cyrl")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/simon/github/peacock-asr/projects/tajik-asr/data/raw/common_voice_25_tg_omni"),
    )
    parser.add_argument("--version", default="0")
    parser.add_argument("--rows-per-file", type=int, default=1000)
    parser.add_argument("--row-group-size", type=int, default=100)
    parser.add_argument("--limit-per-split", type=int, default=0)
    return parser


def audio_to_flac_int8(audio_bytes: bytes, sample_rate: int) -> tuple[list[int], int]:
    waveform, sr = sf.read(io.BytesIO(audio_bytes))
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if sr != OMNI_SAMPLE_RATE:
        waveform = librosa.resample(
            waveform.astype(np.float32), orig_sr=sr, target_sr=OMNI_SAMPLE_RATE
        )
        sr = OMNI_SAMPLE_RATE
    waveform = waveform.astype(np.float32, copy=False)
    buffer = io.BytesIO()
    sf.write(buffer, waveform, sr, format="FLAC")
    encoded = np.frombuffer(buffer.getvalue(), dtype=np.int8)
    return encoded.tolist(), int(waveform.shape[0])


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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    with tarfile.open(args.tarball, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.isfile()]
        all_paths: dict[str, str] = {}
        for m in members:
            if m.name.endswith(".mp3"):
                all_paths[Path(m.name).name] = m.name

        # Find TSV files for each split
        tsv_files: dict[str, str] = {}
        for m in members:
            if not m.name.endswith(".tsv"):
                continue
            stem = Path(m.name).stem
            split = stem.replace("-synced", "").replace("-validated", "")
            tsv_files[split] = m.name

        # Common Voice naming: "validated" is train, "dev" is dev, "test" is test
        split_map = {"validated": "train", "dev": "dev", "test": "test"}
        split_source = None
        for src in split_map:
            if src in tsv_files:
                split_source = src
                break
        if split_source is None:
            print(f"available TSVs: {list(tsv_files.keys())}")
            return 1
        target_split = split_map[split_source]

        # Read TSV
        tsv_path = tsv_files[split_source]
        tsv_file = tar.extractfile(str(tsv_path))
        if tsv_file is None:
            print(f"could not extract {tsv_path}")
            return 1
        reader = DictReader(tsv_file.read().decode("utf-8").splitlines(), delimiter="\t")

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
        progress = tqdm(unit="utt")
        for row in reader:
            clip_name = row.get("path", "")
            if not clip_name:
                continue
            text = normalize_text(row.get("sentence", ""))
            if clip_name not in all_paths:
                continue
            audio_member = tar.extractfile(all_paths[clip_name])
            if audio_member is None:
                continue
            audio_bytes = audio_member.read()
            audio_flac, audio_size = audio_to_flac_int8(audio_bytes, sample_rate=48000)
            shard.append(
                {
                    "text": text,
                    "audio_bytes": audio_flac,
                    "audio_size": audio_size,
                }
            )
            total_audio_size += audio_size
            total_rows += 1
            progress.update()
            if len(shard) >= args.rows_per_file:
                write_shard(shard, out_dir, shard_index, args.row_group_size)
                shard.clear()
                shard_index += 1
            if args.limit_per_split and total_rows >= args.limit_per_split:
                break

        if shard:
            write_shard(shard, out_dir, shard_index, args.row_group_size)
        progress.close()
        hours = total_audio_size / OMNI_SAMPLE_RATE / 3600
        print(f"{target_split}: {total_rows} rows, {hours:.4f} hours")

    # Write stats
    stats_path = args.output_root / f"language_distribution_{args.version}.tsv"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8") as handle:
        handle.write("corpus\tlanguage\thours\n")
        handle.write(f"{args.corpus}\t{args.language}\t{hours:.8f}\n")
    print(f"wrote {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
