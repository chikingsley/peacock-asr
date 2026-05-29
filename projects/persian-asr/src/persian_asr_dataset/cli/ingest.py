from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm

from persian_asr_dataset.dataset.ledger import (
    DEFAULT_LEDGER,
    LedgerSample,
    connect_ledger,
    upsert_sample,
)
from persian_asr_dataset.paths import RAW_ROOT, configure_external_caches
from persian_asr_dataset.vendor.nvidia_stt_fa_fastconformer_hybrid_large import maybe_normalize


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest Persian corpus rows into the ledger.")
    parser.add_argument(
        "--source",
        choices=("fleurs_omni", "thomcles_omni"),
        required=True,
    )
    parser.add_argument("--split", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--ingest-version", default="v0")
    return parser


def duration_from_omni_audio_size(audio_size: int) -> float:
    return float(audio_size / 16_000)


def omni_dataset(source: str) -> tuple[str, str, list[str]]:
    if source == "fleurs_omni":
        return "fleurs", "fleurs_fa_ir_omni", ["train", "dev", "test"]
    return "thomcles_persian_farsi_speech", "thomcles_persian_omni", ["train", "dev"]


def ingest_omni(args: argparse.Namespace) -> int:
    source_name, dataset_dir, default_splits = omni_dataset(args.source)
    splits = args.split or default_splits
    base = RAW_ROOT / dataset_dir / "version=0"
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
    configure_external_caches()
    args = build_parser().parse_args(argv)
    count = ingest_omni(args)
    print(f"ingested {count} rows into {args.ledger}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
