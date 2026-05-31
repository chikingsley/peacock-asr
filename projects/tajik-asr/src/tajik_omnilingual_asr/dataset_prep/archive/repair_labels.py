from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path

from tajik_omnilingual_asr.dataset_prep.text_normalization import normalize_text

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATASET = ROOT / "src/tajik_omnilingual_asr/dataset_prep/artifacts/tajik_asr_combined_v0"
SPLITS = ("train", "dev", "test")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Repair Tajik ASR labels with the project normalizer."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    return parser


def fetch_rows(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    conn.row_factory = sqlite3.Row
    return conn.execute(
        """
        select
          id, split, source, source_split, source_id, audio_filename,
          raw_transcription, audio_bytes, duplicate_count
        from utterances
        order by split, source, source_id, id
        """
    ).fetchall()


def repair_sqlite(conn: sqlite3.Connection, rows: list[sqlite3.Row]) -> int:
    repaired = []
    for row in rows:
        normalized = normalize_text(row["raw_transcription"])
        if not normalized:
            continue
        repaired.append(
            {
                "id": row["id"],
                "transcription": normalized,
                "normalized_text": normalized,
                "characters": len(normalized),
            }
        )

    conn.executemany(
        """
        update utterances
        set transcription = :transcription,
            normalized_text = :normalized_text,
            characters = :characters
        where id = :id
        """,
        repaired,
    )
    return len(repaired)


def write_tsvs(dataset_dir: Path, rows: list[sqlite3.Row]) -> None:
    header = [
        "id",
        "audio_filename",
        "raw_transcription",
        "transcription",
        "normalized_text",
        "characters",
        "audio_bytes",
        "source",
        "source_id",
        "duplicate_count",
    ]
    for split in SPLITS:
        split_rows = [row for row in rows if row["split"] == split]
        with (dataset_dir / split / "data.tsv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(header)
            for row in split_rows:
                normalized = normalize_text(row["raw_transcription"])
                if not normalized:
                    continue
                writer.writerow(
                    [
                        row["id"],
                        row["audio_filename"],
                        row["raw_transcription"],
                        normalized,
                        normalized,
                        len(normalized),
                        row["audio_bytes"],
                        row["source"],
                        row["source_id"],
                        row["duplicate_count"],
                    ]
                )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    db_path = args.dataset_dir / "tajik_asr_combined.sqlite"

    with sqlite3.connect(db_path) as conn:
        rows = fetch_rows(conn)
        repaired_count = repair_sqlite(conn, rows)
        write_tsvs(args.dataset_dir, rows)

    print(f"rows\t{len(rows)}")
    print(f"repaired_rows\t{repaired_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
