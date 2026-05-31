"""Resample the combined Tajik artifact audio to a uniform 16 kHz mono.

The Omnilingual parquet export records ``audio_size = duration * 16000`` for every
clip, but fairseq2's ``AudioDecoder`` decodes at the file's *native* sample rate and
never resamples. Any non-16 kHz clip therefore decodes to more samples than the
length-batcher budgeted, so buckets overshoot ``max_num_elements`` and training OOMs
(and the model is fed the wrong rate). The combined builder copied raw audio verbatim,
so the Common Voice clips landed at 32 kHz.

This rewrites every clip that is not already 16 kHz mono to a 16 kHz mono PCM WAV,
renaming the on-disk file to ``.wav`` and updating the sqlite + per-split ``data.tsv``
references. It is idempotent: already-16 kHz-mono clips are skipped.
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path

import librosa
import soundfile as sf

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATASET = ROOT / "src/tajik_omnilingual_asr/dataset_prep/artifacts/tajik_asr_combined_v0"
TARGET_SAMPLE_RATE = 16_000
SPLITS = ("train", "dev", "test")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resample combined Tajik artifact audio to uniform 16 kHz mono."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without rewriting any files.",
    )
    return parser


def is_already_clean(path: Path) -> bool:
    """True if the clip is already 16 kHz mono (and readable by libsndfile)."""
    try:
        info = sf.info(str(path))
    except Exception:
        return False
    return info.samplerate == TARGET_SAMPLE_RATE and info.channels == 1


def resample_file(path: Path) -> Path:
    """Rewrite ``path`` as a 16 kHz mono PCM WAV; returns the new path (``.wav``)."""
    waveform, _ = librosa.load(str(path), sr=TARGET_SAMPLE_RATE, mono=True)
    target = path.with_suffix(".wav")
    sf.write(str(target), waveform, TARGET_SAMPLE_RATE, subtype="PCM_16")
    if target != path:
        path.unlink()
    return target


def update_sqlite(
    db_path: Path, renames: dict[str, str], sizes: dict[str, int]
) -> None:
    with sqlite3.connect(db_path) as conn:
        for old_rel, new_rel in renames.items():
            conn.execute(
                "update utterances set audio_filename = ?, audio_bytes = ? "
                "where audio_filename = ?",
                (new_rel, sizes[new_rel], old_rel),
            )
        # Files whose name did not change but whose byte size did (rare: same-suffix rewrite).
        for rel, size in sizes.items():
            if rel not in renames.values():
                conn.execute(
                    "update utterances set audio_bytes = ? where audio_filename = ?",
                    (size, rel),
                )


def update_tsv(dataset_dir: Path, renames: dict[str, str]) -> None:
    if not renames:
        return
    for split in SPLITS:
        tsv_path = dataset_dir / split / "data.tsv"
        if not tsv_path.exists():
            continue
        with tsv_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            rows = list(reader)
        if not rows:
            continue
        header = rows[0]
        try:
            col = header.index("audio_filename")
        except ValueError:
            continue
        changed = False
        for row in rows[1:]:
            if col < len(row) and row[col] in renames:
                row[col] = renames[row[col]]
                changed = True
        if changed:
            with tsv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t")
                writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset_dir = args.dataset_dir.resolve()
    db_path = dataset_dir / "tajik_asr_combined.sqlite"

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "select audio_filename from utterances order by audio_filename"
        ).fetchall()

    renames: dict[str, str] = {}
    sizes: dict[str, int] = {}
    examined = converted = skipped = missing = 0

    for row in rows:
        old_rel = str(row["audio_filename"])
        path = dataset_dir / old_rel
        examined += 1
        if not path.exists():
            missing += 1
            print(f"MISSING\t{old_rel}")
            continue
        if is_already_clean(path):
            skipped += 1
            continue
        if args.dry_run:
            converted += 1
            print(f"would-resample\t{old_rel}")
            continue
        new_path = resample_file(path)
        new_rel = new_path.relative_to(dataset_dir).as_posix()
        if new_rel != old_rel:
            renames[old_rel] = new_rel
        sizes[new_rel] = new_path.stat().st_size
        converted += 1

    if not args.dry_run:
        update_sqlite(db_path, renames, sizes)
        update_tsv(dataset_dir, renames)

    print(
        f"examined={examined} converted={converted} skipped={skipped} "
        f"missing={missing} renamed={len(renames)} dry_run={args.dry_run}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
