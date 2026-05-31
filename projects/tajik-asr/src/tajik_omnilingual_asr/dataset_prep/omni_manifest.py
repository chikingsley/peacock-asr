from __future__ import annotations

import argparse
import csv
from pathlib import Path

import librosa

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = ROOT / "src/tajik_omnilingual_asr/dataset_prep/artifacts/tajik_asr_combined_v0"
DEFAULT_OUT = DEFAULT_DATASET / "omni_manifest"
TARGET_SAMPLE_RATE = 16_000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export Tajik ASR TSV/audio data to Omnilingual manifest format."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    return parser


def audio_length_samples(path: Path) -> int:
    duration = librosa.get_duration(path=path)
    return max(1, round(duration * TARGET_SAMPLE_RATE))


def export_split(dataset_dir: Path, output_dir: Path, split: str) -> int:
    input_tsv = dataset_dir / split / "data.tsv"
    output_tsv = output_dir / f"{split}.tsv"
    output_wrd = output_dir / f"{split}.wrd"

    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    with input_tsv.open("r", encoding="utf-8", newline="") as src:
        reader = csv.DictReader(src, delimiter="\t")
        with output_tsv.open("w", encoding="utf-8", newline="") as tsv_out:
            with output_wrd.open("w", encoding="utf-8", newline="") as wrd_out:
                tsv_out.write(f"{dataset_dir.resolve()}\n")
                for row in reader:
                    rel_audio = row["audio_filename"]
                    audio_path = dataset_dir / rel_audio
                    length = audio_length_samples(audio_path)
                    text = row["normalized_text"].strip()
                    if not text:
                        continue
                    tsv_out.write(f"{rel_audio}\t{length}\n")
                    wrd_out.write(f"{text}\n")
                    count += 1

    return count


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()

    counts = {
        split: export_split(dataset_dir, output_dir, split)
        for split in ("train", "dev", "test")
    }
    for split, count in counts.items():
        print(f"{split}\t{count}")
    print(f"manifest_dir\t{output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
