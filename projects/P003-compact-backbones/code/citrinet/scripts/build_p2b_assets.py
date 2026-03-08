#!/usr/bin/env python3
"""Build tokenizer and NeMo manifests for Citrinet P2-B runs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from datasets import Audio, load_dataset

if TYPE_CHECKING:
    from collections.abc import Iterable as IterableRows

HF_DATASET_REPO = "gilkeyio/librispeech-alignments"
TARGET_SAMPLE_RATE = 16_000
STRESS_RE = re.compile(r"[012]$")
REPO_ROOT = Path(__file__).resolve().parents[5]


def strip_stress(phone: str) -> str:
    return STRESS_RE.sub("", phone)


def load_vocab(vocab_json: Path) -> dict[str, int]:
    with vocab_json.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _force_exit_success() -> None:
    os_module = __import__("os")
    exit_fn = os_module.__dict__["_exit"]
    exit_fn(0)


def write_wpe_vocab(vocab_json: Path, output_dir: Path) -> Path:
    vocab = load_vocab(vocab_json)
    ordered = [token for token, _ in sorted(vocab.items(), key=lambda item: item[1])]
    output_dir.mkdir(parents=True, exist_ok=True)
    vocab_txt = output_dir / "vocab.txt"
    vocab_txt.write_text("\n".join(ordered) + "\n", encoding="utf-8")

    metadata = {
        "source_vocab_json": str(vocab_json),
        "tokenizer_type": "wpe",
        "base_token_count": len(ordered),
        "tokens": ordered,
        "sample_rate": TARGET_SAMPLE_RATE,
        "blank_handling": (
            "CTC blank is implicit and not listed in vocab.txt. "
            "NeMo WPE adds [CLS], [SEP], and [MASK] during tokenizer init."
        ),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return vocab_txt


def _iter_split(
    split_name: str,
    *,
    max_samples: int | None,
    cache_dir: Path,
) -> IterableRows[dict[str, Any]]:
    stream = load_dataset(
        HF_DATASET_REPO,
        split=split_name,
        streaming=True,
        cache_dir=str(cache_dir),
    )
    stream = stream.cast_column("audio", Audio(decode=False))
    if max_samples is not None and max_samples > 0:
        return stream.take(max_samples)
    return stream


def _extract_audio_file(row: dict[str, Any], output_dir: Path, index: int) -> Path:
    audio = row["audio"]
    path_hint = str(audio.get("path") or f"sample-{index}.flac")
    suffix = Path(path_hint).suffix or ".flac"
    file_name = f"{row.get('id', f'sample-{index}')}{suffix}"
    target = output_dir / file_name
    if not target.exists():
        payload = audio.get("bytes")
        if not isinstance(payload, (bytes, bytearray)):
            msg = (
                f"Dataset row {row.get('id', index)} does not contain "
                "inline audio bytes"
            )
            raise SystemExit(msg)
        target.write_bytes(bytes(payload))
    return target


def _phones_to_text(row: dict[str, Any], vocab: dict[str, int]) -> str | None:
    phones = [strip_stress(item["phoneme"]) for item in row["phonemes"]]
    phones = [phone for phone in phones if phone in vocab and not phone.startswith("[")]
    if not phones:
        return None
    return " ".join(phones)


def export_manifest(
    *,
    split_names: list[str],
    max_samples: int | None,
    cache_dir: Path,
    audio_dir: Path,
    output_manifest: Path,
    vocab_json: Path,
) -> dict[str, float | int | str]:
    vocab = load_vocab(vocab_json)

    audio_dir.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    total_duration = 0.0
    limit = None if max_samples is None or max_samples <= 0 else max_samples
    with output_manifest.open("w", encoding="utf-8") as handle:
        for split_name in split_names:
            split_limit = None
            if limit is not None:
                remaining = limit - written
                if remaining <= 0:
                    break
                split_limit = remaining
            split_rows = _iter_split(
                split_name,
                max_samples=split_limit,
                cache_dir=cache_dir,
            )
            for row in split_rows:
                text = _phones_to_text(row, vocab)
                if text is None:
                    continue
                audio_path = _extract_audio_file(
                    row,
                    output_dir=audio_dir,
                    index=written,
                )
                duration = None
                for segment in row["phonemes"]:
                    duration = float(segment["end"])
                if duration is None:
                    continue
                record = {
                    "audio_filepath": str(audio_path.resolve()),
                    "duration": duration,
                    "text": text,
                    "utt_id": str(row.get("id", f"sample-{written}")),
                }
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                written += 1
                total_duration += duration
                if limit is not None and written >= limit:
                    break
    return {
        "rows": written,
        "hours": total_duration / 3600.0,
        "manifest": str(output_manifest.resolve()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vocab-json",
        type=Path,
        default=REPO_ROOT / Path(
            "projects/P003-compact-backbones/code/training/vocab.json"
        ),
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=REPO_ROOT / Path(
            "projects/P003-compact-backbones/code/citrinet/tokenizers/arpabet_41_wpe"
        ),
    )
    parser.add_argument(
        "--train-splits",
        nargs="+",
        default=["train_clean_100"],
    )
    parser.add_argument(
        "--eval-split",
        type=str,
        default="dev_clean",
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=64,
        help="Number of train samples to export. Use 0 for the full split.",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=16,
        help="Number of eval samples to export. Use 0 for the full split.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT
        / Path("projects/P003-compact-backbones/experiments/citrinet/preflight"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=REPO_ROOT
        / Path("projects/P003-compact-backbones/.cache/data/hf-datasets"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vocab_txt = write_wpe_vocab(args.vocab_json, args.tokenizer_dir)

    audio_root = args.output_root / "audio"
    manifests_root = args.output_root / "manifests"
    train_manifest = manifests_root / "train.jsonl"
    eval_manifest = manifests_root / "eval.jsonl"

    train_stats = export_manifest(
        split_names=args.train_splits,
        max_samples=args.train_samples,
        cache_dir=args.cache_dir,
        audio_dir=audio_root / "train",
        output_manifest=train_manifest,
        vocab_json=args.vocab_json,
    )
    eval_stats = export_manifest(
        split_names=[args.eval_split],
        max_samples=args.eval_samples,
        cache_dir=args.cache_dir,
        audio_dir=audio_root / "eval",
        output_manifest=eval_manifest,
        vocab_json=args.vocab_json,
    )

    summary_path = args.output_root / "asset_summary.tsv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["artifact", "path"])
        writer.writerow(["tokenizer_vocab", str(vocab_txt.resolve())])
        writer.writerow(["train_manifest", str(train_manifest.resolve())])
        writer.writerow(["train_rows", str(train_stats["rows"])])
        writer.writerow(["train_hours", f"{train_stats['hours']:.3f}"])
        writer.writerow(["eval_manifest", str(eval_manifest.resolve())])
        writer.writerow(["eval_rows", str(eval_stats["rows"])])
        writer.writerow(["eval_hours", f"{eval_stats['hours']:.3f}"])
        writer.writerow(["train_samples_requested", str(args.train_samples)])
        writer.writerow(["eval_samples_requested", str(args.eval_samples)])

    sys.stdout.write(f"{summary_path}\n")
    sys.stdout.flush()
    # Hugging Face datasets / pyarrow teardown is crashing at interpreter
    # finalization in this isolated NeMo env after successful writes.
    _force_exit_success()
    # datasets/pyarrow teardown is crashing at interpreter finalization in this
    # isolated NeMo env after successful writes. Exit immediately once outputs
    # are safely on disk.
    os._exit(0)


if __name__ == "__main__":
    main()
