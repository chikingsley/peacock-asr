# /// script
# requires-python = ">=3.10,<3.11"
# dependencies = [
#   "Cython<3",
#   "huggingface-hub<0.24",
#   "nemo-toolkit[nlp]==1.23.0",
#   "numpy<2",
#   "psutil",
#   "setuptools<81",
#   "transformers==4.36.2",
#   "wheel",
# ]
# ///
"""Restore English punctuation and capitalization with NVIDIA's lexical BERT model."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch
from nemo.collections.nlp.models import PunctuationCapitalizationModel


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_number} is not a JSON object")
            text = row.get("lexical_text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{path}:{line_number} has no lexical_text")
            rows.append(row)
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--model", default="punctuation_en_bert")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-seq-length", type=int, default=64)
    parser.add_argument("--step", type=int, default=8)
    parser.add_argument("--margin", type=int, default=16)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.output_manifest.exists():
        raise FileExistsError(f"immutable output already exists: {args.output_manifest}")
    summary_path = args.output_manifest.with_suffix(".summary.json")
    if summary_path.exists():
        raise FileExistsError(f"immutable output already exists: {summary_path}")
    rows = read_manifest(args.input_manifest)
    model = PunctuationCapitalizationModel.from_pretrained(args.model).to("cuda").eval()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    predictions = model.add_punctuation_capitalization(
        [row["lexical_text"] for row in rows],
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        step=args.step,
        margin=args.margin,
    )
    elapsed = time.monotonic() - started
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("x", encoding="utf-8") as handle:
        for row, prediction in zip(rows, predictions, strict=True):
            restored = dict(row)
            restored["prediction_text"] = prediction
            handle.write(json.dumps(restored, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "model": args.model,
        "rows": len(rows),
        "elapsed_seconds": elapsed,
        "rows_per_second": len(rows) / elapsed if elapsed else 0.0,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "input_manifest": str(args.input_manifest.resolve()),
        "input_sha256": sha256(args.input_manifest),
        "output_manifest": str(args.output_manifest.resolve()),
        "output_sha256": sha256(args.output_manifest),
        "inference": {
            "batch_size": args.batch_size,
            "max_seq_length": args.max_seq_length,
            "step": args.step,
            "margin": args.margin,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
