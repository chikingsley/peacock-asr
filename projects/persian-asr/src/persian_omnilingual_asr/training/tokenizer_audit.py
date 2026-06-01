"""Audit Omnilingual tokenizer coverage over the Persian training texts.

Entry point ``persian-omni-audit-tokenizer``. Mirrors
``tajik_omnilingual_asr.training.tokenizer_audit`` but reads texts from the omni-parquet
shards (Persian has no ``.wrd`` manifests — the data layer emits parquet only). Runs the
shared :func:`omni_finetune_core.tokenizer_audit.audit_texts` to report rows / unknown
rows / unknown tokens, then prints up to ``--max-examples`` rows that contain an unknown
token. Exit code is non-zero if any unknowns are found (so it can gate a build).
"""

from __future__ import annotations

import argparse
import glob
import os
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from fairseq2.data.tokenizers.hub import load_tokenizer
from omni_finetune_core.tokenizer_audit import audit_texts

ROOT = Path(__file__).resolve().parents[3]
# scribe-v4 is the current training line; override with --data-dir for an ablation set.
DEFAULT_DATA = ROOT / "src/finetune_omni/data/training/omnilingual/scribe-v4/version=0"
DEFAULT_TOKENIZER = "omniASR_tokenizer_written_v2"
LANG = "fas_Arab"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit Omnilingual tokenizer coverage.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA, help="version=0 parquet root")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--max-examples", type=int, default=20)
    return parser


def configure_environment() -> None:
    os.environ.setdefault("HF_HOME", str(ROOT / ".hf-cache"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(ROOT / ".hf-cache/datasets"))
    os.environ.setdefault("FAIRSEQ2_CACHE_DIR", str(ROOT / ".fairseq2-cache/assets"))


def iter_text(data_dir: Path):
    pattern = str(data_dir / "corpus=*/split=*/language=*/*.parquet")
    for path in sorted(glob.glob(pattern)):
        split = next((p.split("=")[1] for p in path.split("/") if p.startswith("split=")), "?")
        table = pq.read_table(path, columns=["text"])
        for line_nr, text in enumerate(table.column("text").to_pylist(), start=1):
            if text:
                yield split, line_nr, text


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_environment()

    tokenizer = load_tokenizer(args.tokenizer)
    encoder = tokenizer.create_raw_encoder()
    unk_idx = tokenizer.vocab_info.unk_idx

    # Materialize once so the shared core scorer and the per-example diagnostic agree.
    entries = list(iter_text(args.data_dir))

    report = audit_texts((text for _, _, text in entries), tokenizer)

    char_counts: Counter[str] = Counter()
    examples: list[tuple[str, int, int, str, list[str]]] = []
    for split, line_nr, text in entries:
        char_counts.update(text)
        token_ids = encoder(text).tolist()
        row_unknowns = token_ids.count(unk_idx) if unk_idx is not None else 0
        if row_unknowns and len(examples) < args.max_examples:
            tokens = encoder.encode_as_tokens(text)
            examples.append((split, line_nr, row_unknowns, text, tokens))

    print(f"tokenizer\t{args.tokenizer}")
    print(f"rows\t{report.rows}")
    print(f"unknown_rows\t{report.unk_rows}")
    print(f"unknown_tokens\t{report.unk_tokens}")
    print(f"unique_chars\t{len(char_counts)}")

    if examples:
        print("examples")
        for split, line_nr, count, text, tokens in examples:
            print(f"{split}:{line_nr}\tunknowns={count}\t{text}\t{tokens}")
        return 1

    return 0 if report.clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
