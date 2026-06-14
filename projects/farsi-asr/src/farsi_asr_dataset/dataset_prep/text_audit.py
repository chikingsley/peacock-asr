from __future__ import annotations

import argparse
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow.parquet as pq
import sentencepiece as spm

from farsi_asr_dataset.paths import PROJECT_ROOT

if TYPE_CHECKING:
    from collections.abc import Iterator


DEFAULT_ROOT = PROJECT_ROOT / "src" / "finetune_omni" / "data" / "training" / "omnilingual"
# The Omni char tokenizer, kept as a tracked local asset (override with env
# PERSIAN_OMNI_TOKENIZER). Re-download from:
# dl.fbaipublicfiles.com/mms/omniASR_tokenizer_written_v2.model
_OMNI_ASSETS = PROJECT_ROOT / "src" / "finetune_omni" / "assets"
DEFAULT_TOKENIZER = Path(
    os.environ.get(
        "PERSIAN_OMNI_TOKENIZER",
        str(_OMNI_ASSETS / "omniASR_tokenizer_written_v2.model"),
    )
)
WATCH_CHARS = {
    "\u200c": "ZWNJ",
    "\u2047": "DOUBLE_QUESTION_MARK",
    "\ufeff": "BOM",
    "\u200e": "LRM",
    "\u200f": "RLM",
    "\ufffd": "REPLACEMENT_CHARACTER",
}


@dataclass
class AuditStats:
    rows: int = 0
    char_counts: Counter[str] = field(default_factory=Counter)
    rows_by_char: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    examples: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    unk_rows: int = 0
    unk_pieces: Counter[str] = field(default_factory=Counter)
    unk_examples: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TextRow:
    path: Path
    corpus: str
    split: str
    row_index: int
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit Omnilingual Persian training parquet text for tokenizer-risk characters."
    )
    parser.add_argument("dataset", help="Dataset directory under data/training/omnilingual")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--examples", type=int, default=5)
    parser.add_argument("--max-rows", type=int, default=0)
    return parser.parse_args()


def load_sentencepiece(path: Path) -> spm.SentencePieceProcessor | None:
    if not path.exists():
        return None
    processor = spm.SentencePieceProcessor()
    processor.Load(str(path))
    return processor


def iter_text_rows(dataset_root: Path) -> Iterator[TextRow]:
    for path in sorted(dataset_root.rglob("*.parquet")):
        parts = {part.split("=", 1)[0]: part.split("=", 1)[1] for part in path.parts if "=" in part}
        corpus = parts.get("corpus", "")
        split = parts.get("split", "")
        parquet = pq.ParquetFile(path)
        row_offset = 0
        for batch in parquet.iter_batches(batch_size=2048, columns=["text"]):
            for row_index, row in enumerate(batch.to_pylist()):
                yield TextRow(path, corpus, split, row_offset + row_index, row["text"] or "")
            row_offset += batch.num_rows


def record_watch_chars(
    stats: AuditStats,
    row: TextRow,
    example_limit: int,
) -> None:
    for char, label in WATCH_CHARS.items():
        count = row.text.count(char)
        if count:
            stats.char_counts[label] += count
            stats.rows_by_char[label][f"{row.corpus}/{row.split}"] += 1
            if len(stats.examples[label]) < example_limit:
                stats.examples[label].append(f"{row.path}:{row.row_index}: {row.text}")


def record_tokenizer_unknowns(
    stats: AuditStats,
    tokenizer: spm.SentencePieceProcessor | None,
    row: TextRow,
    example_limit: int,
) -> None:
    if tokenizer is None:
        return
    unk_id = tokenizer.unk_id()
    ids = tokenizer.EncodeAsIds(row.text)
    if unk_id not in ids:
        return
    stats.unk_rows += 1
    pieces = tokenizer.EncodeAsPieces(row.text)
    for piece_id, piece in zip(ids, pieces, strict=True):
        if piece_id == unk_id:
            stats.unk_pieces[piece] += 1
    if len(stats.unk_examples) < example_limit:
        stats.unk_examples.append(f"{row.path}:{row.row_index}: {row.text}")


class TokenizerCoverageError(RuntimeError):
    """Raised when exported training text encodes to tokenizer unknowns."""


def audit_dataset(
    dataset_root: Path,
    tokenizer_path: Path = DEFAULT_TOKENIZER,
    examples: int = 5,
    max_rows: int = 0,
) -> tuple[AuditStats, spm.SentencePieceProcessor | None]:
    tokenizer = load_sentencepiece(tokenizer_path)
    stats = AuditStats()
    for row in iter_text_rows(dataset_root):
        stats.rows += 1
        for char in WATCH_CHARS:
            if row.text.count(char):
                record_watch_chars(stats, row, examples)
                break
        record_tokenizer_unknowns(stats, tokenizer, row, examples)

        if max_rows and stats.rows >= max_rows:
            break
    return stats, tokenizer


def audit(
    args: argparse.Namespace, dataset_root: Path
) -> tuple[AuditStats, spm.SentencePieceProcessor | None]:
    return audit_dataset(dataset_root, args.tokenizer, args.examples, args.max_rows)


def assert_tokenizer_clean(
    dataset_root: Path,
    tokenizer_path: Path = DEFAULT_TOKENIZER,
    examples: int = 5,
) -> AuditStats:
    """Hard gate: fail if any exported row encodes to the tokenizer's <unk>.

    Used as an export preflight so a ZWNJ-contaminated training set can never
    be built. See docs/zwnj-normalization-decision-20260529.md.
    """
    stats, tokenizer = audit_dataset(dataset_root, tokenizer_path, examples=examples)
    if tokenizer is None:
        raise TokenizerCoverageError(
            f"tokenizer unavailable for coverage gate: {tokenizer_path}"
        )
    if stats.unk_rows:
        top_pieces = stats.unk_pieces.most_common(10)
        raise TokenizerCoverageError(
            f"export has {stats.unk_rows} of {stats.rows} rows that encode to <unk> "
            f"(tokenizer={tokenizer_path.name}); top unknown pieces={top_pieces}. "
            "Fix normalization (e.g. ZWNJ -> space) before training. "
            "See docs/zwnj-normalization-decision-20260529.md."
        )
    return stats


def main() -> None:
    args = parse_args()
    dataset_root = args.root / args.dataset
    if not dataset_root.exists():
        raise SystemExit(f"missing dataset root: {dataset_root}")

    stats, tokenizer = audit(args, dataset_root)

    print(f"dataset={args.dataset}")
    print(f"root={dataset_root}")
    print(f"rows={stats.rows}")
    print(f"tokenizer={args.tokenizer if tokenizer is not None else 'unavailable'}")
    if tokenizer is not None:
        print(f"unk_id={tokenizer.unk_id()}")
        print(f"unk_decodes_to={tokenizer.DecodeIds([tokenizer.unk_id()])!r}")
        for char, label in WATCH_CHARS.items():
            print(f"piece_id[{label}]={tokenizer.PieceToId(char)}")
        print(f"unk_rows={stats.unk_rows}")
        print("unk_pieces_top=" + repr(stats.unk_pieces.most_common(20)))
        print("unk_examples:")
        for example in stats.unk_examples:
            print(f"  {example}")

    print("watched_char_counts:")
    for label, count in sorted(stats.char_counts.items()):
        rows_by_source = dict(stats.rows_by_char[label].most_common())
        print(f"  {label}: chars={count} rows_by_corpus_split={rows_by_source}")
        for example in stats.examples[label]:
            print(f"    {example}")


if __name__ == "__main__":
    main()
