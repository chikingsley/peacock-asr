"""Export a Georgian dataset ablation from the curator store to omni-parquet.

  georgian-export v0                 # everything ≤40 s, coverage-gated (the raw baseline)
  georgian-export v1 --max-wer 0.25  # + Scribe-WER gate at NeMo's "acceptable" broadcast tier

An ablation is a recipe over the master pool (``data/curator.sqlite``), not a copy: it writes only
the training rows (``text`` + ``audio_bytes`` + ``audio_size``) under ``data/datasets/<name>``; the
store is never mutated. Every export runs the omni char-tokenizer coverage gate — the normalized
labels must encode with zero ``<unk>`` or the export fails (``--no-strict`` downgrades to a warn).

WER tiers come from ``omni_curator.quality`` (see the package's QUALITY.md): broadcast
excellent/good/acceptable = 0.05/0.15/0.25. ``--max-wer`` reads ``Sample.scribe_wer`` (run verify
first); omitted = the raw baseline (no WER gate). The 40 s ceiling is Omni's hard input limit.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from omni_curator.export import Selection, export_dataset
from omni_curator.quality import OMNI_MAX_DURATION_S
from omni_curator.store import CuratorStore

_ROOT = Path(__file__).resolve().parents[2]
DATA = _ROOT / "data"
DB = DATA / "curator.sqlite"
DATASETS = DATA / "datasets"
TOKENIZER = Path(__file__).resolve().parent / "models" / "omniASR_tokenizer_written_v2.model"


def _coverage_check(texts: list[str]) -> int:
    """Count rows whose normalized text produces a tokenizer ``<unk>`` (the export gate)."""
    from fairseq2.data.tokenizers.char import load_char_tokenizer
    from omni_finetune_core.tokenizer_audit import audit_texts

    tokenizer = load_char_tokenizer(TOKENIZER, None)
    return audit_texts(texts, tokenizer).unk_rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export a Georgian ablation to omni-parquet.")
    parser.add_argument("name", help="ablation dir under data/datasets/ (e.g. v0)")
    parser.add_argument(
        "--max-wer",
        type=float,
        default=None,
        help="drop clips with Scribe WER above this (needs verify run); omit for raw baseline",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=OMNI_MAX_DURATION_S,
        help=f"drop clips longer than this (seconds; default {OMNI_MAX_DURATION_S}, Omni's limit)",
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="warn instead of failing when the coverage gate finds <unk> rows",
    )
    args = parser.parse_args(argv)

    selection = Selection(
        max_duration_seconds=args.max_duration,
        max_scribe_wer=args.max_wer,
    )
    store = CuratorStore(DB)
    stats = export_dataset(
        store,
        DATASETS / args.name,
        version=0,
        selection=selection,
        coverage_check=_coverage_check,
        strict=not args.no_strict,
    )
    store.close()

    print(f"exported {stats.rows} rows ({stats.hours:.2f} h) -> {DATASETS / args.name}")
    print(f"  by corpus: {stats.rows_by_corpus}")
    print(f"  by split:  {stats.rows_by_split}")
    if stats.dropped_quality_total:
        print(f"  dropped by quality filter: {stats.dropped_by_quality}")
    if stats.skipped_missing_audio:
        print(f"  skipped (missing audio): {stats.skipped_missing_audio}")
    print(f"  coverage gate <unk> rows: {stats.unk_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
