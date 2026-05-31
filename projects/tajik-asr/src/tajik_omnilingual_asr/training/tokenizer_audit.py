from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path

from fairseq2.data.tokenizers.hub import load_tokenizer

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = (
    ROOT
    / "src/tajik_omnilingual_asr/dataset_prep/artifacts/tajik_asr_combined_v0/omni_manifest"
)
DEFAULT_TOKENIZER = "omni_asr_tokenizer_written_v2_local"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit Omnilingual tokenizer coverage.")
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--max-examples", type=int, default=20)
    return parser


def configure_environment() -> None:
    os.environ.setdefault("HF_HOME", str(ROOT / ".hf-cache"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(ROOT / ".hf-cache/datasets"))
    os.environ.setdefault("FAIRSEQ2_CACHE_DIR", str(ROOT / ".fairseq2-cache/assets"))


def iter_text(manifest_dir: Path):
    for split in ("train", "dev", "test"):
        wrd = manifest_dir / f"{split}.wrd"
        with wrd.open("r", encoding="utf-8") as fp:
            for line_nr, line in enumerate(fp, start=1):
                text = line.rstrip("\n")
                if text:
                    yield split, line_nr, text


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_environment()

    tokenizer = load_tokenizer(args.tokenizer)
    encoder = tokenizer.create_raw_encoder()
    unk_idx = tokenizer.vocab_info.unk_idx

    rows = 0
    unknown_rows = 0
    unknown_tokens = 0
    char_counts: Counter[str] = Counter()
    examples: list[tuple[str, int, int, str, list[str]]] = []

    for split, line_nr, text in iter_text(args.manifest_dir):
        rows += 1
        char_counts.update(text)
        token_ids = encoder(text).tolist()
        tokens = encoder.encode_as_tokens(text)
        row_unknowns = token_ids.count(unk_idx) if unk_idx is not None else 0
        if row_unknowns:
            unknown_rows += 1
            unknown_tokens += row_unknowns
            if len(examples) < args.max_examples:
                examples.append((split, line_nr, row_unknowns, text, tokens))

    print(f"tokenizer\t{args.tokenizer}")
    print(f"rows\t{rows}")
    print(f"unknown_rows\t{unknown_rows}")
    print(f"unknown_tokens\t{unknown_tokens}")
    print(f"unique_chars\t{len(char_counts)}")

    if examples:
        print("examples")
        for split, line_nr, count, text, tokens in examples:
            print(f"{split}:{line_nr}\tunknowns={count}\t{text}\t{tokens}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
