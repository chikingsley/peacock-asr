"""Build a token-level NGPU-LM for a Parakeet SentencePiece tokenizer."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any

DEFAULT_TOKEN_OFFSET = 100
KENLM_BIN = Path(__file__).resolve().parents[3] / "kenlm/build/bin"


def encode_token(token_id: int, token_offset: int = DEFAULT_TOKEN_OFFSET) -> str:
    """Map a tokenizer ID to the single Unicode character expected by NeMo's ARPA reader."""
    token = chr(token_id + token_offset)
    if token.isspace():
        raise ValueError(f"token ID {token_id} maps to whitespace at offset {token_offset}")
    return token


def write_token_corpus(
    corpus_path: Path,
    output_path: Path,
    tokenizer: Any,
    *,
    token_offset: int = DEFAULT_TOKEN_OFFSET,
) -> tuple[int, int]:
    """Encode one normalized text utterance per line as space-separated NGPU-LM tokens."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines_written = 0
    tokens_written = 0
    with (
        corpus_path.open(encoding="utf-8") as source,
        output_path.open("w", encoding="utf-8") as output,
    ):
        for raw_line in source:
            line = " ".join(raw_line.split())
            if not line:
                continue
            token_ids = tokenizer.encode(line, out_type=int)
            if not token_ids:
                continue
            output.write(" ".join(encode_token(token_id, token_offset) for token_id in token_ids))
            output.write("\n")
            lines_written += 1
            tokens_written += len(token_ids)
    return lines_written, tokens_written


def build_arpa(
    token_corpus_path: Path,
    arpa_path: Path,
    *,
    order: int,
    kenlm_bin: Path = KENLM_BIN,
    memory: str = "40%",
) -> None:
    """Build an unpruned token-level ARPA model with KenLM."""
    lmplz = kenlm_bin / "lmplz"
    if not lmplz.is_file():
        raise FileNotFoundError(lmplz)
    with (
        token_corpus_path.open(encoding="utf-8") as source,
        arpa_path.open("w", encoding="utf-8") as output,
    ):
        subprocess.run(  # noqa: S603 - executable is the validated in-repo KenLM binary
            [str(lmplz), "-o", str(order), "-S", memory],
            stdin=source,
            stdout=output,
            check=True,
        )


def convert_arpa_to_nemo(
    arpa_path: Path,
    nemo_path: Path,
    *,
    vocab_size: int,
    token_offset: int = DEFAULT_TOKEN_OFFSET,
) -> None:
    """Convert a token ARPA into NeMo's faster serialized NGPU-LM form."""
    from nemo.collections.asr.parts.submodules.ngram_lm import (  # ty: ignore[unresolved-import]
        NGramGPULanguageModel,
    )

    model = NGramGPULanguageModel.from_arpa(
        lm_path=arpa_path,
        vocab_size=vocab_size,
        token_offset=token_offset,
    )
    model.save_to(str(nemo_path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an unpruned token-level KenLM order-N ARPA and NeMo NGPU-LM."
    )
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--tokenizer-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", default=None, help="Output stem; default lm{order}np_bpe{vocab}")
    parser.add_argument("--order", type=int, default=6)
    parser.add_argument("--token-offset", type=int, default=DEFAULT_TOKEN_OFFSET)
    parser.add_argument("--kenlm-bin", type=Path, default=KENLM_BIN)
    parser.add_argument("--memory", default="40%", help="KenLM lmplz -S value")
    parser.add_argument("--reuse-token-corpus", action="store_true")
    return parser


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    import sentencepiece as spm  # ty: ignore[unresolved-import]

    if args.order < 1:
        raise ValueError("order must be at least 1")
    tokenizer = spm.SentencePieceProcessor(model_file=str(args.tokenizer_model))
    vocab_size = int(tokenizer.vocab_size())
    stem = args.name or f"lm{args.order}np_bpe{vocab_size}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    token_corpus = args.output_dir / f"corpus_bpe{vocab_size}_tokens.txt"
    arpa_path = args.output_dir / f"{stem}.arpa"
    nemo_path = args.output_dir / f"{stem}.nemo"

    if args.reuse_token_corpus:
        if not token_corpus.is_file():
            raise FileNotFoundError(token_corpus)
        print(f"reusing token corpus {token_corpus}", flush=True)
    else:
        lines, tokens = write_token_corpus(
            args.corpus,
            token_corpus,
            tokenizer,
            token_offset=args.token_offset,
        )
        print(
            f"wrote {token_corpus}: {lines:,} lines, {tokens:,} tokens, vocab={vocab_size}",
            flush=True,
        )

    build_arpa(
        token_corpus,
        arpa_path,
        order=args.order,
        kenlm_bin=args.kenlm_bin,
        memory=args.memory,
    )
    print(f"built {arpa_path} ({arpa_path.stat().st_size / 1e6:.1f} MB)", flush=True)
    convert_arpa_to_nemo(
        arpa_path,
        nemo_path,
        vocab_size=vocab_size,
        token_offset=args.token_offset,
    )
    print(f"built {nemo_path} ({nemo_path.stat().st_size / 1e6:.1f} MB)", flush=True)
    return arpa_path, nemo_path


def build_ngram_lm_main(argv: list[str] | None = None) -> int:
    run(build_parser().parse_args(argv))
    return 0
