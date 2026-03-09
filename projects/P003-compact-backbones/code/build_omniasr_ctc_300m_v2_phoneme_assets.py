#!/usr/bin/env python3
"""Build local OmniASR phoneme tokenizer/model assets for P003."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import sentencepiece as spm

from p003_compact.omni_assets import (
    PHONEME_MODEL_NAME,
    PHONEME_TOKENIZER_NAME,
    STOCK_CHECKPOINT_URL,
    load_ordered_vocab,
    omni_phoneme_assets,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vocab-json",
        type=Path,
        default=None,
        help="Path to the canonical phoneme vocab.json (default: P003 vocab).",
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=None,
        help="Override Omni asset root (default: project-local code/omni).",
    )
    return parser.parse_args()


def build_tokenizer(tokenizer_dir: Path, ordered_tokens: list[str]) -> dict[str, int]:
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    prefix = tokenizer_dir / "arpabet_41"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
        corpus_tokens = [
            token for token in ordered_tokens if not token.startswith("[")
        ]
        handle.write("\n".join(corpus_tokens))
        handle.write("\n")
        corpus_path = Path(handle.name)

    unk_id = ordered_tokens.index("[UNK]")
    pad_id = ordered_tokens.index("[PAD]")
    user_defined_symbols = [
        token for token in ordered_tokens if token not in {"[UNK]", "[PAD]"}
    ]

    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(prefix),
        model_type="word",
        split_by_whitespace=True,
        vocab_size=len(ordered_tokens),
        hard_vocab_limit=False,
        bos_id=-1,
        eos_id=-1,
        unk_id=unk_id,
        unk_piece="[UNK]",
        pad_id=pad_id,
        pad_piece="[PAD]",
        user_defined_symbols=user_defined_symbols,
    )
    corpus_path.unlink(missing_ok=True)

    processor = spm.SentencePieceProcessor(model_file=str(prefix.with_suffix(".model")))
    token_to_id = {token: processor.piece_to_id(token) for token in ordered_tokens}
    if list(token_to_id) != ordered_tokens or any(
        token_to_id[token] != index for index, token in enumerate(ordered_tokens)
    ):
        msg = (
            "SentencePiece token ids do not match the canonical ARPABET order. "
            "Inspect the generated tokenizer before proceeding."
        )
        raise SystemExit(msg)

    metadata = {
        "tokenizer_family": "char_tokenizer",
        "tokens": ordered_tokens,
        "token_to_id": token_to_id,
        "vocab_size": len(ordered_tokens),
        "notes": (
            "SentencePiece word model with fixed token ids matching the canonical "
            "P003 ARPABET vocab."
        ),
    }
    (tokenizer_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (tokenizer_dir / "token_to_id.json").write_text(
        json.dumps(token_to_id, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return token_to_id


def write_cards(asset_root: Path, tokenizer_model: Path, vocab_size: int) -> None:
    assets = omni_phoneme_assets(asset_root)
    assets.card_dir.mkdir(parents=True, exist_ok=True)
    assets.tokenizer_card.write_text(
        (
            f"name: {PHONEME_TOKENIZER_NAME}\n"
            "tokenizer_family: char_tokenizer\n"
            f"tokenizer: {tokenizer_model.resolve()}\n"
        ),
        encoding="utf-8",
    )
    assets.model_card.write_text(
        (
            f"name: {PHONEME_MODEL_NAME}\n"
            "model_family: wav2vec2_asr\n"
            "model_arch: 300m_v2\n"
            f"checkpoint: {STOCK_CHECKPOINT_URL}\n"
            f"tokenizer_ref: {PHONEME_TOKENIZER_NAME}\n"
            "restrict: false\n"
            "model_config_override:\n"
            f"  target_vocab_size: {vocab_size}\n"
        ),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    assets = omni_phoneme_assets(args.asset_root)
    if args.vocab_json is not None:
        ordered_tokens = load_ordered_vocab(args.vocab_json)
    else:
        ordered_tokens = load_ordered_vocab()
    build_tokenizer(assets.tokenizer_dir, ordered_tokens)
    write_cards(
        assets.tokenizer_dir.parents[1],
        assets.tokenizer_model,
        len(ordered_tokens),
    )
    summary = {
        "tokenizer_model": str(assets.tokenizer_model.resolve()),
        "tokenizer_mapping": str(assets.tokenizer_mapping.resolve()),
        "tokenizer_card": str(assets.tokenizer_card.resolve()),
        "model_card": str(assets.model_card.resolve()),
        "model_name": PHONEME_MODEL_NAME,
        "tokenizer_name": PHONEME_TOKENIZER_NAME,
        "vocab_size": len(ordered_tokens),
    }
    sys.stdout.write(f"{json.dumps(summary, indent=2)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
