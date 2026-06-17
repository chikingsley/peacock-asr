"""Extend a pretrained SentencePiece tokenizer with language-specific pieces.

The ``extend-restore`` recipe (see ``extend_restore_recipe.md``) keeps every existing
piece/ID of a base tokenizer (e.g. parakeet-tdt-0.6b-v3's 8192-piece unified BPE) and
*appends* a small set of target-language pieces so the matching pretrained decoder/joint
rows can be restored. Only the appended pieces are new.

Selection (Tajik-tuned, generalizable via ``--marker-chars``):

1. Train a candidate BPE on the omni-normalized transcripts of the train manifest.
2. Keep only candidate pieces that contain >=1 *marker* character (the letters the base
   tokenizer cannot encode, e.g. Tajik ``ғ қ ҳ ҷ ӯ ӣ`` and uppercase forms).
3. Always include the marker singletons themselves (so every marker letter is encodable).
4. Skip pieces already present in the base tokenizer, and skip pieces whose only non-ASCII
   content is *shared* Cyrillic the base already covers (no marker char => dropped by 2).
5. Target ~``--target-k`` pieces by candidate-BPE rank; never pad with shared-only pieces.

The appended pieces are typed ``USER_DEFINED`` so SentencePiece always emits them as whole
units regardless of merge scores -- this guarantees marker coverage. Pieces/IDs 0..N-1 of
the base proto are left byte-identical.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

# Cyrillic letters absent from the v3 (Ru/Uk) unified tokenizer that occur in Tajik text.
# The spec named the six Tajik-specific letters + uppercase; the CPU tokenizer round-trip gate
# also surfaced ``ё``/``Ё`` (Cyrillic yo) as unencodable in v3 yet present in ~40% of training
# lines (the 6th most frequent unencodable char), so it is included to avoid <unk> targets.
TAJIK_MARKER_CHARS: tuple[str, ...] = tuple("ғқҳҷӯӣёҒҚҲҶӮӢЁ")


def _normalize_iter(
    manifest: Path,
    language: str,
    *,
    limit: int | None,
) -> Iterable[str]:
    """Yield omni-normalized ``text`` fields from a JSONL manifest."""
    from omni_curator.process import normalize  # ty: ignore[unresolved-import]

    count = 0
    with manifest.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            text = json.loads(line).get("text", "")
            if not text:
                continue
            yield normalize(text, language)
            count += 1
            if limit is not None and count >= limit:
                return


def _train_candidate_bpe(
    manifest: Path,
    language: str,
    *,
    candidate_vocab: int,
    character_coverage: float,
    sample_limit: int | None,
    workdir: Path,
) -> list[tuple[str, float]]:
    """Train a throwaway candidate BPE and return ``(piece, score)`` in rank order."""
    import sentencepiece as spm  # ty: ignore[unresolved-import]
    import sentencepiece.sentencepiece_model_pb2 as sp_pb2  # ty: ignore[unresolved-import]

    corpus = workdir / "candidate_corpus.txt"
    with corpus.open("w", encoding="utf-8") as handle:
        for text in _normalize_iter(manifest, language, limit=sample_limit):
            handle.write(text + "\n")

    prefix = workdir / "candidate_bpe"
    spm.SentencePieceTrainer.Train(
        input=str(corpus),
        model_prefix=str(prefix),
        vocab_size=candidate_vocab,
        model_type="bpe",
        character_coverage=character_coverage,
        bos_id=-1,
        eos_id=-1,
        unk_id=0,
        pad_id=-1,
    )
    proto = sp_pb2.ModelProto()
    proto.ParseFromString((prefix.with_suffix(".model")).read_bytes())
    # SentencePiece orders pieces by descending merge score after the control tokens;
    # ``pieces`` order is the rank we want.
    normal = sp_pb2.ModelProto.SentencePiece.NORMAL
    return [(p.piece, p.score) for p in proto.pieces if p.type == normal]


def _has_marker(piece: str, markers: frozenset[str]) -> bool:
    return any(ch in markers for ch in piece)


def select_pieces(
    candidates: Sequence[tuple[str, float]],
    base_pieces: frozenset[str],
    markers: Sequence[str],
    *,
    target_k: int,
) -> list[str]:
    """Select target-language pieces to append.

    Always returns the marker singletons first (deduped, in given order), then ranked
    marker-bearing candidate pieces not already in ``base_pieces``, up to ``target_k``
    total. Never pads with non-marker pieces.
    """
    marker_set = frozenset(markers)
    selected: list[str] = []
    seen: set[str] = set()

    for ch in markers:
        if ch in base_pieces or ch in seen:
            continue
        selected.append(ch)
        seen.add(ch)

    for piece, _score in candidates:
        if len(selected) >= target_k:
            break
        if piece in base_pieces or piece in seen:
            continue
        if not _has_marker(piece, marker_set):
            continue  # shared-Cyrillic-only / ASCII: base already covers it
        selected.append(piece)
        seen.add(piece)

    return selected


def _load_base_proto(base_tokenizer_model: Path) -> Any:
    import sentencepiece.sentencepiece_model_pb2 as sp_pb2  # ty: ignore[unresolved-import]

    proto = sp_pb2.ModelProto()
    proto.ParseFromString(base_tokenizer_model.read_bytes())
    return proto


def _append_pieces(proto: Any, pieces: Sequence[str]) -> Any:
    import sentencepiece.sentencepiece_model_pb2 as sp_pb2  # ty: ignore[unresolved-import]

    for piece in pieces:
        sp = proto.pieces.add()
        sp.piece = piece
        sp.score = 0.0
        sp.type = sp_pb2.ModelProto.SentencePiece.USER_DEFINED
    proto.trainer_spec.vocab_size = len(proto.pieces)
    return proto


def _write_nemo_tokenizer_dir(proto: Any, out_dir: Path) -> None:
    """Write a NeMo SentencePiece tokenizer dir: tokenizer.model/.vocab + vocab.txt."""
    import sentencepiece as spm  # ty: ignore[unresolved-import]

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "tokenizer.model"
    model_path.write_bytes(proto.SerializeToString())

    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    # tokenizer.vocab: "<piece>\t<score>" per line (SentencePiece convention).
    with (out_dir / "tokenizer.vocab").open("w", encoding="utf-8") as handle:
        for sp_piece in proto.pieces:
            handle.write(f"{sp_piece.piece}\t{sp_piece.score:g}\n")
    # vocab.txt: NeMo WordPiece-style view (leading-space marker -> bare, else "##").
    with (out_dir / "vocab.txt").open("w", encoding="utf-8") as handle:
        for i in range(sp.get_piece_size()):
            piece = sp.id_to_piece(i)
            if piece.startswith("▁"):
                handle.write(piece[1:] + "\n")
            else:
                handle.write("##" + piece + "\n")


def build_extended_tokenizer(
    *,
    base_tokenizer_model: Path,
    train_manifest: Path,
    language: str,
    out_dir: Path,
    markers: Sequence[str] = TAJIK_MARKER_CHARS,
    target_k: int = 512,
    candidate_vocab: int = 4096,
    character_coverage: float = 1.0,
    sample_limit: int | None = None,
) -> dict[str, Any]:
    """Build the extended tokenizer dir and return a summary.

    Returns a dict with ``old_vocab``, ``k`` (appended pieces), ``new_vocab``, ``added``
    (the appended piece list), and ``out_dir``.
    """
    base_proto = _load_base_proto(base_tokenizer_model)
    old_vocab = len(base_proto.pieces)
    base_set = frozenset(p.piece for p in base_proto.pieces)

    with tempfile.TemporaryDirectory(prefix="tok_extend_") as tmp:
        candidates = _train_candidate_bpe(
            train_manifest,
            language,
            candidate_vocab=candidate_vocab,
            character_coverage=character_coverage,
            sample_limit=sample_limit,
            workdir=Path(tmp),
        )

    added = select_pieces(candidates, base_set, markers, target_k=target_k)
    _append_pieces(base_proto, added)
    _write_nemo_tokenizer_dir(base_proto, out_dir)

    _assert_prefix_identical(base_tokenizer_model, out_dir / "tokenizer.model", old_vocab)

    return {
        "old_vocab": old_vocab,
        "k": len(added),
        "new_vocab": old_vocab + len(added),
        "added": added,
        "out_dir": str(out_dir),
    }


def _assert_prefix_identical(base_model: Path, new_model: Path, old_vocab: int) -> None:
    """Assert pieces/IDs 0..old_vocab-1 are byte-identical between base and extended."""
    import sentencepiece as spm  # ty: ignore[unresolved-import]

    base = spm.SentencePieceProcessor(model_file=str(base_model))
    new = spm.SentencePieceProcessor(model_file=str(new_model))
    if base.get_piece_size() != old_vocab:
        msg = f"base piece size {base.get_piece_size()} != expected old_vocab {old_vocab}"
        raise AssertionError(msg)
    for i in range(old_vocab):
        if base.id_to_piece(i) != new.id_to_piece(i):
            msg = (
                f"piece mismatch at id {i}: base={base.id_to_piece(i)!r} "
                f"new={new.id_to_piece(i)!r}"
            )
            raise AssertionError(msg)


def extract_base_tokenizer_model(nemo_path: Path, out_dir: Path) -> Path:
    """Extract ``tokenizer.model`` from a ``.nemo`` archive (CPU, no model load)."""
    import tarfile

    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(nemo_path) as tar:
        members = [m for m in tar.getmembers() if m.name.endswith("tokenizer.model")]
        if not members:
            msg = f"no tokenizer.model found in {nemo_path}"
            raise FileNotFoundError(msg)
        member = members[0]
        extracted = out_dir / "base_tokenizer.model"
        with tar.extractfile(member) as src:  # type: ignore[union-attr]
            if src is None:
                msg = f"could not read {member.name} from {nemo_path}"
                raise OSError(msg)
            extracted.write_bytes(src.read())
    return extracted


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extend a base SentencePiece tokenizer with target-language pieces."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--base-nemo", type=Path, help="Base .nemo; extract its tokenizer.model.")
    src.add_argument("--base-tokenizer-model", type=Path, help="Path to base tokenizer.model.")
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--language", required=True, help="omni normalizer language code")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--target-k", type=int, default=512)
    parser.add_argument("--candidate-vocab", type=int, default=4096)
    parser.add_argument("--character-coverage", type=float, default=1.0)
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Cap manifest lines used to train the candidate BPE (default: all).",
    )
    parser.add_argument(
        "--markers",
        default="".join(TAJIK_MARKER_CHARS),
        help="String of marker characters the base tokenizer cannot encode.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.base_nemo is not None:
        with tempfile.TemporaryDirectory(prefix="base_tok_") as tmp:
            base_model = extract_base_tokenizer_model(args.base_nemo, Path(tmp))
            summary = build_extended_tokenizer(
                base_tokenizer_model=base_model,
                train_manifest=args.train_manifest,
                language=args.language,
                out_dir=args.out_dir,
                markers=tuple(args.markers),
                target_k=args.target_k,
                candidate_vocab=args.candidate_vocab,
                character_coverage=args.character_coverage,
                sample_limit=args.sample_limit,
            )
    else:
        summary = build_extended_tokenizer(
            base_tokenizer_model=args.base_tokenizer_model,
            train_manifest=args.train_manifest,
            language=args.language,
            out_dir=args.out_dir,
            markers=tuple(args.markers),
            target_k=args.target_k,
            candidate_vocab=args.candidate_vocab,
            character_coverage=args.character_coverage,
            sample_limit=args.sample_limit,
        )
    print(
        f"old_vocab={summary['old_vocab']} K={summary['k']} new_vocab={summary['new_vocab']} "
        f"-> {summary['out_dir']}",
        flush=True,
    )
    print(f"added (first 24): {summary['added'][:24]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
