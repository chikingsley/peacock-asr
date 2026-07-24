"""Tokenizer-coverage adapter for the export gate — the explicit fairseq2 bridge.

:func:`data.export.export_dataset` takes ``coverage_check`` as an *injected* callable so
the curator never depends on the consuming project's tokenizer stack. This module is the one
sanctioned bridge: it builds that callable from a char-tokenizer model file, importing fairseq2 +
omni-finetune-core only when the check actually runs (they live in the project venv, not in this
package). Everything stays explicit: a missing model file or missing training stack fails fast
with a message naming exactly what is absent.
"""

from __future__ import annotations

import tarfile
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def char_tokenizer_coverage(model_path: Path) -> Callable[[list[str]], int]:
    """Build an export ``coverage_check`` from a char-tokenizer ``.model`` file.

    The returned callable counts rows whose text produces a tokenizer ``<unk>``. Raises
    ``FileNotFoundError`` for a missing model file and ``ImportError`` (with the install hint)
    when fairseq2 / omni-finetune-core are absent — never silently skips the gate.
    """

    def check(texts: list[str]) -> int:
        if not model_path.exists():
            msg = f"coverage-gate tokenizer not found: {model_path}"
            raise FileNotFoundError(msg)
        try:
            from fairseq2.data.tokenizers.char import (  # ty: ignore[unresolved-import]
                load_char_tokenizer,
            )
            from omni_finetune_core.tokenizer_audit import (  # ty: ignore[unresolved-import]
                audit_texts,
            )
        except ImportError as exc:
            msg = (
                "the export coverage gate needs fairseq2 + omni-finetune-core in the project "
                "venv (they are not omni-curator dependencies)"
            )
            raise ImportError(msg) from exc
        return audit_texts(texts, load_char_tokenizer(model_path, None)).unk_rows

    return check


def nemo_sentencepiece_coverage(nemo_path: Path) -> Callable[[list[str]], int]:
    """Build an export coverage gate from the SentencePiece model embedded in a ``.nemo``.

    The archive is read in memory and never unpacked into a project-specific tokenizer folder.
    That matters for same-language Parakeet fine-tuning: the export must target the base model's
    exact tokenizer while the trainer preserves its pretrained decoder and joint weights.
    """
    processor: object | None = None

    def check(texts: list[str]) -> int:
        nonlocal processor
        if not nemo_path.exists():
            raise FileNotFoundError(f"coverage-gate NeMo model not found: {nemo_path}")
        try:
            import sentencepiece as spm
        except ImportError as exc:
            raise ImportError(
                "the Parakeet coverage gate needs sentencepiece in the project environment"
            ) from exc
        if processor is None:
            with tarfile.open(nemo_path) as archive:
                members = [
                    member
                    for member in archive.getmembers()
                    if member.isfile() and member.name.endswith("_tokenizer.model")
                ]
                if len(members) != 1:
                    raise RuntimeError(
                        f"expected one embedded SentencePiece model in {nemo_path}, "
                        f"found {len(members)}"
                    )
                handle = archive.extractfile(members[0])
                if handle is None:
                    raise RuntimeError(f"could not read {members[0].name} from {nemo_path}")
                processor = spm.SentencePieceProcessor(model_proto=handle.read())
        unk_id = processor.unk_id()
        return sum(unk_id in processor.encode(text, out_type=int) for text in texts)

    return check
