"""Tokenizer-coverage adapter for the export gate — the explicit fairseq2 bridge.

:func:`omni_curator.export.export_dataset` takes ``coverage_check`` as an *injected* callable so
the curator never depends on the consuming project's tokenizer stack. This module is the one
sanctioned bridge: it builds that callable from a char-tokenizer model file, importing fairseq2 +
omni-finetune-core only when the check actually runs (they live in the project venv, not in this
package). Everything stays explicit: a missing model file or missing training stack fails fast
with a message naming exactly what is absent.
"""

from __future__ import annotations

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
