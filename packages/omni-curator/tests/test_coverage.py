from __future__ import annotations

import io
import sys
import tarfile
import types
from typing import TYPE_CHECKING

import pytest

from omni_curator.audit.coverage import nemo_sentencepiece_coverage

if TYPE_CHECKING:
    from pathlib import Path


class _FakeSentencePieceProcessor:
    def __init__(self, *, model_proto: bytes) -> None:
        assert model_proto == b"embedded-tokenizer"

    @staticmethod
    def unk_id() -> int:
        return 0

    @staticmethod
    def encode(text: str, *, out_type: type[int]) -> list[int]:
        assert out_type is int
        return [0] if "unknown" in text else [1, 2]


def _write_nemo(path: Path, members: dict[str, bytes]) -> None:
    with tarfile.open(path, "w") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def test_nemo_sentencepiece_coverage_uses_embedded_model(monkeypatch, tmp_path: Path) -> None:
    module = types.ModuleType("sentencepiece")
    module.SentencePieceProcessor = _FakeSentencePieceProcessor
    monkeypatch.setitem(sys.modules, "sentencepiece", module)
    model = tmp_path / "base.nemo"
    _write_nemo(model, {"abc_tokenizer.model": b"embedded-tokenizer"})

    check = nemo_sentencepiece_coverage(model)

    assert check(["covered", "unknown glyph"]) == 1
    assert check(["covered again"]) == 0


def test_nemo_sentencepiece_coverage_requires_exactly_one_model(tmp_path: Path) -> None:
    model = tmp_path / "base.nemo"
    _write_nemo(model, {})

    with pytest.raises(RuntimeError, match="expected one"):
        nemo_sentencepiece_coverage(model)(["text"])
