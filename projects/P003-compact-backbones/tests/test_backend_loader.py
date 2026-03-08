from __future__ import annotations

from functools import partial

import pytest

from p003_compact.backend_loader import get_backend


def test_get_backend_accepts_hf_prefix() -> None:
    backend_ctor = get_backend("hf:Peacockery/wav2vec2-base-phoneme-en")
    assert isinstance(backend_ctor, partial)


def test_get_backend_accepts_nemo_prefix() -> None:
    backend_ctor = get_backend("nemo:/tmp/citrinet.nemo")
    assert isinstance(backend_ctor, partial)


def test_get_backend_rejects_unknown_prefix() -> None:
    with pytest.raises(ValueError, match="only supports"):
        get_backend("original")
