from __future__ import annotations

from functools import cache, partial
from importlib import import_module
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Callable

    from p003_compact.backend_protocol import PhonemeBackend


@cache
def _load_hf_backend_class() -> type[PhonemeBackend]:
    module = import_module("p003_compact.backends.hf_ctc")
    return cast("type[PhonemeBackend]", module.HFCTCBackend)


@cache
def _load_nemo_backend_class() -> type[PhonemeBackend]:
    module = import_module("p003_compact.backends.nemo_ctc")
    return cast("type[PhonemeBackend]", module.NemoCTCBackend)


def get_backend(name: str) -> Callable[[], PhonemeBackend]:
    if name.startswith("hf:"):
        hf_backend = _load_hf_backend_class()
        return cast(
            "Callable[[], PhonemeBackend]",
            partial(hf_backend, repo_id=name[3:]),
        )
    if name.startswith("nemo:"):
        nemo_backend = _load_nemo_backend_class()
        return cast(
            "Callable[[], PhonemeBackend]",
            partial(nemo_backend, model_ref=name[5:]),
        )
    msg = "P003 evaluation only supports hf:<repo_id> and nemo:<path-or-repo> backends."
    raise ValueError(msg)


def resolve_backend_name(name: str) -> str:
    return get_backend(name)().name
