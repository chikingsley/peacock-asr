from __future__ import annotations

from functools import cache
from importlib import import_module
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from p001_gop.backend_protocol import PhonemeBackend


_BACKEND_SPECS = {
    "original": ("p001_gop.backends.ctc_gop_original", "OriginalBackend"),
    "xlsr-espeak": ("p001_gop.backends.xlsr_espeak", "XLSREspeakBackend"),
    "zipa": ("p001_gop.backends.zipa", "ZIPABackend"),
}


@cache
def _load_backend_class(name: str) -> type[PhonemeBackend]:
    if name not in _BACKEND_SPECS:
        available = ", ".join(sorted(_BACKEND_SPECS))
        msg = f"Unknown backend {name!r}. Available: {available}"
        raise ValueError(msg)

    module_name, class_name = _BACKEND_SPECS[name]
    module = import_module(module_name)
    return cast("type[PhonemeBackend]", getattr(module, class_name))


def get_backend(name: str) -> type[PhonemeBackend]:
    return _load_backend_class(name)


def resolve_backend_name(name: str) -> str:
    return get_backend(name)().name
