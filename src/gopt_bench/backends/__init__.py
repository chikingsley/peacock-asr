from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gopt_bench.backends.base import PhonemeBackend

BACKEND_REGISTRY: dict[str, type[PhonemeBackend]] = {}


def register_backend(name: str, cls: type[PhonemeBackend]) -> None:
    BACKEND_REGISTRY[name] = cls


def get_backend(name: str) -> type[PhonemeBackend]:
    if name not in BACKEND_REGISTRY:
        available = ", ".join(sorted(BACKEND_REGISTRY))
        msg = f"Unknown backend {name!r}. Available: {available}"
        raise ValueError(msg)
    return BACKEND_REGISTRY[name]


def _register_builtins() -> None:
    from gopt_bench.backends.ctc_gop_original import OriginalBackend  # noqa: PLC0415
    from gopt_bench.backends.xlsr_espeak import XLSREspeakBackend  # noqa: PLC0415

    register_backend("original", OriginalBackend)
    register_backend("xlsr-espeak", XLSREspeakBackend)

    try:
        from gopt_bench.backends.zipa import ZIPABackend  # noqa: PLC0415

        register_backend("zipa", ZIPABackend)
    except ImportError:
        pass


_register_builtins()
