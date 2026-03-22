"""P001 GOP baseline package."""

from __future__ import annotations

import sys
from importlib import import_module


def _register_legacy_module_aliases() -> None:
    """Expose old module paths so legacy torch caches remain loadable."""
    package = sys.modules[__name__]
    sys.modules.setdefault("peacock_asr", package)
    for legacy_name, current_name in {
        "peacock_asr.gopt_model": "p001_gop.gopt_model",
    }.items():
        sys.modules.setdefault(legacy_name, import_module(current_name))


_register_legacy_module_aliases()
