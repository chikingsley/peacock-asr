from __future__ import annotations

import importlib.machinery
import sys
import types
from pathlib import Path

import torchaudio


REPO_ROOT = Path(__file__).resolve().parents[4]
THIRD_PARTY_ROOT = REPO_ROOT / "third_party"
S3PRL_ROOT = THIRD_PARTY_ROOT / "s3prl"
SSL_INTERFACE_ROOT = THIRD_PARTY_ROOT / "SSL_Interface"


def ensure_third_party_on_path() -> None:
    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda *args, **kwargs: None
    if "torchaudio.sox_effects" not in sys.modules:
        sox_effects = types.ModuleType("torchaudio.sox_effects")
        sox_effects.apply_effects_tensor = lambda tensor, sample_rate, effects, channels_first=True: (
            tensor,
            sample_rate,
        )
        sys.modules["torchaudio.sox_effects"] = sox_effects
        torchaudio.sox_effects = sox_effects

    for root in (SSL_INTERFACE_ROOT, S3PRL_ROOT):
        if not root.exists():
            raise FileNotFoundError(f"Missing required third-party checkout: {root}")

        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

    _seed_namespace_package("s3prl.problem", S3PRL_ROOT / "s3prl" / "problem")
    _seed_namespace_package("s3prl.problem.asr", S3PRL_ROOT / "s3prl" / "problem" / "asr")
    _seed_namespace_package("s3prl.problem.common", S3PRL_ROOT / "s3prl" / "problem" / "common")
    _seed_namespace_package("s3prl.problem.asv", S3PRL_ROOT / "s3prl" / "problem" / "asv")


def _seed_namespace_package(name: str, path: Path) -> None:
    if name in sys.modules:
        return

    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    module.__package__ = name
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    module.__spec__.submodule_search_locations = [str(path)]
    sys.modules[name] = module
