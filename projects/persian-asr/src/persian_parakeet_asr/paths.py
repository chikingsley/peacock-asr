from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = Path(
    os.environ.get("PERSIAN_ASR_DATA_ROOT", str(PROJECT_ROOT / "data"))
).expanduser()
DEFAULT_HF_HOME = DEFAULT_DATA_ROOT / "hf-cache"
PARAKEET_ROOT = PROJECT_ROOT / "parakeet"
DEFAULT_NEMO_ROOT = PROJECT_ROOT / "vendor" / "nemo"
DEFAULT_TOKENIZER_ROOT = PROJECT_ROOT / "tokenizers" / "parakeet"
DEFAULT_RUNS_ROOT = PARAKEET_ROOT / "runs"


def configure_external_caches() -> None:
    os.environ.setdefault("HF_HOME", str(DEFAULT_HF_HOME))
    os.environ.setdefault("HF_DATASETS_CACHE", str(DEFAULT_HF_HOME / "datasets"))
