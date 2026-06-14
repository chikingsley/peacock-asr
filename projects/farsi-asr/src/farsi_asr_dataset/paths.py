from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent
# Root data/ holds the flattened canonical datasets (source-of-truth audio).
DEFAULT_DATA_ROOT = Path(
    os.environ.get("PERSIAN_ASR_DATA_ROOT", str(PROJECT_ROOT / "data"))
).expanduser()
DATA_ROOT = DEFAULT_DATA_ROOT
RAW_ROOT = DEFAULT_DATA_ROOT / "raw"
# Curation/labeling data (Scribe ledger + jobs) lives inside the core package,
# gitignored in place.
CURATION_ROOT = PACKAGE_DIR / "data" / "curation"
DEFAULT_LEDGER = CURATION_ROOT / "persian_corpus.sqlite"
DEFAULT_NEMO_RUNS = CURATION_ROOT / "nemo_runs"
DEFAULT_HF_HOME = DEFAULT_DATA_ROOT / "hf-cache"


def configure_external_caches() -> None:
    os.environ.setdefault("HF_HOME", str(DEFAULT_HF_HOME))
    os.environ.setdefault("HF_DATASETS_CACHE", str(DEFAULT_HF_HOME / "datasets"))
