from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path(
    os.environ.get("PERSIAN_ASR_DATA_ROOT", str(PROJECT_ROOT / "data"))
).expanduser()
DEFAULT_HF_HOME = DEFAULT_DATA_ROOT / "hf-cache"
# NeMo example finetune/tokenizer scripts (Apache-2.0) vendored into the package;
# they import only the installed `nemo` package. Override with PERSIAN_NEMO_ROOT
# to point at a full NeMo checkout if needed.
DEFAULT_NEMO_ROOT = Path(
    os.environ.get("PERSIAN_NEMO_ROOT", str(PACKAGE_DIR / "nemo_recipes"))
).expanduser()
# Model + tokenizer artifacts live inside the package (gitignored in place).
DEFAULT_MODEL_ROOT = PACKAGE_DIR / "models"
DEFAULT_CTC_MODEL = DEFAULT_MODEL_ROOT / "ctc.nemo"
DEFAULT_TOKENIZER_ROOT = PACKAGE_DIR / "tokenizer"
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "runs" / "parakeet"


def configure_external_caches() -> None:
    os.environ.setdefault("HF_HOME", str(DEFAULT_HF_HOME))
    os.environ.setdefault("HF_DATASETS_CACHE", str(DEFAULT_HF_HOME / "datasets"))
