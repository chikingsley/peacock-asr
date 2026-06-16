"""Russian ASR — a thin per-language project over the omni toolchain.

Config (``sources.py``) + wiring (``curate.py``, ``train.py``, ``eval.py``) over the shared
packages; all curation/training logic lives in omni-curator / omni-finetune-core.

Russian is high-resource: the omni model knows ``rus_Cyrl`` natively, FLEURS has ``ru_ru``, Common
Voice has ``ru``, and we hold large local corpora (ru_open_stt, SOVA, TIMIT). So this is an
INGEST-heavy project (existing labelled data), not a YouTube scrape — the create path is optional.
"""

from __future__ import annotations

from pathlib import Path

LANGUAGE = "rus_Cyrl"  # Russian in Cyrillic — the curator/omni language code
SCRIPT = "Cyrillic"  # the script the create-pipeline LLM should keep labels in

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
DB = DATA / "curator.sqlite"
