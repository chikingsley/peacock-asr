"""English ASR project configuration."""

from __future__ import annotations

from pathlib import Path

LANGUAGE = "eng_Latn"
SCRIPT = "Latin"

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
DB = DATA / "curator.sqlite"
