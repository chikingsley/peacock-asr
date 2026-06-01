"""Tajik ASR — fine-tuning OmniASR CTC, consuming omni-curator + omni-finetune-core.

A thin per-language project: config (``sources.py``) + wiring (``curate.py``, ``train.py``,
``eval.py``) over the shared packages. The data lives under ``data/`` (gitignored); curation logic
and the Tajik text normalizer (``tgk_Cyrl``) live in omni-curator.
"""

from __future__ import annotations

from pathlib import Path

LANGUAGE = "tgk_Cyrl"  # Tajik (Cyrillic) — the curator language code

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
DB = DATA / "curator.sqlite"
