"""Georgian ASR — the reference per-language project for the omni toolchain.

A thin project: config (``sources.py``) + wiring (``curate.py``, ``train.py``, ``eval.py``) over
the shared packages. All curation/training LOGIC and the Georgian normalizer (``kat_Geor``) live in
omni-curator / omni-finetune-core. The data lives under ``data/`` (gitignored); see the package's
CURATING.md for the layout.
"""

from __future__ import annotations

from pathlib import Path

LANGUAGE = "kat_Geor"  # Georgian in Georgian (Mkhedruli) script — the curator language code

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
DB = DATA / "curator.sqlite"
