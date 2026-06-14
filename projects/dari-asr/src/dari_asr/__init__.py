"""Dari (Afghan Persian) ASR — a thin per-language project over the omni toolchain.

Config (``sources.py``) + wiring (``curate.py``, ``train.py``, ``eval.py``) over the shared
packages; all curation/training logic lives in omni-curator / omni-finetune-core.

Language code: Dari is ISO ``prs`` (Afghan Persian), but the omni CTC model's language table has
NO ``prs`` — the only Perso-Arabic Persian it knows is ``fas_Arab`` (shared with Iranian Farsi). So
the code wires ``fas_Arab`` everywhere it touches the model/tokenizer/normalizer; the Dari *variety*
is kept separate at the DATA/project level (its own channels, its own fine-tune), not via a distinct
model tag. The contaminant to gate against here is Pashto, not Russian.
"""

from __future__ import annotations

from pathlib import Path

LANGUAGE = "fas_Arab"  # Perso-Arabic; the omni model has no `prs`, so Dari rides on fas_Arab
SCRIPT = "Perso-Arabic"  # the script the create-pipeline LLM should keep labels in

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
DB = DATA / "curator.sqlite"
