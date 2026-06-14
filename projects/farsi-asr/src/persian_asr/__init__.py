"""Persian ASR — the template-shaped project package (omni-curator + omni-finetune-core).

The thin per-language standard (matches tajik/georgian): config (``sources.py``) + wiring
(``curate.py``, ``train.py``, ``eval.py``, ``assets.py``) over the shared packages. The Persian
normalizer (``fas_Arab``: hazm + num2words) lives in omni-curator.

This package is ADDITIVE — the pre-template monoliths (``persian_asr_dataset``,
``persian_omnilingual_asr``, ``finetune_omni``, ``finetune_parakeet``) and their CLIs keep
working untouched. New-pipeline data lands under ``data/`` in the curator layout
(``create/``, ``channels/``, ``canonical_audio/``, ``raw/``, ``datasets/``, ``clips/``,
``curator.sqlite``); those subdir names do not collide with the legacy canonical datasets that
already cohabit ``data/`` (``fleurs/``, ``common_voice_25/``, ``mana_tts/``, ...).
"""

from __future__ import annotations

from pathlib import Path

LANGUAGE = "fas_Arab"  # Persian (Perso-Arabic script) — the omni-curator NORMALIZERS key
SCRIPT = "Perso-Arabic"  # the script the create-pipeline LLM should keep labels in

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
DB = DATA / "curator.sqlite"
