"""Georgian ASR — a fresh-language reference project for the omni toolchain.

Consumes omni-curator (ingest/process/store) to build the dataset and omni-finetune-core to train.
The data lives under ``data/`` (gitignored); see the package's CURATING.md for the layout.
"""

from __future__ import annotations

LANGUAGE = "kat_Geor"  # Georgian in Georgian (Mkhedruli) script
