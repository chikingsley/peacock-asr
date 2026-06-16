"""fairseq2 asset cards for the Russian ASR project, registered in-process.

The training/eval configs refer to model/tokenizer/dataset by name; fairseq2 resolves those
through its asset store, populated from every package exposing a ``fairseq2.extension`` entry
point. Card *shapes* are the typed models in :mod:`omni_finetune_core.assets`; this module supplies
the Russian values. Wired via the entry point in ``pyproject.toml`` -> ``setup_fairseq2_extension``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from omni_finetune_core.assets import (
    MixtureParquetDatasetCard,
    ModelCard,
    TokenizerCard,
    register_cards,
)

if TYPE_CHECKING:
    from fairseq2.runtime.dependency import DependencyContainer

_PKG = Path(__file__).resolve().parent
_PROJECT = Path(__file__).resolve().parents[2]
_MODELS = _PKG / "models"
_PARQUET = _PROJECT / "data" / "datasets" / "v0" / "version=0"

TOKENIZER_NAME = "omni_asr_tokenizer_written_v2_local"

CARDS = [
    TokenizerCard(
        name=TOKENIZER_NAME,
        tokenizer=_MODELS / "omniASR_tokenizer_written_v2.model",
    ),
    ModelCard(
        name="omni_ctc_300m_v2_russian_base",
        checkpoint=_MODELS / "omniASR-CTC-300M-v2.pt",
        tokenizer_ref=TOKENIZER_NAME,
    ),
    MixtureParquetDatasetCard(
        name="russian_asr_corpus",
        data=_PARQUET,
        tokenizer_ref=TOKENIZER_NAME,
    ),
    # Trained-checkpoint cards (omni_ctc_300m_v2_russian_v0_step_NNNNN) added after first v0 run.
]


def setup_fairseq2_extension(container: DependencyContainer) -> None:
    register_cards(container, "russian_asr", CARDS)
