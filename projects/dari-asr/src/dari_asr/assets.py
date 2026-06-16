"""fairseq2 asset cards for the Dari ASR project, registered in-process.

The training/eval configs refer to model/tokenizer/dataset by name; fairseq2 resolves those
through its asset store, populated from every package exposing a ``fairseq2.extension`` entry
point. Card *shapes* are the typed models in :mod:`omni_finetune_core.assets`; this module supplies
the Dari values. Wired via the entry point in ``pyproject.toml`` -> ``setup_fairseq2_extension``.
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
_OMNI_BASE = Path(__file__).resolve().parents[4] / "base_models" / "omni"
_PARQUET = _PROJECT / "data" / "datasets" / "v0" / "version=0"

TOKENIZER_NAME = "omni_asr_tokenizer_written_v2_local"

# The Farsi (Iranian Persian) production checkpoint — a strong warm-start source for Dari, which
# shares the fas_Arab script/model code. Train both (cold omni base + this warm start) and compare.
_FARSI = Path(__file__).resolve().parents[3] / "farsi-asr"
_FARSI_PROD = (
    _FARSI / "src/finetune_omni/runs/scribe-v4-rewarm10k/ws_1.5318420d"
    / "checkpoints/step_7000/model/pp_00/tp_00/sdp_00.pt"
)

CARDS = [
    TokenizerCard(
        name=TOKENIZER_NAME,
        tokenizer=_OMNI_BASE / "omniASR_tokenizer_written_v2.model",
    ),
    ModelCard(
        name="omni_ctc_300m_v2_dari_base",
        checkpoint=_OMNI_BASE / "omniASR-CTC-300M-v2.pt",
        tokenizer_ref=TOKENIZER_NAME,
    ),
    # Farsi production (omni_ctc_300m_v2_farsi_v4_step_41000) as the warm-start source.
    ModelCard(
        name="omni_ctc_300m_v2_farsi_v4_step_41000",
        checkpoint=_FARSI_PROD,
        tokenizer_ref=TOKENIZER_NAME,
    ),
    MixtureParquetDatasetCard(
        name="dari_asr_corpus",
        data=_PARQUET,
        tokenizer_ref=TOKENIZER_NAME,
    ),
    # Trained-checkpoint cards (omni_ctc_300m_v2_dari_v0_step_NNNNN) added after the first v0 run.
]


def setup_fairseq2_extension(container: DependencyContainer) -> None:
    register_cards(container, "dari_asr", CARDS)
