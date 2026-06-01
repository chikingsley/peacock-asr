"""fairseq2 asset cards for the Georgian ASR project, registered in-process.

The training/eval configs refer to the model/tokenizer/dataset by name
(``omni_ctc_300m_v2_georgian_base``, ``omni_asr_tokenizer_written_v2_local``,
``georgian_asr_corpus``); fairseq2 resolves those names through its asset store, which is
populated from every package exposing a ``fairseq2.extension`` entry point. We register
our cards here so the definitions live next to the code that uses them, and so the on-disk
paths are derived from ``__file__`` instead of hardcoded absolute paths.

The card *shapes* (the three card kinds and their key sets) are the typed Pydantic models in
:mod:`omni_finetune_core.assets`; this module only supplies the Georgian values. Wired up via
the ``[project.entry-points."fairseq2.extension"]`` entry point in ``pyproject.toml`` ->
``setup_fairseq2_extension``.
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

# This file lives at src/georgian_asr/assets.py, so the package directory holds models/.
_PKG = Path(__file__).resolve().parent
_PROJECT = Path(__file__).resolve().parents[2]
_MODELS = _PKG / "models"
# The exported omni-parquet (92k clips, ~145 h) — the version=0 partition is a fairseq2 layout
# requirement, not the version axis (the axis is the datasets/vN dir). Mixture weights come from
# the sibling language_distribution_0.tsv.
_PARQUET = _PROJECT / "data" / "datasets" / "v0" / "version=0"

TOKENIZER_NAME = "omni_asr_tokenizer_written_v2_local"

CARDS = [
    TokenizerCard(
        name=TOKENIZER_NAME,
        tokenizer=_MODELS / "omniASR_tokenizer_written_v2.model",
    ),
    ModelCard(
        name="omni_ctc_300m_v2_georgian_base",
        checkpoint=_MODELS / "omniASR-CTC-300M-v2.pt",
        tokenizer_ref=TOKENIZER_NAME,
    ),
    MixtureParquetDatasetCard(
        name="georgian_asr_corpus",
        data=_PARQUET,
        tokenizer_ref=TOKENIZER_NAME,
    ),
]


def setup_fairseq2_extension(container: DependencyContainer) -> None:
    register_cards(container, "georgian_asr", CARDS)
