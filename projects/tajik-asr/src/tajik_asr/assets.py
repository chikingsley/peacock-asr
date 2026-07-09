"""fairseq2 asset cards for the Tajik ASR project, registered in-process.

Replaces the hand-maintained ``.fairseq2-assets/`` YAML directory. The training
config refers to the model/tokenizer/dataset by name (``omni_ctc_300m_v2_base``,
``omni_asr_tokenizer_written_v2_local``, ``tajik_asr_corpus``); fairseq2 resolves
those names through its asset store, which is populated from every package that
exposes a ``fairseq2.extension`` entry point. We register our cards here so the
definitions live next to the code that uses them, and so the on-disk paths are derived
from ``__file__`` instead of being hardcoded absolute paths.

The card *shapes* (the three card kinds and their key sets) are the typed Pydantic
models in :mod:`omni_finetune_core.assets`; this module only supplies the Tajik values.
Wired up via the ``[project.entry-points."fairseq2.extension"]`` entry point in
``pyproject.toml`` -> ``setup_fairseq2_extension``.
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
_RESTORED_MODELS = _PROJECT / "data" / "models"
# v0 = the baseline-model provenance corpus (v1/v2 superseded, recorded in EXPERIMENTS.md and
# deleted; kept datasets live on the Peacockery HF org). Canonical data lives under data/, not src.
_PARQUET = _PROJECT / "data" / "datasets" / "v0" / "version=0"
# v3 = v2's corpus minus a held-out conversational test set (157 whole videos carved to
# split=test); the only difference from v2 is that those videos are out of train. The fair
# conversational benchmark — v3 has never seen the held-out clips it is scored on.
_PARQUET_V3 = _PROJECT / "data" / "datasets" / "v3" / "version=0"

TOKENIZER_NAME = "omni_asr_tokenizer_written_v2_local"

CARDS = [
    TokenizerCard(
        name=TOKENIZER_NAME,
        tokenizer=_OMNI_BASE / "omniASR_tokenizer_written_v2.model",
    ),
    ModelCard(
        name="omni_ctc_300m_v2_base",
        checkpoint=_OMNI_BASE / "omniASR-CTC-300M-v2.pt",
        tokenizer_ref=TOKENIZER_NAME,
    ),
    ModelCard(
        name="omni_ctc_300m_v2_tajik_step_1800",
        checkpoint=(
            _PROJECT
            / "runs/omni-ctc-300m-tajik-asr-corpus-v0/ws_1.3dcb9e0b/"
            / "checkpoints/step_1800/model/pp_00/tp_00/sdp_00.pt"
        ),
        tokenizer_ref=TOKENIZER_NAME,
    ),
    # v2 (full new-pipeline corpus, ~1,070 h) best dev-WER ckpt (step_19500, FLEURS dev 17.06).
    ModelCard(
        name="omni_ctc_300m_v2_tajik_v2_step_19500",
        checkpoint=(
            _PROJECT
            / "runs/omni-ctc-300m-tajik-asr-corpus-v2-r2/ws_1.a8b9ba67/"
            / "checkpoints/step_19500/model/pp_00/tp_00/sdp_00.pt"
        ),
        tokenizer_ref=TOKENIZER_NAME,
    ),
    MixtureParquetDatasetCard(
        name="tajik_asr_corpus",
        data=_PARQUET,
        tokenizer_ref=TOKENIZER_NAME,
    ),
    MixtureParquetDatasetCard(
        name="tajik_asr_corpus_v3",
        data=_PARQUET_V3,
        tokenizer_ref=TOKENIZER_NAME,
    ),
    # v3 (v2 corpus minus the held-out conversational test) best dev-WER ckpt (step_20000,
    # FLEURS dev 17.41). Restored from Peacockery/omni-ctc-300m-tajik at pinned Hub revision
    # cafa6e9fb394f7cef29caf79385feb96bcfc05ae; the original training-run directory was retired.
    # This is the fair model for the conversational held-out — it never saw those clips.
    ModelCard(
        name="omni_ctc_300m_v2_tajik_v3_step_20000",
        checkpoint=_RESTORED_MODELS / "omni-ctc-300m-tajik" / "model.pt",
        tokenizer_ref=TOKENIZER_NAME,
    ),
]


def setup_fairseq2_extension(container: DependencyContainer) -> None:
    register_cards(container, "tajik_asr", CARDS)
