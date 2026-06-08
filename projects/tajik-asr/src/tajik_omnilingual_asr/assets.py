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

# This file lives at src/tajik_omnilingual_asr/fairseq2_assets.py, so the package
# directory holds models/ and dataset_prep/.
_PKG = Path(__file__).resolve().parent
_PROJECT = Path(__file__).resolve().parents[2]
_MODELS = _PKG / "models"
# The training parquet lives inside its source artifact (sibling of omni_manifest/).
# Versioning is by artifact dir: point this at tajik_asr_combined_v1/... for v1, etc.
# The parquet's own version=0 partition is a fairseq2 layout requirement, not the axis.
_PARQUET = (
    _PKG / "dataset_prep" / "artifacts" / "tajik_asr_combined_v0" / "omni_parquet" / "version=0"
)
# v1 = v0's real Tajik corpora + a transliterated-Persian (FLEURS) augmentation corpus
# (corpus=persian_translit_fleurs). Same version=0 layout; mixture weights come from the
# sibling language_distribution_0.tsv. DEAD END (kept for provenance) — see
# docs/persian-augmentation-experiment-20260530.md; builder archived under
# dataset_prep/archive/build_persian_augmentation.py.
_PARQUET_V1 = (
    _PKG / "dataset_prep" / "artifacts" / "tajik_asr_combined_v1" / "omni_parquet" / "version=0"
)
# v2 = the new curator pipeline's first full export: FLEURS + 41 YouTube channels
# (~1,400 h), Scribe-verified with script-aware scoring, WER <= 0.35 + junk filter.
# Produced by `tajik-curate export v2 --max-wer 0.35`; lives in the project data dir.
_PARQUET_V2 = _PROJECT / "data" / "datasets" / "v2" / "version=0"

TOKENIZER_NAME = "omni_asr_tokenizer_written_v2_local"

CARDS = [
    TokenizerCard(
        name=TOKENIZER_NAME,
        tokenizer=_MODELS / "omniASR_tokenizer_written_v2.model",
    ),
    ModelCard(
        name="omni_ctc_300m_v2_base",
        checkpoint=_MODELS / "omniASR-CTC-300M-v2.pt",
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
    # v1 (Persian-augmented) best dev-WER checkpoint (step_4000).
    ModelCard(
        name="omni_ctc_300m_v2_tajik_v1_step_4000",
        checkpoint=(
            _PROJECT
            / "runs/omni-ctc-300m-tajik-asr-corpus-v1/ws_1.fbafaafe/"
            / "checkpoints/step_4000/model/pp_00/tp_00/sdp_00.pt"
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
        name="tajik_asr_corpus_v1",
        data=_PARQUET_V1,
        tokenizer_ref=TOKENIZER_NAME,
    ),
    MixtureParquetDatasetCard(
        name="tajik_asr_corpus_v2",
        data=_PARQUET_V2,
        tokenizer_ref=TOKENIZER_NAME,
    ),
]


def setup_fairseq2_extension(container: DependencyContainer) -> None:
    register_cards(container, "tajik_omnilingual_asr", CARDS)
