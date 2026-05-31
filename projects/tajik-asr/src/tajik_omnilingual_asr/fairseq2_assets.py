"""fairseq2 asset cards for the Tajik ASR project, registered in-process.

Replaces the hand-maintained ``.fairseq2-assets/`` YAML directory. The training
config refers to the model/tokenizer/dataset by name (``omni_ctc_300m_v2_base``,
``omni_asr_tokenizer_written_v2_local``, ``tajik_asr_corpus``); fairseq2 resolves
those names through its asset store, which is populated from every package that
exposes a ``fairseq2.extension`` entry point. We register our three cards here so the
definitions live next to the code that uses them, and so the on-disk paths are derived
from ``__file__`` instead of being hardcoded absolute paths.

Wired up via the ``[project.entry-points."fairseq2.extension"]`` entry point in
``pyproject.toml`` -> ``setup_fairseq2_extension``.
"""

from __future__ import annotations

from pathlib import Path

from fairseq2.composition.assets import register_in_memory_assets
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

TOKENIZER_NAME = "omni_asr_tokenizer_written_v2_local"

ASSET_CARDS: list[dict[str, object]] = [
    {
        "name": TOKENIZER_NAME,
        "tokenizer_family": "char_tokenizer",
        "tokenizer": str(_MODELS / "omniASR_tokenizer_written_v2.model"),
    },
    {
        "name": "omni_ctc_300m_v2_base",
        "model_family": "wav2vec2_asr",
        "model_arch": "300m_v2",
        "checkpoint": str(_MODELS / "omniASR-CTC-300M-v2.pt"),
        "tokenizer_ref": TOKENIZER_NAME,
    },
    {
        "name": "omni_ctc_300m_v2_tajik_step_1800",
        "model_family": "wav2vec2_asr",
        "model_arch": "300m_v2",
        "checkpoint": str(
            _PROJECT
            / "runs/omni-ctc-300m-tajik-asr-corpus-v0/ws_1.3dcb9e0b/"
            / "checkpoints/step_1800/model/pp_00/tp_00/sdp_00.pt"
        ),
        "tokenizer_ref": TOKENIZER_NAME,
    },
    {
        # v1 (Persian-augmented) best dev-WER checkpoint (step_4000).
        "name": "omni_ctc_300m_v2_tajik_v1_step_4000",
        "model_family": "wav2vec2_asr",
        "model_arch": "300m_v2",
        "checkpoint": str(
            _PROJECT
            / "runs/omni-ctc-300m-tajik-asr-corpus-v1/ws_1.fbafaafe/"
            / "checkpoints/step_4000/model/pp_00/tp_00/sdp_00.pt"
        ),
        "tokenizer_ref": TOKENIZER_NAME,
    },
    {
        "name": "tajik_asr_corpus",
        "dataset_family": "mixture_parquet_asr_dataset",
        "dataset_config": {"data": str(_PARQUET)},
        "tokenizer_ref": TOKENIZER_NAME,
    },
    {
        "name": "tajik_asr_corpus_v1",
        "dataset_family": "mixture_parquet_asr_dataset",
        "dataset_config": {"data": str(_PARQUET_V1)},
        "tokenizer_ref": TOKENIZER_NAME,
    },
]


def setup_fairseq2_extension(container: DependencyContainer) -> None:
    register_in_memory_assets(container, "tajik_omnilingual_asr", ASSET_CARDS)
