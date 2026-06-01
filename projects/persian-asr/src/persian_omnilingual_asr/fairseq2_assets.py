"""fairseq2 asset cards for the Persian ASR project, registered in-process.

Replaces the hand-maintained ``.fairseq2-assets/`` YAML directory (which the old
``finetune_omni`` training path resolved via ``FAIRSEQ2_ASSET_DIR``). The training
config refers to model/dataset by name; fairseq2 resolves those names through its asset
store, populated from every package that exposes a ``fairseq2.extension`` entry point.
We register our cards here so the definitions live next to the code that uses them and
the on-disk paths derive from ``__file__`` instead of being hardcoded absolute paths.

The base model (``omniASR_CTC_300M_v2``) and the tokenizer (``omniASR_tokenizer_written_v2``)
come from the upstream ``omnilingual-asr`` package's own cards (auto-downloaded from the
HF Hub), so we do not register them here — every card below references the tokenizer by
that upstream name. We register only the Persian *fine-tuned* model cards and the
project's mixture-parquet dataset cards.

The card *shapes* are the typed Pydantic models in :mod:`omni_finetune_core.assets`;
this module only supplies the Persian values. Wired up via the
``[project.entry-points."fairseq2.extension"]`` entry point in ``pyproject.toml`` ->
``setup_fairseq2_extension``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from omni_finetune_core.assets import (
    MixtureParquetDatasetCard,
    ModelCard,
    register_cards,
)

if TYPE_CHECKING:
    from fairseq2.runtime.dependency import DependencyContainer

# This file lives at src/persian_omnilingual_asr/fairseq2_assets.py, so parents[2] is the
# persian-asr project root. All asset paths below derive from it.
_PROJECT = Path(__file__).resolve().parents[2]

# Checkpoint roots. Exported "best/final" checkpoints are collected under
# models/omnilingual/checkpoints/<name>/model.pt; live training runs keep the raw
# fairseq2 sharded layout under runs/.../checkpoints/step_*/model/pp_00/tp_00/sdp_00.pt.
_CKPTS = _PROJECT / "models/omnilingual/checkpoints"
_FT_RUNS = _PROJECT / "src/finetune_omni/runs"
_RUNS = _PROJECT / "runs"

# Dataset (mixture-parquet) roots. Most curated training sets live under the project-root
# data/ tree; scribe-v4 currently sits inside the old finetune_omni/data/ tree.
_DATA = _PROJECT / "data"
_FT_DATA = _PROJECT / "src/finetune_omni/data"

# Upstream tokenizer card name (provided by the omnilingual-asr package).
TOKENIZER_NAME = "omniASR_tokenizer_written_v2"


def _sharded(run: Path, step: int) -> Path:
    """Path to a raw fairseq2 single-shard checkpoint inside a training run."""
    return run / f"checkpoints/step_{step}/model/pp_00/tp_00/sdp_00.pt"


def _model(name: str, checkpoint: Path) -> ModelCard:
    return ModelCard(name=name, checkpoint=checkpoint, tokenizer_ref=TOKENIZER_NAME)


def _dataset(name: str, data: Path) -> MixtureParquetDatasetCard:
    return MixtureParquetDatasetCard(name=name, data=data, tokenizer_ref=TOKENIZER_NAME)


MODEL_CARDS = [
    # --- FLEURS / Thomcles baselines (exported best/final checkpoints) ---
    _model("omni_ctc_300m_v2_fleurs_fa_ir_best", _CKPTS / "omni_ctc_300m_v2_fleurs_fa_ir_best/model.pt"),
    _model("omni_ctc_300m_v2_fleurs_fa_ir_final", _CKPTS / "omni_ctc_300m_v2_fleurs_fa_ir_final/model.pt"),
    _model(
        "omni_ctc_300m_v2_fleurs_fa_ir_thomcles_final",
        _CKPTS / "omni_ctc_300m_v2_fleurs_fa_ir_thomcles_final/model.pt",
    ),
    # --- 100h curated-filter ablations ---
    _model(
        "omni_ctc_300m_v2_persian_balanced_100h_filter_best",
        _CKPTS / "omni_ctc_300m_v2_persian_balanced_100h_filter_best/model.pt",
    ),
    _model(
        "omni_ctc_300m_v2_persian_clean_100h_filter_best",
        _CKPTS / "omni_ctc_300m_v2_persian_clean_100h_filter_best/model.pt",
    ),
    _model(
        "omni_ctc_300m_v2_persian_target_100h_filter_best",
        _CKPTS / "omni_ctc_300m_v2_persian_target_100h_filter_best/model.pt",
    ),
    _model(
        "omni_ctc_300m_v2_persian_wer35_fastconformer_best",
        _CKPTS / "omni_ctc_300m_v2_persian_wer35_fastconformer_best/model.pt",
    ),
    # --- Scribe exact-match (step_34000 = best == final for this run) ---
    _model(
        "omni_ctc_300m_v2_persian_scribe_exact_match_20260525_best",
        _sharded(_FT_RUNS / "omni-ctc-300m-scribe-exact-match-20260525/ws_1.c5ad768d", 34000),
    ),
    _model(
        "omni_ctc_300m_v2_persian_scribe_exact_match_20260525_final",
        _sharded(_FT_RUNS / "omni-ctc-300m-scribe-exact-match-20260525/ws_1.c5ad768d", 34000),
    ),
    # --- Scribe v3 max (continued from exact-match best) ---
    _model(
        "omni_ctc_300m_v2_persian_scribe_v3_max_20260527_best_retained_step38000",
        _sharded(_RUNS / "omni-ctc-300m-scribe-v3-max-20260527-from-exact-best/ws_1.519fd28b", 38000),
    ),
    _model(
        "omni_ctc_300m_v2_persian_scribe_v3_max_20260527_final_step40000",
        _sharded(_RUNS / "omni-ctc-300m-scribe-v3-max-20260527-from-exact-best/ws_1.519fd28b", 40000),
    ),
    # --- Scribe v4 (current line): re-warm best + pinned pre-rewarm baseline ---
    _model(
        "omni_ctc_300m_v2_scribe_v4_20260530_best",
        _sharded(_FT_RUNS / "scribe-v4-rewarm10k/ws_1.5318420d", 7000),
    ),
    _model(
        "omni_ctc_300m_v2_scribe_v4_baseline_step34000",
        _sharded(_FT_RUNS / "scribe-v4/ws_1.760d57f2", 34000),
    ),
]

DATASET_CARDS = [
    _dataset("fleurs_fa_ir", _DATA / "raw/fleurs_fa_ir_omni/version=0"),
    _dataset("thomcles_persian_farsi_speech", _DATA / "raw/thomcles_persian_omni/version=0"),
    _dataset(
        "persian_asr_balanced_100h_filter",
        _DATA / "training/omnilingual/persian_asr_balanced_100h_filter/version=0",
    ),
    _dataset(
        "persian_asr_clean_100h_filter",
        _DATA / "training/omnilingual/persian_asr_clean_100h_filter/version=0",
    ),
    _dataset(
        "persian_asr_target_100h_filter",
        _DATA / "training/omnilingual/persian_asr_target_100h_filter/version=0",
    ),
    _dataset(
        "persian_asr_scribe_exact_match_20260525",
        _DATA / "training/omnilingual/persian_asr_scribe_exact_match_20260525/version=0",
    ),
    _dataset(
        "persian_asr_scribe_v3_max_20260527",
        _DATA / "training/omnilingual/persian_asr_scribe_v3_max_20260527/version=0",
    ),
    _dataset("scribe_v4", _FT_DATA / "training/omnilingual/scribe-v4/version=0"),
]

CARDS = [*MODEL_CARDS, *DATASET_CARDS]


def setup_fairseq2_extension(container: DependencyContainer) -> None:
    register_cards(container, "persian_omnilingual_asr", CARDS)
