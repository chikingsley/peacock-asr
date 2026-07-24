"""English Parakeet TDT 110M configuration."""

from __future__ import annotations

from parakeet_finetune_core import ParakeetProject

from english_asr import DATA, LANGUAGE, ROOT

_BASE = ROOT.parents[1] / "base_models" / "parakeet"
_PARAKEET = DATA / "parakeet"

PROJECT = ParakeetProject(
    name="english",
    language=LANGUAGE,
    root=ROOT,
    data_root=DATA,
    nemo_root=_BASE / "nemo_recipes",
    model_root=_PARAKEET / "models",
    runs_root=ROOT / "runs" / "parakeet",
    default_hybrid_model=_BASE / "parakeet-tdt_ctc-110m.nemo",
    default_tdt_model=_BASE / "parakeet-tdt_ctc-110m.nemo",
    default_dataset_root=DATA / "datasets" / "pilot-v0" / "version=0",
    default_materialized_root=_PARAKEET / "materialized" / "pilot-v0",
    default_train_manifest=_PARAKEET / "materialized" / "pilot-v0" / "manifests" / "train.jsonl",
    default_validation_manifest=_PARAKEET / "materialized" / "pilot-v0" / "manifests" / "dev.jsonl",
    default_tokenizer_dir=None,
    default_eval_tdt_model=_BASE / "parakeet-tdt_ctc-110m.nemo",
    default_eval_kind="tdt",
    default_eval_normalizer="english_asr.evaluation:normalize_wer",
    default_eval_normalizer_language=None,
    default_tdt_run_name="english-parakeet-tdt-110m-pilot-v0",
)
