"""Tajik Parakeet model family — project config over :mod:`parakeet_finetune_core`.

Training commands live in `tajik_asr.parakeet.train`, evaluation in `tajik_asr.parakeet.eval`.
Canonical artifacts live under `data/parakeet/` (manifests, tokenizers, final models); base models +
NeMo recipes come from the repo-level shared cache `base_models/parakeet/` (no cross-project paths).
See `artifacts.json` for the tracked pointer (sha256 / source command / HF target).
"""

from __future__ import annotations

from parakeet_finetune_core import ParakeetProject

from tajik_asr import DATA, LANGUAGE, ROOT

_BASE = ROOT.parents[1] / "base_models" / "parakeet"  # repo-level shared base/recipe cache
_PK = DATA / "parakeet"

PROJECT = ParakeetProject(
    name="tajik",
    language=LANGUAGE,
    root=ROOT,
    data_root=DATA,
    nemo_root=_BASE / "nemo_recipes",
    model_root=_PK / "models",
    tokenizer_root=_PK / "tokenizers",
    runs_root=ROOT / "runs" / "parakeet",
    default_ctc_model=_BASE / "ctc.nemo",
    default_hybrid_model=_BASE / "parakeet-tdt_ctc-110m-base-hybrid.nemo",
    default_tdt_model=_BASE / "parakeet-tdt_ctc-110m-base-hybrid.nemo",
    # eval defaults to the promoted final model (a stable .nemo), not a runs/ checkpoint
    default_eval_tdt_model=_PK / "final" / "parakeet_tdt_110m_tajik_v3_step_100000.nemo",
    default_train_manifest=_PK / "v3-safe" / "manifests" / "train.jsonl",
    default_validation_manifest=_PK / "v3-safe" / "manifests" / "dev.jsonl",
    heldout_manifest=ROOT / "manifests" / "heldout_test_videos.json",
    evaluation_manifests=(_PK / "v3-safe" / "manifests" / "test.jsonl",),
    default_dataset_root=DATA / "datasets" / "v3" / "version=0",
    default_materialized_root=_PK / "v3-safe",
    default_tokenizer_dir=_PK / "tokenizers" / "v3-safe" / "tokenizer_spe_bpe_v1024",
    default_tokenizer_name="tgk_cyrl_spe_bpe_v1024",
    default_eval_kind="tdt",
    default_eval_normalizer="omni_curator.process:normalize",
    default_eval_normalizer_language=LANGUAGE,
    default_ctc_run_name="tajik-parakeet-ctc-110m",
    default_tdt_run_name="tajik-parakeet-tdt-110m",
)
