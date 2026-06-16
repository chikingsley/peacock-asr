"""Persian Parakeet model family — project config over :mod:`parakeet_finetune_core`.

Training commands live in `farsi_asr.parakeet.train`, evaluation in `farsi_asr.parakeet.eval`.
Base models + NeMo recipes come from the repo-level shared cache `base_models/parakeet/` (no
cross-project paths, no legacy `src/finetune_parakeet` fallback).
"""

from __future__ import annotations

from parakeet_finetune_core import ParakeetProject

from farsi_asr import DATA, LANGUAGE, ROOT

_BASE = ROOT.parents[1] / "base_models" / "parakeet"  # repo-level shared base/recipe cache
_PK = DATA / "parakeet"

PROJECT = ParakeetProject(
    name="farsi",
    language=LANGUAGE,
    root=ROOT,
    data_root=DATA,
    nemo_root=_BASE / "nemo_recipes",
    model_root=_PK / "models",
    tokenizer_root=_PK / "tokenizers",
    runs_root=ROOT / "runs" / "parakeet",
    default_ctc_model=_BASE / "ctc.nemo",
    default_hybrid_model=_BASE / "parakeet-tdt_ctc-110m-base-hybrid.nemo",
    default_tdt_model="nvidia/parakeet-tdt_ctc-110m",
    default_tokenizer_name="fa_spe_bpe_v1024",
    default_eval_kind="ctc",
    default_ctc_run_name="parakeet-110m-ctc-persian",
    default_tdt_run_name="parakeet",
    env_prefix="PERSIAN",
)
