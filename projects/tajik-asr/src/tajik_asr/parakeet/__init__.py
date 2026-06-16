"""Tajik Parakeet model family — project config over :mod:`parakeet_finetune_core`.

Training commands live in `tajik_asr.parakeet.train`, evaluation in `tajik_asr.parakeet.eval`.
"""

from __future__ import annotations

from parakeet_finetune_core import ParakeetProject

from tajik_asr import DATA, LANGUAGE, ROOT

_PEACOCK = ROOT.parents[1]
_FARSI_PARAKEET = _PEACOCK / "projects" / "farsi-asr" / "src" / "finetune_parakeet"
_TDT_EXPERIMENT = ROOT / "experiments" / "tdt"

PROJECT = ParakeetProject(
    name="tajik",
    language=LANGUAGE,
    root=ROOT,
    data_root=DATA,
    nemo_root=_FARSI_PARAKEET / "nemo_recipes",
    model_root=DATA / "models" / "parakeet",
    tokenizer_root=DATA / "tokenizers" / "parakeet",
    runs_root=ROOT / "runs" / "parakeet",
    default_ctc_model=_FARSI_PARAKEET / "models" / "ctc.nemo",
    default_hybrid_model=_FARSI_PARAKEET / "models" / "parakeet-tdt_ctc-110m-base-hybrid.nemo",
    default_tdt_model=_FARSI_PARAKEET / "models" / "parakeet-tdt_ctc-110m-base-hybrid.nemo",
    default_tdt_checkpoint=ROOT
    / "runs"
    / "parakeet"
    / "tajik-parakeet-tdt-110m"
    / "checkpoints"
    / "last.ckpt",
    default_train_manifest=_TDT_EXPERIMENT / "data" / "train_big.jsonl",
    default_validation_manifest=_TDT_EXPERIMENT / "data" / "dev_big.jsonl",
    default_tokenizer_dir=_TDT_EXPERIMENT / "data" / "tok_big" / "tokenizer_spe_bpe_v1024",
    default_tokenizer_name="tgk_cyrl_spe_bpe_v1024",
    default_eval_kind="tdt",
    default_eval_normalizer="omni_curator.process:normalize",
    default_eval_normalizer_language=LANGUAGE,
    default_ctc_run_name="tajik-parakeet-ctc-110m",
    default_tdt_run_name="tajik-parakeet-tdt-110m",
)
