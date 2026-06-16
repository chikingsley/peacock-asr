"""Tajik Parakeet fine-tuning commands over :mod:`parakeet_finetune_core`."""

from __future__ import annotations

from parakeet_finetune_core import ParakeetProject
from parakeet_finetune_core.ctc import train_ctc_main
from parakeet_finetune_core.nemo_recipe import train_nemo_recipe_main
from parakeet_finetune_core.tdt import train_tdt_main
from parakeet_finetune_core.tokenizer import train_tokenizer_main

from tajik_omnilingual_asr import DATA, LANGUAGE, ROOT

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
    default_train_manifest=_TDT_EXPERIMENT / "data" / "train_big.jsonl",
    default_validation_manifest=_TDT_EXPERIMENT / "data" / "dev_big.jsonl",
    default_tokenizer_dir=_TDT_EXPERIMENT / "data" / "tok_big" / "tokenizer_spe_bpe_v1024",
    default_tokenizer_name="tgk_cyrl_spe_bpe_v1024",
    default_ctc_run_name="tajik-parakeet-ctc-110m",
    default_tdt_run_name="tajik-parakeet-tdt-110m",
)


def train_tokenizer(argv: list[str] | None = None) -> int:
    return train_tokenizer_main(PROJECT, argv)


def train_ctc(argv: list[str] | None = None) -> int:
    return train_ctc_main(PROJECT, argv)


def train_nemo_recipe(argv: list[str] | None = None) -> int:
    return train_nemo_recipe_main(PROJECT, argv)


def train_tdt(argv: list[str] | None = None) -> int:
    return train_tdt_main(PROJECT, argv)
