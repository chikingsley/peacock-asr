"""Farsi Parakeet fine-tuning commands over :mod:`parakeet_finetune_core`."""

from __future__ import annotations

from parakeet_finetune_core import ParakeetProject
from parakeet_finetune_core.ctc import train_ctc_main
from parakeet_finetune_core.nemo_recipe import train_nemo_recipe_main
from parakeet_finetune_core.tokenizer import train_tokenizer_main

from farsi_asr import DATA, LANGUAGE, ROOT

_LEGACY_PACKAGE = ROOT / "src" / "finetune_parakeet"

PROJECT = ParakeetProject(
    name="farsi",
    language=LANGUAGE,
    root=ROOT,
    data_root=DATA,
    nemo_root=_LEGACY_PACKAGE / "nemo_recipes",
    model_root=_LEGACY_PACKAGE / "models",
    tokenizer_root=_LEGACY_PACKAGE / "tokenizer",
    runs_root=_LEGACY_PACKAGE / "runs",
    default_ctc_model=_LEGACY_PACKAGE / "models" / "ctc.nemo",
    default_tdt_model="nvidia/parakeet-tdt_ctc-110m",
    default_tokenizer_name="fa_spe_bpe_v1024",
    default_ctc_run_name="parakeet-110m-ctc-persian",
    default_tdt_run_name="parakeet",
    env_prefix="PERSIAN",
)


def train_tokenizer(argv: list[str] | None = None) -> int:
    return train_tokenizer_main(PROJECT, argv)


def train_ctc(argv: list[str] | None = None) -> int:
    return train_ctc_main(PROJECT, argv)


def train_nemo_recipe(argv: list[str] | None = None) -> int:
    return train_nemo_recipe_main(PROJECT, argv)
