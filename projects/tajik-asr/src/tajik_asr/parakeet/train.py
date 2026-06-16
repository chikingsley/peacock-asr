"""Tajik Parakeet training commands (tokenizer / CTC / TDT / NeMo recipe)."""

from __future__ import annotations

from parakeet_finetune_core.ctc import train_ctc_main
from parakeet_finetune_core.nemo_recipe import train_nemo_recipe_main
from parakeet_finetune_core.tdt import train_tdt_main
from parakeet_finetune_core.tokenizer import train_tokenizer_main

from tajik_asr.parakeet import PROJECT


def train_tokenizer(argv: list[str] | None = None) -> int:
    return train_tokenizer_main(PROJECT, argv)


def train_ctc(argv: list[str] | None = None) -> int:
    return train_ctc_main(PROJECT, argv)


def train_nemo_recipe(argv: list[str] | None = None) -> int:
    return train_nemo_recipe_main(PROJECT, argv)


def train_tdt(argv: list[str] | None = None) -> int:
    return train_tdt_main(PROJECT, argv)
