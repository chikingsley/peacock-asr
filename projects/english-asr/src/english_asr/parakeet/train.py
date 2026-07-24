"""English Parakeet TDT training command."""

from __future__ import annotations

from parakeet_finetune_core.tdt import train_tdt_main

from english_asr.parakeet import PROJECT


def train_tdt(argv: list[str] | None = None) -> int:
    return train_tdt_main(PROJECT, argv)
