"""English Parakeet evaluation command."""

from __future__ import annotations

from parakeet_finetune_core.eval import eval_main

from english_asr.parakeet import PROJECT


def evaluate(argv: list[str] | None = None) -> int:
    return eval_main(PROJECT, argv)
