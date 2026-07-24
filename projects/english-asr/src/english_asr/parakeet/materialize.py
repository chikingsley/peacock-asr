"""Materialize English omni-parquet exports for NeMo."""

from __future__ import annotations

from parakeet_finetune_core.materialize import materialize_main

from english_asr.parakeet import PROJECT


def main(argv: list[str] | None = None) -> int:
    return materialize_main(PROJECT, argv)
