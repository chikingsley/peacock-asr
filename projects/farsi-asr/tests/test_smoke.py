"""Smoke test: every console-entry callable imports (catches broken-import regressions)."""

from __future__ import annotations


def test_cli_entrypoints_import() -> None:
    from farsi_asr.curate import main as curate
    from farsi_asr.omni.eval import main as omni_eval
    from farsi_asr.omni.train import main as omni_train
    from farsi_asr.parakeet.eval import evaluate
    from farsi_asr.parakeet.train import (
        train_ctc,
        train_nemo_recipe,
        train_tdt,
        train_tokenizer,
    )

    fns = (curate, omni_train, omni_eval, evaluate,
           train_ctc, train_nemo_recipe, train_tdt, train_tokenizer)
    assert all(callable(f) for f in fns)
