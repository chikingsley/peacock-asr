"""Smoke test: every console-entry callable imports (catches broken-import regressions)."""

from __future__ import annotations


def test_cli_entrypoints_import() -> None:
    from russian_asr.curate import main as curate
    from russian_asr.omni.eval import main as omni_eval
    from russian_asr.omni.train import main as omni_train

    assert all(callable(f) for f in (curate, omni_train, omni_eval))
