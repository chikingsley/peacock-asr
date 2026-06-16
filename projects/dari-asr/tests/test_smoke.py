"""Smoke test: every console-entry callable imports (catches broken-import regressions)."""

from __future__ import annotations


def test_cli_entrypoints_import() -> None:
    from dari_asr.curate import main as curate
    from dari_asr.omni.eval import main as omni_eval
    from dari_asr.omni.train import main as omni_train

    assert all(callable(f) for f in (curate, omni_train, omni_eval))
