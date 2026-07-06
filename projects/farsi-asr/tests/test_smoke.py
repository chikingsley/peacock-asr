"""Smoke test: every console-entry callable imports (catches broken-import regressions)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def test_cli_entrypoints_import() -> None:
    from farsi_asr.curate import main as curate
    from farsi_asr.omni.eval import main as omni_eval
    from farsi_asr.omni.lm_eval import main as omni_eval_lm
    from farsi_asr.omni.train import main as omni_train
    from farsi_asr.parakeet.eval import evaluate
    from farsi_asr.parakeet.train import (
        train_ctc,
        train_nemo_recipe,
        train_tdt,
        train_tokenizer,
    )

    fns = (
        curate,
        omni_train,
        omni_eval,
        omni_eval_lm,
        evaluate,
        train_ctc,
        train_nemo_recipe,
        train_tdt,
        train_tokenizer,
    )
    assert all(callable(f) for f in fns)


def test_lm_eval_load_rows_honors_limit(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    from farsi_asr.omni.lm_eval import load_rows

    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    pq.write_table(
        pa.table({"audio_bytes": [b"a", b"b"], "normalized_text": ["one", "two"]}),
        first,
    )
    pq.write_table(
        pa.table({"audio_bytes": [b"c", b"d"], "normalized_text": ["three", "four"]}),
        second,
    )

    audio, refs = load_rows([first, second], limit=3)

    assert audio == [b"a", b"b", b"c"]
    assert refs == ["one", "two", "three"]


def test_curator_export_schema_matches_finetune_reader() -> None:
    from omni_curator.data.export import EXPORT_SCHEMA
    from omni_finetune_core.parquet import OMNI_SCHEMA

    assert EXPORT_SCHEMA.names[: len(OMNI_SCHEMA.names)] == OMNI_SCHEMA.names
    assert EXPORT_SCHEMA.types[: len(OMNI_SCHEMA.types)] == OMNI_SCHEMA.types
