from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from p004_training_from_scratch.canonical.train_smoke import (
    ResumeCheckpoint,
    _build_ctc_log_probs,
    _edit_distance,
    _load_resume_checkpoint,
    _mark_cudagraph_step_begin,
    _normalize_model_state_dict,
    _uses_flex_attention,
    load_phone_token_table,
    read_smoke_cuts,
    run_canonical_train_smoke,
)


def test_load_phone_token_table_parses_ids(tmp_path: Path) -> None:
    tokens_path = tmp_path / "tokens.txt"
    tokens_path.write_text("<eps> 0\nAA 1\nBB 2\n", encoding="utf-8")

    token_table = load_phone_token_table(tokens_path)

    assert token_table == {"<eps>": 0, "AA": 1, "BB": 2}


def test_read_smoke_cuts_reads_manifest_and_validates_tokens(tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake")
    manifest_path = tmp_path / "cuts.jsonl.gz"
    payload = {
        "id": "cut-1",
        "duration": 1.25,
        "recording": {"sources": [{"source": str(audio_path)}]},
        "supervisions": [{"text": "AA BB"}],
    }
    with gzip.open(manifest_path, "wt", encoding="utf-8") as handle:
        handle.write(f"{json.dumps(payload)}\n")

    cuts = read_smoke_cuts(
        manifest_path,
        limit=1,
        token_table={"<eps>": 0, "AA": 1, "BB": 2},
    )

    assert len(cuts) == 1
    assert cuts[0].cut_id == "cut-1"
    assert cuts[0].phones == ("AA", "BB")
    assert cuts[0].duration_seconds == pytest.approx(1.25)


def test_read_smoke_cuts_rejects_unknown_phones(tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake")
    manifest_path = tmp_path / "cuts.jsonl.gz"
    payload = {
        "id": "cut-1",
        "recording": {"sources": [{"source": str(audio_path)}]},
        "supervisions": [{"text": "AA CC"}],
    }
    with gzip.open(manifest_path, "wt", encoding="utf-8") as handle:
        handle.write(f"{json.dumps(payload)}\n")

    with pytest.raises(ValueError):
        read_smoke_cuts(
            manifest_path,
            limit=1,
            token_table={"<eps>": 0, "AA": 1, "BB": 2},
        )


def test_edit_distance_counts_insertions_deletions_and_substitutions() -> None:
    assert _edit_distance([1, 2, 3], [1, 2, 3]) == 0
    assert _edit_distance([1, 3], [1, 2, 3]) == 1
    assert _edit_distance([1, 2, 3, 4], [1, 2, 3]) == 1
    assert _edit_distance([1, 4, 3], [1, 2, 3]) == 1


def test_load_resume_checkpoint_parses_expected_fields() -> None:
    payload = {
        "epoch": 2,
        "model_state": {"weight": [1, 2, 3]},
        "optimizer_state": {"state": {}},
    }

    checkpoint = _load_resume_checkpoint(payload)

    assert checkpoint == ResumeCheckpoint(
        epoch=2,
        model_state={"weight": [1, 2, 3]},
        optimizer_state={"state": {}},
    )


def test_load_resume_checkpoint_rejects_legacy_payload() -> None:
    with pytest.raises(ValueError, match="model_state"):
        _load_resume_checkpoint({"epoch": 1})


def test_normalize_model_state_dict_strips_compiled_prefix() -> None:
    normalized = _normalize_model_state_dict(
        {
            "_orig_mod.encoder.weight": [1, 2],
            "_orig_mod.encoder.bias": [3],
        }
    )

    assert normalized == {
        "encoder.weight": [1, 2],
        "encoder.bias": [3],
    }


def test_uses_flex_attention_matches_backend_contract() -> None:
    assert _uses_flex_attention("mha") is False
    assert _uses_flex_attention("flex_auto") is True
    assert _uses_flex_attention("flex_triton") is True
    assert _uses_flex_attention("flex_flash") is True


def test_mark_cudagraph_step_begin_uses_compiler_hook() -> None:
    calls: list[str] = []

    class Compiler:
        @staticmethod
        def cudagraph_mark_step_begin() -> None:
            calls.append("marked")

    class TorchStub:
        compiler = Compiler()

    _mark_cudagraph_step_begin(
        torch=TorchStub(),
        compile_enabled=True,
        attention_backend="flex_triton",
    )

    assert calls == ["marked"]


def test_build_ctc_log_probs_can_promote_to_float32() -> None:
    class LogitsStub:
        def __init__(self) -> None:
            self.float_called = False

        def float(self) -> LogitsStub:
            self.float_called = True
            return self

        def log_softmax(self, dim: int) -> tuple[str, int]:
            return ("log_softmax", dim)

    logits = LogitsStub()

    result = _build_ctc_log_probs(logits=logits, loss_compute_dtype="float32")

    assert logits.float_called is True
    assert result == ("log_softmax", -1)


def test_build_ctc_log_probs_keeps_model_dtype_by_default() -> None:
    class LogitsStub:
        def __init__(self) -> None:
            self.float_called = False

        def float(self) -> LogitsStub:
            self.float_called = True
            return self

        def log_softmax(self, dim: int) -> tuple[str, int]:
            return ("log_softmax", dim)

    logits = LogitsStub()

    result = _build_ctc_log_probs(logits=logits, loss_compute_dtype="model")

    assert logits.float_called is False
    assert result == ("log_softmax", -1)


def test_run_canonical_train_smoke_rejects_flex_without_compile(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="flex attention backends require"):
        run_canonical_train_smoke(
            output_dir=tmp_path / "smoke",
            attention_backend="flex_triton",
            enable_compile=False,
        )
