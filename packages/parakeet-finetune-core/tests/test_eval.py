from __future__ import annotations

import json
import sys
import types

from parakeet_finetune_core.eval import (
    coerce_hypotheses,
    compute_wer_percent,
    default_checkpoint_for_kind,
    default_model_for_kind,
    load_manifest,
    make_normalizer,
    write_predictions,
)
from parakeet_finetune_core.project import ParakeetProject


def test_eval_defaults_are_kind_specific(tmp_path):
    project = ParakeetProject(
        name="tajik",
        language="tgk_Cyrl",
        root=tmp_path,
        default_ctc_model=tmp_path / "ctc.nemo",
        default_tdt_model=tmp_path / "tdt.nemo",
        default_ctc_checkpoint=tmp_path / "ctc.ckpt",
        default_tdt_checkpoint=tmp_path / "tdt.ckpt",
    )

    assert default_model_for_kind(project, "ctc") == tmp_path / "ctc.nemo"
    assert default_model_for_kind(project, "tdt") == tmp_path / "tdt.nemo"
    assert default_checkpoint_for_kind(project, "ctc") == tmp_path / "ctc.ckpt"
    assert default_checkpoint_for_kind(project, "tdt") == tmp_path / "tdt.ckpt"


def test_load_manifest_filters_duration_and_limit(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    rows = [
        {"audio_filepath": "a.flac", "text": "one", "duration": 1.0},
        {"audio_filepath": "b.flac", "text": "two", "duration": 40.0},
        {"audio_filepath": "c.flac", "text": "three", "duration": 2.0},
    ]
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    loaded = load_manifest(
        manifest,
        audio_field="audio_filepath",
        text_field="text",
        duration_field="duration",
        max_duration=10.0,
        limit=1,
    )

    assert len(loaded) == 1
    assert loaded[0].audio_filepath == "a.flac"
    assert loaded[0].text == "one"
    assert loaded[0].duration == 1.0


def test_make_normalizer_supports_one_arg_and_two_arg_functions(monkeypatch):
    module = types.ModuleType("fake_normalizers")

    def one_arg(text):
        return text.upper()

    def two_arg(text, language):
        return f"{language}:{text.lower()}"

    def none_arg(_text):
        return None

    module.__dict__["one_arg"] = one_arg
    module.__dict__["two_arg"] = two_arg
    module.__dict__["none_arg"] = none_arg
    monkeypatch.setitem(sys.modules, "fake_normalizers", module)

    assert make_normalizer("fake_normalizers:one_arg", "tgk_Cyrl")("AbC") == "ABC"
    assert make_normalizer("fake_normalizers:two_arg", "tgk_Cyrl")("AbC") == "tgk_Cyrl:abc"
    assert make_normalizer("fake_normalizers:none_arg", None)("AbC") == ""


def test_compute_wer_percent_uses_normalized_text():
    def normalize(text: str) -> str:
        return text.lower()

    assert compute_wer_percent(["HELLO WORLD"], ["hello there"], normalize) == 50.0


def test_coerce_hypotheses_accepts_strings_and_hypothesis_objects():
    class Hypothesis:
        text = "object text"

    assert coerce_hypotheses(["plain", Hypothesis()]) == ["plain", "object text"]


def test_write_predictions_jsonl(tmp_path):
    output = tmp_path / "predictions" / "eval.jsonl"
    rows = load_manifest(
        _write_manifest(tmp_path, [{"audio_filepath": "a.flac", "text": "ref"}]),
        audio_field="audio_filepath",
        text_field="text",
        duration_field="duration",
        max_duration=None,
        limit=0,
    )

    write_predictions(output, rows, ["hyp"])

    assert json.loads(output.read_text(encoding="utf-8")) == {
        "audio_filepath": "a.flac",
        "text": "ref",
        "hypothesis": "hyp",
        "duration": None,
    }


def _write_manifest(tmp_path, rows: list[dict[str, str]]):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return manifest
