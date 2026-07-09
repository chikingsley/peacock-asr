from __future__ import annotations

import argparse
import json
import sys
import types

import pytest

from parakeet_finetune_core.eval import (
    coerce_hypotheses,
    compute_error_rates,
    compute_wer_percent,
    default_checkpoint_for_kind,
    default_model_for_kind,
    load_manifest,
    load_model,
    make_normalizer,
    replacement_tokenizer_dir,
    write_predictions,
    write_summary,
)
from parakeet_finetune_core.project import ParakeetProject


class FakeEvalModel:
    def __init__(self):
        self.events = []

    def change_vocabulary(self, **kwargs):
        self.events.append(("change_vocabulary", kwargs))

    def load_state_dict(self, state_dict, *, strict):
        self.events.append(("load_state_dict", state_dict, strict))
        return [], []

    def to(self, device):
        self.events.append(("to", device))
        return self

    def eval(self):
        self.events.append(("eval",))
        return self


def _install_fake_model_runtime(monkeypatch, model):
    torch = types.ModuleType("torch")

    def load_checkpoint(*_args, **_kwargs):
        return {"state_dict": {"weight": "trained"}}

    torch.__dict__["load"] = load_checkpoint

    modules = {
        name: types.ModuleType(name)
        for name in [
            "nemo",
            "nemo.collections",
            "nemo.collections.asr",
            "nemo.collections.asr.models",
        ]
    }

    class FakeASRModel:
        @classmethod
        def restore_from(cls, *_args, **_kwargs):
            return model

        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return model

    modules["nemo.collections.asr.models"].__dict__["ASRModel"] = FakeASRModel
    modules["nemo"].__dict__["collections"] = modules["nemo.collections"]
    modules["nemo.collections"].__dict__["asr"] = modules["nemo.collections.asr"]
    modules["nemo.collections.asr"].__dict__["models"] = modules["nemo.collections.asr.models"]
    monkeypatch.setitem(sys.modules, "torch", torch)
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


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


def test_load_model_keeps_final_nemo_vocabulary(monkeypatch, tmp_path):
    model = FakeEvalModel()
    _install_fake_model_runtime(monkeypatch, model)
    final_model = tmp_path / "final.nemo"
    final_model.touch()
    args = argparse.Namespace(
        checkpoint=None,
        device="cpu",
        replace_tokenizer=False,
        tokenizer_dir=tmp_path / "tokenizer",
        tokenizer_type="bpe",
    )

    loaded = load_model(args, final_model)

    assert loaded is model
    assert not any(event[0] == "change_vocabulary" for event in model.events)


def test_load_model_replaces_base_vocabulary_before_checkpoint(monkeypatch, tmp_path):
    model = FakeEvalModel()
    _install_fake_model_runtime(monkeypatch, model)
    base_model = tmp_path / "base.nemo"
    base_model.touch()
    checkpoint = tmp_path / "best.ckpt"
    tokenizer = tmp_path / "tokenizer"
    args = argparse.Namespace(
        checkpoint=checkpoint,
        device="cpu",
        replace_tokenizer=True,
        tokenizer_dir=tokenizer,
        tokenizer_type="bpe",
    )

    load_model(args, base_model)

    assert model.events[0] == (
        "change_vocabulary",
        {
            "new_tokenizer_dir": str(tokenizer.resolve()),
            "new_tokenizer_type": "bpe",
        },
    )
    assert model.events[1] == ("load_state_dict", {"weight": "trained"}, True)


def test_load_model_refuses_tokenizer_replacement_without_checkpoint(tmp_path):
    args = argparse.Namespace(
        checkpoint=None,
        device="cpu",
        replace_tokenizer=True,
        tokenizer_dir=tmp_path / "tokenizer",
        tokenizer_type="bpe",
    )

    with pytest.raises(SystemExit, match="requires --checkpoint"):
        load_model(args, tmp_path / "final.nemo")


def test_compute_error_rates_reports_wer_cer_and_empty_hypotheses():
    rates = compute_error_rates(["one two", "abc"], ["one", ""], str)

    assert rates["wer_percent"] == 66.66666666666666
    assert rates["cer_percent"] > 0
    assert rates["empty_hypotheses"] == 1


def test_write_summary_creates_parent_and_json(tmp_path):
    output = tmp_path / "nested" / "summary.json"

    write_summary(output, {"wer": 12.5, "rows": 10})

    assert json.loads(output.read_text()) == {"wer": 12.5, "rows": 10}


def test_dry_run_validation_refuses_tokenizer_replacement_without_checkpoint(tmp_path):
    args = argparse.Namespace(
        checkpoint=None,
        replace_tokenizer=True,
        tokenizer_dir=tmp_path / "tokenizer",
    )

    with pytest.raises(SystemExit, match="requires --checkpoint"):
        replacement_tokenizer_dir(args)


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
