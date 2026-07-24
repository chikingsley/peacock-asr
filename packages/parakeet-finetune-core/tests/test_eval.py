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
    enable_memory_efficient_subsampling,
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

    def change_attention_model(self, attention_model, context):
        self.events.append(("change_attention_model", attention_model, context))


class FakeHybridEvalModel(FakeEvalModel):
    cur_decoder = "rnnt"

    def change_decoding_strategy(self, config, *, decoder_type):
        self.events.append(("change_decoding_strategy", config, decoder_type))


class FakeStandaloneEvalModel(FakeEvalModel):
    def change_decoding_strategy(self, config):
        self.events.append(("change_decoding_strategy", config))


def _install_fake_model_runtime(monkeypatch, model):
    torch = types.ModuleType("torch")
    torch.__dict__["cuda"] = types.SimpleNamespace(empty_cache=lambda: None)

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


def test_load_model_configures_official_longform_attention_and_bf16(monkeypatch, tmp_path):
    model = FakeEvalModel()
    _install_fake_model_runtime(monkeypatch, model)
    sys.modules["torch"].__dict__["bfloat16"] = "bf16"
    final_model = tmp_path / "final.nemo"
    final_model.touch()
    args = argparse.Namespace(
        checkpoint=None,
        device="cuda",
        inference_dtype="bf16",
        longform_attention_context=128,
        load_model_on_cpu=True,
        disable_cuda_graph_decoder=False,
        memory_efficient_subsampling=False,
        replace_tokenizer=False,
        tokenizer_dir=tmp_path / "tokenizer",
        tokenizer_type="bpe",
    )

    loaded = load_model(args, final_model)

    assert loaded is model
    assert ("change_attention_model", "rel_pos_local_attn", [128, 128]) in model.events
    assert ("to", "bf16") in model.events
    assert ("to", "cuda") in model.events


def test_load_model_disables_cuda_graph_decoder(monkeypatch, tmp_path):
    from omegaconf import OmegaConf

    model = FakeHybridEvalModel()
    model.cfg = OmegaConf.create(
        {
            "decoding": {"strategy": "greedy_batch", "greedy": {"max_symbols": 10}},
            "aux_ctc": {"decoding": {}},
        }
    )
    _install_fake_model_runtime(monkeypatch, model)
    final_model = tmp_path / "final.nemo"
    final_model.touch()
    args = argparse.Namespace(
        checkpoint=None,
        device="cpu",
        kind="tdt",
        ngram_lm=None,
        beam_size=0,
        inference_dtype="fp32",
        longform_attention_context=0,
        load_model_on_cpu=True,
        disable_cuda_graph_decoder=True,
        memory_efficient_subsampling=False,
        replace_tokenizer=False,
        tokenizer_dir=tmp_path / "tokenizer",
        tokenizer_type="bpe",
    )

    load_model(args, final_model)

    event = next(event for event in model.events if event[0] == "change_decoding_strategy")
    assert event[2] == "rnnt"
    assert event[1].greedy.use_cuda_graph_decoder is False


def test_memory_efficient_subsampling_replaces_nemo_channel_concat():
    class Subsampler:
        def channel_chunked_conv(self, *_args):
            raise AssertionError("original implementation should be replaced")

    subsampler = Subsampler()
    model = types.SimpleNamespace(encoder=types.SimpleNamespace(pre_encode=subsampler))
    torch = types.SimpleNamespace()

    enable_memory_efficient_subsampling(model, torch)

    assert subsampler.channel_chunked_conv.__self__ is subsampler
    assert subsampler.channel_chunked_conv.__name__ == "channel_chunked_conv"


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


@pytest.mark.parametrize(("kind", "decoder_type"), [("tdt", "rnnt"), ("ctc", "ctc")])
def test_load_model_selects_requested_hybrid_head(monkeypatch, tmp_path, kind, decoder_type):
    model = FakeHybridEvalModel()
    _install_fake_model_runtime(monkeypatch, model)
    final_model = tmp_path / "final.nemo"
    final_model.touch()
    args = argparse.Namespace(
        checkpoint=None,
        device="cpu",
        kind=kind,
        ngram_lm=None,
        replace_tokenizer=False,
        tokenizer_dir=tmp_path / "tokenizer",
        tokenizer_type="bpe",
    )

    load_model(args, final_model)

    assert ("change_decoding_strategy", None, decoder_type) in model.events


def test_load_model_configures_tdt_batched_beam_ngpu_lm(monkeypatch, tmp_path):
    from omegaconf import OmegaConf

    model = FakeHybridEvalModel()
    model.cfg = OmegaConf.create(
        {
            "decoding": {
                "strategy": "greedy_batch",
                "greedy": {"ngram_lm_model": None, "ngram_lm_alpha": 0.0},
                "beam": {
                    "beam_size": 2,
                    "return_best_hypothesis": False,
                    "ngram_lm_model": None,
                    "ngram_lm_alpha": 0.0,
                    "pruning_mode": "LATE",
                    "blank_lm_score_mode": "LM_WEIGHTED_FULL",
                },
            },
            "aux_ctc": {"decoding": {}},
        }
    )
    _install_fake_model_runtime(monkeypatch, model)
    final_model = tmp_path / "final.nemo"
    final_model.touch()
    lm = tmp_path / "lm.nemo"
    args = argparse.Namespace(
        checkpoint=None,
        device="cpu",
        kind="tdt",
        ngram_lm=lm,
        ngram_lm_alpha=0.3,
        beam_size=8,
        beam_beta=0.0,
        replace_tokenizer=False,
        tokenizer_dir=tmp_path / "tokenizer",
        tokenizer_type="bpe",
    )

    load_model(args, final_model)

    event = next(event for event in model.events if event[0] == "change_decoding_strategy")
    config = event[1]
    assert event[2] == "rnnt"
    assert config.strategy == "malsd_batch"
    assert config.beam.beam_size == 8
    assert config.beam.ngram_lm_model == str(lm)
    assert config.beam.ngram_lm_alpha == 0.3
    assert config.beam.pruning_mode == "late"
    assert config.beam.blank_lm_score_mode == "lm_weighted_full"


def test_load_model_configures_standalone_tdt_batched_beam(monkeypatch, tmp_path):
    from omegaconf import OmegaConf

    model = FakeStandaloneEvalModel()
    model.cfg = OmegaConf.create(
        {
            "decoding": {
                "strategy": "greedy_batch",
                "greedy": {"ngram_lm_model": None, "ngram_lm_alpha": 0.0},
                "beam": {
                    "beam_size": 2,
                    "return_best_hypothesis": False,
                    "ngram_lm_model": None,
                    "ngram_lm_alpha": 0.0,
                    "pruning_mode": "LATE",
                    "blank_lm_score_mode": "LM_WEIGHTED_FULL",
                },
            }
        }
    )
    _install_fake_model_runtime(monkeypatch, model)
    final_model = tmp_path / "final.nemo"
    final_model.touch()
    args = argparse.Namespace(
        checkpoint=None,
        device="cpu",
        kind="tdt",
        ngram_lm=tmp_path / "lm.nemo",
        ngram_lm_alpha=0.3,
        beam_size=4,
        beam_beta=0.0,
        replace_tokenizer=False,
        tokenizer_dir=tmp_path / "tokenizer",
        tokenizer_type="bpe",
    )

    load_model(args, final_model)

    event = next(event for event in model.events if event[0] == "change_decoding_strategy")
    assert event[1].strategy == "malsd_batch"
    assert event[1].beam.beam_size == 4


def test_compute_error_rates_reports_wer_cer_and_empty_hypotheses():
    rates = compute_error_rates(["one two", "abc"], ["one", ""], str)

    assert rates["wer_percent"] == 66.66666666666666
    assert rates["cer_percent"] > 0
    assert rates["empty_hypotheses"] == 1
    assert rates["scored_rows"] == 2
    assert rates["excluded_empty_references"] == 0


def test_compute_error_rates_accounts_for_references_normalized_to_empty():
    def normalize(text: str) -> str:
        return "" if text == "[noise]" else text.lower()

    rates = compute_error_rates(["HELLO", "[noise]"], ["hello", "noise"], normalize)

    assert rates["wer_percent"] == 0.0
    assert rates["scored_rows"] == 1
    assert rates["excluded_empty_references"] == 1


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
