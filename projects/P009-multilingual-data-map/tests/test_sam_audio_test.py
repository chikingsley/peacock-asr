import importlib.util
import sys
import types
from pathlib import Path

import torch
import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "code" / "sam_audio_test.py"
)


def load_module():
    if "torchaudio" not in sys.modules:
        sys.modules["torchaudio"] = types.SimpleNamespace(
            functional=types.SimpleNamespace(
                resample=lambda waveform, src_sr, dst_sr: waveform
            ),
            load=lambda path: (_ for _ in ()).throw(NotImplementedError(path)),
            save=lambda path, waveform, sample_rate: None,
        )
    spec = importlib.util.spec_from_file_location("sam_audio_test_module", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_load_judge_uses_official_judge_classes(monkeypatch):
    module = load_module()
    monkeypatch.setattr(module, "_patch_sam_audio_compat", lambda: None)

    class FakeJudgeModel:
        calls = []

        @classmethod
        def from_pretrained(cls, model_id):
            cls.calls.append(model_id)
            return cls()

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            self.evaluated = True
            return self

    class FakeJudgeProcessor:
        calls = []

        @classmethod
        def from_pretrained(cls, model_id):
            cls.calls.append(model_id)
            return cls()

    class FakeSeparator:
        @classmethod
        def from_pretrained(cls, model_id):
            raise AssertionError("separator loader should not be used for judge")

    class FakeSeparatorProcessor:
        @classmethod
        def from_pretrained(cls, model_id):
            raise AssertionError("separator processor should not be used for judge")

    fake_sam_audio = types.SimpleNamespace(
        SAMAudio=FakeSeparator,
        SAMAudioProcessor=FakeSeparatorProcessor,
        SAMAudioJudgeModel=FakeJudgeModel,
        SAMAudioJudgeProcessor=FakeJudgeProcessor,
    )
    monkeypatch.setitem(sys.modules, "sam_audio", fake_sam_audio)

    judge, proc = module.load_judge(torch.device("cpu"))

    assert isinstance(judge, FakeJudgeModel)
    assert isinstance(proc, FakeJudgeProcessor)
    assert FakeJudgeModel.calls == ["facebook/sam-audio-judge"]
    assert FakeJudgeProcessor.calls == ["facebook/sam-audio-judge"]


def test_load_separator_disables_optional_multimodal_components(monkeypatch):
    module = load_module()
    monkeypatch.setattr(module, "_patch_sam_audio_compat", lambda: None)
    monkeypatch.setattr(module, "_patch_no_imagebind", lambda: None)

    class FakeSeparatorModel:
        calls = []

        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            cls.calls.append((model_id, kwargs))
            return cls()

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            self.evaluated = True
            return self

    class FakeProcessor:
        calls = []
        audio_sampling_rate = 48_000

        @classmethod
        def from_pretrained(cls, model_id):
            cls.calls.append(model_id)
            return cls()

    fake_sam_audio = types.SimpleNamespace(
        SAMAudio=FakeSeparatorModel,
        SAMAudioProcessor=FakeProcessor,
    )
    monkeypatch.setitem(sys.modules, "sam_audio", fake_sam_audio)

    model, processor = module.load_separator("facebook/sam-audio-small", torch.device("cpu"))

    assert isinstance(model, FakeSeparatorModel)
    assert isinstance(processor, FakeProcessor)
    assert FakeSeparatorModel.calls == [
        (
            "facebook/sam-audio-small",
            {
                "span_predictor": None,
                "text_ranker": None,
                "visual_ranker": None,
            },
        )
    ]
    assert FakeProcessor.calls == ["facebook/sam-audio-small"]


def test_run_judge_uses_official_processor_fields(tmp_path):
    module = load_module()

    original = tmp_path / "original.wav"
    target = tmp_path / "target.wav"
    original.write_bytes(b"")
    target.write_bytes(b"")

    class FakeBatch:
        def __init__(self, payload):
            self.device = None
            self.payload = payload

        def to(self, device):
            self.device = device
            return self.payload

    class FakeProcessor:
        def __init__(self):
            self.calls = []

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            return FakeBatch(kwargs)

    class FakeJudge:
        def __call__(self, **kwargs):
            return types.SimpleNamespace(overall=torch.tensor([[0.75]]))

    processor = FakeProcessor()
    score = module.run_judge(
        FakeJudge(),
        processor,
        torch.device("cpu"),
        "person speaking",
        original,
        target,
    )

    assert score == pytest.approx(0.75)
    assert processor.calls == [
        {
            "text": ["person speaking"],
            "input_audio": [str(original)],
            "separated_audio": [str(target)],
        }
    ]


def test_process_file_saves_outputs_at_model_sample_rate(monkeypatch, tmp_path):
    module = load_module()

    monkeypatch.setattr(
        module.torchaudio,
        "load",
        lambda _: (torch.ones(2, 24_000), 24_000),
    )
    monkeypatch.setattr(
        module.torchaudio.functional,
        "resample",
        lambda waveform, src_sr, dst_sr: torch.ones(
            waveform.shape[0],
            int(waveform.shape[1] * dst_sr / src_sr),
        ),
    )

    saved = []

    def fake_save(path, waveform, sample_rate):
        saved.append((Path(path).name, tuple(waveform.shape), sample_rate))

    monkeypatch.setattr(module.torchaudio, "save", fake_save)
    monkeypatch.setattr(
        module,
        "separate_chunked",
        lambda *args, **kwargs: torch.ones(1, 48_000),
    )
    monkeypatch.setattr(module, "run_judge", lambda *args, **kwargs: None)

    result = module.process_file(
        audio_path=tmp_path / "clip.wav",
        separator=object(),
        sep_proc=object(),
        judge=None,
        judge_proc=None,
        device=torch.device("cpu"),
        description="person speaking",
        out_dir=tmp_path,
        sample_rate=48_000,
        chunk_s=25.0,
        overlap_s=2.0,
    )

    assert result["dur"] == 1.0
    assert saved == [
        ("clip_original.wav", (1, 48_000), 48_000),
        ("clip_target.wav", (1, 48_000), 48_000),
    ]


def test_run_single_file_uses_cached_runtime_and_returns_output_paths(monkeypatch, tmp_path):
    module = load_module()

    audio_path = tmp_path / "clip.wav"
    audio_path.write_bytes(b"")

    runtime_calls = []

    def fake_get_runtime(model_id, with_judge):
        runtime_calls.append((model_id, with_judge))
        return (
            object(),
            types.SimpleNamespace(audio_sampling_rate=48_000),
            None,
            None,
            torch.device("cpu"),
            48_000,
        )

    process_calls = []

    def fake_process_file(**kwargs):
        process_calls.append(kwargs)
        return {"file": "clip.wav", "dur": 1.0, "score": None}

    monkeypatch.setattr(module, "_get_runtime", fake_get_runtime)
    monkeypatch.setattr(module, "process_file", fake_process_file)

    result = module.run_single_file(
        audio_path=audio_path,
        description="a person speaking",
        out_dir=tmp_path,
        model_id="facebook/sam-audio-small",
        with_judge=False,
    )

    assert runtime_calls == [("facebook/sam-audio-small", False)]
    assert process_calls[0]["audio_path"] == audio_path
    assert process_calls[0]["description"] == "a person speaking"
    assert process_calls[0]["sample_rate"] == 48_000
    assert result == {
        "file": "clip.wav",
        "dur": 1.0,
        "score": None,
        "original_path": tmp_path / "clip_original.wav",
        "target_path": tmp_path / "clip_target.wav",
    }
