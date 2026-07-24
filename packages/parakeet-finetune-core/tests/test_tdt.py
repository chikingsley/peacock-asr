from __future__ import annotations

import json
import sys
import types
from types import SimpleNamespace
from typing import Any

import pytest

from parakeet_finetune_core.project import ParakeetProject
from parakeet_finetune_core.tdt import (
    JsonlTrainLogger,
    JsonlValLogger,
    _add_l2sp_gradients,
    _audio_seconds_from_batch,
    apply_loss_init_fix,
    build_parser,
    configure_aux_ctc_weight,
    configure_fused_tdt_loss,
    configure_spec_augmentation,
    enable_eval_loss,
    export_training_artifacts,
    freeze_encoder,
    load_and_prepare_model,
    parse_train_sources,
    shutdown_lhotse_training_sampler,
    spec_augment_config,
    train_ds,
    val_ds,
    validation_checkpoint_config,
    validation_schedule,
)


class FakeRNNTLoss:
    def __init__(self, *, num_classes, loss_name, loss_kwargs, reduction):
        self.num_classes = num_classes
        self.loss_name = loss_name
        self.loss_kwargs = loss_kwargs
        self.reduction = reduction


def test_audio_seconds_from_standard_nemo_batch() -> None:
    assert _audio_seconds_from_batch((object(), [16_000, 8_000])) == 1.5
    assert _audio_seconds_from_batch({"audio_signal_length": [32_000]}) == 2.0
    assert _audio_seconds_from_batch(None) == 0.0


def _install_fake_rnnt_loss(monkeypatch):
    modules = {
        name: types.ModuleType(name)
        for name in [
            "nemo",
            "nemo.collections",
            "nemo.collections.asr",
            "nemo.collections.asr.losses",
            "nemo.collections.asr.losses.rnnt",
        ]
    }
    modules["nemo.collections.asr.losses.rnnt"].__dict__["RNNTLoss"] = FakeRNNTLoss
    modules["nemo"].__dict__["collections"] = modules["nemo.collections"]
    modules["nemo.collections"].__dict__["asr"] = modules["nemo.collections.asr"]
    modules["nemo.collections.asr"].__dict__["losses"] = modules["nemo.collections.asr.losses"]
    modules["nemo.collections.asr.losses"].__dict__["rnnt"] = modules[
        "nemo.collections.asr.losses.rnnt"
    ]
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


class FakeJoint:
    def __init__(self, *, num_classes_with_blank=1030, num_extra_outputs=5):
        self.num_classes_with_blank = num_classes_with_blank
        self.num_extra_outputs = num_extra_outputs
        self.fused_batch_size = None
        self.fuse_loss_wer = None

    def set_fused_batch_size(self, value):
        self.fused_batch_size = value

    def set_fuse_loss_wer(self, **kwargs):
        self.fuse_loss_wer = kwargs


class FakeModel:
    def __init__(self, joint):
        self.joint = joint
        self.cfg = {"loss": {"tdt_kwargs": {"durations": [0, 1, 2, 3, 4]}}}
        self.compute_eval_loss = False
        self.loss: Any = object()
        self.wer = object()

    def extract_rnnt_loss_cfg(self, cfg):
        return "tdt", cfg


def test_apply_loss_init_fix_excludes_blank_and_tdt_duration_bins(monkeypatch):
    _install_fake_rnnt_loss(monkeypatch)
    model = FakeModel(FakeJoint(num_classes_with_blank=1030, num_extra_outputs=5))

    num_classes = apply_loss_init_fix(model)

    assert num_classes == 1024
    assert model.loss.num_classes == 1024
    assert model.loss.loss_name == "tdt"
    assert model.loss.loss_kwargs == {"tdt_kwargs": {"durations": [0, 1, 2, 3, 4]}}
    assert model.loss.reduction == "mean_batch"


def test_apply_loss_init_fix_handles_plain_rnnt_shape(monkeypatch):
    _install_fake_rnnt_loss(monkeypatch)
    model = FakeModel(FakeJoint(num_classes_with_blank=1025, num_extra_outputs=0))

    assert apply_loss_init_fix(model) == 1024
    assert model.loss.num_classes == 1024


def test_configure_fused_tdt_loss_passes_model_loss_and_wer():
    model = FakeModel(FakeJoint())

    configure_fused_tdt_loss(model, fused_batch_size=3)

    assert model.joint.fused_batch_size == 3
    assert model.joint.fuse_loss_wer == {
        "fuse_loss_wer": True,
        "loss": model.loss,
        "metric": model.wer,
    }


def test_enable_eval_loss_updates_runtime_flag_and_model_cfg():
    model = FakeModel(FakeJoint())

    enable_eval_loss(model)

    assert model.compute_eval_loss is True
    assert model.cfg["compute_eval_loss"] is True


def test_configure_aux_ctc_weight_updates_runtime_flag_and_model_cfg():
    model = FakeModel(FakeJoint())
    model.ctc_loss_weight = 0.3

    configured = configure_aux_ctc_weight(model, 0.1)

    assert configured == 0.1
    assert model.ctc_loss_weight == 0.1
    assert model.cfg["aux_ctc"]["ctc_loss_weight"] == 0.1


def test_spec_augment_profiles_have_controlled_mask_counts():
    assert spec_augment_config("off") is None
    assert spec_augment_config("half")["freq_masks"] == 1
    assert spec_augment_config("half")["time_masks"] == 5
    assert spec_augment_config("current")["freq_masks"] == 2
    assert spec_augment_config("current")["time_masks"] == 10


def test_configure_spec_augmentation_updates_forward_path_and_model_cfg():
    model = FakeModel(FakeJoint())

    class FakeModuleFactory:
        @staticmethod
        def from_config_dict(config):
            return ("configured", config)

    configure_spec_augmentation(model, "half", FakeModuleFactory)

    assert model.spec_augmentation[0] == "configured"
    assert model.spec_augmentation[1]["time_masks"] == 5
    assert model.cfg["spec_augment"]["freq_masks"] == 1

    configure_spec_augmentation(model, "off", FakeModuleFactory)

    assert model.spec_augmentation is None
    assert model.cfg["spec_augment"] is None


class FakeParameter:
    def __init__(self, size):
        self.size = size
        self.requires_grad = False

    def requires_grad_(self, *, requires_grad):
        self.requires_grad = requires_grad
        return self

    def numel(self):
        return self.size


class FakeLayer:
    def __init__(self, *parameters):
        self._parameters = list(parameters)
        self.train_called = False

    def parameters(self):
        return iter(self._parameters)

    def train(self):
        self.train_called = True


class FakeEncoder:
    def __init__(self, layers):
        self.layers = layers
        self.freeze_called = False

    def freeze(self):
        self.freeze_called = True


class FakeEncoderModel:
    def __init__(self, layers):
        self.encoder = FakeEncoder(layers)

    def parameters(self):
        for layer in self.encoder.layers:
            yield from layer.parameters()


def test_freeze_encoder_can_unfreeze_only_top_layers():
    low = FakeLayer(FakeParameter(1_000_000))
    mid = FakeLayer(FakeParameter(1_000_000))
    high_param = FakeParameter(2_000_000)
    high = FakeLayer(high_param)
    model = FakeEncoderModel([low, mid, high])

    message = freeze_encoder(model, unfreeze_top=1)

    assert message == "encoder top 1/3 unfrozen (2M trainable)"
    assert model.encoder.freeze_called is True
    assert low.train_called is False
    assert mid.train_called is False
    assert high.train_called is True
    assert high_param.requires_grad is True


def test_tdt_dataset_configs_use_lhotse_train_and_plain_validation(tmp_path):
    train_config = train_ds(
        tmp_path / "train.jsonl",
        max_dur=30.0,
        batch_dur=120.0,
        num_workers=8,
        seed=17,
    )
    validation_config = val_ds(tmp_path / "dev.jsonl", max_dur=25.0, num_workers=2)

    assert train_config["use_lhotse"] is True
    assert train_config["concurrent_bucketing"] is True
    assert train_config["batch_duration"] == 120.0
    assert train_config["batch_size"] is None
    assert train_config["num_workers"] == 8
    assert train_config["seed"] == 17
    assert train_config["shard_seed"] == 17
    assert validation_config["batch_size"] == 2
    assert validation_config["num_workers"] == 2
    assert validation_config["max_duration"] == 25.0


def test_shutdown_lhotse_training_sampler_stops_producer():
    sampler = SimpleNamespace(_source_exhausted=False, _producer_thread=None)

    class Producer:
        alive = True
        timeout = None

        def is_alive(self):
            return self.alive

        def join(self, timeout):
            assert sampler._source_exhausted is True  # noqa: SLF001
            self.timeout = timeout
            self.alive = False

    producer = Producer()
    sampler._producer_thread = producer  # noqa: SLF001
    # Reproduce Lightning's wrapped active-loader graph. The stale model loader is a decoy;
    # the live Lhotse producer belongs to the trainer's nested DataLoader.
    model = SimpleNamespace(_train_dl=SimpleNamespace(sampler=SimpleNamespace()))
    trainer = SimpleNamespace(
        train_dataloader=SimpleNamespace(
            loaders={
                "train": SimpleNamespace(
                    dataset=SimpleNamespace(sampler=sampler),
                    sampler=SimpleNamespace(),
                )
            }
        )
    )

    assert shutdown_lhotse_training_sampler(model, trainer=trainer, join_timeout=2.0) is True
    assert producer.timeout == 2.0
    assert sampler._producer_thread is producer  # noqa: SLF001


def test_shutdown_lhotse_training_sampler_supports_direct_model_loader():
    producer = SimpleNamespace(is_alive=lambda: False)
    sampler = SimpleNamespace(_producer_thread=producer)
    model = SimpleNamespace(_train_dl=SimpleNamespace(sampler=sampler))

    assert shutdown_lhotse_training_sampler(model) is False


def test_shutdown_lhotse_training_sampler_recovers_detached_thread_owner(monkeypatch):
    sampler = SimpleNamespace(_source_exhausted=False, _producer_thread=None)

    class Producer:
        alive = True

        def is_alive(self):
            return self.alive

        def join(self, timeout):
            assert sampler._source_exhausted is True  # noqa: SLF001
            assert timeout == 5.0
            self.alive = False

    producer = Producer()

    def target():
        return sampler

    producer._target = target  # noqa: SLF001 - mirrors threading.Thread's closure target
    sampler._producer_thread = producer  # noqa: SLF001
    monkeypatch.setattr("parakeet_finetune_core.tdt.threading.enumerate", lambda: [producer])

    assert shutdown_lhotse_training_sampler(SimpleNamespace()) is True
    assert sampler._producer_thread is producer  # noqa: SLF001


def test_shutdown_lhotse_training_sampler_is_noop_without_producer():
    assert shutdown_lhotse_training_sampler(SimpleNamespace()) is False


def test_tdt_dataset_config_supports_explicit_source_weights(tmp_path):
    train_config = train_ds(
        [(tmp_path / "one.jsonl", 0.5), (tmp_path / "two.jsonl", 0.5)],
        max_dur=30.0,
        batch_dur=120.0,
        num_workers=8,
        seed=17,
    )

    assert train_config["input_cfg"] == [
        {"type": "nemo", "manifest_filepath": str(tmp_path / "one.jsonl"), "weight": 0.5},
        {"type": "nemo", "manifest_filepath": str(tmp_path / "two.jsonl"), "weight": 0.5},
    ]


def test_parse_train_sources_rejects_bad_weights_and_duplicates(tmp_path):
    one = tmp_path / "one.jsonl"

    assert parse_train_sources([f"{one}=0.25"]) == [(one, 0.25)]
    with pytest.raises(ValueError, match="expected PATH=WEIGHT"):
        parse_train_sources([str(one)])
    with pytest.raises(ValueError, match="must be positive"):
        parse_train_sources([f"{one}=0"])
    with pytest.raises(ValueError, match="duplicate"):
        parse_train_sources([f"{one}=0.5", f"{one}=0.5"])


def test_l2sp_callback_adds_anchor_gradient() -> None:
    class FakeGradient:
        def __init__(self) -> None:
            self.value = 0.0

        def add_(self, difference, *, alpha) -> None:
            self.value += difference * alpha

    class FakeParameter:
        def __init__(self, value) -> None:
            self.value = value
            self.grad = FakeGradient()

        def detach(self):
            return self.value

    parameter = FakeParameter(2.0)
    anchors = SimpleNamespace(anchor_0=0.0)

    _add_l2sp_gradients([("weight", parameter, "anchor_0")], anchors, weight=0.25)

    assert parameter.grad.value == 0.5


def test_tdt_parser_exposes_gradient_accumulation(tmp_path):
    project = ParakeetProject(name="farsi", language="fas_Arab", root=tmp_path)

    args = build_parser(project).parse_args(["--accumulate-grad-batches", "2"])

    assert args.accumulate_grad_batches == 2


def test_tdt_parser_allows_embedded_tokenizer(tmp_path):
    project = ParakeetProject(name="english", language="eng_Latn", root=tmp_path)

    args = build_parser(project).parse_args([])

    assert args.tokenizer_dir is None


def test_load_and_prepare_model_preserves_embedded_tokenizer(monkeypatch, tmp_path):
    class PreparedModel(FakeModel):
        def __init__(self):
            super().__init__(FakeJoint())
            self.changed_vocabulary = False

        def change_vocabulary(self, **_kwargs):
            self.changed_vocabulary = True

    model = PreparedModel()

    class FakeASRModel:
        @staticmethod
        def restore_from(_path, *, map_location):
            assert map_location == "cpu"
            return model

    model_path = tmp_path / "base.nemo"
    model_path.touch()
    args = build_parser(
        ParakeetProject(name="english", language="eng_Latn", root=tmp_path)
    ).parse_args([])
    monkeypatch.setattr(
        "parakeet_finetune_core.tdt.apply_loss_init_fix", lambda *_args, **_kwargs: 1024
    )
    monkeypatch.setattr("parakeet_finetune_core.tdt.configure_fused_tdt_loss", lambda *_args: None)

    prepared = load_and_prepare_model(FakeASRModel, str(model_path), None, args)

    assert prepared is model
    assert model.changed_vocabulary is False
    assert model.compute_eval_loss is True


def test_extend_restore_rejects_missing_tokenizer(monkeypatch, tmp_path):
    class FakeASRModel:
        @staticmethod
        def restore_from(_path, *, map_location):
            assert map_location == "cpu"
            return FakeModel(FakeJoint())

    model_path = tmp_path / "base.nemo"
    model_path.touch()
    args = build_parser(
        ParakeetProject(name="english", language="eng_Latn", root=tmp_path)
    ).parse_args(["--recipe", "extend-restore"])

    with pytest.raises(ValueError, match="requires --tokenizer-dir"):
        load_and_prepare_model(FakeASRModel, str(model_path), None, args)


def test_validation_checkpoint_monitors_eval_loss(tmp_path):
    config = validation_checkpoint_config(tmp_path / "checkpoints")

    assert config["monitor"] == "val_loss"
    assert config["mode"] == "min"
    assert config["filename"] == "best-valloss-step{step}-{val_loss:.3f}"


def test_validation_schedule_counts_globally_across_iterable_epochs():
    assert validation_schedule(100, 2) == {
        "val_check_interval": 200,
        "check_val_every_n_epoch": None,
    }


class FakeArtifactModel:
    def __init__(self):
        self.events = []

    def save_to(self, path):
        self.events.append(("save_to", path))

    def load_state_dict(self, state_dict, *, strict):
        self.events.append(("load_state_dict", state_dict, strict))


def test_export_training_artifacts_saves_last_and_best_validation_models(tmp_path):
    model = FakeArtifactModel()
    checkpoint = tmp_path / "checkpoints" / "best.ckpt"
    checkpoint.parent.mkdir()
    checkpoint.touch()

    final_path, best_path = export_training_artifacts(
        model,
        tmp_path,
        "tajik-tdt",
        checkpoint,
        checkpoint_loader=lambda _path: {"state_dict": {"weight": "best"}},
    )

    assert final_path == tmp_path / "tajik-tdt_final.nemo"
    assert best_path == tmp_path / "tajik-tdt_best-valloss.nemo"
    assert model.events == [
        ("save_to", str(final_path)),
        ("load_state_dict", {"weight": "best"}, True),
        ("save_to", str(best_path)),
    ]


def test_export_training_artifacts_allows_no_validation_checkpoint(tmp_path):
    model = FakeArtifactModel()

    final_path, best_path = export_training_artifacts(model, tmp_path, "tajik-tdt", None)

    assert final_path == tmp_path / "tajik-tdt_final.nemo"
    assert best_path is None
    assert model.events == [("save_to", str(final_path))]


def test_jsonl_train_logger_writes_rounded_loss_and_lr(tmp_path):
    log_path = tmp_path / "train.jsonl"
    logger = JsonlTrainLogger(log_path, every=5)
    logger.t0 = 0
    trainer = SimpleNamespace(
        global_step=5,
        callback_metrics={"train_loss": 12.34567},
        optimizers=[SimpleNamespace(param_groups=[{"lr": 0.000123456}])],
    )

    logger.on_train_batch_end(trainer, None)
    logger.on_train_batch_end(trainer, None)

    records = log_path.read_text(encoding="utf-8").splitlines()
    assert len(records) == 1
    record = json.loads(records[0])
    assert record["step"] == 5
    assert record["loss"] == 12.3457
    assert record["lr"] == 0.00012346


def test_jsonl_val_logger_uses_null_for_missing_loss_and_records_wer(tmp_path):
    log_path = tmp_path / "val.jsonl"
    logger = JsonlValLogger(log_path)
    trainer = SimpleNamespace(
        global_step=7,
        callback_metrics={"val_wer": 1.23456, "val_wer_ctc": 2.34567},
    )

    logger.on_validation_end(trainer, None)

    record = json.loads(log_path.read_text(encoding="utf-8"))
    assert record == {
        "step": 7,
        "val_loss": None,
        "val_wer_bf16": 1.2346,
        "val_wer_ctc_bf16": 2.3457,
    }
