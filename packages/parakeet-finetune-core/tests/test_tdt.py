from __future__ import annotations

import json
import sys
import types
from types import SimpleNamespace
from typing import Any

from parakeet_finetune_core.tdt import (
    JsonlTrainLogger,
    JsonlValLogger,
    apply_loss_init_fix,
    configure_fused_tdt_loss,
    enable_eval_loss,
    export_training_artifacts,
    freeze_encoder,
    train_ds,
    val_ds,
    validation_checkpoint_config,
)


class FakeRNNTLoss:
    def __init__(self, *, num_classes, loss_name, loss_kwargs, reduction):
        self.num_classes = num_classes
        self.loss_name = loss_name
        self.loss_kwargs = loss_kwargs
        self.reduction = reduction


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
    train_config = train_ds(tmp_path / "train.jsonl", max_dur=30.0, batch_dur=120.0, num_workers=8)
    validation_config = val_ds(tmp_path / "dev.jsonl", max_dur=25.0, num_workers=2)

    assert train_config["use_lhotse"] is True
    assert train_config["batch_duration"] == 120.0
    assert train_config["batch_size"] is None
    assert train_config["num_workers"] == 8
    assert validation_config["batch_size"] == 2
    assert validation_config["num_workers"] == 2
    assert validation_config["max_duration"] == 25.0


def test_validation_checkpoint_monitors_eval_loss(tmp_path):
    config = validation_checkpoint_config(tmp_path / "checkpoints")

    assert config["monitor"] == "val_loss"
    assert config["mode"] == "min"
    assert config["filename"] == "best-valloss-step{step}-{val_loss:.3f}"


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

    record = json.loads(log_path.read_text(encoding="utf-8"))
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
