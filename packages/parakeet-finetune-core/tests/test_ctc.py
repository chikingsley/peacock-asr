from __future__ import annotations

from argparse import Namespace

from parakeet_finetune_core.ctc import (
    add_early_stopping,
    build_config,
    disable_checkpoint_saves,
    parse_val_check_interval,
)


def _args(tmp_path, **overrides):
    defaults = {
        "train_manifest": tmp_path / "train.jsonl",
        "validation_manifest": tmp_path / "dev.jsonl",
        "tokenizer_dir": tmp_path / "tok",
        "tokenizer_type": "bpe",
        "batch_size": 4,
        "validation_batch_size": None,
        "num_workers": 2,
        "min_duration": 0.1,
        "max_duration": 25.0,
        "use_lhotse": False,
        "batch_duration": 700.0,
        "quadratic_duration": 15.0,
        "num_buckets": 30,
        "learning_rate": 1e-4,
        "warmup_steps": 500,
        "min_lr": 5e-6,
        "devices": "1",
        "accelerator": "gpu",
        "min_epochs": 0,
        "max_epochs": 3,
        "min_steps": None,
        "max_steps": 100,
        "val_check_interval": "25",
        "precision": "bf16",
        "accumulate_grad_batches": 2,
        "log_every_n_steps": 10,
        "disable_progress_bar": True,
        "exp_dir": tmp_path / "runs",
        "name": "ctc-test",
        "early_stopping": False,
        "early_stopping_patience": 7,
        "early_stopping_min_delta": 0.02,
        "no_save_checkpoints": False,
    }
    defaults.update(overrides)
    return Namespace(**defaults)


def test_val_check_interval_parsing_keeps_integer_steps():
    assert parse_val_check_interval("25") == 25
    assert parse_val_check_interval("1.0") == 1.0
    assert parse_val_check_interval("0.5") == 0.5


def test_build_config_uses_standard_manifest_batching(tmp_path):
    args = _args(tmp_path, validation_batch_size=2)

    cfg = build_config(args)

    assert cfg.model.train_ds.manifest_filepath == str((tmp_path / "train.jsonl").resolve())
    assert cfg.model.train_ds.batch_size == 4
    assert cfg.model.validation_ds.batch_size == 2
    assert cfg.model.validation_ds.shuffle is False
    assert cfg.model.tokenizer.dir == str((tmp_path / "tok").resolve())
    assert cfg.trainer.accelerator == "gpu"
    assert cfg.trainer.precision == "bf16"
    assert cfg.trainer.val_check_interval == 25
    assert cfg.exp_manager.checkpoint_callback_params.monitor == "val_wer"


def test_build_config_uses_lhotse_duration_batching(tmp_path):
    args = _args(tmp_path, use_lhotse=True, batch_duration=123.0)

    cfg = build_config(args)

    assert cfg.model.train_ds.use_lhotse is True
    assert cfg.model.train_ds.use_bucketing is True
    assert cfg.model.train_ds.batch_size is None
    assert cfg.model.train_ds.batch_duration == 123.0
    assert cfg.model.train_ds.num_buckets == 30


def test_early_stopping_and_checkpoint_flags_mutate_config(tmp_path):
    args = _args(tmp_path, early_stopping=True, no_save_checkpoints=True)
    cfg = build_config(args)

    add_early_stopping(cfg, args)
    disable_checkpoint_saves(cfg, args)

    assert cfg.exp_manager.create_checkpoint_callback is False
    assert cfg.exp_manager.create_early_stopping_callback is True
    assert cfg.exp_manager.early_stopping_callback_params.monitor == "val_wer"
    assert cfg.exp_manager.early_stopping_callback_params.patience == 7
