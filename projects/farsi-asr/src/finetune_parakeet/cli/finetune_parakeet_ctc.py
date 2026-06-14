from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import lightning.pytorch as pl
import torch
from nemo.collections.asr.models import ASRModel
from nemo.utils import model_utils
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg
from omegaconf import OmegaConf

from finetune_parakeet.paths import (
    DEFAULT_CTC_MODEL,
    DEFAULT_RUNS_ROOT,
    configure_external_caches,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune the converted Parakeet 110M CTC model.")
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--init-from-nemo-model", type=Path, default=DEFAULT_CTC_MODEL)
    parser.add_argument("--tokenizer-type", default="bpe", choices=["bpe", "wpe", "agg"])
    parser.add_argument("--exp-dir", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--name", default="parakeet-110m-ctc-persian")
    parser.add_argument("--devices", default="1")
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--min-epochs", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--min-steps", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--validation-batch-size", type=int, default=None)
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=5000)
    parser.add_argument("--min-lr", type=float, default=5e-6)
    parser.add_argument("--val-check-interval", default="1.0")
    parser.add_argument("--log-every-n-steps", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--min-duration", type=float, default=0.1)
    parser.add_argument("--max-duration", type=float, default=20.0)
    parser.add_argument("--use-lhotse", action="store_true")
    parser.add_argument("--batch-duration", type=float, default=600.0)
    parser.add_argument("--quadratic-duration", type=float, default=15.0)
    parser.add_argument("--num-buckets", type=int, default=30)
    parser.add_argument("--early-stopping", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.001)
    parser.add_argument("--no-save-checkpoints", action="store_true")
    parser.add_argument("--disable-progress-bar", action="store_true")
    parser.add_argument("--disable-cudnn", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def parse_val_check_interval(raw_value: str) -> float | int:
    value = float(raw_value)
    if "." not in raw_value and value.is_integer() and value >= 1:
        return int(value)
    return value


def dataset_config(
    args: argparse.Namespace,
    manifest: Path,
    batch_size: int | None,
    *,
    shuffle: bool,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "manifest_filepath": str(manifest.resolve()),
        "sample_rate": 16_000,
        "shuffle": shuffle,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "min_duration": args.min_duration,
        "max_duration": args.max_duration,
        "is_tarred": False,
        "tarred_audio_filepaths": None,
        "shuffle_n": 2048,
        "bucketing_strategy": "fully_randomized",
        "bucketing_batch_size": None,
    }
    if args.use_lhotse and batch_size is not None:
        config.update(
            {
                "use_lhotse": True,
                "use_bucketing": True,
                "batch_size": None,
                "batch_duration": args.batch_duration,
                "quadratic_duration": args.quadratic_duration,
                "num_buckets": args.num_buckets,
                "bucket_duration_bins": None,
                "bucket_buffer_size": 20_000,
                "shuffle_buffer_size": 10_000,
            }
        )
    else:
        config["batch_size"] = batch_size
    return config


def build_config(args: argparse.Namespace) -> Any:
    validation_batch_size = args.validation_batch_size or args.batch_size
    return OmegaConf.create(
        {
            "model": {
                "train_ds": dataset_config(
                    args,
                    args.train_manifest,
                    args.batch_size,
                    shuffle=True,
                ),
                "validation_ds": dataset_config(
                    args,
                    args.validation_manifest,
                    validation_batch_size,
                    shuffle=False,
                ),
                "test_ds": dataset_config(
                    args,
                    args.validation_manifest,
                    validation_batch_size,
                    shuffle=False,
                ),
                "tokenizer": {
                    "dir": str(args.tokenizer_dir.resolve()),
                    "type": args.tokenizer_type,
                },
                "spec_augment": {
                    "_target_": "nemo.collections.asr.modules.SpectrogramAugmentation",
                    "freq_masks": 2,
                    "time_masks": 10,
                    "freq_width": 27,
                    "time_width": 0.05,
                },
                "optim": {
                    "name": "adamw",
                    "lr": args.learning_rate,
                    "betas": [0.9, 0.98],
                    "weight_decay": 1e-3,
                    "sched": {
                        "name": "CosineAnnealing",
                        "warmup_steps": args.warmup_steps,
                        "warmup_ratio": None,
                        "min_lr": args.min_lr,
                    },
                },
            },
            "trainer": {
                "devices": args.devices,
                "accelerator": args.accelerator,
                "min_epochs": args.min_epochs,
                "max_epochs": args.max_epochs,
                "min_steps": args.min_steps,
                "max_steps": args.max_steps,
                "val_check_interval": parse_val_check_interval(args.val_check_interval),
                "precision": args.precision,
                "accumulate_grad_batches": args.accumulate_grad_batches,
                "gradient_clip_val": 0.0,
                "log_every_n_steps": args.log_every_n_steps,
                "num_sanity_val_steps": 0,
                "enable_checkpointing": False,
                "enable_progress_bar": not args.disable_progress_bar,
                "logger": False,
                "benchmark": False,
            },
            "exp_manager": {
                "exp_dir": str(args.exp_dir.resolve()),
                "name": args.name,
                "create_tensorboard_logger": True,
                "create_checkpoint_callback": True,
                "checkpoint_callback_params": {
                    "monitor": "val_wer",
                    "mode": "min",
                    "save_top_k": 5,
                    "always_save_nemo": True,
                },
                "resume_if_exists": False,
                "resume_ignore_no_checkpoint": False,
                "create_wandb_logger": False,
            },
        }
    )


def add_early_stopping(cfg: Any, args: argparse.Namespace) -> None:
    if not args.early_stopping:
        return
    cfg.exp_manager.create_early_stopping_callback = True
    cfg.exp_manager.early_stopping_callback_params = {
        "monitor": "val_wer",
        "mode": "min",
        "patience": args.early_stopping_patience,
        "min_delta": args.early_stopping_min_delta,
        "check_on_train_epoch_end": False,
    }


def disable_checkpoint_saves(cfg: Any, args: argparse.Namespace) -> None:
    if args.no_save_checkpoints:
        cfg.exp_manager.create_checkpoint_callback = False


def load_ctc_model(args: argparse.Namespace) -> ASRModel:
    model_path = args.init_from_nemo_model.resolve()
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    model = ASRModel.restore_from(str(model_path))
    if not hasattr(model, "change_vocabulary"):
        raise TypeError(f"{type(model).__name__} does not support tokenizer replacement")
    model.change_vocabulary(
        new_tokenizer_dir=str(args.tokenizer_dir.resolve()),
        new_tokenizer_type=args.tokenizer_type,
    )
    return model


def run(args: argparse.Namespace) -> None:
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
    cfg = build_config(args)
    add_early_stopping(cfg, args)
    disable_checkpoint_saves(cfg, args)
    print(OmegaConf.to_yaml(cfg))
    if args.dry_run:
        return

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    trainer_kwargs = cast("Mapping[str, Any]", resolve_trainer_cfg(cfg.trainer))
    trainer = pl.Trainer(**trainer_kwargs)
    exp_manager(trainer, cfg.exp_manager)
    model = load_ctc_model(args)
    model.set_trainer(trainer)
    resolved_model_cfg = model_utils.convert_model_config_to_dict_config(cfg)
    model.setup_training_data(resolved_model_cfg.model.train_ds)
    model.setup_multiple_validation_data(resolved_model_cfg.model.validation_ds)
    model.setup_multiple_test_data(resolved_model_cfg.model.test_ds)
    model.setup_optimization(resolved_model_cfg.model.optim)
    model.spec_augment = ASRModel.from_config_dict(resolved_model_cfg.model.spec_augment)
    trainer.fit(model)


def main(argv: list[str] | None = None) -> int:
    configure_external_caches()
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
