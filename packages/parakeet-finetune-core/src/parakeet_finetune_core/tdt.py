"""Shared helpers and CLI for Parakeet TDT fine-tuning."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from parakeet_finetune_core.project import ParakeetProject


class JsonlTrainLogger:
    """Factory for a Lightning callback that appends train loss records to JSONL."""

    def __init__(self, path: Path, every: int) -> None:
        self.path = path
        self.every = every
        self.t0 = time.time()

    def on_train_batch_end(self, trainer: Any, _pl_module: Any, *_args: object) -> None:
        step = trainer.global_step
        if step % self.every != 0:
            return
        metrics = trainer.callback_metrics
        loss = metrics.get("train_loss", metrics.get("loss"))
        loss_value = float(loss) if loss is not None else float("nan")
        lr = trainer.optimizers[0].param_groups[0]["lr"] if trainer.optimizers else float("nan")
        record = {
            "step": step,
            "loss": round(loss_value, 4),
            "lr": round(lr, 8),
            "t": round(time.time() - self.t0, 1),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")


class JsonlValLogger:
    """Factory for a Lightning callback that logs held-out TDT/CTC metrics."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def on_validation_end(self, trainer: Any, _pl_module: Any) -> None:
        metrics = trainer.callback_metrics

        def get_metric(key: str) -> float | None:
            value = metrics.get(key)
            return float(value) if value is not None else None

        def round_metric(value: float | None) -> float | None:
            return round(value, 4) if value is not None else None

        record = {
            "step": trainer.global_step,
            "val_loss": round_metric(get_metric("val_loss")),
            "val_wer_bf16": round_metric(get_metric("val_wer")),
            "val_wer_ctc_bf16": round_metric(get_metric("val_wer_ctc")),
        }
        print(
            f"  [val @ step {record['step']}] val_loss={record['val_loss']} "
            f"val_wer(bf16,noisy)={record['val_wer_bf16']}",
            flush=True,
        )
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")


def apply_loss_init_fix(model: Any) -> int:
    """Rebuild TDT RNNTLoss with duration-bin outputs excluded from ``num_classes``."""
    from nemo.collections.asr.losses.rnnt import RNNTLoss  # ty: ignore[unresolved-import]

    num_classes = model.joint.num_classes_with_blank - 1
    if model.joint.num_extra_outputs > 0:
        num_classes -= model.joint.num_extra_outputs
    loss_name, loss_kwargs = model.extract_rnnt_loss_cfg(model.cfg.get("loss", None))
    model.loss = RNNTLoss(
        num_classes=num_classes,
        loss_name=loss_name,
        loss_kwargs=loss_kwargs,
    )
    return int(num_classes)


def configure_fused_tdt_loss(model: Any, fused_batch_size: int) -> None:
    model.joint.set_fused_batch_size(fused_batch_size)
    model.joint.set_fuse_loss_wer(fuse_loss_wer=True, loss=model.loss, metric=model.wer)


def enable_eval_loss(model: Any) -> None:
    model.compute_eval_loss = True
    cfg = getattr(model, "cfg", None)
    if cfg is None:
        return
    if isinstance(cfg, dict):
        cfg["compute_eval_loss"] = True
        return
    try:
        cfg.compute_eval_loss = True
    except (AttributeError, TypeError, ValueError):
        from omegaconf import open_dict

        with open_dict(cfg):
            cfg.compute_eval_loss = True


def freeze_encoder(model: Any, *, unfreeze_top: int = 0) -> str:
    model.encoder.freeze()
    if unfreeze_top <= 0:
        return "encoder frozen"
    for layer in model.encoder.layers[-unfreeze_top:]:
        for parameter in layer.parameters():
            parameter.requires_grad_(requires_grad=True)
        layer.train()
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return (
        f"encoder top {unfreeze_top}/{len(model.encoder.layers)} unfrozen "
        f"({trainable / 1e6:.0f}M trainable)"
    )


def train_ds(manifest: Path, max_dur: float, batch_dur: float, num_workers: int) -> dict[str, Any]:
    return {
        "manifest_filepath": str(manifest),
        "sample_rate": 16_000,
        "use_lhotse": True,
        "use_bucketing": True,
        "num_buckets": 30,
        "batch_duration": batch_dur,
        "quadratic_duration": 15.0,
        "batch_size": None,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": True,
        "min_duration": 0.5,
        "max_duration": max_dur,
    }


def val_ds(manifest: Path, max_dur: float, num_workers: int) -> dict[str, Any]:
    return {
        "manifest_filepath": str(manifest),
        "sample_rate": 16_000,
        "batch_size": 2,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": True,
        "min_duration": 0.5,
        "max_duration": max_dur,
    }


def validation_checkpoint_config(ckpt_dir: Path) -> dict[str, Any]:
    return {
        "dirpath": str(ckpt_dir),
        "save_top_k": 2,
        "monitor": "val_loss",
        "mode": "min",
        "filename": "best-valloss-step{step}-{val_loss:.3f}",
        "auto_insert_metric_name": False,
    }


def build_parser(project: ParakeetProject) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Fine-tune a Parakeet TDT or TDT+CTC model for {project.name}."
    )
    parser.add_argument(
        "--name",
        default=project.default_tdt_run_name or f"{project.name}-parakeet-tdt",
    )
    parser.add_argument(
        "--model-name",
        default=str(project.default_tdt_model or project.default_hybrid_model or ""),
    )
    parser.add_argument("--train-manifest", type=Path, default=project.default_train_manifest)
    parser.add_argument(
        "--validation-manifest",
        type=Path,
        default=project.default_validation_manifest,
    )
    parser.add_argument("--tokenizer-dir", type=Path, default=project.default_tokenizer_dir)
    parser.add_argument("--tokenizer-type", default="bpe")
    parser.add_argument("--exp-dir", type=Path, default=project.runs)
    parser.add_argument("--batch-dur", type=float, default=120.0)
    parser.add_argument("--max-dur", type=float, default=30.0)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--val-every", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--unfreeze-top", type=int, default=0)
    parser.add_argument("--fused-batch-size", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def require(value: Any, label: str) -> Any:
    if value in (None, ""):
        raise SystemExit(f"{label} is required")
    return value


def run(args: argparse.Namespace) -> None:
    train_manifest = require(args.train_manifest, "--train-manifest")
    validation_manifest = require(args.validation_manifest, "--validation-manifest")
    tokenizer_dir = require(args.tokenizer_dir, "--tokenizer-dir")
    model_name = str(require(args.model_name, "--model-name"))
    run_dir = args.exp_dir / args.name
    ckpt_dir = run_dir / "checkpoints"

    print(
        f"model={model_name} train={train_manifest} dev={validation_manifest} "
        f"tokenizer={tokenizer_dir} run={run_dir}",
        flush=True,
    )
    if args.dry_run:
        return

    import lightning.pytorch as pl  # ty: ignore[unresolved-import]
    from lightning.pytorch.callbacks import ModelCheckpoint  # ty: ignore[unresolved-import]
    from nemo.collections.asr.models import ASRModel  # ty: ignore[unresolved-import]
    from nemo.utils import model_utils  # ty: ignore[unresolved-import]
    from omegaconf import OmegaConf

    class TrainLogger(JsonlTrainLogger, pl.Callback):
        pass

    class ValLogger(JsonlValLogger, pl.Callback):
        pass

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(model_name)
    if model_path.exists():
        model = ASRModel.restore_from(str(model_path))
    else:
        model = ASRModel.from_pretrained(model_name)
    print(
        f"loaded {type(model).__name__} num_extra_outputs={model.joint.num_extra_outputs}",
        flush=True,
    )
    model.change_vocabulary(
        new_tokenizer_dir=str(Path(tokenizer_dir).resolve()),
        new_tokenizer_type=args.tokenizer_type,
    )
    num_classes = apply_loss_init_fix(model)
    configure_fused_tdt_loss(model, args.fused_batch_size)
    enable_eval_loss(model)
    print(
        f"RNNTLoss num_classes={num_classes}; fused_batch_size={args.fused_batch_size}; "
        f"compute_eval_loss={model.compute_eval_loss}",
        flush=True,
    )
    if args.freeze_encoder or args.unfreeze_top > 0:
        print(freeze_encoder(model, unfreeze_top=args.unfreeze_top), flush=True)

    checkpoint_train = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        save_last=True,
        save_top_k=3,
        monitor="train_loss",
        mode="min",
        every_n_train_steps=args.val_every,
        filename="step{step}-loss{train_loss:.3f}",
        auto_insert_metric_name=False,
    )
    checkpoint_val = ModelCheckpoint(
        **validation_checkpoint_config(ckpt_dir),
    )
    trainer = pl.Trainer(
        devices=1,
        accelerator="gpu",
        precision=args.precision,
        max_steps=args.max_steps,
        val_check_interval=args.val_every,
        num_sanity_val_steps=0,
        gradient_clip_val=1.0,
        log_every_n_steps=args.log_every,
        enable_checkpointing=True,
        logger=False,
        enable_progress_bar=False,
        callbacks=[
            checkpoint_train,
            checkpoint_val,
            TrainLogger(run_dir / "train_log.jsonl", args.log_every),
            ValLogger(run_dir / "val_log.jsonl"),
        ],
    )
    model.set_trainer(trainer)
    config = OmegaConf.create(
        {
            "model": {
                "train_ds": train_ds(
                    Path(train_manifest), args.max_dur, args.batch_dur, args.num_workers
                ),
                "validation_ds": val_ds(
                    Path(validation_manifest), args.max_dur, args.num_workers
                ),
                "optim": {
                    "name": "adamw",
                    "lr": args.lr,
                    "betas": [0.9, 0.98],
                    "weight_decay": 1e-3,
                    "sched": {
                        "name": "CosineAnnealing",
                        "warmup_steps": args.warmup,
                        "min_lr": 1e-5,
                        "max_steps": args.max_steps,
                    },
                },
                "spec_augment": {
                    "_target_": "nemo.collections.asr.modules.SpectrogramAugmentation",
                    "freq_masks": 2,
                    "time_masks": 10,
                    "freq_width": 27,
                    "time_width": 0.05,
                },
            }
        }
    )
    resolved = model_utils.convert_model_config_to_dict_config(config)
    model.setup_training_data(resolved.model.train_ds)
    model.setup_multiple_validation_data(resolved.model.validation_ds)
    model.setup_optimization(resolved.model.optim)
    model.spec_augment = ASRModel.from_config_dict(resolved.model.spec_augment)

    resume_ckpt = (
        str(ckpt_dir / "last.ckpt")
        if args.resume and (ckpt_dir / "last.ckpt").exists()
        else None
    )
    trainer.fit(model, ckpt_path=resume_ckpt)
    model.save_to(str(run_dir / f"{args.name}_final.nemo"))


def train_tdt_main(project: ParakeetProject, argv: list[str] | None = None) -> int:
    project.configure_environment()
    run(build_parser(project).parse_args(argv))
    return 0
