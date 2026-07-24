"""Shared helpers and CLI for Parakeet TDT fine-tuning."""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from parakeet_finetune_core.project import ParakeetProject

_AUDIO_SAMPLE_RATE = 16_000


def _audio_seconds_from_batch(batch: object) -> float:
    """Return unpadded audio exposure from a standard NeMo ASR batch."""
    if isinstance(batch, dict):
        lengths = batch.get("audio_signal_length")
    elif isinstance(batch, (list, tuple)) and len(batch) > 1:
        lengths = batch[1]
    else:
        return 0.0
    if lengths is None:
        return 0.0
    total = lengths.sum() if hasattr(lengths, "sum") else sum(lengths)
    if hasattr(total, "item"):
        total = total.item()
    return float(total) / _AUDIO_SAMPLE_RATE


class JsonlTrainLogger:
    """Factory for a Lightning callback that appends train loss records to JSONL."""

    def __init__(self, path: Path, every: int) -> None:
        self.path = path
        self.every = every
        self.t0 = time.time()
        self._last_logged_step: int | None = None
        self.audio_seconds_seen = 0.0

    def on_train_batch_end(self, trainer: Any, _pl_module: Any, *_args: object) -> None:
        batch = _args[1] if len(_args) > 1 else None
        self.audio_seconds_seen += _audio_seconds_from_batch(batch)
        step = trainer.global_step
        if step % self.every != 0 or step == self._last_logged_step:
            return
        self._last_logged_step = step
        metrics = trainer.callback_metrics
        loss = metrics.get("train_loss", metrics.get("loss"))
        loss_value = float(loss) if loss is not None else float("nan")
        lr = trainer.optimizers[0].param_groups[0]["lr"] if trainer.optimizers else float("nan")
        record = {
            "step": step,
            "loss": round(loss_value, 4),
            "lr": round(lr, 8),
            "t": round(time.time() - self.t0, 1),
            "audio_hours_seen": round(self.audio_seconds_seen / 3600, 4),
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


def apply_loss_init_fix(model: Any, reduction_override: str | None = None) -> int:
    """Rebuild TDT RNNTLoss with duration-bin outputs excluded from ``num_classes``."""
    from nemo.collections.asr.losses.rnnt import RNNTLoss  # ty: ignore[unresolved-import]

    num_classes = model.joint.num_classes_with_blank - 1
    if model.joint.num_extra_outputs > 0:
        num_classes -= model.joint.num_extra_outputs
    loss_name, loss_kwargs = model.extract_rnnt_loss_cfg(model.cfg.get("loss", None))
    # extract_rnnt_loss_cfg drops rnnt_reduction; v3 cfg uses mean_volume while RNNTLoss defaults
    # to mean_batch. mean_volume normalizes by total tokens (~40x smaller loss/grads than
    # mean_batch) so it needs a co-scaled lr; allow an explicit override so a run can pin the
    # validated mean_batch+lr combo instead of silently flipping loss scale.
    reduction = reduction_override or model.cfg.get("rnnt_reduction", "mean_batch")
    model.loss = RNNTLoss(
        num_classes=num_classes,
        loss_name=loss_name,
        loss_kwargs=loss_kwargs,
        reduction=reduction,
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


def configure_aux_ctc_weight(model: Any, weight: float | None) -> float | None:
    """Override the auxiliary CTC share on a hybrid model when explicitly requested."""
    if weight is None:
        return None
    if not 0.0 <= weight <= 1.0:
        raise ValueError("ctc_loss_weight must be between 0.0 and 1.0")
    if not hasattr(model, "ctc_loss_weight"):
        raise ValueError("ctc_loss_weight requires a hybrid TDT+CTC model")
    model.ctc_loss_weight = weight
    cfg = getattr(model, "cfg", None)
    if cfg is not None:
        if isinstance(cfg, dict):
            cfg.setdefault("aux_ctc", {})["ctc_loss_weight"] = weight
        else:
            from omegaconf import open_dict

            with open_dict(cfg.aux_ctc):
                cfg.aux_ctc.ctc_loss_weight = weight
    return weight


def spec_augment_config(profile: str) -> dict[str, Any] | None:
    """Return the controlled SpecAugment profile used by TDT experiments."""
    if profile == "off":
        return None
    if profile == "half":
        freq_masks, time_masks = 1, 5
    elif profile == "current":
        freq_masks, time_masks = 2, 10
    else:
        raise ValueError(f"unknown SpecAugment profile: {profile}")
    return {
        "_target_": "nemo.collections.asr.modules.SpectrogramAugmentation",
        "freq_masks": freq_masks,
        "time_masks": time_masks,
        "freq_width": 27,
        "time_width": 0.05,
    }


def configure_spec_augmentation(model: Any, profile: str, module_factory: Any) -> None:
    """Install the selected profile on NeMo's actual forward-path attribute and config."""
    config = spec_augment_config(profile)
    model.spec_augmentation = None if config is None else module_factory.from_config_dict(config)
    cfg = getattr(model, "cfg", None)
    if cfg is None:
        return
    if isinstance(cfg, dict):
        cfg["spec_augment"] = config
        return
    from omegaconf import open_dict

    with open_dict(cfg):
        cfg.spec_augment = config


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


def _change_vocabulary(model: Any, tokenizer_dir: Path, tokenizer_type: str) -> None:
    model.change_vocabulary(
        new_tokenizer_dir=str(Path(tokenizer_dir).resolve()),
        new_tokenizer_type=tokenizer_type,
    )


def load_and_prepare_model(
    asr_model_cls: Any,
    model_name: str,
    tokenizer_dir: Path | None,
    args: argparse.Namespace,
) -> Any:
    """Load the base model, apply the chosen recipe, and configure the TDT loss."""
    model_path = Path(model_name)
    if model_path.exists():
        # Restore on CPU so preflight/model construction does not opportunistically compete with
        # an unrelated resident GPU service. Lightning moves the prepared model onto the GPU.
        model = asr_model_cls.restore_from(str(model_path), map_location="cpu")
    else:
        model = asr_model_cls.from_pretrained(model_name, map_location="cpu")
    print(
        f"loaded {type(model).__name__} num_extra_outputs={model.joint.num_extra_outputs}",
        flush=True,
    )
    if args.recipe == "extend-restore":
        if tokenizer_dir is None:
            raise ValueError("extend-restore requires --tokenizer-dir")
        from parakeet_finetune_core.extend_restore import (
            restore_extended_decoder_joint,
            snapshot_decoder_joint,
        )

        snapshot = snapshot_decoder_joint(model)
        _change_vocabulary(model, tokenizer_dir, args.tokenizer_type)
        info = restore_extended_decoder_joint(model, snapshot, old_vocab=args.old_vocab)
        print(
            f"extend-restore: old_vocab={info['old_vocab']} K={info['k']} "
            f"new_blank={info['new_blank']} "
            f"num_classes_with_blank={info['num_classes_with_blank']}",
            flush=True,
        )
    elif tokenizer_dir is not None:
        _change_vocabulary(model, tokenizer_dir, args.tokenizer_type)
    else:
        print(
            "tokenizer=embedded (preserving the base decoder/joint vocabulary weights)",
            flush=True,
        )
    num_classes = apply_loss_init_fix(model, args.reduction or None)
    configure_fused_tdt_loss(model, args.fused_batch_size)
    enable_eval_loss(model)
    ctc_loss_weight = configure_aux_ctc_weight(model, args.ctc_loss_weight)
    effective_ctc_loss_weight = (
        getattr(model, "ctc_loss_weight", None) if ctc_loss_weight is None else ctc_loss_weight
    )
    print(
        f"RNNTLoss num_classes={num_classes}; fused_batch_size={args.fused_batch_size}; "
        f"compute_eval_loss={model.compute_eval_loss}; "
        f"ctc_loss_weight={effective_ctc_loss_weight}",
        flush=True,
    )
    if args.freeze_encoder or args.unfreeze_top > 0:
        print(freeze_encoder(model, unfreeze_top=args.unfreeze_top), flush=True)
    return model


def parse_train_sources(values: list[str] | None) -> list[tuple[Path, float]]:
    """Parse repeated ``PATH=WEIGHT`` source specifications for Lhotse multiplexing."""
    sources: list[tuple[Path, float]] = []
    seen: set[Path] = set()
    for value in values or []:
        path_text, separator, weight_text = value.rpartition("=")
        if not separator or not path_text or not weight_text:
            raise ValueError(f"invalid --train-source {value!r}; expected PATH=WEIGHT")
        path = Path(path_text)
        try:
            weight = float(weight_text)
        except ValueError as error:
            raise ValueError(
                f"invalid --train-source weight {weight_text!r} in {value!r}"
            ) from error
        if weight <= 0:
            raise ValueError(f"--train-source weight must be positive: {value!r}")
        if path in seen:
            raise ValueError(f"duplicate --train-source path: {path}")
        seen.add(path)
        sources.append((path, weight))
    return sources


def _add_l2sp_gradients(entries: list[tuple[str, Any, str]], anchors: Any, weight: float) -> None:
    for _name, parameter, buffer_name in entries:
        if parameter.grad is None:
            continue
        difference = parameter.detach() - getattr(anchors, buffer_name)
        parameter.grad.add_(difference, alpha=weight)


def make_l2sp_callback(callback_base: type, model: Any, weight: float) -> Any:
    """Anchor trainable parameters to their restored base values with L2-SP gradients."""
    if weight <= 0:
        raise ValueError("l2sp_weight must be positive")

    import torch  # ty: ignore[unresolved-import]

    anchors = torch.nn.Module()
    entries: list[tuple[str, Any, str]] = []
    parameter_count = 0
    for index, (name, parameter) in enumerate(model.named_parameters()):
        if not parameter.requires_grad:
            continue
        buffer_name = f"anchor_{index}"
        anchors.register_buffer(buffer_name, parameter.detach().clone(), persistent=False)
        entries.append((name, parameter, buffer_name))
        parameter_count += parameter.numel()
    if not entries:
        raise ValueError("L2-SP requires at least one trainable parameter")
    model.add_module("l2sp_anchors", anchors)

    class L2SPCallback(callback_base):
        def __init__(self) -> None:
            super().__init__()
            self._last_reported_step: int | None = None

        def on_before_optimizer_step(self, _trainer: Any, pl_module: Any, _optimizer: Any) -> None:
            anchor_module = pl_module.l2sp_anchors
            with torch.no_grad():
                _add_l2sp_gradients(entries, anchor_module, weight)

        def on_validation_end(self, trainer: Any, pl_module: Any) -> None:
            step = trainer.global_step
            if step == self._last_reported_step:
                return
            self._last_reported_step = step
            anchor_module = pl_module.l2sp_anchors
            squared_l2 = 0.0
            with torch.no_grad():
                for _name, parameter, buffer_name in entries:
                    difference = (
                        parameter.detach().float() - getattr(anchor_module, buffer_name).float()
                    )
                    squared_l2 += float(difference.square().sum())
            half_squared_l2 = 0.5 * squared_l2
            rms = (squared_l2 / parameter_count) ** 0.5
            print(
                f"  [l2sp @ step {step}] weight={weight:g} "
                f"half_squared_l2={half_squared_l2:.4f} rms_drift={rms:.8f} "
                f"effective_penalty={weight * half_squared_l2:.4f}",
                flush=True,
            )

    print(
        f"L2-SP enabled: weight={weight:g} anchored_parameters={parameter_count}",
        flush=True,
    )
    return L2SPCallback()


def train_ds(
    manifest: Path | list[tuple[Path, float]],
    max_dur: float,
    batch_dur: float,
    num_workers: int,
    seed: int,
) -> dict[str, Any]:
    source_config: dict[str, Any]
    if isinstance(manifest, Path):
        source_config = {"manifest_filepath": str(manifest)}
    else:
        if not manifest:
            raise ValueError("at least one weighted training manifest is required")
        source_config = {
            "input_cfg": [
                {"type": "nemo", "manifest_filepath": str(path), "weight": weight}
                for path, weight in manifest
            ]
        }
    return {
        **source_config,
        "sample_rate": 16_000,
        "use_lhotse": True,
        "use_bucketing": True,
        "concurrent_bucketing": True,
        "num_buckets": 30,
        "batch_duration": batch_dur,
        "quadratic_duration": 15.0,
        "batch_size": None,
        "shuffle": True,
        "seed": seed,
        "shard_seed": seed,
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


def validation_schedule(val_every: int, accumulate_grad_batches: int) -> dict[str, Any]:
    """Return a global optimizer-step validation schedule for an iterable training loader."""
    return {
        "val_check_interval": val_every * accumulate_grad_batches,
        "check_val_every_n_epoch": None,
    }


def _loader_graph(*roots: Any) -> Iterator[Any]:
    """Yield the bounded loader/sampler object graph used by Lightning and NeMo."""
    pending = list(roots)
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, dict):
            pending.extend(current.values())
            continue
        if isinstance(current, (list, tuple)):
            pending.extend(current)
            continue
        yield current
        for attribute in (
            "sampler",
            "dataset",
            "batch_sampler",
            "iterables",
            "loaders",
            "_loaders",
        ):
            child = getattr(current, attribute, None)
            if child is not None:
                pending.append(child)


def _thread_owned_lhotse_samplers() -> Iterator[Any]:
    """Recover detached DynamicBucketer owners from Lhotse producer closures."""
    for thread in threading.enumerate():
        target = getattr(thread, "_target", None)
        for cell in getattr(target, "__closure__", None) or ():
            try:
                owner = cell.cell_contents
            except ValueError:
                continue
            if getattr(owner, "_producer_thread", None) is thread:
                yield owner


def shutdown_lhotse_training_sampler(
    model: Any, *, trainer: Any | None = None, join_timeout: float = 5.0
) -> bool:
    """Stop Lhotse's non-daemon concurrent bucketing producer after bounded training."""
    # Lightning may wrap or replace the model's original loader. Start with its active
    # loader graph, then retain NeMo's ``_train_dl`` as a fallback for direct callers.
    active_dataloader = getattr(trainer, "train_dataloader", None)
    model_dataloader = getattr(model, "_train_dl", None)
    candidates = [
        *_loader_graph(active_dataloader, model_dataloader),
        *_thread_owned_lhotse_samplers(),
    ]
    seen: set[int] = set()
    for sampler in candidates:
        if id(sampler) in seen:
            continue
        seen.add(id(sampler))
        producer = getattr(sampler, "_producer_thread", None)
        if producer is None or not producer.is_alive():
            continue

        sampler._source_exhausted = True  # noqa: SLF001 - mirrors Lhotse's own cleanup
        producer.join(timeout=join_timeout)
        if producer.is_alive():
            raise RuntimeError("Lhotse concurrent bucketing producer did not stop")
        # Keep the joined thread object until Lhotse's generator finalizer observes that it
        # is no longer alive; clearing it here makes that finalizer call ``None.is_alive()``.
        return True
    return False


def export_training_artifacts(
    model: Any,
    run_dir: Path,
    run_name: str,
    best_checkpoint_path: str | Path | None,
    *,
    checkpoint_loader: Callable[[Path], Any] | None = None,
) -> tuple[Path, Path | None]:
    """Save last-step weights and, when available, a distinct best-validation model."""
    final_path = run_dir / f"{run_name}_final.nemo"
    model.save_to(str(final_path))

    if not best_checkpoint_path:
        print("no best val_loss checkpoint was produced; saved last-step final only", flush=True)
        return final_path, None

    checkpoint_path = Path(best_checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    loader = checkpoint_loader
    if loader is None:
        import torch  # ty: ignore[unresolved-import]

        def load_checkpoint(path: Path) -> Any:
            return torch.load(path, map_location="cpu", weights_only=False)

        loader = load_checkpoint
    checkpoint = loader(checkpoint_path)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    best_path = run_dir / f"{run_name}_best-valloss.nemo"
    model.save_to(str(best_path))
    print(
        f"saved last-step model={final_path} and best-val_loss model={best_path} "
        f"from checkpoint={checkpoint_path}",
        flush=True,
    )
    return final_path, best_path


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
        "--train-source",
        action="append",
        default=[],
        metavar="PATH=WEIGHT",
        help=(
            "Repeat to sample separate NeMo manifests with explicit Lhotse weights. "
            "When set, these sources replace --train-manifest."
        ),
    )
    parser.add_argument(
        "--validation-manifest",
        type=Path,
        default=project.default_validation_manifest,
    )
    parser.add_argument("--tokenizer-dir", type=Path, default=project.default_tokenizer_dir)
    parser.add_argument("--tokenizer-type", default="bpe")
    parser.add_argument("--exp-dir", type=Path, default=project.runs)
    parser.add_argument("--batch-dur", type=float, default=120.0)
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)
    parser.add_argument("--max-dur", type=float, default=30.0)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--optim", default="adamw", help="adamw | adafactor")
    parser.add_argument("--reduction", default="", help="rnnt reduction override; empty=model cfg")
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--val-every", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--l2sp-weight",
        type=float,
        default=0.0,
        help=(
            "Gradient coefficient anchoring trainable parameters to their restored base values. "
            "0 disables L2-SP."
        ),
    )
    parser.add_argument(
        "--ctc-loss-weight",
        type=float,
        default=None,
        help="Auxiliary CTC share for a hybrid model; empty keeps the base-model value.",
    )
    parser.add_argument(
        "--spec-augment",
        choices=["off", "half", "current"],
        default="current",
        help="Controlled SpecAugment profile: off, half masks, or the current base recipe.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--unfreeze-top", type=int, default=0)
    parser.add_argument("--fused-batch-size", type=int, default=2)
    parser.add_argument(
        "--recipe",
        choices=["simple", "extend-restore"],
        default="simple",
        help="simple: preserve the embedded tokenizer, or replace it when --tokenizer-dir is set. "
        "extend-restore: extend the base tokenizer and restore pretrained decoder/joint rows.",
    )
    parser.add_argument(
        "--old-vocab",
        type=int,
        default=8192,
        help="Base tokenizer size for extend-restore row mapping (v3 = 8192).",
    )
    parser.add_argument(
        "--freeze-warmup-steps",
        type=int,
        default=0,
        help="extend-restore: freeze encoder until this global step, then unfreeze "
        "(all, or --unfreeze-top N). 0 disables the warmup callback.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help=(
            "Restore and prepare the model on CPU, print the live contract, then exit before "
            "Trainer."
        ),
    )
    return parser


def require(value: Any, label: str) -> Any:
    if value in (None, ""):
        raise SystemExit(f"{label} is required")
    return value


def run(args: argparse.Namespace) -> None:
    train_sources = parse_train_sources(args.train_source)
    train_manifest = train_sources or require(args.train_manifest, "--train-manifest")
    validation_manifest = require(args.validation_manifest, "--validation-manifest")
    tokenizer_dir = args.tokenizer_dir
    model_name = str(require(args.model_name, "--model-name"))
    run_dir = args.exp_dir / args.name
    ckpt_dir = run_dir / "checkpoints"

    print(
        f"model={model_name} train={train_manifest} dev={validation_manifest} "
        f"tokenizer={tokenizer_dir or 'embedded'} run={run_dir} batch_dur={args.batch_dur} "
        f"accumulate_grad_batches={args.accumulate_grad_batches}",
        flush=True,
    )
    if args.dry_run:
        return

    import lightning.pytorch as pl  # ty: ignore[unresolved-import]
    from lightning.pytorch.callbacks import ModelCheckpoint  # ty: ignore[unresolved-import]
    from nemo.collections.asr.models import ASRModel  # ty: ignore[unresolved-import]
    from nemo.utils import model_utils  # ty: ignore[unresolved-import]
    from omegaconf import OmegaConf

    pl.seed_everything(args.seed, workers=True)

    class TrainLogger(JsonlTrainLogger, pl.Callback):
        pass

    class ValLogger(JsonlValLogger, pl.Callback):
        pass

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model = load_and_prepare_model(ASRModel, model_name, tokenizer_dir, args)
    if args.prepare_only:
        parameters = sum(parameter.numel() for parameter in model.parameters())
        vocab_size = getattr(getattr(model, "tokenizer", None), "vocab_size", "unknown")
        print(
            f"prepared model on CPU: parameters={parameters} tokenizer_vocab={vocab_size} "
            f"compute_eval_loss={model.compute_eval_loss}",
            flush=True,
        )
        return

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
    callbacks: list[Any] = [
        checkpoint_train,
        checkpoint_val,
        TrainLogger(run_dir / "train_log.jsonl", args.log_every),
        ValLogger(run_dir / "val_log.jsonl"),
    ]
    if args.l2sp_weight < 0:
        raise ValueError("l2sp_weight must be nonnegative")
    if args.l2sp_weight > 0:
        callbacks.append(make_l2sp_callback(pl.Callback, model, args.l2sp_weight))
    if args.recipe == "extend-restore" and args.freeze_warmup_steps > 0:
        from parakeet_finetune_core.extend_restore import make_freeze_warmup_callback

        callbacks.append(make_freeze_warmup_callback(args.freeze_warmup_steps, args.unfreeze_top))
        print(
            f"freeze-warmup enabled: encoder frozen until step {args.freeze_warmup_steps} "
            f"(unfreeze_top={args.unfreeze_top})",
            flush=True,
        )
    trainer = pl.Trainer(
        devices=1,
        accelerator="gpu",
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        max_steps=args.max_steps,
        # Lightning counts val_check_interval in microbatches, while max_steps and our CLI
        # contract count optimizer updates. check_val_every_n_epoch=None makes the integer
        # interval global across iterable-loader epoch boundaries.
        **validation_schedule(args.val_every, args.accumulate_grad_batches),
        num_sanity_val_steps=0,
        gradient_clip_val=1.0,
        log_every_n_steps=args.log_every,
        enable_checkpointing=True,
        logger=False,
        enable_progress_bar=False,
        callbacks=callbacks,
    )
    model.set_trainer(trainer)
    config = OmegaConf.create(
        {
            "model": {
                "train_ds": train_ds(
                    train_manifest if isinstance(train_manifest, list) else Path(train_manifest),
                    args.max_dur,
                    args.batch_dur,
                    args.num_workers,
                    args.seed,
                ),
                "validation_ds": val_ds(Path(validation_manifest), args.max_dur, args.num_workers),
                "optim": {
                    "name": args.optim,
                    "lr": args.lr,
                    "weight_decay": 1e-3,
                    # AdamW keeps fp32 first+second moments (~12 B/param); adafactor factors the
                    # second moment + drops the first (tiny state) so 0.6B full-FT fits 12 GB.
                    # adafactor: relative_step off so it honors our manual lr + cosine schedule.
                    **(
                        {"betas": [0.9, 0.98]}
                        if args.optim == "adamw"
                        else {
                            "relative_step": False,
                            "scale_parameter": False,
                            "warmup_init": False,
                        }
                    ),
                    "sched": {
                        "name": "CosineAnnealing",
                        "warmup_steps": args.warmup,
                        "min_lr": 1e-5,
                        "max_steps": args.max_steps,
                    },
                },
            }
        }
    )
    resolved = model_utils.convert_model_config_to_dict_config(config)
    model.setup_training_data(resolved.model.train_ds)
    model.setup_multiple_validation_data(resolved.model.validation_ds)
    model.setup_optimization(resolved.model.optim)
    configure_spec_augmentation(model, args.spec_augment, ASRModel)
    print(f"SpecAugment profile={args.spec_augment}", flush=True)

    resume_ckpt = (
        str(ckpt_dir / "last.ckpt") if args.resume and (ckpt_dir / "last.ckpt").exists() else None
    )
    try:
        trainer.fit(model, ckpt_path=resume_ckpt)
    finally:
        shutdown_lhotse_training_sampler(model, trainer=trainer)
    export_training_artifacts(model, run_dir, args.name, checkpoint_val.best_model_path)


def train_tdt_main(project: ParakeetProject, argv: list[str] | None = None) -> int:
    project.configure_environment()
    run(build_parser(project).parse_args(argv))
    return 0
