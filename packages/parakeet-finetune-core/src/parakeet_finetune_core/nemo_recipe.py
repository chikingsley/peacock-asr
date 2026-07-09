"""CTC-only wrapper for NeMo's generic speech-to-text fine-tune example."""

from __future__ import annotations

import argparse
import re
import runpy
import sys
import tarfile
from pathlib import Path
from typing import TYPE_CHECKING, Never

if TYPE_CHECKING:
    from parakeet_finetune_core.project import ParakeetProject

TRANSDUCER_MODEL_HINTS = ("tdt", "rnnt", "transducer")
TRUSTED_REMOTE_CTC = re.compile(r"^nvidia/parakeet-ctc[-_].+$", re.IGNORECASE)
PURE_CTC_TARGETS = {
    "nemo.collections.asr.models.ctc_models.EncDecCTCModel",
    "nemo.collections.asr.models.ctc_bpe_models.EncDecCTCModelBPE",
}


def build_parser(project: ParakeetProject) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Launch NeMo speech-to-text fine-tuning for {project.name}."
    )
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-kind", choices=["ctc", "tdt", "rnnt"], required=True)
    parser.add_argument("--tokenizer-type", default="bpe", choices=["bpe", "wpe", "agg"])
    parser.add_argument("--nemo-root", type=Path, default=project.default_nemo_root())
    parser.add_argument("--exp-dir", type=Path, default=project.runs)
    parser.add_argument("--name", default=project.default_tdt_run_name or "parakeet")
    parser.add_argument("--devices", default="1")
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--validation-batch-size", type=int, default=None)
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)
    parser.add_argument("--learning-rate", default="1e-4")
    parser.add_argument("--warmup-steps", type=int, default=5000)
    parser.add_argument("--val-check-interval", default="1.0")
    parser.add_argument("--log-every-n-steps", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--min-duration", type=float, default=0.1)
    parser.add_argument("--max-duration", type=float, default=20.0)
    parser.add_argument("--early-stopping", action="store_true")
    parser.add_argument("--early-stopping-monitor", choices=["val_wer"], default="val_wer")
    parser.add_argument("--early-stopping-mode", choices=["min"], default="min")
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.001)
    parser.add_argument("--early-stopping-check-on-train-epoch-end", action="store_true")
    parser.add_argument("--disable-cudnn", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def validate_ctc_only(args: argparse.Namespace) -> None:
    model_name = str(args.model_name)
    refused = args.model_kind != "ctc" or any(
        hint in model_name.lower() for hint in TRANSDUCER_MODEL_HINTS
    )
    model_path = Path(model_name).expanduser()
    if not refused and model_path.is_file():
        refused = _local_nemo_target(model_path) not in PURE_CTC_TARGETS
    elif not refused and (model_path.suffix == ".nemo" or model_path.is_absolute()):
        raise SystemExit(f"local NeMo model does not exist: {model_path}")
    elif not refused:
        refused = TRUSTED_REMOTE_CTC.fullmatch(model_name) is None
    if refused:
        _raise_ctc_only()


def _local_nemo_target(path: Path) -> str:
    """Read the model class from a local NeMo archive without restoring its weights."""
    try:
        with tarfile.open(path, mode="r:*") as archive:
            members = [
                member
                for member in archive.getmembers()
                if member.name.endswith("model_config.yaml")
            ]
            if len(members) != 1:
                _raise_ctc_only()
            handle = archive.extractfile(members[0])
            if handle is None:
                _raise_ctc_only()
            config = handle.read().decode("utf-8")
    except (OSError, tarfile.TarError, UnicodeDecodeError) as exc:
        raise SystemExit(f"could not inspect local NeMo model {path}: {exc}") from exc
    match = re.search(r"(?m)^target:\s*([^\s#]+)", config)
    return match.group(1) if match else ""


def _raise_ctc_only() -> Never:
    raise SystemExit(
        "the generic NeMo recipe is CTC-only; TDT/RNNT models must use the dedicated "
        "<language>-parakeet-train-tdt command so loss repair, compute_eval_loss=True, "
        "and val_loss checkpoint selection cannot be bypassed"
    )


def build_script_args(args: argparse.Namespace) -> tuple[Path, list[str]]:
    validate_ctc_only(args)
    script = args.nemo_root / "examples/asr/speech_to_text_finetune.py"
    if not script.exists():
        raise FileNotFoundError(script)
    config_path = args.nemo_root / "examples/asr/conf/asr_finetune"
    validation_batch_size = args.validation_batch_size or args.batch_size
    script_args = [
        f"--config-path={config_path}",
        "--config-name=speech_to_text_finetune",
        f"model.train_ds.manifest_filepath={args.train_manifest.resolve()}",
        f"model.validation_ds.manifest_filepath={args.validation_manifest.resolve()}",
        f"model.test_ds.manifest_filepath={args.validation_manifest.resolve()}",
        "model.tokenizer.update_tokenizer=true",
        f"model.tokenizer.dir={args.tokenizer_dir.resolve()}",
        f"model.tokenizer.type={args.tokenizer_type}",
        f"model.train_ds.batch_size={args.batch_size}",
        f"model.validation_ds.batch_size={validation_batch_size}",
        f"model.test_ds.batch_size={validation_batch_size}",
        f"model.train_ds.num_workers={args.num_workers}",
        f"model.validation_ds.num_workers={args.num_workers}",
        f"model.test_ds.num_workers={args.num_workers}",
        f"model.train_ds.min_duration={args.min_duration}",
        f"model.train_ds.max_duration={args.max_duration}",
        f"model.optim.lr={args.learning_rate}",
        f"model.optim.sched.warmup_steps={args.warmup_steps}",
        f"trainer.devices={args.devices}",
        f"trainer.accelerator={args.accelerator}",
        f"trainer.precision={args.precision}",
        f"trainer.max_epochs={args.max_epochs}",
        f"trainer.max_steps={args.max_steps}",
        f"trainer.accumulate_grad_batches={args.accumulate_grad_batches}",
        f"trainer.val_check_interval={args.val_check_interval}",
        f"trainer.log_every_n_steps={args.log_every_n_steps}",
        f"exp_manager.exp_dir={args.exp_dir}",
        f"exp_manager.name={args.name}",
        "exp_manager.create_wandb_logger=false",
        "exp_manager.checkpoint_callback_params.monitor=val_wer",
        "exp_manager.checkpoint_callback_params.mode=min",
    ]
    model_path = Path(str(args.model_name)).expanduser()
    init_key = "init_from_nemo_model" if model_path.is_file() else "init_from_pretrained_model"
    script_args.append(f"+{init_key}={args.model_name}")
    if args.early_stopping:
        script_args.extend(
            [
                "+exp_manager.create_early_stopping_callback=true",
                f"+exp_manager.early_stopping_callback_params.monitor={args.early_stopping_monitor}",
                f"+exp_manager.early_stopping_callback_params.mode={args.early_stopping_mode}",
                f"+exp_manager.early_stopping_callback_params.patience={args.early_stopping_patience}",
                f"+exp_manager.early_stopping_callback_params.min_delta={args.early_stopping_min_delta}",
                "+exp_manager.early_stopping_callback_params.check_on_train_epoch_end="
                f"{str(args.early_stopping_check_on_train_epoch_end).lower()}",
            ]
        )
    return script, script_args


def build_command(args: argparse.Namespace) -> list[str]:
    script, script_args = build_script_args(args)
    return ["runpy", str(script), *script_args]


def run_script(script: Path, script_args: list[str], *, disable_cudnn: bool) -> None:
    if disable_cudnn:
        import torch  # ty: ignore[unresolved-import]

        torch.backends.cudnn.enabled = False
    old_argv = sys.argv
    try:
        sys.argv = [str(script), *script_args]
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = old_argv


def train_nemo_recipe_main(project: ParakeetProject, argv: list[str] | None = None) -> int:
    project.configure_environment()
    args = build_parser(project).parse_args(argv)
    script, script_args = build_script_args(args)
    command = ["runpy", str(script), *script_args]
    print(" ".join(command))
    if args.dry_run:
        return 0
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    run_script(script, script_args, disable_cudnn=args.disable_cudnn)
    return 0
