from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from finetune_parakeet.paths import (
    DEFAULT_NEMO_ROOT,
    DEFAULT_RUNS_ROOT,
    configure_external_caches,
)

DEFAULT_MODEL = "nvidia/parakeet-tdt_ctc-110m"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch NeMo fine-tuning for a Parakeet checkpoint with a Persian tokenizer."
    )
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer-type", default="bpe", choices=["bpe", "wpe", "agg"])
    parser.add_argument("--nemo-root", type=Path, default=DEFAULT_NEMO_ROOT)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--exp-dir", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--name", default="parakeet")
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
    parser.add_argument("--early-stopping-monitor", default="val_wer")
    parser.add_argument("--early-stopping-mode", default="min")
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.001)
    parser.add_argument("--early-stopping-check-on-train-epoch-end", action="store_true")
    parser.add_argument("--disable-cudnn", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def build_command(args: argparse.Namespace) -> list[str]:
    script = args.nemo_root / "examples/asr/speech_to_text_finetune.py"
    if not script.exists():
        raise FileNotFoundError(script)
    config_path = args.nemo_root / "examples/asr/conf/asr_finetune"
    train_manifest = args.train_manifest.resolve()
    validation_manifest = args.validation_manifest.resolve()
    tokenizer_dir = args.tokenizer_dir.resolve()
    validation_batch_size = args.validation_batch_size or args.batch_size
    script_args = [
        f"--config-path={config_path}",
        "--config-name=speech_to_text_finetune",
        f"model.train_ds.manifest_filepath={train_manifest}",
        f"model.validation_ds.manifest_filepath={validation_manifest}",
        f"model.test_ds.manifest_filepath={validation_manifest}",
        "model.tokenizer.update_tokenizer=true",
        f"model.tokenizer.dir={tokenizer_dir}",
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
        f"+init_from_pretrained_model={args.model_name}",
    ]
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
    if args.disable_cudnn:
        runner = (
            "import runpy, sys, torch; "
            "torch.backends.cudnn.enabled = False; "
            "sys.argv = sys.argv[1:]; "
            "runpy.run_path(sys.argv[0], run_name='__main__')"
        )
        return [args.python, "-c", runner, str(script), *script_args]
    return [args.python, str(script), *script_args]


def main(argv: list[str] | None = None) -> int:
    configure_external_caches()
    args = build_parser().parse_args(argv)
    command = build_command(args)
    print(" ".join(command))
    if args.dry_run:
        return 0
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True)  # noqa: S603 — trusted operator CLI: list-form argv (no shell); interpreter is the operator-supplied --python path
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
