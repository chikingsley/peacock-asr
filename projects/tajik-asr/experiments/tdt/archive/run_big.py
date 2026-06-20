"""Big TDT run: 110M Parakeet hybrid (TDT+CTC), full fine-tune on real ~1197h Tajik.

Validated pipeline (gate_decode_check proved decode works; loss-init fix applied):
  fresh Tajik BPE-1024 (built on the full 232k-transcript corpus) + RNNTLoss num_classes fix
  + fused joint loss/WER (12GB) + spec-augment. bf16 train; train_loss is the reliable signal
  (bf16 TDT val_wer can be noisy -> do a clean fp32 transcribe eval at a checkpoint separately).

Logs {step, loss, lr, t} to runs/<name>/train_log.jsonl every log step (cron reads the tail).
Checkpoints to runs/<name>/checkpoints (save_last + top-3 by train_loss) for resume / fp32 eval.

  uv run --project /home/simon/github/peacock-asr/projects/tajik-asr \
    tajik-parakeet-train-tdt --name big_110m --batch-dur 120 --max-dur 30 --max-steps 100000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from nemo.collections.asr.losses.rnnt import RNNTLoss
from nemo.collections.asr.models import ASRModel
from nemo.utils import model_utils
from omegaconf import OmegaConf

HERE = Path(__file__).resolve().parent
PEACOCK = HERE.parents[3]
HYBRID = PEACOCK / "projects/farsi-asr/src/finetune_parakeet/models/parakeet-tdt_ctc-110m-base-hybrid.nemo"
TOK = HERE / "data/tok_big/tokenizer_spe_bpe_v1024"
TRAIN, DEV = HERE / "data/train_big.jsonl", HERE / "data/dev_big.jsonl"


class JsonlLogger(pl.Callback):
    """Append {step, loss, lr, t} to a jsonl every log_every_n_steps (cron reads the tail)."""

    def __init__(self, path: Path, every: int) -> None:
        self.path = path
        self.every = every
        self.t0 = time.time()

    def on_train_batch_end(self, trainer, pl_module, *_a) -> None:
        step = trainer.global_step
        if step % self.every != 0:
            return
        m = trainer.callback_metrics
        loss = m.get("train_loss", m.get("loss"))
        loss = float(loss) if loss is not None else float("nan")
        lr = trainer.optimizers[0].param_groups[0]["lr"] if trainer.optimizers else float("nan")
        rec = {"step": step, "loss": round(loss, 4), "lr": round(lr, 8),
               "t": round(time.time() - self.t0, 1)}
        with self.path.open("a") as f:
            f.write(json.dumps(rec) + "\n")


class ValLogger(pl.Callback):
    """Surface held-out dev metrics: print + append to val_log.jsonl each validation.

    NOTE: bf16 TDT val_wer is unreliable (bf16 decode bug) — val_loss is the trustworthy held-out
    signal during training; run eval_ckpt.py (fp32 transcribe) for a real WER number.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def on_validation_epoch_end(self, trainer, _pl) -> None:
        m = trainer.callback_metrics

        def g(k: str) -> float:
            v = m.get(k)
            return float(v) if v is not None else float("nan")

        rec = {"step": trainer.global_step, "val_loss": round(g("val_loss"), 4),
               "val_wer_bf16": round(g("val_wer"), 4), "val_wer_ctc_bf16": round(g("val_wer_ctc"), 4)}
        print(f"  [val @ step {rec['step']}] val_loss={rec['val_loss']} "
              f"val_wer(bf16,noisy)={rec['val_wer_bf16']}", flush=True)
        with self.path.open("a") as f:
            f.write(json.dumps(rec) + "\n")


def train_ds(manifest: Path, max_dur: float, batch_dur: float) -> dict:
    # lhotse dynamic bucketing: cap audio-seconds/batch so 30s clips auto-batch small (12GB + TDT joint)
    return {"manifest_filepath": str(manifest), "sample_rate": 16000,
            "use_lhotse": True, "use_bucketing": True, "num_buckets": 30,
            "batch_duration": batch_dur, "quadratic_duration": 15.0, "batch_size": None,
            "shuffle": True, "num_workers": 8, "pin_memory": True,
            "min_duration": 0.5, "max_duration": max_dur}


def val_ds(manifest: Path, max_dur: float) -> dict:
    return {"manifest_filepath": str(manifest), "sample_rate": 16000, "batch_size": 2,
            "shuffle": False, "num_workers": 4, "pin_memory": True,
            "min_duration": 0.5, "max_duration": max_dur}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="big_110m")
    ap.add_argument("--batch-dur", type=float, default=120.0, help="lhotse audio-seconds per batch")
    ap.add_argument("--max-dur", type=float, default=30.0)
    ap.add_argument("--max-steps", type=int, default=100000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--precision", default="bf16")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    run = HERE / "runs" / args.name
    ckpt_dir = run / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = run / "train_log.jsonl"

    model = ASRModel.restore_from(str(HYBRID))
    print(f"loaded {type(model).__name__}  num_extra_outputs={model.joint.num_extra_outputs}",
          flush=True)
    model.change_vocabulary(new_tokenizer_dir=str(TOK), new_tokenizer_type="bpe")
    # loss-init fix (RNNTLoss num_classes must exclude blank AND duration bins)
    nc = model.joint.num_classes_with_blank - 1 - model.joint.num_extra_outputs
    loss_name, loss_kwargs = model.extract_rnnt_loss_cfg(model.cfg.get("loss", None))
    model.loss = RNNTLoss(num_classes=nc, loss_name=loss_name, loss_kwargs=loss_kwargs)
    model.joint.set_fused_batch_size(2)
    model.joint.set_fuse_loss_wer(fuse_loss_wer=True, loss=model.loss, metric=model.wer)
    print(f"  RNNTLoss num_classes={nc}; tokenizer={TOK.name}", flush=True)

    ckpt_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir), save_last=True, save_top_k=3, monitor="train_loss", mode="min",
        every_n_train_steps=args.val_every, filename="step{step}-loss{train_loss:.3f}",
        auto_insert_metric_name=False)
    # also keep the best-on-held-out-dev checkpoints (val_loss is the trustworthy bf16 signal)
    ckpt_cb_val = ModelCheckpoint(
        dirpath=str(ckpt_dir), save_top_k=2, monitor="val_loss", mode="min",
        filename="best-valloss-step{step}-{val_loss:.3f}", auto_insert_metric_name=False)
    trainer = pl.Trainer(
        devices=1, accelerator="gpu", precision=args.precision,
        max_steps=args.max_steps, val_check_interval=args.val_every,
        num_sanity_val_steps=0, gradient_clip_val=1.0, log_every_n_steps=args.log_every,
        enable_checkpointing=True, logger=False, enable_progress_bar=False,
        callbacks=[ckpt_cb, ckpt_cb_val, JsonlLogger(log_path, args.log_every),
                   ValLogger(run / "val_log.jsonl")])
    model.set_trainer(trainer)
    cfg = OmegaConf.create({"model": {
        "train_ds": train_ds(TRAIN, args.max_dur, args.batch_dur),
        "validation_ds": val_ds(DEV, args.max_dur),
        "optim": {"name": "adamw", "lr": args.lr, "betas": [0.9, 0.98], "weight_decay": 1e-3,
                  "sched": {"name": "CosineAnnealing", "warmup_steps": args.warmup,
                            "min_lr": 1e-5, "max_steps": args.max_steps}},
        "spec_augment": {"_target_": "nemo.collections.asr.modules.SpectrogramAugmentation",
                         "freq_masks": 2, "time_masks": 10, "freq_width": 27, "time_width": 0.05},
    }})
    rc = model_utils.convert_model_config_to_dict_config(cfg)
    model.setup_training_data(rc.model.train_ds)
    model.setup_multiple_validation_data(rc.model.validation_ds)
    model.setup_optimization(rc.model.optim)
    model.spec_augment = ASRModel.from_config_dict(rc.model.spec_augment)

    resume_ckpt = str(ckpt_dir / "last.ckpt") if (args.resume and (ckpt_dir / "last.ckpt").exists()) else None
    print(f"\n=== big run '{args.name}': batch_dur{args.batch_dur} max_dur{args.max_dur} lr{args.lr} "
          f"max_steps{args.max_steps} precision{args.precision} resume={bool(resume_ckpt)} ===",
          flush=True)
    trainer.fit(model, ckpt_path=resume_ckpt)
    print(f"\n=== big run '{args.name}' done at step {trainer.global_step} ===", flush=True)
    model.save_to(str(run / f"{args.name}_final.nemo"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
