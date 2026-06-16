"""TDT fine-tune ablation harness (gate 3): does fixing the loss-init bug make the TDT head train?

Arms (cumulative; loss-init fix is in every *trainable* arm — A0 is the broken control):
  A0  stock change_vocabulary (buggy RNNTLoss num_classes) — expect TDT stall
  B   + loss-init fix (RNNTLoss num_classes = num_classes_with_blank-1 - num_extra_outputs)
  C1  + extend-tokenizer (TODO)        C2  + weight-restore (TODO)        D  + encoder-freeze (TODO)

Logs TDT-head (val_wer) and CTC-head (val_wer_ctc) separately. 110M-hybrid first (cheap, has a
CTC head as a sanity floor), then the winning arm on v3.

  uv run --project /home/simon/github/peacock-asr/projects/tajik-asr \
    tajik-parakeet-train-tdt --name gate-b --max-steps 2000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lightning.pytorch as pl
from nemo.collections.asr.losses.rnnt import RNNTLoss
from nemo.collections.asr.models import ASRModel
from nemo.utils import model_utils
from omegaconf import OmegaConf

HERE = Path(__file__).resolve().parent
PEACOCK = HERE.parents[3]
HYBRID = PEACOCK / "projects/farsi-asr/src/finetune_parakeet/models/parakeet-tdt_ctc-110m-base-hybrid.nemo"
TOK = HERE / "data/tok/tokenizer_spe_bpe_v1024"
TRAIN, DEV = HERE / "data/train.jsonl", HERE / "data/dev.jsonl"


class HeadWERPrinter(pl.Callback):
    def on_validation_epoch_end(self, trainer: pl.Trainer, _pl: pl.LightningModule) -> None:
        m = trainer.callback_metrics
        def g(k: str) -> float:
            v = m.get(k)
            return float(v) if v is not None else float("nan")
        loss = next((g(k) for k in ("train_loss", "loss", "train_backward_loss")
                     if not (g(k) != g(k))), float("nan"))  # first non-nan
        print(f"  [val @ step {trainer.global_step}]  train_loss={loss:.3f}  "
              f"TDT val_wer={g('val_wer'):.4f}  CTC val_wer_ctc={g('val_wer_ctc'):.4f}", flush=True)


def ds(manifest: Path, bs: int, max_dur: float, *, shuffle: bool) -> dict:
    return {"manifest_filepath": str(manifest), "sample_rate": 16000, "batch_size": bs,
            "shuffle": shuffle, "num_workers": 4, "pin_memory": True,
            "min_duration": 0.1, "max_duration": max_dur}


def apply_loss_init_fix(model) -> None:  # noqa: ANN001  — arm B: mirror the constructor's TDT branch
    nc = model.joint.num_classes_with_blank - 1
    if model.joint.num_extra_outputs > 0:
        nc -= model.joint.num_extra_outputs
    loss_name, loss_kwargs = model.extract_rnnt_loss_cfg(model.cfg.get("loss", None))
    model.loss = RNNTLoss(num_classes=nc, loss_name=loss_name, loss_kwargs=loss_kwargs)
    print(f"  [arm B] rebuilt RNNTLoss num_classes={nc} "
          f"(was {model.joint.num_classes_with_blank - 1}; -{model.joint.num_extra_outputs} durations)",
          flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="B", choices=["A0", "B"])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--model-name", default=str(HYBRID),
                    help="path to .nemo, or HF id e.g. nvidia/parakeet-tdt-0.6b-v3")
    ap.add_argument("--overfit", action="store_true", help="validate on TRAIN (memorization gate)")
    ap.add_argument("--max-dur", type=float, default=16.0)
    ap.add_argument("--precision", default="bf16", help="bf16 | 32 (fp32 fixes the TDT bf16-decode bug)")
    ap.add_argument("--freeze-encoder", action="store_true", help="freeze whole encoder")
    ap.add_argument("--unfreeze-top", type=int, default=0,
                    help="freeze encoder but unfreeze top-N FastConformer layers (fits 0.6B on 12GB)")
    args = ap.parse_args()

    print(f"=== arm {args.arm}  steps {args.steps}  model {args.model_name} ===", flush=True)
    if Path(args.model_name).exists():
        model = ASRModel.restore_from(args.model_name)
    else:
        model = ASRModel.from_pretrained(args.model_name)  # cached from gate 0b
    print(f"loaded {type(model).__name__}  num_extra_outputs={model.joint.num_extra_outputs}",
          flush=True)
    model.change_vocabulary(new_tokenizer_dir=str(TOK), new_tokenizer_type="bpe")  # stock (buggy)
    if args.arm != "A0":
        apply_loss_init_fix(model)
    # transducer joint memory fix (compute loss/WER in fused chunks)
    model.joint.set_fused_batch_size(min(4, args.bs))
    model.joint.set_fuse_loss_wer(fuse_loss_wer=True, loss=model.loss, metric=model.wer)
    if args.freeze_encoder or args.unfreeze_top > 0:
        model.encoder.freeze()
        msg = "encoder FROZEN (decoder+joint only)"
        if args.unfreeze_top > 0:
            for layer in model.encoder.layers[-args.unfreeze_top:]:
                for p in layer.parameters():
                    p.requires_grad_(requires_grad=True)
                layer.train()
            n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
            msg = (f"encoder: top {args.unfreeze_top}/{len(model.encoder.layers)} layers unfrozen "
                   f"+ decoder/joint  ({n_tr / 1e6:.0f}M trainable)")
        print(f"  {msg}", flush=True)

    trainer = pl.Trainer(
        devices=1, accelerator="gpu", precision=args.precision,
        max_steps=args.steps, check_val_every_n_epoch=5, num_sanity_val_steps=0,
        gradient_clip_val=1.0, log_every_n_steps=50,
        enable_checkpointing=False, logger=False, enable_progress_bar=False,
        callbacks=[HeadWERPrinter()],
    )
    model.set_trainer(trainer)
    cfg = OmegaConf.create({"model": {
        "train_ds": ds(TRAIN, args.bs, args.max_dur, shuffle=True),
        "validation_ds": ds(TRAIN if args.overfit else DEV, args.bs, args.max_dur, shuffle=False),
        "optim": {"name": "adamw", "lr": 1e-4, "betas": [0.9, 0.98], "weight_decay": 1e-3,
                  "sched": {"name": "CosineAnnealing", "warmup_steps": 100, "min_lr": 1e-6}},
        "spec_augment": {"_target_": "nemo.collections.asr.modules.SpectrogramAugmentation",
                         "freq_masks": 2, "time_masks": 10, "freq_width": 27, "time_width": 0.05},
    }})
    rc = model_utils.convert_model_config_to_dict_config(cfg)
    model.setup_training_data(rc.model.train_ds)
    model.setup_multiple_validation_data(rc.model.validation_ds)
    model.setup_optimization(rc.model.optim)
    model.spec_augment = ASRModel.from_config_dict(rc.model.spec_augment)
    print(f"\n=== training arm {args.arm} ===", flush=True)
    trainer.fit(model)
    print(f"\n=== arm {args.arm} done ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
