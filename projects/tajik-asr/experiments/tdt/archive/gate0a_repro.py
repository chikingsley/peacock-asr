"""Gate 0a — reproduce the TDT-head stagnation on the 110M hybrid (stock change_vocabulary).

Loads the 110M hybrid (TDT+CTC), stock `change_vocabulary` to a fresh Tajik BPE-1024, and
fine-tunes briefly on FLEURS-Tajik, logging the TDT-head WER (`val_wer`) and CTC-head WER
(`val_wer_ctc`) SEPARATELY each validation. Expected (issue #14140): CTC drops while TDT
stalls high. The loss-init bug (change_vocabulary omits `- num_extra_outputs`) is already
confirmed by source inspection; this shows its training effect.

  uv run --project /home/simon/github/peacock-asr/projects/farsi-asr \
    python projects/tajik-asr/experiments/tdt/gate0a_repro.py
"""

from __future__ import annotations

from pathlib import Path

import lightning.pytorch as pl
from nemo.collections.asr.models import ASRModel
from nemo.utils import model_utils
from omegaconf import OmegaConf

HERE = Path(__file__).resolve().parent
PEACOCK = HERE.parents[3]
HYBRID = PEACOCK / "projects/farsi-asr/src/finetune_parakeet/models/parakeet-tdt_ctc-110m-base-hybrid.nemo"
TOK = HERE / "data/tok/tokenizer_spe_bpe_v1024"
TRAIN = HERE / "data/train.jsonl"
DEV = HERE / "data/dev.jsonl"


class HeadWERPrinter(pl.Callback):
    """Print TDT-head (val_wer) and CTC-head (val_wer_ctc) separately each validation."""

    def on_validation_epoch_end(self, trainer: pl.Trainer, _pl_module: pl.LightningModule) -> None:
        m = trainer.callback_metrics
        def g(k: str) -> float:
            v = m.get(k)
            return float(v) if v is not None else float("nan")
        print(f"  [val @ step {trainer.global_step}]  TDT-head val_wer = {g('val_wer'):.4f}   "
              f"CTC-head val_wer_ctc = {g('val_wer_ctc'):.4f}", flush=True)


def ds(manifest: Path, bs: int, *, shuffle: bool) -> dict:
    return {
        "manifest_filepath": str(manifest), "sample_rate": 16000, "batch_size": bs,
        "shuffle": shuffle, "num_workers": 4, "pin_memory": True,
        "min_duration": 0.1, "max_duration": 16.0,
    }


def main() -> int:
    print(f"hybrid: {HYBRID.name}\ntokenizer: {TOK}\n", flush=True)
    model = ASRModel.restore_from(str(HYBRID))
    print(f"loaded {type(model).__name__}", flush=True)
    model.change_vocabulary(new_tokenizer_dir=str(TOK), new_tokenizer_type="bpe")  # STOCK (buggy for TDT)
    # transducer joint (B×T×U×V) is memory-heavy on 12 GB → compute loss/WER in small fused chunks
    model.cfg.joint.fused_batch_size = 4
    model.joint.set_fused_batch_size(4)
    model.joint.set_fuse_loss_wer(fuse_loss_wer=True, loss=model.loss, metric=model.wer)

    trainer = pl.Trainer(
        devices=1, accelerator="gpu", precision="bf16",
        max_steps=400, val_check_interval=1.0, num_sanity_val_steps=0,
        accumulate_grad_batches=1, gradient_clip_val=0.0, log_every_n_steps=25,
        enable_checkpointing=False, logger=False, enable_progress_bar=False,
        callbacks=[HeadWERPrinter()],
    )
    model.set_trainer(trainer)
    cfg = OmegaConf.create({"model": {
        "train_ds": ds(TRAIN, 8, shuffle=True),
        "validation_ds": ds(DEV, 8, shuffle=False),
        "optim": {"name": "adamw", "lr": 1e-4, "betas": [0.9, 0.98], "weight_decay": 1e-3,
                  "sched": {"name": "CosineAnnealing", "warmup_steps": 50, "min_lr": 1e-6}},
        "spec_augment": {"_target_": "nemo.collections.asr.modules.SpectrogramAugmentation",
                         "freq_masks": 2, "time_masks": 10, "freq_width": 27, "time_width": 0.05},
    }})
    rc = model_utils.convert_model_config_to_dict_config(cfg)
    model.setup_training_data(rc.model.train_ds)
    model.setup_multiple_validation_data(rc.model.validation_ds)
    model.setup_optimization(rc.model.optim)
    model.spec_augment = ASRModel.from_config_dict(rc.model.spec_augment)
    print("\n=== training (stock change_vocabulary; watch TDT vs CTC) ===", flush=True)
    trainer.fit(model)
    print("\n=== gate 0a done — TDT stalls high while CTC drops => bug reproduced ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
