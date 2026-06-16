"""Decode-trust check: is the stuck ~0.95 WER a DECODE artifact or real under-learning?

Overfit a TINY set (~32 rows) HARD in fp32 (high LR, many steps) so train loss -> near 0, then
`transcribe()` those exact clips and print hyp vs ref. Decisive:
  - loss -> low AND hyp ≈ ref  => decode is FINE; the earlier "stuck WER" was slow transducer
    learning on 400 cold rows (path = real data + more steps).
  - loss -> low BUT hyp = garbage => decode is BROKEN (TDT decode-after-change_vocabulary) =>
    fix decode (cfg / patch) before any big run.

  uv run --project /home/simon/github/peacock-asr/projects/farsi-asr \
    python projects/tajik-asr/experiments/tdt/gate_decode_check.py
"""

from __future__ import annotations

import json
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
TRAIN = HERE / "data/train.jsonl"
N = 32


def main() -> int:
    rows = [json.loads(l) for l in TRAIN.read_text().splitlines()][:N]
    tiny = HERE / "data/tiny32.jsonl"
    tiny.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")

    model = ASRModel.restore_from(str(HYBRID))
    model.change_vocabulary(new_tokenizer_dir=str(TOK), new_tokenizer_type="bpe")
    nc = model.joint.num_classes_with_blank - 1 - model.joint.num_extra_outputs
    loss_name, loss_kwargs = model.extract_rnnt_loss_cfg(model.cfg.get("loss", None))
    model.loss = RNNTLoss(num_classes=nc, loss_name=loss_name, loss_kwargs=loss_kwargs)
    model.joint.set_fused_batch_size(4)
    model.joint.set_fuse_loss_wer(fuse_loss_wer=True, loss=model.loss, metric=model.wer)

    trainer = pl.Trainer(devices=1, accelerator="gpu", precision=32, max_steps=1500,
                         check_val_every_n_epoch=10**9, num_sanity_val_steps=0,
                         gradient_clip_val=1.0, enable_checkpointing=False, logger=False,
                         enable_progress_bar=False)
    model.set_trainer(trainer)
    ds = {"manifest_filepath": str(tiny), "sample_rate": 16000, "batch_size": 8,
          "shuffle": True, "num_workers": 4, "min_duration": 0.1, "max_duration": 16.0}
    cfg = OmegaConf.create({"model": {"train_ds": ds, "optim": {
        "name": "adamw", "lr": 1e-3, "betas": [0.9, 0.98], "weight_decay": 0.0,
        "sched": {"name": "CosineAnnealing", "warmup_steps": 50, "min_lr": 1e-5}}}})
    rc = model_utils.convert_model_config_to_dict_config(cfg)
    model.setup_training_data(rc.model.train_ds)
    model.setup_optimization(rc.model.optim)
    print(f"=== overfitting {len(rows)} rows hard (fp32, lr 1e-3, 1500 steps) ===", flush=True)
    trainer.fit(model)
    print(f"final train_loss ~ {float(trainer.callback_metrics.get('train_loss', float('nan'))):.3f}",
          flush=True)

    # decode the exact training clips and compare
    from omni_curator.process import normalize
    from omni_finetune_core.metrics import compute_measures
    paths = [r["audio_filepath"] for r in rows]
    refs = [r["text"] for r in rows]
    model.eval()
    hyps = model.transcribe(paths, batch_size=8)
    hyps = [h.text if hasattr(h, "text") else h for h in hyps]  # NeMo Hypothesis or str
    print("\n=== hyp vs ref on memorized training clips ===", flush=True)
    for h, r in list(zip(hyps, refs, strict=False))[:6]:
        print(f"  REF: {r[:60]}", flush=True)
        print(f"  HYP: {h[:60]}\n", flush=True)
    wer = compute_measures([normalize(r, "tgk_Cyrl") for r in refs],
                           [normalize(h, "tgk_Cyrl") for h in hyps]).wer
    print(f"=== TRAIN WER on the 32 memorized clips: {wer:.2f}%  "
          f"({'DECODE OK (was under-learning)' if wer < 50 else 'DECODE BROKEN or not memorized'}) ===",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
