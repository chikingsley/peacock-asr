"""Overfit-32 gate for the extend-restore recipe (GPU — DO NOT run on CPU/while a run is live).

This is the *final* dip-a-toe gate from extend_restore_recipe.md, to run at swap time on a
free GPU. It loads v3, builds the extend-restore model (snapshot -> change_vocabulary to the
extended tokenizer -> restore decoder/joint -> loss-init fix), overfits 32 fixed clips HARD in
fp32, then transcribes those exact clips. PASS = train_loss collapses AND >=5/6 fixed clips
decode exact (WER alone is insufficient — print hyp vs ref).

Run (only when the GPU is free):
  uv run python projects/tajik-asr/experiments/tdt/gate_extend_restore.py
"""

from __future__ import annotations

import json
from pathlib import Path

import lightning.pytorch as pl
from nemo.collections.asr.models import ASRModel
from nemo.utils import model_utils
from omegaconf import OmegaConf
from parakeet_finetune_core.extend_restore import (
    restore_extended_decoder_joint,
    snapshot_decoder_joint,
)
from parakeet_finetune_core.tdt import apply_loss_init_fix, configure_fused_tdt_loss

HERE = Path(__file__).resolve().parent
PEACOCK = HERE.parents[3]
V3 = PEACOCK / "base_models/parakeet/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.nemo"
TOK = PEACOCK / "projects/tajik-asr/data/parakeet/tokenizers/tok_extend_v3"
TRAIN = PEACOCK / "projects/tajik-asr/data/parakeet/manifests/train_big.jsonl"
OLD_VOCAB = 8192
N = 32


def main() -> int:
    rows = [json.loads(line) for line in TRAIN.read_text().splitlines()[:N]]
    tiny = HERE / "data" / "tiny32_extend.jsonl"
    tiny.parent.mkdir(parents=True, exist_ok=True)
    tiny.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")

    model = ASRModel.restore_from(str(V3))
    snapshot = snapshot_decoder_joint(model)
    model.change_vocabulary(new_tokenizer_dir=str(TOK), new_tokenizer_type="bpe")
    info = restore_extended_decoder_joint(model, snapshot, old_vocab=OLD_VOCAB)
    print(f"restored: {info}", flush=True)
    apply_loss_init_fix(model)
    configure_fused_tdt_loss(model, fused_batch_size=4)

    trainer = pl.Trainer(
        devices=1,
        accelerator="gpu",
        precision=32,
        max_steps=1500,
        check_val_every_n_epoch=10**9,
        num_sanity_val_steps=0,
        gradient_clip_val=1.0,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
    )
    model.set_trainer(trainer)
    ds = {
        "manifest_filepath": str(tiny),
        "sample_rate": 16000,
        "batch_size": 8,
        "shuffle": True,
        "num_workers": 4,
        "min_duration": 0.1,
        "max_duration": 16.0,
    }
    cfg = OmegaConf.create(
        {
            "model": {
                "train_ds": ds,
                "optim": {
                    "name": "adamw",
                    "lr": 1e-3,
                    "betas": [0.9, 0.98],
                    "weight_decay": 0.0,
                    "sched": {"name": "CosineAnnealing", "warmup_steps": 50, "min_lr": 1e-5},
                },
            }
        }
    )
    resolved = model_utils.convert_model_config_to_dict_config(cfg)
    model.setup_training_data(resolved.model.train_ds)
    model.setup_optimization(resolved.model.optim)
    print("=== overfitting 32 rows hard (fp32, lr 1e-3, 1500 steps) ===", flush=True)
    trainer.fit(model)
    final_loss = float(trainer.callback_metrics.get("train_loss", float("nan")))
    print(f"final train_loss ~ {final_loss:.3f}", flush=True)

    from omni_curator.process import normalize
    from omni_finetune_core.metrics import compute_measures

    paths = [r["audio_filepath"] for r in rows]
    refs = [r["text"] for r in rows]
    model.eval()
    hyps = model.transcribe(paths, batch_size=8)
    hyps = [h.text if hasattr(h, "text") else h for h in hyps]
    exact = 0
    print("\n=== hyp vs ref on memorized clips (first 6) ===", flush=True)
    for i, (h, r) in enumerate(zip(hyps, refs, strict=False)):
        if normalize(h, "tgk_Cyrl") == normalize(r, "tgk_Cyrl"):
            exact += 1
        if i < 6:
            print(f"  REF: {r[:60]}\n  HYP: {h[:60]}\n", flush=True)
    wer = compute_measures(
        [normalize(r, "tgk_Cyrl") for r in refs],
        [normalize(h, "tgk_Cyrl") for h in hyps],
    ).wer
    print(f"exact decode: {exact}/{len(refs)}  train WER: {wer:.2f}%", flush=True)
    overfit_ok = final_loss < 5.0 and exact >= 5
    print(f"=== OVERFIT-32 GATE: {'PASS' if overfit_ok else 'FAIL'} ===", flush=True)
    return 0 if overfit_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
