# P003 Final Sweep Specs

Canonical compact-backbone sweep specs:

- `train_wav2vec2_base.yaml`
- `train_wav2vec2_large.yaml`
- `eval_wav2vec2_base.yaml`
- `eval_wav2vec2_large.yaml`
- `train_hubert_base.yaml`
- `eval_hubert_base.yaml`

Migration rule:

- New docs should reference these project-local specs first.
- Root-level compatibility sweep copies have been removed.

Current focus:

- `wav2vec2-base` is the first completed compact-backbone point.
- `HuBERT-base` is staged to follow the same 95M recipe and writes checkpoints
  under `projects/P003-compact-backbones/experiments/checkpoints/`.
- `wav2vec2-large` is the remaining Phase 1 size-control backbone.
