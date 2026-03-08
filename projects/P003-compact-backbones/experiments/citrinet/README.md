# Citrinet Experiments

This directory holds Phase 2 experiment artifacts only.

Current subdirectories:

- `data/`
  exported NeMo manifests plus copied audio for real P2-B runs
- `checkpoints/`
  Citrinet-only checkpoints and `.nemo` exports
- `sweeps/`
  Citrinet-only training/eval sweeps
- `logs/`
  run logs
- `reports/`
  stock-feasibility notes, benchmark summaries, and result tables

This keeps Citrinet artifacts out of the Phase 1 backbone comparison paths.

Current real-run dataset:

- `train_clean_100_full`
  - `28,538` train utterances
  - `98.595` hours
  - `2,703` eval utterances
  - `5.133` hours
  - artifact root:
    [data/train_clean_100_full](/home/simon/github/peacock-asr/projects/P003-compact-backbones/experiments/citrinet/data/train_clean_100_full)

Canonical commands:

```bash
/home/simon/github/peacock-asr/projects/P003-compact-backbones/env/citrinet/.venv/bin/python \
  /home/simon/github/peacock-asr/projects/P003-compact-backbones/code/citrinet/scripts/build_p2b_assets.py \
  --train-samples 0 \
  --eval-samples 0 \
  --output-root /home/simon/github/peacock-asr/projects/P003-compact-backbones/experiments/citrinet/data/train_clean_100_full
```

```bash
/home/simon/github/peacock-asr/projects/P003-compact-backbones/env/citrinet/.venv/bin/python \
  /home/simon/github/peacock-asr/projects/P003-compact-backbones/code/citrinet/scripts/train_citrinet_p2b.py \
  --train-manifest /home/simon/github/peacock-asr/projects/P003-compact-backbones/experiments/citrinet/data/train_clean_100_full/manifests/train.jsonl \
  --eval-manifest /home/simon/github/peacock-asr/projects/P003-compact-backbones/experiments/citrinet/data/train_clean_100_full/manifests/eval.jsonl \
  --output-dir /home/simon/github/peacock-asr/projects/P003-compact-backbones/experiments/citrinet/checkpoints/citrinet_256_p2b_train_clean_100 \
  --run-name citrinet-256-p2b-train-clean-100-v1
```
