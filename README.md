# Peacock-ASR

Pronunciation assessment research repo centered on CTC phoneme posteriors,
segmentation-free GOP, and downstream scoring on SpeechOcean762.

## Current Status

As of March 5, 2026:

- `P001` established the main GOP baseline: `xlsr-espeak + GOPT` reached
  `0.6774 +/- 0.0127` phone-level PCC on SpeechOcean762.
- `P002` showed that ConPCO-style loss alone adds little on the 42-d GOP-SF
  stack; the main remaining question is whether richer features justify
  continuing that line.
- `P003` is the active next paper: `wav2vec2-base` reached
  `0.640 +/- 0.009` PCC with 3.3x fewer backbone parameters than the 300M
  baseline.

## Start Here

- Project registry: [projects/INDEX.yaml](/home/simon/github/peacock-asr/projects/INDEX.yaml)
- Program board: [projects/PROGRAM_BOARD.md](/home/simon/github/peacock-asr/projects/PROGRAM_BOARD.md)
- Research map: [docs/research/META_PLAN.md](/home/simon/github/peacock-asr/docs/research/META_PLAN.md)

## Active Projects

- `P001` GOP baselines:
  [docs](/home/simon/github/peacock-asr/projects/P001-gop-baselines/docs/README.md)
- `P002` ConPCO scoring:
  [docs](/home/simon/github/peacock-asr/projects/P002-conpco-scoring/docs/README.md)
- `P003` compact backbones:
  [docs](/home/simon/github/peacock-asr/projects/P003-compact-backbones/docs/README.md)

Planning/incubation workspaces:

- `P004` training from scratch
- `P005` LLM pronunciation
- `P006` realtime streaming
- `P008` CAPT runtime
- `P009` multilingual data map

## Common Commands

```bash
uv run --project projects/P001-gop-baselines peacock-asr --help
uv run --project projects/P001-gop-baselines peacock-asr batch --config projects/P001-gop-baselines/experiments/track05_phase1_baseline.yaml --output-dir artifacts/batches
uv run --project projects/P003-compact-backbones wandb sweep projects/P003-compact-backbones/experiments/sweeps/final/eval_wav2vec2_base.yaml
```

## Repo Shape

- `projects/P###-*/` paper-grade project workspaces
- `projects/P###-*/pyproject.toml` isolated per-project environments
- `projects/P###-*/.env.example` checked-in env template for secrets and W&B/HF defaults
- `projects/P###-*/scripts/push_env.py` project-local helper to copy `.env` onto a remote machine
- `projects/P###-*/code/` project-owned code, even when duplicated across projects
- `projects/P003-compact-backbones/code/training/` project-local phoneme-head training, preprocessing, and benchmark modules
- `docs/research/` lab notebook, decisions, and archived track narratives
- `docs/papers/` mirrored papers and extracted markdown
