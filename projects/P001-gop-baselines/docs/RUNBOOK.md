# P001 Runbook (GOP Baselines)

Moved from legacy path: `docs/research/track05_paper/RUNBOOK.md`.

This runbook now has two completed phases on one frozen backend (`xlsr-espeak`):
- Phase 1 baseline trio:
`A1` scalar GOP + poly, `A2` GOP features + SVR, `A3` GOP features + GOPT.
- Phase 2 scalar logit ablation:
`B1` gop_sf, `B2` logit_margin, `B3-B5` logit_combined alpha sweep.
- Phase 2b dense cached alpha sweep:
`alpha=0.00..1.00` step `0.05` from cached scalar variants.

Canonical project configs:
- [`projects/P001-gop-baselines/experiments/track05_phase1_baseline.yaml`](/home/simon/github/peacock-asr/projects/P001-gop-baselines/experiments/track05_phase1_baseline.yaml)
- [`projects/P001-gop-baselines/experiments/track05_phase2_logit_scalar.yaml`](/home/simon/github/peacock-asr/projects/P001-gop-baselines/experiments/track05_phase2_logit_scalar.yaml)
- Historical archived configs:
  - [`track05_phase1_baseline.yaml`](/home/simon/github/peacock-asr/projects/P001-gop-baselines/experiments/legacy/configs/track05_phase1_baseline.yaml)
  - [`track05_phase2_logit_scalar.yaml`](/home/simon/github/peacock-asr/projects/P001-gop-baselines/experiments/legacy/configs/track05_phase2_logit_scalar.yaml)

## 1) Validate Configs Load

```bash
uv run --project projects/P001-gop-baselines python -c "from pathlib import Path; from p001_gop.batch_config import load_batch_spec; print(load_batch_spec(Path('projects/P001-gop-baselines/experiments/track05_phase1_baseline.yaml')).name)"
uv run --project projects/P001-gop-baselines python -c "from pathlib import Path; from p001_gop.batch_config import load_batch_spec; print(load_batch_spec(Path('projects/P001-gop-baselines/experiments/track05_phase2_logit_scalar.yaml')).name)"
```

Expected output:
- `track05_phase1_baseline`
- `track05_phase2_logit_scalar`

## 2) Execute Phase-1 Baseline Batch

```bash
uv run --project projects/P001-gop-baselines peacock-asr batch \
  --config projects/P001-gop-baselines/experiments/track05_phase1_baseline.yaml \
  --output-dir projects/P001-gop-baselines/experiments/final/batches
```

## 3) Execute Phase-2 Scalar Logit Batch

```bash
MLFLOW_TRACKING_URI=https://mlflow.peacockery.studio \
MLFLOW_EXPERIMENT_NAME=peacock-asr-track05 \
uv run --project projects/P001-gop-baselines peacock-asr batch \
  --config projects/P001-gop-baselines/experiments/track05_phase2_logit_scalar.yaml \
  --output-dir projects/P001-gop-baselines/experiments/final/batches
```

## 4) Output Artifacts

The canonical destination for new `P001` reruns is:
- `projects/P001-gop-baselines/experiments/final/batches/`
- `projects/P001-gop-baselines/experiments/final/results/`

New canonical reruns write:
- `projects/P001-gop-baselines/experiments/final/batches/<timestamp>_<batch_name>/summary.tsv`
- `projects/P001-gop-baselines/experiments/final/batches/<timestamp>_<batch_name>/aggregates.tsv`
- per-run logs in the same directory
- campaign-wide summaries in:
  - [`aggregate_summary.tsv`](/home/simon/github/peacock-asr/projects/P001-gop-baselines/experiments/final/results/aggregate_summary.tsv)
  - [`per_run_summary.tsv`](/home/simon/github/peacock-asr/projects/P001-gop-baselines/experiments/final/results/per_run_summary.tsv)
  - [`alpha_best.tsv`](/home/simon/github/peacock-asr/projects/P001-gop-baselines/experiments/final/results/alpha_best.tsv)

Historical archived folders:
- `projects/P001-gop-baselines/experiments/legacy/batches/2026-03-03_001037_track05_phase1_baseline/`
- `projects/P001-gop-baselines/experiments/legacy/batches/2026-03-03_045426_track05_phase2_logit_scalar/`
- `projects/P001-gop-baselines/experiments/legacy/alpha_sweeps/2026-03-03_080157_alpha_sweep_xlsr-espeak__wav2vec2-xlsr-53-espeak-cv-ft/`

## 5) Execute Dense Cached Alpha Sweep (Phase 2b)

```bash
uv run --project projects/P001-gop-baselines peacock-asr sweep-alpha \
  --backend xlsr-espeak \
  --alpha-start 0.0 \
  --alpha-stop 1.0 \
  --alpha-step 0.05 \
  --output-dir projects/P001-gop-baselines/experiments/final/alpha_sweeps
```

Expected best point for current stack:
- `alpha=0.25`
- `PCC=0.3452`
- `MSE=0.5981`

## 6) Optional Fast Smoke Test (Not for Claims)

```bash
uv run --project projects/P001-gop-baselines peacock-asr batch \
  --config projects/P001-gop-baselines/experiments/track05_phase2_logit_scalar.yaml \
  --limit 50 \
  --output-dir projects/P001-gop-baselines/experiments/final/batches
```

## 7) Final P001 Campaign (W&B-First Sweeps)

Canonical W&B project for final-paper runs:
- `peacockery/peacock-asr-p001-gop-baselines`

Canonical naming, metadata, and retention policy:
- [`projects/P001-gop-baselines/docs/FINAL_CAMPAIGN_SPEC.md`](/home/simon/github/peacock-asr/projects/P001-gop-baselines/docs/FINAL_CAMPAIGN_SPEC.md)

Preferred campaign launcher:

```bash
uv run --project projects/P001-gop-baselines python projects/P001-gop-baselines/experiments/final/launch_p001_paper_close.py
```

Final campaign status:
- completed on 2026-03-06
- final aggregate summary:
  [`aggregate_summary.tsv`](/home/simon/github/peacock-asr/projects/P001-gop-baselines/experiments/final/results/aggregate_summary.tsv)
- dense alpha best points:
  [`alpha_best.tsv`](/home/simon/github/peacock-asr/projects/P001-gop-baselines/experiments/final/results/alpha_best.tsv)

Useful variants:

```bash
uv run --project projects/P001-gop-baselines python projects/P001-gop-baselines/experiments/final/launch_p001_paper_close.py --phase phase1
uv run --project projects/P001-gop-baselines python projects/P001-gop-baselines/experiments/final/launch_p001_paper_close.py --phase phase2
uv run --project projects/P001-gop-baselines python projects/P001-gop-baselines/experiments/final/launch_p001_paper_close.py --phase phase2b
uv run --project projects/P001-gop-baselines python projects/P001-gop-baselines/experiments/final/launch_p001_paper_close.py --phase all --start-at phase2_original_b3_logit_combined_a025
```

Sweep specs:
- [`projects/P001-gop-baselines/experiments/sweeps/final/README.md`](/home/simon/github/peacock-asr/projects/P001-gop-baselines/experiments/sweeps/final/README.md)
- [`projects/P001-gop-baselines/experiments/sweeps/final/phase1_original_a1_scalar.yaml`](/home/simon/github/peacock-asr/projects/P001-gop-baselines/experiments/sweeps/final/phase1_original_a1_scalar.yaml)
- [`projects/P001-gop-baselines/experiments/sweeps/final/phase1_original_a2_feats.yaml`](/home/simon/github/peacock-asr/projects/P001-gop-baselines/experiments/sweeps/final/phase1_original_a2_feats.yaml)
- [`projects/P001-gop-baselines/experiments/sweeps/final/phase1_original_a3_gopt.yaml`](/home/simon/github/peacock-asr/projects/P001-gop-baselines/experiments/sweeps/final/phase1_original_a3_gopt.yaml)
- [`projects/P001-gop-baselines/experiments/sweeps/final/phase1_xlsr_a1_scalar.yaml`](/home/simon/github/peacock-asr/projects/P001-gop-baselines/experiments/sweeps/final/phase1_xlsr_a1_scalar.yaml)
- [`projects/P001-gop-baselines/experiments/sweeps/final/phase1_xlsr_a2_feats.yaml`](/home/simon/github/peacock-asr/projects/P001-gop-baselines/experiments/sweeps/final/phase1_xlsr_a2_feats.yaml)
- [`projects/P001-gop-baselines/experiments/sweeps/final/phase1_xlsr_a3_gopt.yaml`](/home/simon/github/peacock-asr/projects/P001-gop-baselines/experiments/sweeps/final/phase1_xlsr_a3_gopt.yaml)
- [`projects/P001-gop-baselines/experiments/sweeps/final/phase2_original_b1_gopsf.yaml`](/home/simon/github/peacock-asr/projects/P001-gop-baselines/experiments/sweeps/final/phase2_original_b1_gopsf.yaml)
- [`projects/P001-gop-baselines/experiments/sweeps/final/phase2_original_b2_logit_margin.yaml`](/home/simon/github/peacock-asr/projects/P001-gop-baselines/experiments/sweeps/final/phase2_original_b2_logit_margin.yaml)
- [`projects/P001-gop-baselines/experiments/sweeps/final/phase2_original_b3_logit_combined_a025.yaml`](/home/simon/github/peacock-asr/projects/P001-gop-baselines/experiments/sweeps/final/phase2_original_b3_logit_combined_a025.yaml)
- [`projects/P001-gop-baselines/experiments/sweeps/final/phase2_original_b4_logit_combined_a050.yaml`](/home/simon/github/peacock-asr/projects/P001-gop-baselines/experiments/sweeps/final/phase2_original_b4_logit_combined_a050.yaml)
- [`projects/P001-gop-baselines/experiments/sweeps/final/phase2_original_b5_logit_combined_a075.yaml`](/home/simon/github/peacock-asr/projects/P001-gop-baselines/experiments/sweeps/final/phase2_original_b5_logit_combined_a075.yaml)
- [`projects/P001-gop-baselines/experiments/sweeps/final/phase2_xlsr_b1_gopsf.yaml`](/home/simon/github/peacock-asr/projects/P001-gop-baselines/experiments/sweeps/final/phase2_xlsr_b1_gopsf.yaml)
- [`projects/P001-gop-baselines/experiments/sweeps/final/phase2_xlsr_b2_logit_margin.yaml`](/home/simon/github/peacock-asr/projects/P001-gop-baselines/experiments/sweeps/final/phase2_xlsr_b2_logit_margin.yaml)
- [`projects/P001-gop-baselines/experiments/sweeps/final/phase2_xlsr_b3_logit_combined_a025.yaml`](/home/simon/github/peacock-asr/projects/P001-gop-baselines/experiments/sweeps/final/phase2_xlsr_b3_logit_combined_a025.yaml)
- [`projects/P001-gop-baselines/experiments/sweeps/final/phase2_xlsr_b4_logit_combined_a050.yaml`](/home/simon/github/peacock-asr/projects/P001-gop-baselines/experiments/sweeps/final/phase2_xlsr_b4_logit_combined_a050.yaml)
- [`projects/P001-gop-baselines/experiments/sweeps/final/phase2_xlsr_b5_logit_combined_a075.yaml`](/home/simon/github/peacock-asr/projects/P001-gop-baselines/experiments/sweeps/final/phase2_xlsr_b5_logit_combined_a075.yaml)

Create a sweep:

```bash
uv run --project projects/P001-gop-baselines wandb sweep projects/P001-gop-baselines/experiments/sweeps/final/phase1_xlsr_a3_gopt.yaml
```

Start an agent:

```bash
mkdir -p projects/P001-gop-baselines/experiments/final/agents
nohup uv run --project projects/P001-gop-baselines wandb agent peacockery/peacock-asr-p001-gop-baselines/<SWEEP_ID> > projects/P001-gop-baselines/experiments/final/agents/sweep_<SWEEP_ID>.log 2>&1 &
```

Notes:
- Prefer the Python `uv` launcher above instead of running individual sweep
  commands by hand when you want the full campaign.
- These sweeps use `--limit=0` and do not pass `--no-cache`, so current cache behavior is preserved.
- Group/run naming is encoded per sweep through env vars in each YAML.
- The CLI now logs campaign metadata (`track`, `project_id`, `phase`, `job_id`,
  dataset revision, cache hits, feature dim, device, git SHA) into W&B config.
- `gopt` runs also log checkpoint directories as W&B model artifacts and write
  local checkpoint files under `projects/P001-gop-baselines/experiments/final/checkpoints/`.

## 8) Final P001 Phase-2b Dense Alpha Sweeps

Use the canonical `phase2b` commands from:
- [`projects/P001-gop-baselines/docs/FINAL_CAMPAIGN_SPEC.md`](/home/simon/github/peacock-asr/projects/P001-gop-baselines/docs/FINAL_CAMPAIGN_SPEC.md)

These runs now log:
- one W&B `analysis` run per backend
- a full alpha table
- a versioned analysis artifact containing `alpha_sweep.tsv` and
  `alpha_sweep_meta.json`

## 9) Prewarm the k2 topology cache

`P001` still defaults to the Python scalar backend, but when you want to run the
`k2` scalar path, prewarm the denominator topology cache once before the real
run:

```bash
uv run --project projects/P001-gop-baselines python -m p001_gop.cli \
  prewarm-k2 \
  --backend xlsr-espeak \
  --split both
```

This populates persistent topology files under:

```text
projects/P001-gop-baselines/.cache/k2_topologies/
```
