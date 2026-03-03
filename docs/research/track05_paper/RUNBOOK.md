# Track 05 Runbook (Narrow Scope)

This runbook executes the baseline ablation trio on one frozen backend (`xlsr-espeak`):
- `A1`: scalar GOP + polynomial regression
- `A2`: GOP features + SVR
- `A3`: GOP features + GOPT (5 seeds)

Config:
- [`runs/track05_phase1_baseline.yaml`](/home/simon/github/peacock-asr/runs/track05_phase1_baseline.yaml)

## 1) Validate Config Loads

```bash
uv run python -c "from pathlib import Path; from peacock_asr.batch_config import load_batch_spec; print(load_batch_spec(Path('runs/track05_phase1_baseline.yaml')).name)"
```

Expected output:
- `track05_phase1_baseline`

## 2) Execute Phase-1 Baseline Batch

```bash
uv run peacock-asr batch --config runs/track05_phase1_baseline.yaml --output-dir runs
```

Artifacts are written to:
- `runs/<timestamp>_track05_phase1_baseline/summary.tsv`
- `runs/<timestamp>_track05_phase1_baseline/aggregates.tsv`
- per-run logs in the same directory.

## 3) Optional Fast Smoke Test (Tiny Subset)

```bash
uv run peacock-asr batch --config runs/track05_phase1_baseline.yaml --limit 50 --output-dir runs
```

Use this only to verify plumbing quickly; do not use for paper claims.

