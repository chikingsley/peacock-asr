# Track 05 Runbook (Narrow Scope)

This runbook now has two completed phases on one frozen backend (`xlsr-espeak`):
- Phase 1 baseline trio:
`A1` scalar GOP + poly, `A2` GOP features + SVR, `A3` GOP features + GOPT.
- Phase 2 scalar logit ablation:
`B1` gop_sf, `B2` logit_margin, `B3-B5` logit_combined alpha sweep.
- Phase 2b dense cached alpha sweep:
`alpha=0.00..1.00` step `0.05` from cached scalar variants.

Configs:
- [`runs/track05_phase1_baseline.yaml`](/home/simon/github/peacock-asr/runs/track05_phase1_baseline.yaml)
- [`runs/track05_phase2_logit_scalar.yaml`](/home/simon/github/peacock-asr/runs/track05_phase2_logit_scalar.yaml)

## 1) Validate Configs Load

```bash
uv run python -c "from pathlib import Path; from peacock_asr.batch_config import load_batch_spec; print(load_batch_spec(Path('runs/track05_phase1_baseline.yaml')).name)"
uv run python -c "from pathlib import Path; from peacock_asr.batch_config import load_batch_spec; print(load_batch_spec(Path('runs/track05_phase2_logit_scalar.yaml')).name)"
```

Expected output:
- `track05_phase1_baseline`
- `track05_phase2_logit_scalar`

## 2) Execute Phase-1 Baseline Batch

```bash
uv run peacock-asr batch --config runs/track05_phase1_baseline.yaml --output-dir runs
```

## 3) Execute Phase-2 Scalar Logit Batch

```bash
MLFLOW_TRACKING_URI=https://mlflow.peacockery.studio \
MLFLOW_EXPERIMENT_NAME=peacock-asr-track05 \
uv run peacock-asr batch --config runs/track05_phase2_logit_scalar.yaml --output-dir runs
```

## 4) Output Artifacts

Each batch writes:
- `runs/<timestamp>_<batch_name>/summary.tsv`
- `runs/<timestamp>_<batch_name>/aggregates.tsv`
- per-run logs in the same directory.

Latest completed folders:
- `runs/2026-03-03_001037_track05_phase1_baseline/`
- `runs/2026-03-03_045426_track05_phase2_logit_scalar/`
- `runs/2026-03-03_080157_alpha_sweep_xlsr-espeak__wav2vec2-xlsr-53-espeak-cv-ft/`

## 5) Execute Dense Cached Alpha Sweep (Phase 2b)

```bash
uv run peacock-asr sweep-alpha \
  --backend xlsr-espeak \
  --alpha-start 0.0 \
  --alpha-stop 1.0 \
  --alpha-step 0.05 \
  --output-dir runs
```

Expected best point for current stack:
- `alpha=0.25`
- `PCC=0.3452`
- `MSE=0.5981`

## 6) Optional Fast Smoke Test (Not for Claims)

```bash
uv run peacock-asr batch --config runs/track05_phase2_logit_scalar.yaml --limit 50 --output-dir runs
```
