# P001 Experiment Path Compatibility

Canonical project paths:

- `projects/P001-gop-baselines/experiments/track05_phase1_baseline.yaml`
- `projects/P001-gop-baselines/experiments/track05_phase2_logit_scalar.yaml`
- `projects/P001-gop-baselines/experiments/legacy/configs/track05_phase1_baseline.yaml`
- `projects/P001-gop-baselines/experiments/legacy/configs/track05_phase2_logit_scalar.yaml`

## Migration rule

- New docs and scripts should reference canonical project paths first.
- Root-level `runs/` paths are no longer part of the repo contract.
- Historical configs and outputs live under `experiments/legacy/`.

## Current command examples

Canonical project path:

```bash
uv run --project projects/P001-gop-baselines peacock-asr batch --config projects/P001-gop-baselines/experiments/track05_phase1_baseline.yaml --output-dir projects/P001-gop-baselines/experiments/final/batches
```
