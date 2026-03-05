# P001 Experiment Path Compatibility

Canonical project paths:

- `projects/P001-gop-baselines/experiments/track05_phase1_baseline.yaml`
- `projects/P001-gop-baselines/experiments/track05_phase2_logit_scalar.yaml`

Legacy compatibility paths (still valid during migration):

- `runs/track05_phase1_baseline.yaml`
- `runs/track05_phase2_logit_scalar.yaml`

## Migration rule

- New docs and scripts should reference canonical project paths first.
- Existing scripts that use legacy `runs/` paths are allowed until cutover.
- Do not delete legacy files until cutover is explicitly approved.

## Current command examples

Legacy path:

```bash
uv run peacock-asr batch --config runs/track05_phase1_baseline.yaml --output-dir runs
```

Canonical project path:

```bash
uv run peacock-asr batch --config projects/P001-gop-baselines/experiments/track05_phase1_baseline.yaml --output-dir runs
```
