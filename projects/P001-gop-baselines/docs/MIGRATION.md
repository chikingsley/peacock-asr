# P001 Migration Notes

## Objective

Consolidate GOP baseline work into `P001` while preserving reproducibility.

## Current Canonical Inputs (legacy paths)

- Docs: `docs/research/track05_paper/`
- Run specs:
  - `runs/track05_phase1_baseline.yaml`
  - `runs/track05_phase2_logit_scalar.yaml`
- Typical outputs:
  - `runs/<timestamp>_track05_phase1_baseline/`
  - `runs/<timestamp>_track05_phase2_logit_scalar/`

## Near-term plan

1. Copy P001-specific run manifests into `projects/P001-gop-baselines/experiments/`.
2. Keep legacy run specs as compatibility entry points.
3. Move P001 docs incrementally into `projects/P001-gop-baselines/docs/` with backlinks.
4. Keep shared implementation in `src/peacock_asr/` unless tightly project-specific.
