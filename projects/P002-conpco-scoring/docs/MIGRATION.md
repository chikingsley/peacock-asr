# P002 Migration Notes

## Objective

Consolidate ConPCO reproduction work into `P002` without disrupting active sweeps.

## Current Compatibility Constraints

No root-level runtime shims remain for `P002`.

## Current Canonical Inputs

- Docs: `projects/P002-conpco-scoring/docs/`
- Code: `projects/P002-conpco-scoring/code/`
- Upstream reference repo: `projects/P002-conpco-scoring/third_party/ConPCO/`
- Run specs:
  - `projects/P002-conpco-scoring/experiments/sweeps/final/reproduce_conpco_v3.yaml`
  - `projects/P002-conpco-scoring/experiments/sweeps/final/reproduce_conpco_v4_rng_fix.yaml`
  - `projects/P002-conpco-scoring/experiments/sweeps/final/track09_p1_ablation.yaml`

## Compatibility Cleanup Target

- Keep all new automation on the project-local entrypoints only.
