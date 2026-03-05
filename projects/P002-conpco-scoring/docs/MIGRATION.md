# P002 Migration Notes

## Objective

Consolidate ConPCO reproduction work into `P002` without disrupting active sweeps.

## Freeze Constraints (active reproduction)

Do not move these yet:

- `references/ConPCO/`
- `runs/reproduce_conpco.py`

## Current Canonical Inputs (legacy paths)

- Docs: `docs/research/track09_conpco_scoring/`
- Run specs:
  - `runs/sweep_conpco_v3_paper.yaml`
  - `runs/sweep_conpco_v4_rng_fix.yaml`

## Post-sweep target layout

- Upstream reference repo target:
  - `projects/P002-conpco-scoring/third_party/conpco-upstream/`
- Reproduction script wrapper target:
  - `projects/P002-conpco-scoring/code/reproduce_conpco.py` (or project wrapper that calls legacy path)
