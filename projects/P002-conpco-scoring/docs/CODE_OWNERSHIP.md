# P002 Code Ownership

Purpose:

- Distinguish what is actually shared across projects from what is currently
  living in shared paths only for convenience or historical reasons.

## Canonical P002-owned Paths

- `projects/P002-conpco-scoring/code/reproduce_conpco.py`
- `projects/P002-conpco-scoring/code/track09_conpco_ablation.py`
- `projects/P002-conpco-scoring/code/benchmark_reproduce_conpco.py`
- `projects/P002-conpco-scoring/third_party/ConPCO/`
- `projects/P002-conpco-scoring/experiments/sweeps/final/`

These are project-local and should remain the main execution surface for
ConPCO / HierCB reproduction and feature-enrichment work.

## Project-Local Runtime

- `projects/P002-conpco-scoring/code/p002_conpco/evaluate.py`
- `projects/P002-conpco-scoring/code/p002_conpco/settings.py`
- `projects/P002-conpco-scoring/code/p002_conpco/gopt_model.py`
  - `GOPTModel`
  - `GoptDataset`
  - `train_and_evaluate_gopt()`
- `projects/P002-conpco-scoring/code/p002_conpco/gopt_track09.py`
- `projects/P002-conpco-scoring/code/p002_conpco/conpco_losses.py`

The project no longer depends on a root shared runtime package.

## Recommended Next Cutover

1. Keep the scoring/runtime stack inside `projects/P002-conpco-scoring/code/`.
2. Duplicate code intentionally if `P001` or `P003` need similar logic.
3. Do not reintroduce root-level runtime wrappers.
