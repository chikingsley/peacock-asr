# P001 Migration Notes

## Objective

Consolidate GOP baseline work into `P001` while preserving reproducibility.

## Current Canonical Inputs

- Docs: `projects/P001-gop-baselines/docs/`
- Run specs:
  - `projects/P001-gop-baselines/experiments/track05_phase1_baseline.yaml`
  - `projects/P001-gop-baselines/experiments/track05_phase2_logit_scalar.yaml`
- Legacy archived specs:
  - `projects/P001-gop-baselines/experiments/legacy/configs/track05_phase1_baseline.yaml`
  - `projects/P001-gop-baselines/experiments/legacy/configs/track05_phase2_logit_scalar.yaml`
- Typical outputs:
  - `projects/P001-gop-baselines/experiments/final/batches/<timestamp>_<batch>/`
  - `projects/P001-gop-baselines/experiments/legacy/batches/<timestamp>_<batch>/`

## Current status

- Canonical configs already live in project workspace:
  - `projects/P001-gop-baselines/experiments/track05_phase1_baseline.yaml`
  - `projects/P001-gop-baselines/experiments/track05_phase2_logit_scalar.yaml`
- Moved into project workspace:
  - `projects/P001-gop-baselines/docs/RUNBOOK.md`
  - `projects/P001-gop-baselines/docs/{README.md,ABLATION_PLAN.md,EVIDENCE_LEDGER.md,PAPER_CLOSE_CHECKLIST.md,manuscript.md,refs.bib}`
- Compatibility note:
  - `projects/P001-gop-baselines/experiments/COMPATIBILITY.md`
- Historical `runs/` artifacts have been archived under
  `projects/P001-gop-baselines/experiments/legacy/`.

## Next plan

1. Keep manuscript, evidence, and results tied to `experiments/final/results/`.
2. Keep historical ad hoc evidence under `experiments/legacy/` only.
3. Keep implementation project-local unless it is deliberately duplicated or extracted later.
