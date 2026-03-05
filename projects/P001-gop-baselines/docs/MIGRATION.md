# P001 Migration Notes

## Objective

Consolidate GOP baseline work into `P001` while preserving reproducibility.

## Current Canonical Inputs (legacy paths)

- Docs: `projects/P001-gop-baselines/docs/`
- Run specs:
  - `runs/track05_phase1_baseline.yaml`
  - `runs/track05_phase2_logit_scalar.yaml`
- Typical outputs:
  - `runs/<timestamp>_track05_phase1_baseline/`
  - `runs/<timestamp>_track05_phase2_logit_scalar/`

## Current status

- Copied into project workspace:
  - `projects/P001-gop-baselines/experiments/track05_phase1_baseline.yaml`
  - `projects/P001-gop-baselines/experiments/track05_phase2_logit_scalar.yaml`
- Moved into project workspace:
  - `projects/P001-gop-baselines/docs/RUNBOOK.md`
  - `projects/P001-gop-baselines/docs/{README.md,ABLATION_PLAN.md,EVIDENCE_LEDGER.md,PAPER_CLOSE_CHECKLIST.md,manuscript.md,refs.bib}`
- Compatibility note:
  - `projects/P001-gop-baselines/experiments/COMPATIBILITY.md`
- Legacy `runs/*.yaml` files remain in place by policy.

## Next plan

1. Move remaining P001 docs incrementally into `projects/P001-gop-baselines/docs/` with backlinks.
2. Update P001 runbook references and command examples as P001 naming is adopted.
3. Keep shared implementation in `src/peacock_asr/` unless tightly project-specific.
4. Execute one explicit cutover commit later to remove legacy duplicates.
