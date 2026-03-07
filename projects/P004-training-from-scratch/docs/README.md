# P004 Docs

The operational source of truth for this workspace is `../README.md`.
This folder is for evidence, ablations, and manuscript support only.

## Keep Here

- `EVIDENCE_LEDGER.md`: what has been validated, what failed, and why
- `ABLATION_PLAN.md`: the next bounded runs and their acceptance gates
- `manuscript.md`: paper-writing scratch space, only after the experiments justify it
- `refs.bib`: bibliography backing the manuscript and evidence notes

## Working Rule

Do not add planning documents here unless they directly change the next experiment.
The default pattern is:

1. update the workspace README if the operating model changed
2. run an experiment or validator
3. log the result in `EVIDENCE_LEDGER.md`
