# ASR Program Board

Canonical project IDs are `P###-slug`.
Legacy `trackXX` names remain historical aliases only.

## Current Program Snapshot

- `P001-gop-baselines`: mostly complete; baseline evidence and runbook are stable.
- `P002-conpco-scoring`: active; reproduction sweep in progress.
- `P003-compact-backbones`: active/next; compact model experiments.
- `P004-training-from-scratch`: planned.
- `P005-llm-pronunciation`: planned/research notes largely drafted.
- `P006-realtime-streaming`: planned/research notes largely drafted.

## Current Execution Rule

- Do not move or alter these while ConPCO sweeps are active:
  - `references/ConPCO/`
  - `runs/reproduce_conpco.py`

## Migration Waves

1. Wave A (now): project scaffolds, registry, status board.
2. Wave B: migrate docs and sweep manifests by project with compatibility pointers.
3. Wave C: move project-specific code wrappers under each project `code/`.
4. Wave D: move upstream/reference repos into each project `third_party/` after active runs finish.
