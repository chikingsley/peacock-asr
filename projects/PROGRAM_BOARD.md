# ASR Program Board

Canonical project IDs are `P###-slug`.
Legacy `trackXX` names remain historical aliases only.

## Current Program Snapshot

- `P001-gop-baselines`: mostly complete; baseline evidence is stable. Remaining
  cleanup is original-backend scalar closeout and final paper framing.
- `P002-conpco-scoring`: phase-1 conclusion is in; loss-only gains are small on
  the 42-d GOP-SF stack. Feature enrichment is the remaining decision point.
- `P003-compact-backbones`: active; `wav2vec2-base` result landed, `HuBERT-base`
  is the clearest next backbone.
- `P004-training-from-scratch`: planned.
- `P005-llm-pronunciation`: planned/research notes largely drafted.
- `P006-realtime-streaming`: planned/research notes largely drafted.

## Current Execution Rule

- Prefer canonical project-local paths for new `P002` work:
  - `projects/P002-conpco-scoring/code/`
  - `projects/P002-conpco-scoring/third_party/ConPCO/`
- Prefer project-local sweep specs for new work.
- Root-level `runs/` has been removed; keep legacy evidence under each
  project's `experiments/legacy/` tree instead.

## Migration Waves

1. Wave A (now): project scaffolds, registry, status board.
2. Wave B: migrate docs and sweep manifests by project with compatibility pointers.
3. Wave C: move project-specific code wrappers under each project `code/`.
4. Wave D: remove legacy compatibility shims once project-local paths are stable in automation.
