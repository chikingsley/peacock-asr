# P001 Final Campaign Spec

This document defines the canonical final rerun campaign for `P001`.

## 1. Scope

`P001` is the controlled GOP-SF scoring-layer paper:

- frozen dataset protocol on SpeechOcean762
- fixed backend comparison (`original`, `xlsr-espeak`)
- controlled scorer comparison (`scalar`, `feats`, `gopt`)
- scalar score-variant ablation (`gop_sf`, `logit_margin`, `logit_combined`)

Keep these out of `P001`:

- ConPCO reproduction
- duration / energy / SSL feature enrichment
- broader architecture changes whose main claim is richer feature spaces

Those belong in `P002`, because they test a different question than the
baseline scoring-layer ablation.

Local evidence:

- `docs/PAPER_CLOSE_CHECKLIST.md`
- `docs/ABLATION_PLAN.md`
- `docs/manuscript.md`
- `docs/EVIDENCE_LEDGER.md`

## 2. Canonical W&B Naming

Workspace project:

- `peacockery/peacock-asr-p001-gop-baselines`

Sweep group format:

- `p001-paper-close-{phase}-{backend}-{job_id}-{mode_or_variant}`

Examples:

- `p001-paper-close-phase1-xlsr-a3-gopt`
- `p001-paper-close-phase2-original-b2-logit-margin`
- `p001-paper-close-phase2b-original-alpha-sweep`

Run name format:

- `p001-paper-close | {phase} | {job_id} | {eval_name} | seed={seed}`

Examples:

- `p001-paper-close | phase1 | a3 | wav2vec2-xlsr-53-espeak-cv-ft (GOPT) | seed=501`
- `p001-paper-close | phase2 | b2 | original | seed=501`

Campaign label:

- `paper-close`

W&B job types:

- `eval` for the main `peacock-asr run` matrix
- `analysis` for dense alpha sweep summary runs

Tags:

- always include `p001`, `track05`, backend, phase, and job id
- add the run family tag (`scalar`, `feats`, `gopt`, `gop_sf`,
  `logit_margin`, `logit_combined`, `alpha-sweep`)

Why this shape:

- W&B uses the project as the top-level namespace for runs, then `group`,
  `job_type`, and `config` for the substructure inside that namespace.
- `peacock-asr-p001-gop-baselines` is stable and maps exactly to the repo
  workspace and project registry entry.
- `paper-close` is the campaign label, not the permanent project identity.
- `P001` already needs phase/job ids to align manuscript tables, evidence
  ledger, and rerun manifests.

References:

- W&B init and run metadata:
  https://docs.wandb.ai/ref/python/init
- W&B run grouping:
  https://docs.wandb.ai/models/runs/grouping
- W&B config:
  https://docs.wandb.ai/models/runs/config

## 3. Required W&B Metadata

Every final `P001` eval run should log:

- `track=track05`
- `project_id=P001`
- `phase`
- `job_id`
- `backend`
- `backend_vocab_size`
- `mode`
- `score_variant`
- `score_alpha`
- `seed`
- `dataset_revision`
- `feature_dim`
- `device`
- `use_cache`
- `cache_train_hit`
- `cache_test_hit`
- `cache_hits`
- `cache_misses`
- `train_utterances`
- `test_utterances`
- `workers`
- `limit`
- `git_sha`

Final summary metrics:

- `pcc`
- `pcc_low`
- `pcc_high`
- `mse`
- `n_phones`
- `duration_s`

Diagnostics:

- per-phone PCC table
- full GOPT epoch history for `A3`
- final GOPT epoch summary in run summary

Why these fields:

- `P001` manuscript and ablation docs already define PCC, CI, MSE, and
  reproducibility as the core reporting surface.
- cache, dataset revision, device, and git SHA are the minimum provenance
  fields needed to explain why two nominally similar runs differ.

References:

- `docs/manuscript.md`
- `docs/ABLATION_PLAN.md`
- `docs/PAPER_CLOSE_CHECKLIST.md`
- W&B summary metrics:
  https://docs.wandb.ai/models/runs/summary
- W&B tables:
  https://docs.wandb.ai/models/tables/
- W&B environment variables:
  https://docs.wandb.ai/models/track/environment-variables

## 4. Artifact Policy

Canonical artifacts to keep:

- W&B runs for metrics and config provenance
- W&B model artifacts for `A3` GOPT checkpoints
- local final manifests:
  - `projects/P001-gop-baselines/experiments/final/manifests/latest_host_manifest.json`
  - `projects/P001-gop-baselines/experiments/final/manifests/latest_run_context.json`
  - `projects/P001-gop-baselines/experiments/final/batches/<timestamp>_<batch>/summary.tsv`
  - `projects/P001-gop-baselines/experiments/final/batches/<timestamp>_<batch>/aggregates.tsv`
  - `projects/P001-gop-baselines/experiments/final/alpha_sweeps/<timestamp>_alpha_sweep_<backend>/alpha_sweep.tsv`
  - `projects/P001-gop-baselines/experiments/final/alpha_sweeps/<timestamp>_alpha_sweep_<backend>/alpha_sweep_meta.json`
  - `projects/P001-gop-baselines/experiments/final/checkpoints/<timestamp>_<backend>_<seed>/...`

Recommended remote provenance fields for RunPod-style training or reruns:

- `pod_id`
- `template_id`
- `image_name`
- `gpu_type`
- `gpu_count`
- `hourly_cost_usd`
- `volume_gb`
- `desired_status`
- SSH/public endpoint info when appropriate for internal reproducibility notes

Disposable local state:

- `.cache/features/...`
- `.cache/models/...` when recoverable elsewhere
- raw agent logs after the campaign is closed

Why:

- W&B is the best system of record for metric history, grouping, and comparison.
- Project-local TSV/JSON manifests are still needed for paper tables, evidence
  ledgers, and long-term reproducibility without depending on one hosted UI.
- Feature caches are accelerators, not evidence.

References:

- `docs/EVIDENCE_LEDGER.md`
- W&B artifacts:
  https://docs.wandb.ai/models/artifacts/construct-an-artifact/

## 5. Final P001 Matrix

### Phase 1

- `A1` scalar + poly
- `A2` features + SVR
- `A3` features + GOPT, seeds `501..505`

Backends:

- `original`
- `xlsr-espeak`

### Phase 2

- `B1` `gop_sf`
- `B2` `logit_margin`
- `B3` `logit_combined`, `alpha=0.25`
- `B4` `logit_combined`, `alpha=0.50`
- `B5` `logit_combined`, `alpha=0.75`

Backends:

- `original`
- `xlsr-espeak`

### Phase 2b

Dense alpha sweep on both backends:

- `alpha=0.00..1.00`, step `0.05`
- log one W&B `analysis` run per backend
- keep local `alpha_sweep.tsv` and `alpha_sweep_meta.json`

## 6. Execution Notes

Canonical main sweep specs live in:

- `projects/P001-gop-baselines/experiments/sweeps/final/`

Preferred campaign launcher:

- `uv run --project projects/P001-gop-baselines python projects/P001-gop-baselines/experiments/final/launch_p001_paper_close.py`
- supports `--phase {all,phase1,phase2,phase2b}` and `--start-at <sweep_name>`
- capture host + run-context provenance with:
  `uv run --project projects/P001-gop-baselines python projects/P001-gop-baselines/experiments/final/capture_machine_manifest.py`
- for remote pods, capture RunPod control-plane metadata with:
  `uv run --project projects/P001-gop-baselines python projects/P001-gop-baselines/experiments/final/capture_machine_manifest.py --pod-id <POD_ID>`
- the same script can be reused for future project-local remote training by
  overriding `--project-id`, `--project-slug`, `--output-root`, and
  `--wandb-group-prefix`

Reason:

- keeps campaign orchestration in Python rather than shell
- stays project-local and reproducible on local or remote GPU hosts
- makes partial reruns and resume points explicit

Canonical dense alpha sweep commands:

```bash
PEACOCK_WANDB_GROUP=p001-paper-close-phase2b-original-alpha-sweep \
PEACOCK_WANDB_TRACK=track05 \
PEACOCK_WANDB_PROJECT_ID=P001 \
PEACOCK_WANDB_PHASE=phase2b \
PEACOCK_WANDB_JOB_ID=alpha-sweep \
PEACOCK_WANDB_RUN_PREFIX=p001-paper-close \
PEACOCK_WANDB_JOB_TYPE=analysis \
PEACOCK_WANDB_TAGS=p001,paper-close,phase2b,original,alpha-sweep \
uv run --project projects/P001-gop-baselines peacock-asr sweep-alpha \
  --backend original \
  --alpha-start 0.0 \
  --alpha-stop 1.0 \
  --alpha-step 0.05 \
  --output-dir projects/P001-gop-baselines/experiments/final/alpha_sweeps
```

```bash
PEACOCK_WANDB_GROUP=p001-paper-close-phase2b-xlsr-alpha-sweep \
PEACOCK_WANDB_TRACK=track05 \
PEACOCK_WANDB_PROJECT_ID=P001 \
PEACOCK_WANDB_PHASE=phase2b \
PEACOCK_WANDB_JOB_ID=alpha-sweep \
PEACOCK_WANDB_RUN_PREFIX=p001-paper-close \
PEACOCK_WANDB_JOB_TYPE=analysis \
PEACOCK_WANDB_TAGS=p001,paper-close,phase2b,xlsr,alpha-sweep \
uv run --project projects/P001-gop-baselines peacock-asr sweep-alpha \
  --backend xlsr-espeak \
  --alpha-start 0.0 \
  --alpha-stop 1.0 \
  --alpha-step 0.05 \
  --output-dir projects/P001-gop-baselines/experiments/final/alpha_sweeps
```
