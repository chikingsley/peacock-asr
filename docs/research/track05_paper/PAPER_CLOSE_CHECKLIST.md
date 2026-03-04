# Track05 Paper-Close Checklist (W&B-First)

Scope:
- Close Track05 to lab-standard rigor.
- Re-run the full Track05 package with Weights & Biases as the primary tracker.
- Keep claims limited to what the reruns support.

Related standard:
- `docs/research/lab_research_methodology.md`

## 0) W&B References (Canonical)

- W&B Quickstart / run tracking:
  - https://docs.wandb.ai/quickstart
- W&B Python `init` + run configuration:
  - https://docs.wandb.ai/ref/python/init
- W&B environment variables:
  - https://docs.wandb.ai/guides/track/environment-variables
- W&B sweeps overview and setup:
  - https://docs.wandb.ai/guides/sweeps
- Sweep config keys:
  - https://docs.wandb.ai/models/sweeps/sweep-config-keys
- W&B tracking limits / performance guidance:
  - https://docs.wandb.ai/guides/track/limits
- W&B pricing / tracked-hours + storage framing:
  - https://wandb.ai/site/pricing

## 1) Migration Preconditions

- [ ] Confirm W&B workspace/org and project names for Track05 (single project target).
- [ ] Confirm authentication on run host (`wandb login`) and non-interactive env strategy.
- [ ] Define mandatory run metadata fields:
  - `track=track05`
  - `backend`
  - `mode` (`scalar` / `feats` / `gopt` / `sweep`)
  - `score_variant`
  - `score_alpha`
  - `seed`
  - `dataset_revision`
  - `cache_hit` / `cache_miss`
- [ ] Define grouping convention:
  - `group=<timestamp>_<batch_name>`
  - `job_type=<mode>`
  - tags include: `track05`, backend, phase (`phase1`, `phase2`, `phase2b`).

## 2) Required Code/CLI Changes (Before Reruns)

- [ ] Add first-class W&B logging path to Track05 CLI flows (same coverage currently in MLflow path).
- [ ] Ensure every run logs final summary metrics:
  - `pcc`, `pcc_low`, `pcc_high`, `mse`, `n_phones`, `duration_sec`.
- [ ] Log full config payload to W&B config (backend, score variant, alpha, seed, eval mode).
- [ ] Log artifact links/paths for:
  - `summary.tsv`, `aggregates.tsv`, per-run logs, sweep TSV/JSON.
- [ ] Keep local `runs/...` outputs unchanged so manuscript/evidence links stay stable.

## 3) Rerun Matrix (Paper-Close Minimum)

### 3.1 Phase-1 Baseline Trio

- [ ] Re-run `A1` scalar + poly (>=3 seeds if stochastic path exists).
- [ ] Re-run `A2` feats + SVR (>=3 seeds if stochastic path exists).
- [ ] Re-run `A3` feats + GOPT (>=5 seeds; keep current standard).

### 3.2 Phase-2 Scalar Variant Set

- [ ] Re-run `B1` (`gop_sf`) with >=3 seeds.
- [ ] Re-run `B2` (`logit_margin`) with >=3 seeds.
- [ ] Re-run `B3` (`logit_combined`, alpha=0.25) with >=3 seeds.
- [ ] Re-run `B4` (`logit_combined`, alpha=0.50) with >=3 seeds.
- [ ] Re-run `B5` (`logit_combined`, alpha=0.75) with >=3 seeds.

### 3.3 Phase-2b Dense Alpha Sweep

- [ ] Re-run dense cached sweep on `xlsr-espeak` (`0.00..1.00`, step `0.05`) and log as W&B sweep or grouped runs.
- [ ] Run missing dense sweep on `original` backend after required caches exist.
- [ ] Record best-alpha per backend and compare backend sensitivity.

## 4) Compute + Reproducibility Requirements (Lab Standard)

- [ ] For each reported table row, record compute metadata:
  - wall-clock, hardware class, cache usage, and rerun count.
- [ ] Report mean ± std for all headline claims (not single-seed points only).
- [ ] Keep exact commands/config files in runbook for every reported result.
- [ ] Pin environment snapshot used for reruns (toolchain + deps).
- [ ] Include failed/aborted run accounting in internal evidence notes.

## 5) Manuscript / Evidence Updates After Reruns

- [ ] Update `manuscript.md` tables with W&B-backed rerun aggregates.
- [ ] Update `EVIDENCE_LEDGER.md` with W&B run URLs + local artifact paths.
- [ ] Update `RUNBOOK.md` with exact W&B-first execution commands.
- [ ] Update `docs/EXPERIMENTS.md` and `docs/TODO.md` to reflect final status.
- [ ] Verify all IDs/metrics are consistent across manuscript, runbook, and ledger.

## 6) Claim Lock (What We Are Allowed to Claim)

- [ ] If both backends show best alpha near the same region and consistent deltas:
  - claim scorer-variant effect is robust in this setup.
- [ ] If effects diverge strongly by backend:
  - claim stack-specific behavior and narrow scope.
- [ ] Keep generalization claims out unless supported by extra datasets/backends.

## 7) Exit Criteria (Track05 Complete)

- [ ] All required reruns complete and logged in W&B project.
- [ ] All headline results reported as mean ± std with uncertainty.
- [ ] Compute and reproducibility disclosures present.
- [ ] Paper tables/figures and evidence ledger fully aligned.
- [ ] No unresolved TODOs for Track05 critical path.
