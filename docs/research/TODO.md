# TODO

Best historical single PCC: **0.6958**  
Best historical stochastic mean: **0.6774 +/- 0.0127** (`xlsr-espeak + GOPT`, `P001` final)  
Canonical active workspace: [`projects/P001-gop-baselines/`](../../projects/P001-gop-baselines/)
Lab north star: [`docs/research/LAB_GOALS.md`](./LAB_GOALS.md)

## Active Now

- [x] Finish `P001` paper-close live batch under
      `projects/P001-gop-baselines/experiments/final/`
- [x] Confirm and export the final `P001` result set:
      `aggregate_summary.tsv`, `per_run_summary.tsv`, `alpha_best.tsv`,
      machine manifests, W&B artifact links, and final manuscript tables
- [x] Reconcile the final batch outputs back into:
      `docs/research/EXPERIMENTS.md`,
      `projects/P001-gop-baselines/docs/RUNBOOK.md`,
      `projects/P001-gop-baselines/docs/EVIDENCE_LEDGER.md`, and
      `projects/P001-gop-baselines/docs/manuscript.md`
- [x] Close the remaining original-backend scalar claim:
      low-weight logit mixing helps modestly on both backends, but scalar gains
      remain far below feature-based scoring

## Immediate Next Tranche

- [x] Remove the root `runs/` contract after the project-local campaign outputs
      were verified; keep legacy artifacts under each project workspace
- [x] Delete the legacy shell launcher; keep the `uv` launcher as the only
      canonical campaign entrypoint
- [x] Add a remote/pod machine manifest companion for RunPod-style runs:
      provider, pod/template IDs, GPU type/count, hourly price, mounted volume,
      and launch notes now fit into the project-local manifest flow
- [ ] Expand remote manifest coverage with image hash and any provider-specific
      fields that `runpodctl` exposes on future pods
- [ ] Upgrade `ruff` and `ty` after the live batch, then rerun lint + types
- [ ] Add `vulture --min-confidence 100` as a non-blocking hygiene pass
- [x] Split the old root CLI/runtime surface into project-local packages for
      `P001`, `P002`, and `P003`

## Repo Structure Cleanup

- [x] Remove the root `src/peacock_asr/` package once `P001`/`P002`/`P003`
      pass independently
- [ ] Move project-specific scripts, references, and paper helpers into their
      owning project workspaces where possible
- [ ] Audit project-local `third_party/` ownership and keep upstream repos
      only where they are actually used
- [ ] Move standalone training entrypoints into project-owned locations unless
      they are truly shared infrastructure
- [ ] Standardize project-local launcher patterns so `P001`, `P002`, and `P003`
      use similar metadata conventions without requiring a shared root package

## Acceleration Backlog

- [ ] Profile the current GOP pipeline again after `P001` closes so the next
      speed pass is based on current evidence, not older track assumptions
- [ ] Investigate deeper `_ctc_forward_denom` acceleration for scalar GOP:
      vectorization, Rust/PyO3, or a more specialized graph/runtime path
- [ ] Review and extract repo-wide actions from
      `projects/P001-gop-baselines/docs/PERFORMANCE_CODE_AUDIT_2026-03-05.md`,
      `projects/P002-conpco-scoring/docs/PERFORMANCE_ACCELERATION_PLAYBOOK.md`
      and `projects/P002-conpco-scoring/docs/PERFORMANCE_CODE_AUDIT_2026-03-05.md`
- [ ] Keep `k2/icefall` as a strategic path for graph-based training work,
      not as the first fix for the current CPU-bound scalar/eval bottlenecks

## Project Roadmap

- [x] `P001`: finish paper-close batch, freeze canonical manifests, and lock
      manuscript claims
- [ ] `P002`: decide whether feature enrichment
      (duration, energy, SSL embeddings) becomes the main continuation path
- [ ] `P003`: finish full cutover to project-local sweeps / manifests and rerun
      any final compact-backbone confirmations under canonical naming
- [ ] `P004-P006`: keep as incubators until their evidence ledgers justify
      promotion to active-paper status
- [ ] Reframe `P006` around unscripted / ASR-conditioned CAPT and decide the
      first paper-grade experiment contract for that track
- [ ] Define the product-facing conversational CAPT layer
      (pronunciation + intelligibility + semantic/coherence judgment) as a
      separate evaluation problem, not as a hidden extension of the current
      pronunciation metrics

## Research / Paper Infrastructure

- [ ] Build a lightweight paper-management flow for citations, PDFs, notes, and
      project-local reference mapping
- [ ] Standardize what goes in the main paper vs supplement vs repo manifests:
      hardware, timing, seeds, configs, prompts, figures, and failure counts
- [ ] Decide whether the repo should export a single paper-grade result manifest
      format shared across projects

## Key Papers To Process

- [x] ZIPA (2505.23170)
- [x] POWSM (2510.24992)
- [x] PRiSM (2601.14046)
- [x] CTC-based-GOP (2507.16838)
- [x] GOPT Transformer paper (Gong et al. ICASSP 2022)
- [ ] Xu et al. 2022 (`wav2vec2-xlsr-53-espeak-cv-ft`)
- [ ] Allosaurus (ICASSP 2020)
- [ ] Enhancing GOP with Phonological Knowledge (2506.02080)
- [ ] Logit-based GOP Scores (2506.12067)
- [ ] Original GOP paper (Witt & Young 2000)
- [ ] SpeechOcean762 dataset paper (2104.01378)
