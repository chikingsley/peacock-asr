# P004 Ablation Plan

The point of this file is not to predict the whole paper. It is to define the
next bounded experiments so we can validate the path one step at a time.

## Core Rule

- `reference lane` first, `canonical lane` second
- one change per run when the change can affect correctness or throughput
- no new architecture branch until the previous gate is green
- no GPU should be running while we are still deciding what the run is

## Run Classes

- `P0 local preflight`: zero-rental checks, dry-runs, env validation, and
  manifest inspection
- `S1 smoke run`: first real train launch, capped to prove one optimizer step
  and recoverable outputs
- `B1 bounded validation`: short repeatable training window used to prove loss
  movement and restart or resume behavior
- `A1 ablation`: one controlled change against a known-good baseline

## `G0` Rental Readiness Gate

Goal:
- make the next GPU run exact enough that turning a GPU on is a deliberate act,
  not part of exploration

Required before any rented or dedicated GPU is activated:
- the exact command is written down
- the target machine class is chosen
- the output directory and run ID are chosen
- the success signal is written down
- the stop or kill conditions are written down
- the monitoring cadence is written down
- the next action after pass and the next action after fail are written down

Acceptance:
- all seven items above are filled in for the next run
- if any item is missing, the answer is `not ready`

Blackwell note:
- the local Blackwell GPU is not the first validation target for the reference
  lane
- the reference lane is still the older `icefall` + `k2` stack pinned around
  `torch 2.4.1+cu124`
- treat that lane as a compatibility proof, not as the final performance stack

## Reference Lane

### `R0` Workspace Validation

Goal:
- prove the P004 reference workspace is materially ready to train

Checks:
- `icefall` recipe exists locally
- `.venv-icefall` can import `torch`, `k2`, and `lhotse` when launched with the
  required runtime env
- phone manifests and `lang_phone` exist and are readable
- current GPU state is known before any training run starts

Artifact:
- `experiments/validation/reference_setup.json`

Acceptance:
- all required paths exist
- prepared-env import probe succeeds
- manifest summary and first-record checks succeed

Move-on rule:
- `R0` is only good enough to unblock `G0`; it is not enough to justify a long
  run by itself

### `R1` Conformer Smoke Train

Goal:
- prove the upstream `icefall` Conformer CTC loop can take phone manifests,
  start stepping, and write artifacts

Command shape:
- `code/run_conformer_phone_ctc.py -- --num-epochs 1 ...`

Acceptance:
- train log progresses past initialization
- at least one optimizer step is logged
- at least one checkpoint or recoverable artifact is written

Notes:
- run only on a compatibility-first GPU or a dedicated cheap node
- do not mix in new kernels, W&B, or refactors here
- the run should be treated as `S1 smoke`, not as a performance run
- current state on `2026-03-07`: `R1` is green on rented `RTX 4090` and failed
  on rented `RTX PRO 4000 Blackwell`, so the lane is proven but still
  compatibility-bound

Required pre-commits before launch:
- exact launch command written into the ledger or run notes
- output directory reserved
- first failure owner decided in advance: env, recipe, data, or hardware

Immediate stop conditions:
- import or startup failure from `torch`, `k2`, or data paths
- explicit `sm_120` / `no kernel image is available` failure on Blackwell with
  the old `torch 2.4.1+cu124` reference env
- no first optimizer step within the allowed smoke window
- repeated OOM after one reduced-load retry
- no recoverable output written to the target experiment directory

Smoke window:
- the purpose is only to prove `start -> step -> write`
- if we need more time than that to decide whether the run is healthy, the gate
  was not defined tightly enough

### `R2` Stable Small Reference Run

Goal:
- get a clean small-scale run that can be repeated without hand-fixing the env

Acceptance:
- same command can be re-run from a clean shell
- resume or restart behavior is understood
- train and dev loss both move in the expected direction

Required evidence:
- one clean fresh run
- one clean rerun or resume
- recorded note on whether the environment still needs special shell prep beyond
  the launcher itself

Current state on `2026-03-07`:
- `R2` is green on a RunPod `RTX A5000`
- the fresh run wrote `epoch-0.pt` and `epoch-1.pt`
- the explicit resume run loaded `epoch-1.pt` and wrote `epoch-2.pt`
- the old reference lane is now repeatable on compatible non-Blackwell hardware

## Canonical Lane

### `C0` Canonical Workspace Bootstrap

Goal:
- lift the validated path into the lab-standard project layout

Scope:
- `uv` project config and lockfile
- exact dependency groups
- run metadata and artifact layout
- cleaner launch entrypoints

Acceptance:
- canonical env can be reproduced from scratch
- run configuration is explicit and versioned
- the reference lane result that is being lifted is identified by run ID
- current local status on `2026-03-07`: canonical env and CUDA preflight are
  running on the `RTX 5070`; the lane also has a real-data local smoke and a
  bounded validation run using the WAV fallback loader, but native
  `torchcodec` runtime is still a workstation-specific blocker

### `C1` Tracking And Artifacts

Goal:
- add tracking once the reference lane is stable enough that logs are worth
  keeping

Scope:
- local machine-manifest capture
- W&B runs and artifacts
- optional HF Hub milestone uploads

Acceptance:
- run config, metrics, checkpoints, and dataset identity are recoverable
- local run metadata and remote tracker metadata agree on the run identity
- current local status on `2026-03-07`: machine manifest capture, offline W&B
  runs, metrics logs, and checkpoints are already written for canonical local
  smoke runs; canonical resume is now also proven, so the next gap is optional
  online or Hub sync

### `C2` Blackwell / New-Kernel Branch

Goal:
- evaluate newer PyTorch and Blackwell-oriented kernels without contaminating
  the stable reference lane

Scope:
- current stable PyTorch lane
- separate nightly/experimental lane for newer attention kernels

Acceptance:
- correctness check against the stable lane
- measured speed delta
- measured memory delta
- no silent training instability during a bounded run
- no contamination of the reference lane command or environment

Immediate sequence after `R2`:
- `C2.0`: keep the stable local lane on `torch 2.9.1+cu128` and the WAV
  fallback loader; swap the tiny smoke model for the first structured encoder
  variant before touching kernels
- `C2.1`: prove the structured encoder path can run fresh and resume on the
  local `RTX 5070`
- `C2.2`: benchmark stable-lane `scaled_dot_product_attention` and
  `torch.compile` as the local baseline
- `C2.3`: open a separate nightly branch for Blackwell-specific kernel work
  such as FA4 or newer attention paths
- `C2.4`: only keep a new kernel path if the bounded run is numerically sane
  and the measured speed or memory delta is real

Current local status on `2026-03-07`:
- `C2.0` is green with a `conformer_like` encoder (`1.7M` params) on the local
  `RTX 5070`
- `C2.1` is green with explicit resume from `epoch-1.pt` into later epochs on
  the same stack
- `C2.2` is green on the local `RTX 5070`
- on this structured-smoke benchmark, eager beat `torch.compile` in both total
  elapsed time and post-warmup mean step time
- the stable local baseline for `C2.3` should therefore keep
  `enable_compile=false`
- `C2.3` is now open locally in `env/nightly-fa4`
- compiled nightly `flex_attention` passes with `AUTO` and explicit `TRITON`
  on the local `RTX 5070`
- current `FLASH` / FA4 fails on this local device with
  `Unsupported compute capability. Supported: 9.x, 10.x, 11.x`
- the trainer-level `flex_triton` smoke passes locally
- the bounded `flex_triton` validation is not numerically sane yet: it goes
  non-finite at `epoch=0, batch_index=1`
- the same bounded failure reproduced on an isolated Vast `RTX PRO 4000
  Blackwell`, so the instability is in the current trainer path, not just local
  workstation interference
- `C2.5` is now complete and red on isolated Vast Blackwell hardware:
  lower LR (`1e-4`, `1e-5`), `flex_auto`, and `float32` CTC loss all failed
  with the same post-update `NaN` at `epoch=0`, `batch_index=1`
- the next branch is no longer a cheap sweep; it is either
  full-fp32 / first-step finiteness diagnostics or demotion of nightly flex
  training back to benchmark-only status
- keep a separate remote FA4 validation target for `H100/H200/B200`-class
  hardware

## Architecture Branches

### `Z0` Zipformer

Start only after `R2` is green.

Reason:
- Zipformer may be the stronger long-term efficiency bet
- Conformer is still the shorter path for proving the data and recipe flow

Acceptance:
- reuse the same manifests, vocab, and eval wiring
- change the architecture, not the whole stack

### `H0` Attach-Head / Fine-Tune Modes

Start only after the from-scratch reference lane is stable.

Reason:
- otherwise we will not know whether a failure came from the encoder strategy
  or the training stack itself

## Decision Gates

- If `G0` is incomplete, do not turn on a GPU.
- If `R0` fails, fix the reference setup before running more training.
- If `R1` cannot get past initialization, stop adding complexity and repair the
  reference lane.
- If `R2` is not repeatable, do not start the canonical lane yet.
- If `C2` makes training faster but less stable, keep it experimental.
- If Zipformer requires widespread stack changes, defer it until after the
  canonical lane is clean.

## What Counts As Ready

The workspace is ready for a GPU only when:

- `G0` is green for the next run
- the command and kill conditions are already fixed
- the result will clearly answer one question
- passing the run would immediately unlock the next bounded step

If any of those are false, the correct action is another `P0` local preflight,
not another GPU hour.
