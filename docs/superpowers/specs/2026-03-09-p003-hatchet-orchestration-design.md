# P003 Hatchet Orchestration Design

## Goal

Replace the current `P003` orchestration layer of launcher scripts, watcher scripts,
and implicit W&B-driven chaining with an explicit Hatchet workflow system that owns
run sequencing, step timing, retries, failure reporting, and operator notifications.
W&B remains the experiment tracking system of record for metrics, configs, sweeps,
artifacts, and comparisons.

## Context

The current `P003` setup mixes together several responsibilities:

- W&B sweep definitions
- local launcher scripts
- post-train chaining scripts
- watcher scripts that poll logs or W&B state
- ad hoc environment variable conventions

That makes the run surface difficult to reason about and hard to trust. The user
wants:

- explicit workflow steps with durations and failures
- a visible execution graph
- fewer orchestration scripts
- stable commands that can be run either manually or from a workflow system
- W&B used for experiment results, not as the main scheduler

## Scope

This design covers the first real migration for `projects/P003-compact-backbones`.
It does not yet define a repo-wide orchestration platform for all projects.

The first implementation target is one concrete `P003` backbone workflow, starting
with the original `XLSR-53` path the user called out, while shaping the code so the
other existing backbone workflows can be migrated next.

## Recommended Architecture

### System boundaries

The system should be split into three layers:

1. Stable project commands
2. Hatchet workflows and tasks
3. W&B experiment logging

#### Stable project commands

`P003` should expose a small, explicit set of commands for the units of work that
already exist in practice:

- training a backbone
- prewarming the backend and cache path used by evaluation
- running evaluation for a configured backend
- creating and running an eval sweep when eval fan-out is still delegated to W&B

These commands should be stable enough to call:

- directly by a human
- from Hatchet tasks
- from future automation or remote execution wrappers

These commands must not contain hidden orchestration logic such as "launch the next
thing when this succeeds".

#### Hatchet workflows and tasks

Hatchet should become the orchestration layer for `P003`.

Hatchet owns:

- workflow graph definition
- step boundaries
- retries
- run timing
- failure state
- operator notifications
- optional manual approval checkpoints

Hatchet does not own model metrics, sweep charts, or experiment artifacts. Those stay
in W&B.

For the first migration, Hatchet is the top-level orchestrator but **not yet** the
fan-out engine for eval seeds. Hatchet will explicitly run the W&B sweep lifecycle as
workflow steps:

- create eval sweep
- launch sweep agent
- monitor sweep completion
- notify operator of outcome

This keeps the first cut smaller while still removing hidden orchestration from local
launcher scripts.

#### W&B experiment logging

W&B remains responsible for:

- train metrics
- eval metrics
- sweep comparisons where useful
- artifacts
- summaries
- run grouping and tagging

W&B is intentionally demoted from "scheduler plus tracker plus partial orchestrator"
back to "tracker".

## Migration Strategy

### Keep

Keep the actual implementation code for:

- training
- evaluation
- prewarm logic
- W&B logging

### Replace

Replace the current orchestration chain:

- watcher scripts
- scripts whose job is to poll another job and launch the next one
- scripts whose only purpose is chaining phases together
- hidden auto-follow behavior embedded in local launchers

### Reshape

Reshape the command surface so workflow steps are explicit and narrow. Each workflow
step should call one stable command and return structured status to Hatchet.

## Canonical Workflow Contract

The first migration needs one concrete workflow contract so state is not passed around
implicitly the way it is today.

### Workflow input

Each Hatchet workflow invocation should accept a single typed payload with the fields
needed to identify the work unambiguously:

- `project_id`: logical project identifier such as `P003`
- `workflow_kind`: `evaluate_existing_backbone` or `train_then_evaluate`
- `label`: short stable workflow label such as `xlsr53`
- `backend_ref`: backend identifier used by `P003` evaluation, such as `hf:<repo>`
- `eval_yaml`: canonical eval sweep YAML path
- `train_command`: optional explicit training command for train-plus-eval flows
- `project_root`: absolute path to `projects/P003-compact-backbones`
- `repo_root`: absolute repo root
- `device`: target device such as `cuda`
- `split`: evaluation split, default `both`
- `notify_target`: logical notification target or channel name
- `metadata`: freeform metadata for tags, notes, or operator context

### Step result payload

Each Hatchet task should return structured data rather than relying on stdout parsing
alone. The minimum payload should include:

- `status`: `success` or `failure`
- `step_name`: stable step identifier
- `started_at`
- `finished_at`
- `duration_s`
- `command`: exact command executed when applicable
- `log_path`: log file path when applicable
- `artifacts`: list of important output paths
- `wandb_refs`: run IDs, sweep refs, or URLs when available
- `outputs`: step-specific structured values such as `sweep_id` or `backend_ref`
- `error`: normalized error summary when failed

### Workflow state passed between steps

The canonical state object passed through the workflow should include:

- `workflow_run_id`: Hatchet workflow run identifier
- `label`
- `backend_ref`
- `eval_yaml`
- `prewarm_log_path`
- `sweep_id`
- `sweep_ref`
- `agent_pid` if a local process is spawned
- `agent_log_path`
- `wandb_project`
- `wandb_entity`
- `notification_summary`

This is the contract that replaces today’s implicit JSON state files and ad hoc
launcher conventions.

## First-Cut Workflow Model

For a backbone that already exists and needs evaluation, the workflow should look
like this:

1. validate workflow input
2. resolve backend/checkpoint identity
3. prewarm backend caches
4. run evaluation
5. collect key outputs and W&B references
6. notify operator with success or failure

For a train-plus-eval flow, the workflow should look like this:

1. validate workflow input
2. run training
3. resolve produced checkpoint/backend reference
4. prewarm backend caches
5. run evaluation
6. notify operator
7. optionally pause at a manual approval step before any subsequent workflow is
   launched

The first cut should prefer explicit sequential flows over a generalized dynamic
pipeline engine.

### Exact first migration path

The first concrete migration path is:

- existing backbone: original `XLSR-53`
- flow type: evaluate existing backbone
- orchestration target: replace the current launcher/watcher chain with one Hatchet
  workflow
- acceptance condition: Hatchet shows prewarm, sweep creation, agent execution, and
  final notification as visible steps with timings; W&B still records the underlying
  eval sweep and run metrics correctly

The first migrated workflow should replace one real end-to-end path, not a toy demo.

## Location and Layout

The first Hatchet setup should live inside this repo so `P003` can move quickly.

Recommended layout:

- `infra/hatchet/`
- `infra/hatchet/README.md`
- `infra/hatchet/docker-compose.yml` or equivalent local runtime files
- `infra/hatchet/worker/`
- `infra/hatchet/workflows/`
- `infra/hatchet/tasks/`

`P003` runtime code remains under:

- `projects/P003-compact-backbones/code/`

This keeps orchestration local to the repo while preserving the option to extract the
Hatchet runtime layout later if it proves broadly reusable.

### Execution model

For the first cut:

- Hatchet control-plane services may run via repo-local Docker or compose files
- the Hatchet worker that launches `P003` jobs should run on the local machine with
  direct access to:
  - the checked-out repo
  - local `uv` environments
  - GPUs
  - local caches
  - checkpoint directories
  - local log paths

The first worker should not depend on containerized GPU execution. That would add
complexity before the new orchestration path is proven.

## Command Surface Design

The new command surface should favor a few explicit CLI entrypoints over many narrow
launcher scripts.

Examples:

- `peacock-asr train-backbone ...`
- `peacock-asr prewarm-backend ...`
- `peacock-asr run-eval ...`
- `peacock-asr resolve-backend ...` if needed for workflow plumbing

The exact names can vary, but the key requirement is that the commands describe the
real unit of work and can be safely called independently.

The current launcher scripts may be retained only as short-term compatibility shims
while migration is in progress, but the target state is to remove them.

## Notifications

The operator wants completion and failure visibility, not blind automatic chaining.

The first cut should support:

- success notification
- failure notification
- links or paths to relevant logs and outputs
- links or identifiers for the corresponding W&B run or sweep when available
- Slack-style push notification in addition to the Hatchet UI

Manual decision-making after notification is preferred over auto-launching the next
experiment.

### Approval model

The first cut should not auto-launch downstream experiments after completion.

Instead:

- Hatchet finishes the workflow
- a Slack-style notification is sent with status and relevant references
- the operator decides whether to trigger the next workflow manually

If approval checkpoints are implemented in the first cut, they should default to
timing out into a safe no-op rather than launching follow-up work automatically.

## Deletion Policy

Delete orchestration code only after the Hatchet replacement is working end-to-end for
at least one real `P003` workflow.

Likely deletion targets after migration:

- `watch_*` orchestration scripts
- `trigger_post_train_scoring.py`
- `start_*_then_queue_*` scripts
- local launcher scripts whose main job is phase chaining

Do not bundle unrelated repo cleanup such as deleting `papers`, `scripts`, or local
environment directories into this migration unless separately audited and approved.

## Risks

### Risk: over-generalizing too early

Trying to build a fully generic workflow platform before migrating one real `P003`
path would slow delivery and reintroduce abstraction sprawl.

Mitigation:

- migrate one real path first
- use real task boundaries found during migration
- generalize only after the first workflow is stable

### Risk: preserving hidden orchestration

If Hatchet simply wraps the existing chaining scripts, the complexity is only moved,
not removed.

Mitigation:

- define stable commands for real units of work
- keep sequencing only in Hatchet

### Risk: retrying expensive or non-idempotent ML steps

Blind retries can duplicate runs, burn GPU time, or create duplicate W&B sweeps.

Mitigation:

- validation and monitoring steps may retry automatically
- prewarm may retry automatically once
- sweep creation should not auto-retry unless it can prove no sweep was created
- training and sweep-agent launch should default to manual retry only
- notifications may retry automatically

### Risk: metadata drift between Hatchet and W&B

If workflow identity, model identity, and W&B run metadata are derived in different
places, the UI will stay inconsistent.

Mitigation:

- centralize run metadata derivation
- pass a workflow/job identifier through all steps

## Testing Strategy

The migration should be validated in layers:

1. unit tests for any new command-surface helpers
2. task-level tests for Hatchet task wrappers where practical
3. a local end-to-end dry run for one `P003` workflow
4. one real workflow run with visible Hatchet timing and expected W&B output

## Success Criteria

The migration is successful when:

- one real `P003` backbone workflow runs through Hatchet end-to-end
- step timing and failures are visible in Hatchet
- W&B contains clean train/eval tracking without acting as the orchestration layer
- the old watcher/chaining path is no longer needed for that migrated workflow
- the resulting command surface is smaller and easier to understand than the current
  launcher-based setup
