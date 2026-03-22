# P003 Hatchet Orchestration Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current `P003` launcher/watcher orchestration path with a repo-local Hatchet workflow that runs one real `XLSR-53` evaluation pipeline end-to-end while keeping W&B as the experiment tracker.

**Architecture:** Add a repo-local Hatchet runtime under `infra/hatchet`, expose a smaller set of explicit `P003` CLI commands for prewarm and eval orchestration, and have Hatchet call those commands as visible timed steps. Keep W&B sweeps for eval fan-out in the first cut, but move creation, agent launch, monitoring, and notifications into Hatchet-owned workflow steps.

**Tech Stack:** Python 3.13, `uv`, Hatchet, existing `p003_compact` CLI/runtime code, W&B CLI/API, Slack-style webhook notification, pytest.

---

## Chunk 1: Inventory And Freeze The First Migration Path

### Task 1: Capture the exact legacy `XLSR-53` path to replace

**Files:**

- Create: `docs/superpowers/notes/2026-03-09-p003-xlsr53-legacy-path.md`
- Modify: `docs/superpowers/specs/2026-03-09-p003-hatchet-orchestration-design.md`

- [ ] **Step 1: Identify the current `XLSR-53` evaluation entrypoint and every script it calls**

Run searches for:

- current `XLSR-53` references
- prewarm command
- eval YAML
- sweep creation
- agent launch
- watcher scripts

Expected: one concrete path mapped from human entrypoint to final notification/logging behavior.

- [ ] **Step 2: Write the legacy path note**

Document:

- old entry command
- files involved
- expected outputs
- logs written
- W&B objects created

- [ ] **Step 3: Tighten the design doc with the exact legacy path reference**

Update the spec if the actual `XLSR-53` path differs from assumptions.

- [ ] **Step 4: Review the note and spec for ambiguity**

Expected: no unresolved “maybe this script is used” uncertainty for the first migration path.

## Chunk 2: Create The Repo-Local Hatchet Skeleton

### Task 2: Add repo-local Hatchet runtime layout

**Files:**

- Create: `infra/hatchet/README.md`
- Create: `infra/hatchet/.env.example`
- Create: `infra/hatchet/docker-compose.yml`
- Create: `infra/hatchet/pyproject.toml`
- Create: `infra/hatchet/worker/__init__.py`
- Create: `infra/hatchet/worker/main.py`
- Create: `infra/hatchet/workflows/__init__.py`
- Create: `infra/hatchet/workflows/p003.py`
- Create: `infra/hatchet/tasks/__init__.py`
- Create: `infra/hatchet/tasks/p003.py`
- Test: `infra/hatchet/tests/test_imports.py`

- [ ] **Step 1: Write a failing import smoke test**

```python
def test_hatchet_modules_import():
    __import__("infra.hatchet.worker.main")
    __import__("infra.hatchet.workflows.p003")
    __import__("infra.hatchet.tasks.p003")
```

- [ ] **Step 2: Run the smoke test to verify it fails**

Run: `uv run --project infra/hatchet pytest infra/hatchet/tests/test_imports.py -v`

Expected: FAIL because the Hatchet package layout does not exist yet.

- [ ] **Step 3: Create the minimal Hatchet project files**

Add:

- local project metadata
- worker entrypoint
- empty workflow/task modules
- README with local startup instructions

- [ ] **Step 4: Re-run the smoke test**

Run: `uv run --project infra/hatchet pytest infra/hatchet/tests/test_imports.py -v`

Expected: PASS.

- [ ] **Step 5: Add a minimal local runtime command section to the README**

Document how to:

- start Hatchet backing services
- run the worker locally
- trigger a workflow manually

## Chunk 3: Define The Canonical Workflow Payload And Step Result Types

### Task 3: Add typed workflow contracts

**Files:**

- Create: `infra/hatchet/tasks/contracts.py`
- Test: `infra/hatchet/tests/test_contracts.py`

- [ ] **Step 1: Write failing tests for workflow payload validation**

Cover:

- required fields present
- missing `backend_ref` rejected
- missing `eval_yaml` rejected for eval flow
- default values for `split` and `device`

- [ ] **Step 2: Run the contract tests to verify they fail**

Run: `uv run --project infra/hatchet pytest infra/hatchet/tests/test_contracts.py -v`

Expected: FAIL because the contract module does not exist yet.

- [ ] **Step 3: Implement the minimal typed payload and step result models**

Include:

- workflow input payload
- step result payload
- shared workflow state shape

- [ ] **Step 4: Re-run the contract tests**

Run: `uv run --project infra/hatchet pytest infra/hatchet/tests/test_contracts.py -v`

Expected: PASS.

## Chunk 4: Expose Stable P003 Commands For Workflow Use

### Task 4: Add explicit CLI units for Hatchet to call

**Files:**

- Modify: `projects/P003-compact-backbones/code/p003_compact/cli.py`
- Modify: `projects/P003-compact-backbones/code/p003_compact/scoring/runtime.py`
- Create: `projects/P003-compact-backbones/tests/test_p003_cli_workflow_commands.py`

- [ ] **Step 1: Write failing tests for the new command surface**

Cover command parsing for:

- `prewarm-backend`
- `create-eval-sweep`
- `launch-eval-agent`
- `watch-eval-sweep`
- `notify-workflow` or equivalent minimal notification command

- [ ] **Step 2: Run the CLI tests to verify they fail**

Run: `uv run --project projects/P003-compact-backbones pytest projects/P003-compact-backbones/tests/test_p003_cli_workflow_commands.py -v`

Expected: FAIL because the commands do not exist yet.

- [ ] **Step 3: Implement minimal command parsing and adapters**

Rules:

- commands must be explicit
- each command does one thing
- no hidden “launch next thing” behavior
- stdout should be machine-readable when practical

- [ ] **Step 4: Re-run the CLI tests**

Run: `uv run --project projects/P003-compact-backbones pytest projects/P003-compact-backbones/tests/test_p003_cli_workflow_commands.py -v`

Expected: PASS.

### Task 5: Remove hidden chaining from the retained launchers

**Files:**

- Modify: `projects/P003-compact-backbones/code/_launcher_lib.py`
- Modify: `projects/P003-compact-backbones/code/launch_hubert_base_local.py`
- Modify: `projects/P003-compact-backbones/code/launch_wav2vec2_large_local.py`
- Modify: `projects/P003-compact-backbones/code/launch_parakeet_ctc_0_6b_local.py`
- Modify: `projects/P003-compact-backbones/code/omniasr/launch_train_local.py`
- Test: `projects/P003-compact-backbones/tests/test_legacy_launchers_no_chain.py`

- [ ] **Step 1: Write failing tests asserting launchers no longer auto-chain**

Expected behavior:

- launchers may still launch their primary job
- launchers do not automatically trigger post-train orchestration

- [ ] **Step 2: Run the launcher tests to verify they fail**

Run: `uv run --project projects/P003-compact-backbones pytest projects/P003-compact-backbones/tests/test_legacy_launchers_no_chain.py -v`

Expected: FAIL against the current auto-chain behavior.

- [ ] **Step 3: Remove auto-follow orchestration from the retained launchers**

Keep only the primary unit of work in each launcher.

- [ ] **Step 4: Re-run the launcher tests**

Run: `uv run --project projects/P003-compact-backbones pytest projects/P003-compact-backbones/tests/test_legacy_launchers_no_chain.py -v`

Expected: PASS.

## Chunk 5: Implement Hatchet Tasks Around Explicit P003 Commands

### Task 6: Add local command-runner helpers for Hatchet tasks

**Files:**

- Modify: `infra/hatchet/tasks/p003.py`
- Create: `infra/hatchet/tests/test_p003_tasks.py`

- [ ] **Step 1: Write failing task tests for command execution wrappers**

Cover:

- command construction
- captured log path
- structured success result
- structured failure result

- [ ] **Step 2: Run the task tests to verify they fail**

Run: `uv run --project infra/hatchet pytest infra/hatchet/tests/test_p003_tasks.py -v`

Expected: FAIL because the task wrappers are not implemented.

- [ ] **Step 3: Implement the task wrappers**

Implement task helpers for:

- validate input
- prewarm backend
- create eval sweep
- launch eval agent
- watch eval sweep
- send notification

- [ ] **Step 4: Re-run the task tests**

Run: `uv run --project infra/hatchet pytest infra/hatchet/tests/test_p003_tasks.py -v`

Expected: PASS.

### Task 7: Set retry and idempotency rules

**Files:**

- Modify: `infra/hatchet/tasks/p003.py`
- Modify: `infra/hatchet/README.md`
- Test: `infra/hatchet/tests/test_retry_policy.py`

- [ ] **Step 1: Write failing tests for retry classification**

Cover:

- validation can retry
- prewarm retries once
- sweep creation manual retry only by default
- agent launch manual retry only by default
- notification retries automatically

- [ ] **Step 2: Run the retry tests to verify they fail**

Run: `uv run --project infra/hatchet pytest infra/hatchet/tests/test_retry_policy.py -v`

Expected: FAIL because retry policy is not encoded yet.

- [ ] **Step 3: Implement retry metadata or helper policy**

Add a clear place where retry behavior is defined and documented.

- [ ] **Step 4: Re-run the retry tests**

Run: `uv run --project infra/hatchet pytest infra/hatchet/tests/test_retry_policy.py -v`

Expected: PASS.

## Chunk 6: Build The First Real XLSR-53 Workflow

### Task 8: Implement the `evaluate_existing_backbone` workflow

**Files:**

- Modify: `infra/hatchet/workflows/p003.py`
- Test: `infra/hatchet/tests/test_p003_workflow_definition.py`

- [ ] **Step 1: Write a failing workflow definition test**

Assert that the workflow contains these ordered steps:

- validate
- prewarm
- create eval sweep
- launch eval agent
- watch eval sweep
- notify

- [ ] **Step 2: Run the workflow definition test to verify it fails**

Run: `uv run --project infra/hatchet pytest infra/hatchet/tests/test_p003_workflow_definition.py -v`

Expected: FAIL because the workflow is not defined yet.

- [ ] **Step 3: Implement the workflow**

The workflow should accept the canonical payload and use the step result/state
contracts from earlier tasks.

- [ ] **Step 4: Re-run the workflow definition test**

Run: `uv run --project infra/hatchet pytest infra/hatchet/tests/test_p003_workflow_definition.py -v`

Expected: PASS.

### Task 9: Add the concrete `XLSR-53` workflow trigger entry

**Files:**

- Create: `infra/hatchet/workflows/p003_inputs.py`
- Modify: `infra/hatchet/README.md`
- Test: `infra/hatchet/tests/test_p003_inputs.py`

- [ ] **Step 1: Write failing tests for the `XLSR-53` input builder**

Cover:

- expected `backend_ref`
- expected eval YAML path
- expected default device and split
- stable workflow label

- [ ] **Step 2: Run the input-builder tests to verify they fail**

Run: `uv run --project infra/hatchet pytest infra/hatchet/tests/test_p003_inputs.py -v`

Expected: FAIL because the builder does not exist yet.

- [ ] **Step 3: Implement the concrete workflow input builder**

Create one obvious way to trigger the `XLSR-53` eval workflow without hand-assembling
JSON payloads each time.

- [ ] **Step 4: Re-run the input-builder tests**

Run: `uv run --project infra/hatchet pytest infra/hatchet/tests/test_p003_inputs.py -v`

Expected: PASS.

## Chunk 7: Add Notifications And Operator Visibility

### Task 10: Add Slack-style notification delivery

**Files:**

- Create: `infra/hatchet/tasks/notify.py`
- Modify: `infra/hatchet/tasks/p003.py`
- Modify: `infra/hatchet/.env.example`
- Modify: `infra/hatchet/README.md`
- Test: `infra/hatchet/tests/test_notify.py`

- [ ] **Step 1: Write failing notification tests**

Cover:

- success payload includes workflow label, duration, W&B refs, and log links
- failure payload includes failed step and error summary
- missing webhook config degrades safely

- [ ] **Step 2: Run the notification tests to verify they fail**

Run: `uv run --project infra/hatchet pytest infra/hatchet/tests/test_notify.py -v`

Expected: FAIL because notification support is not implemented.

- [ ] **Step 3: Implement the minimal notification sender**

Requirements:

- Slack-compatible webhook payload
- safe failure mode if webhook is unset
- no secret values logged

- [ ] **Step 4: Re-run the notification tests**

Run: `uv run --project infra/hatchet pytest infra/hatchet/tests/test_notify.py -v`

Expected: PASS.

## Chunk 8: Remove Replaced P003 Orchestration Code

### Task 11: Delete legacy watcher/chaining scripts that Hatchet replaces

**Files:**

- Delete: `projects/P003-compact-backbones/code/orchestration/watch_sweep_and_launch.py`
- Delete: `projects/P003-compact-backbones/code/orchestration/watch_agent_log_and_launch.py`
- Delete: `projects/P003-compact-backbones/code/orchestration/trigger_post_train_scoring.py`
- Delete: `projects/P003-compact-backbones/code/orchestration/start_p004_then_queue_parakeet.py`
- Delete: `projects/P003-compact-backbones/code/orchestration/start_parakeet_then_queue_omni.py`
- Modify: `projects/P003-compact-backbones/docs/RUNBOOK.md`
- Modify: `projects/P003-compact-backbones/docs/README.md`
- Test: `projects/P003-compact-backbones/tests/test_orchestration_imports.py`

- [ ] **Step 1: Write a failing test or assertion for the new canonical orchestration path**

Assert docs and imports point to Hatchet, not the deleted scripts.

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --project projects/P003-compact-backbones pytest projects/P003-compact-backbones/tests/test_orchestration_imports.py -v`

Expected: FAIL because docs and references still point to the old scripts.

- [ ] **Step 3: Delete the replaced scripts and update docs**

Make Hatchet the documented path for the migrated workflow.

- [ ] **Step 4: Re-run the test**

Run: `uv run --project projects/P003-compact-backbones pytest projects/P003-compact-backbones/tests/test_orchestration_imports.py -v`

Expected: PASS.

## Chunk 9: Verify The Real Workflow End-To-End

### Task 12: Run local verification for the migrated path

**Files:**

- Modify: `infra/hatchet/README.md`
- Create: `docs/superpowers/notes/2026-03-09-p003-hatchet-verification.md`

- [ ] **Step 1: Start the local Hatchet runtime**

Run the documented local startup command.

Expected: control-plane services healthy and worker connected.

- [ ] **Step 2: Trigger the concrete `XLSR-53` eval workflow**

Expected: workflow appears in Hatchet UI with visible step boundaries.

- [ ] **Step 3: Verify each workflow step completes with timing**

Check:

- prewarm visible
- W&B sweep created
- agent launch visible
- sweep monitoring visible
- final notification sent

- [ ] **Step 4: Verify W&B outputs remain correct**

Check:

- sweep exists in the expected W&B project
- eval runs are grouped as expected
- logs/refs reported by Hatchet match real outputs

- [ ] **Step 5: Record verification evidence**

Write down:

- commands run
- workflow ID
- sweep ref
- any issues found

## Chunk 10: Final Cleanup And Handoff

### Task 13: Summarize the new canonical path

**Files:**

- Modify: `projects/P003-compact-backbones/docs/RUNBOOK.md`
- Modify: `projects/P003-compact-backbones/docs/README.md`
- Modify: `infra/hatchet/README.md`

- [ ] **Step 1: Add the new canonical operator flow**

Document:

- how to start Hatchet locally
- how to trigger the `XLSR-53` workflow
- where to view workflow state
- where to view W&B results
- how manual follow-up decisions are made

- [ ] **Step 2: Remove obsolete instructions for the migrated path**

Delete or rewrite instructions that tell operators to use the old watcher/chain flow.

- [ ] **Step 3: Run the targeted test suite**

Run:

- `uv run --project infra/hatchet pytest infra/hatchet/tests -v`
- `uv run --project projects/P003-compact-backbones pytest projects/P003-compact-backbones/tests -v`

Expected: all new and updated tests pass.

- [ ] **Step 4: Run lint checks on touched Python files**

Run:

- `uv run --project infra/hatchet ruff check infra/hatchet`
- `uv run --project projects/P003-compact-backbones ruff check projects/P003-compact-backbones/code projects/P003-compact-backbones/tests`

Expected: no new lint failures in touched files.

- [ ] **Step 5: Prepare the implementation handoff**

Record:

- what was migrated
- what old scripts were deleted
- what still remains on W&B sweeps for the first cut
- what the next migration target should be after `XLSR-53`
