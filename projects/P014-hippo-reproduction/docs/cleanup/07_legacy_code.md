# P014 — Legacy / Deprecated Code Assessment

Research-only audit. **No source edited.** Scope: `src/p014/` (~4,395 LOC across 18 modules), `tests/` (8 files), `configs/` (3 YAML), `docs/`.

## 1. Executive Summary

The P014 tree is **exceptionally clean for a paper-faithful reproduction project**. Mechanical searches for the usual legacy smells came back almost empty:

| Probe | Hits |
|---|---|
| `rg -n 'TODO|FIXME' src/ tests/` | **0** |
| `rg -n '_old|_v1|_v2|_legacy|_orig|_backup|_deprecated' src/` | **0** |
| `rg -ni 'deprecated|legacy|backup|# old|# was |# previously|# replaced|# retained|# backcompat|hack|XXX' src/ tests/` | **0** (only doc-side hits) |
| `rg -n 'if.*version|sys\.version|torch\.__version|transformers\.__version' src/` | **0** |

Recent git history (`git log --oneline -- src/p014/`) is empty — `src/p014/` arrived whole in commit `4dcd2ca` ("Add P013/P014 HMamba projects") and has not been iteratively replaced. There is no buried abandoned experiment under a different subpackage: `find . -maxdepth 1 -type d \( -name hippo -o -name scripts -o -name data \)` returned nothing.

The only real findings are (a) one stale claim in `docs/IMPLEMENTATION_NOTES.md`, (b) a mild naming split between paper-pinned and runtime flags for CONO / curriculum, and (c) a handful of `# pragma: no cover` fallbacks in trackio init that are legitimate. Nothing warrants a "delete this file" recommendation.

## 2. TODO / FIXME Inventory

**Zero** TODO / FIXME / XXX / HACK markers anywhere in `src/` or `tests/`. This is unusual and worth noting as a positive signal — the author is not leaving scaffolding in-tree.

## 3. Deprecated / Legacy Findings

### 3.1 Stale "legacy branch" claim in IMPLEMENTATION_NOTES.md

- **Location**: `docs/IMPLEMENTATION_NOTES.md:29-31`
- **Evidence**: The doc says:
  > The preserved legacy branch under `hippo/`, `data/`, and `scripts/` still contains placeholder logic and should not be used for reproduction work.
  Those directories **do not exist** at the project root (verified by `find . -maxdepth 2 -type d`). Only `src/`, `configs/`, `tests/`, `docs/`, and `artifacts/` are present. The legacy branch was already cleaned up (likely in one of the repo-wide `chore: remove outdated experiment and project artifacts` / `chore: remove P001 spike experiments` commits upstream of the P014 directory).
- **Current status**: Doc is stale, code is fine.
- **Proposed action**: In a future doc-cleanup pass, delete or rewrite this paragraph. **No code change.**
- **Confidence**: **HIGH**

### 3.2 Paper-pinned vs runtime flag naming split: CONO and curriculum

- **Location**:
  - Paper YAML side: `src/p014/config.py:89-90` (`TrainingConfig.use_cono_regularizer`, `use_curriculum_learning`) — loaded from `configs/base_config.yaml`, validated in `validate_paper_consistency` (`config.py:131-145`), asserted by `tests/test_config.py:15-24`.
  - Runtime side: `src/p014/config.py:213` (`TrainingRunSettings.use_cono`) and `config.py:222` (`use_curriculum`). These are the ones that actually gate training behavior:
    - `training.py:469` and `training.py:546`: `if settings.use_cono:` guards the `cono_loss` call.
    - `training.py:235`: `curriculum_active = ... and settings.use_curriculum`.
    - `cli.py:36-37, 58-66, 118-121`: `--use-cono` / `--use-curriculum` map to the runtime fields only.
- **Why this looks like legacy**: Two different names for what is conceptually the same toggle (`use_cono_regularizer` vs `use_cono`, `use_curriculum_learning` vs `use_curriculum`). On first read it smells like one supersedes the other.
- **Actual status**: **NOT legacy — deliberate split.** `ExperimentConfig` is the paper contract (immutable, YAML-pinned, used to cross-check reproduction fidelity). `TrainingRunSettings` is the per-run runtime knob (overridable via env / CLI). The two Pydantic models are layered intentionally. The docstring at `config.py:190-196` explicitly states: *"Distinct from `ExperimentConfig`, which is the YAML-pinned paper spec. This one tracks knobs that change per run."*
- **Risk**: They are not kept in sync. An operator could set `P014_USE_CONO=false` at runtime while the paper-pinned YAML says `use_cono_regularizer: true` for free-speaking — nothing cross-validates them, so a free-speaking run can silently drift from the paper spec.
- **Proposed action**: **Keep the split**; add a one-liner consistency check in `train_hippo` that warns when the runtime toggle contradicts the paper-pinned flag for the active scenario. Not a legacy-code removal.
- **Confidence**: **HIGH** that it is deliberate; **MEDIUM** that the lack of cross-check is a minor bug (see agent 1 deduplication report, §3 — likely overlaps).

### 3.3 `# pragma: no cover` fallbacks in trackio init

- **Location**: `src/p014/training.py:70-84, 89-114` (three `try/except Exception` blocks).
- **Evidence**: Catch-all `except Exception as exc:` for `trackio.init`, `.log`, `.finish`, each suppressed from coverage with `# pragma: no cover - network/runtime issues`.
- **Current status**: Live. Gracefully degrades to local-only logging if trackio is unreachable or uninstalled. Not a stale fallback — trackio is an optional network dependency and MEMORY.md rule says "Everything logs to W&B" but the repo uses trackio (HF's replacement for W&B-on-HF-Spaces), not W&B proper.
- **Proposed action**: **Keep as-is.** Narrow `except Exception` to `except (OSError, RuntimeError, ImportError, AttributeError)` if tightening is desired, but this is a quality-of-life fallback, not legacy cruft.
- **Confidence**: **HIGH**

### 3.4 Phone-ID fallback in CTC GOP extraction

- **Location**: `src/p014/features/ctc_gop.py:198, 208`.
- **Evidence**:

  ```python
  fallback_id = vocab.get("[UNK]", vocab.get("<unk>", 1))
  ...
  ids.append(vocab.get(first_char, fallback_id))
  ```

  The docstring (line 194) explicitly documents the fallback: *"Unknown tokens fall back to `[UNK]` if present, else the first non-blank token."*
- **Current status**: Live and correct. This is handling vocabulary mismatch between SpeechOcean phone inventory and the CTC model's tokenizer — not a legacy code path.
- **Proposed action**: **Keep.** Add a `logger.debug` when the fallback triggers if debugging paper-faithfulness on unknown tokens.
- **Confidence**: **HIGH**

### 3.5 Empty-word placeholder in free-speaking annotation

- **Location**: `src/p014/freespeak/annotation.py:132-138`.
- **Evidence**: When ASR produces no parseable words, code emits a single `""` placeholder scored at zero. Comment says: *"Pathological case — learner produced no parseable speech. We still need a non-empty annotation so downstream collation doesn't break."*
- **Current status**: Live. Handles a real edge case (Whisper outputs empty transcript), not legacy.
- **Proposed action**: **Keep.**
- **Confidence**: **HIGH**

## 4. Duplicate Implementations

None of the substantive kind. Candidates considered and dismissed:

- **`TrainingConfig` vs `TrainingRunSettings`** (§3.2). Two Pydantic models with overlapping fields, but serve different purposes (paper spec vs runtime) and coexist deliberately.
- **`AttentionPoolingConfig` dataclass** (`blocks.py:197-204`) alongside `AttentionPooling` class that takes kwargs directly. Already flagged in `docs/cleanup/03_unused_code.md §3.1` as dead; not a duplicate implementation per se — the dataclass was scaffolding that never got wired up.
- **`evaluate_predictions()` helper** (`metrics.py:38-74`) alongside the inline metrics computation in `training.py`. Already flagged in `03_unused_code.md §3.3`. This *is* a duplicate-implementation case: `training.py` reinvents what `evaluate_predictions` would do. Cross-reference that report for the consolidation recommendation. **The winner is the inline version in `training.py`** because it integrates with the per-epoch best-selection loop; `evaluate_predictions` should be deleted, not consolidated into.

## 5. Dead Config Flags

Checked every field in `TrainingConfig` and `TrainingRunSettings` against YAML and CLI:

| Flag | Definition | Bound in YAML? | Selected at runtime? | Status |
|---|---|---|---|---|
| `training.use_cono_regularizer` | `config.py:89` | all 3 YAMLs | validation-only | **live, not dead** |
| `training.use_curriculum_learning` | `config.py:90` | all 3 YAMLs | validation-only | **live, not dead** |
| `training.trials: 5` | `config.py:85` | `base_config.yaml:32` | **not consumed** — `TrainingRunSettings.seeds` drives trial count | **paper-spec-pinned, purely declarative; keep as doc** |
| `training.selection_metric: "phone_mse"` | `config.py:86` | YAML | read nowhere | same as above |
| `training.word_score_range` / `utterance_score_range` | `config.py:87-88` | YAML | read nowhere | same as above |
| `data.free_speaking_assignment` | `config.py:108` | YAML | read nowhere | same as above |

**None of these are dead flags gating legacy behavior.** They are paper-config declarative surface (documented in `docs/cleanup/03_unused_code.md §4.2` as pydantic false-positives). They exist to enforce YAML shape and document the paper's numeric choices, not to drive branching. **Do not delete.**

No flag was found that gates a never-selected old code path. No `if use_legacy_X:` branches anywhere.

## 6. Cross-Cutting Notes

- **Overlap with `03_unused_code.md`**: Their five HIGH-confidence removals (§3.1 `AttentionPoolingConfig`, §3.2 `SSL_UTT_DIM`, §3.3 `evaluate_predictions`, §3.4 `phones_to_canonical_ids`, §3.5 `asr_word` arg) are all scaffolding that never got wired up, not legacy in the "replaced by newer impl" sense. They overlap with this report's framing only in that `evaluate_predictions` has an inline rival in `training.py` (§4 above). **This report adds no new removal candidates beyond that set.**
- **Overlap with deduplication report (agent 1)**: The `use_cono_regularizer` / `use_cono` split (§3.2 here) is the main cross-cut. If agent 1 flagged the `TrainingConfig` / `TrainingRunSettings` overlap, the verdict is: **keep both, add a consistency assertion**, not merge.
- **The `official_repo_public_state: "README only on public main branch as of 2026-04-20"` config field** (`base_config.yaml:10`) is an unusual piece of metadata that will drift as the upstream repo evolves. Not legacy code, but a doc-string maintenance hazard worth noting.

## 7. Bottom Line

There is effectively **no legacy code** to remove in P014. The tree was added in a single commit, has no churn history, zero TODOs, and the handful of fallback paths that do exist are load-bearing (vocabulary fallbacks, empty-ASR guards, optional network logging). The removal surface lives entirely in the unused-code report. The only action items from this pass are documentation (fix `IMPLEMENTATION_NOTES.md:29-31`) and a minor defense-in-depth suggestion (cross-check paper-pinned vs runtime CONO/curriculum flags).
