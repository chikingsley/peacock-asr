# P014 — Unused Code Assessment

Research-only audit. No source files edited. Scope: `src/p014/` (~4,395 LOC across 18 modules) and `tests/` (8 test files). YAML configs under `configs/` checked for dynamic references.

## 1. Executive Summary

The codebase is **remarkably clean**. `vulture --min-confidence 80` returned **zero** findings. `ruff F401/F811/F841/ARG` produced **one** real hit. `deadcode` produced 50 findings, but after manual `rg` verification **all but 5 are false positives** (Pydantic fields bound by YAML and `nn.Module.forward` methods invoked by PyTorch's `__call__`).

No whole dead modules exist. Every file in `src/p014/` is imported from either `cli.py`, `training.py`, a test file, or a public re-export in `__init__.py`.

**Genuinely removable dead code is ~30–40 lines total**, covering one unused dataclass, one unused constant, two unused helper functions, and one unused function argument.

## 2. Tool Output Summary

| Tool | Command | Raw findings | After dedup vs false-positive classes |
|---|---|---|---|
| `vulture` | `--min-confidence 80` | 0 | 0 |
| `vulture` | `--min-confidence 60` | 50 | 5 real |
| `ruff` | `F401,F811,F841,ARG` | 1 | 1 real |
| `deadcode` | default | 50 | 5 real |

Vulture and deadcode overlap completely at 60% confidence. Their noise comes from (a) Pydantic `model_config = ConfigDict(...)` class vars (26 hits), (b) Pydantic model fields that are populated from YAML (14 hits), (c) `forward()` methods on `nn.Module` subclasses (12 hits) — none of which are actually dead.

## 3. Confirmed Removals

### 3.1 `AttentionPoolingConfig` dataclass

- **Location**: `src/p014/blocks.py:197-204`
- **Confidence**: **HIGH**
- **Evidence**: Flagged by deadcode (DC03) and vulture (60%). `rg '\bAttentionPoolingConfig\b'` across `src/`, `tests/`, `configs/`, `docs/`, and the whole peacock-asr repo returns only the definition site. The live class `AttentionPooling` (line 206) takes individual keyword arguments directly; nothing constructs `AttentionPoolingConfig`.
- **Safe to remove?** **Yes.** Orphan scaffolding. Deleting the 8-line `@dataclass` block has no dependents.

### 3.2 `SSL_UTT_DIM` module constant

- **Location**: `src/p014/data.py:22`
- **Confidence**: **HIGH**
- **Evidence**: Flagged by deadcode (DC01) and vulture (60%). `rg '\bSSL_UTT_DIM\b'` across the whole repo returns only the definition. Public sibling constants `UTTERANCE_SCORE_KEYS` / `WORD_SCORE_KEYS` are heavily referenced; `SSL_UTT_DIM` is not.
- **Safe to remove?** **Yes.** One-line deletion. Callers that need the SSL feature dim read it from `ModelConfig.ssl_feature_dim` (YAML-bound).

### 3.3 `evaluate_predictions()` helper

- **Location**: `src/p014/metrics.py:38-74`
- **Confidence**: **HIGH**
- **Evidence**: Flagged by deadcode (DC02) and vulture (60%). `rg '\bevaluate_predictions\b'` finds only the definition. Not re-exported from `__init__.py`. `training.py` imports `EvaluationMetrics`, `masked_mse`, `safe_pcc` from `metrics` but not this function — it computes evaluation metrics inline in its own loop.
- **Safe to remove?** **Yes.** 37-line function, entirely dead. `EvaluationMetrics`, `safe_pcc`, `masked_mse` in the same file are live and must stay.

### 3.4 `phones_to_canonical_ids()` wrapper

- **Location**: `src/p014/features/ctc_gop.py:558-568`
- **Confidence**: **MEDIUM**
- **Evidence**: Flagged by deadcode (DC02) and vulture (60%). `rg '\bphones_to_canonical_ids\b'` finds only the definition. Not re-exported from `src/p014/features/__init__.py`.
- **Safe to remove?** **Yes, but read docstring first.** The docstring explicitly advertises it as a public wrapper for "the free-speaking loader" to drive GOP extraction from externally-supplied phones. The current free-speak path in `data.py` does not call it — it passes phones via the `phones_per_utterance` kwarg to `extract_gop_for_split` instead. So the wrapper is orphaned but was a deliberate API surface. **Recommend removal** with a note; the private `_canonical_phone_ids` is still available and already used by the tests (`tests/test_ctc_gop_helpers.py`).

### 3.5 Unused function argument `asr_word` in `_assign_word`

- **Location**: `src/p014/freespeak/annotation.py:47`
- **Confidence**: **HIGH**
- **Evidence**: Flagged by ruff (ARG001). Manual read of function body (lines 45-90): `asr_word` is declared but the body only references `asr_phones`, `ref_word_index`, `reference`. The callsite at line 113 passes `asr_word` but also appends it separately to `emitted_words` at line 122 — that append uses the local variable directly, not the return from `_assign_word`.
- **Safe to remove?** **Yes.** Remove the parameter from the signature and drop the `asr_word=asr_word` kwarg at the call site (line 118). No behavior change.

## 4. False Positives (DO NOT remove during cleanup)

The following are **flagged by deadcode/vulture** but are **live**. Implementation phase must leave these alone.

### 4.1 Every `model_config = ConfigDict(...)` / `SettingsConfigDict(...)` (14 hits)

All occurrences in `config.py` (lines 46, 58, 79, 94, 102, 114, 123, 198) and `features/ctc_gop.py:94`. These are the Pydantic v2 **class-level configuration attribute** — Pydantic reads it via metaclass. Tools can't see the metaclass binding.

### 4.2 Pydantic model fields on `PaperReference`, `ModelConfig`, `TrainingConfig`, `DataConfig`, `ScenarioConfig`

Fields flagged: `title`, `doi`, `acl_url`, `arxiv_url`, `official_repo_url`, `official_repo_checked_on`, `official_repo_public_state`, `attention_pool_heads`, `cnn_kernel_size`, `fusion_weight_conv`, `word_embedding_backend`, `ssl_feature_dim`, `ssl_models`, `selection_metric`, `word_score_range`, `utterance_score_range`, `corpus`, `sampling_rate_hz`, `asr_model`, `g2p_model`, `free_speaking_assignment`, `requires_reference_text`, `notes`, `paper`.

All of these **are bound** in `configs/base_config.yaml` (verified by direct read). They pass through `load_experiment_config` → `ExperimentConfig.model_validate(merged)`. The paper-faithful YAML audit in `test_config.py` exercises the validation path.

### 4.3 `validate_paper_consistency` model_validator

- **Location**: `src/p014/config.py:131`
- Invoked automatically by Pydantic through `@model_validator(mode="after")`. Tools don't resolve decorators.

### 4.4 All `forward()` methods (12 hits across `blocks.py` and `model.py`)

Every `nn.Module` subclass has a `forward` that tools flag. PyTorch dispatches these through `Module.__call__`. Must all stay.

### 4.5 `artifact_dir` field on `TrialSummary`

- **Location**: `src/p014/training.py:46`
- Deadcode flags the field declaration. It is populated at `training.py:361` (`artifact_dir=str(trial_dir)`) and the dataclass is serialized via `asdict`. Live.

### 4.6 `word_embedding_backend` (not flagged by deadcode but shown in YAML)

Populated from `configs/base_config.yaml`. Currently read only by Pydantic validation, but is part of the paper-pinned config contract — do not drop.

## 5. Whole-File Candidates

**None.** Verified via `rg 'from p014\.|import p014'` that every module under `src/p014/` has at least one importer:

- `cli.py`, `__main__.py` — entrypoint wiring
- `config.py`, `metrics.py`, `cono.py`, `data.py`, `model.py`, `blocks.py`, `training.py` — imported by training/CLI and tests
- `features/ctc_gop.py`, `features/ssl_utterance.py` — re-exported by `features/__init__.py`, consumed in `data.py` and `cli.py`
- `freespeak/align.py`, `freespeak/annotation.py`, `freespeak/asr.py`, `freespeak/g2p.py` — re-exported by `freespeak/__init__.py`, imported lazily in `data.py:182-188` for the free-speaking scenario, and covered by tests

The `freespeak/` subpackage is only activated for `scenario=free_speaking`, so a superficial scan of `training.py` alone could miss its usage — this is the subtle case to flag to the implementer: **do not delete `freespeak/` thinking it is dead; its consumers are guarded behind a YAML scenario toggle and a lazy import at `data.py:182`**.

## 6. Aggregate Removal Estimate

If items 3.1–3.5 are all removed, expected change:

- `blocks.py`: -8 lines (AttentionPoolingConfig)
- `data.py`: -1 line (SSL_UTT_DIM)
- `metrics.py`: -37 lines (evaluate_predictions)
- `features/ctc_gop.py`: -11 lines (phones_to_canonical_ids)
- `freespeak/annotation.py`: -2 lines (param + callsite kwarg)

**Total: ~59 lines of dead code in a 4,395-line source tree (~1.3%).** This is an unusually healthy ratio for a research reproduction codebase and reflects the paper-faithful scaffolding discipline evident in `config.py`.
