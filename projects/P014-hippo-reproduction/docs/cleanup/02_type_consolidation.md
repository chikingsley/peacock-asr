# P014 Type Definition Consolidation — Critical Assessment

## 1. Executive summary

Overall type discipline is **moderate-to-good**: ~4,400 LOC of source uses
modern `from __future__ import annotations`, 3.10+ generics (`list[…]`,
`tuple[…]`, `X | None`), and leans on frozen `@dataclass` for in-process
records. There is zero `Tuple`/`List`/`Optional` from `typing`, no `TypeVar`
soup, and no legacy forward-reference strings.

However, the project has **no central `types.py` / `schemas.py`**. Types are
scattered across every module that needs them, organised by feature rather
than by concern. Three specific weak spots:

1. **Two parallel type systems**: `pydantic.BaseModel` is used exclusively
   for YAML/env-backed config (`config.py`) and `@dataclass(frozen=True)`
   for in-process records (everything else). The split is consistent, but
   the **cache sidecar records** (`_GopCacheSidecar`, `_GopPartsManifest`,
   `_SslCacheSidecar`, and the ad-hoc word-embedding sidecar in `data.py`)
   all hand-roll `to_dict()` serialisation — exactly what a `BaseModel`
   would give for free. This is the single biggest consolidation opportunity.
2. **`ReadAloudSample` vs `ReadAloudBatch` duplication**: 8 of 10 fields are
   identical aside from the batching dim. `collate_read_aloud_batch` and
   `move_batch_to_device` re-list every field twice; adding one field
   requires edits in four places.
3. **`dict[str, Any]` smell** in `training.py` for trackio/history payloads
   (7 sites). These are fixed-shape records (loss terms, eval metrics,
   curriculum state) that deserve a `TypedDict` or small dataclass.

No circular-import risk in any of the proposed moves: the leaf types have
zero intra-package dependencies.

## 2. Inventory of non-trivial type definitions

| Name | Kind | Location | Purpose |
|------|------|----------|---------|
| `Probability`, `ScoreRange` | `Annotated` alias | `config.py:22-23` | Pydantic-validated float/list |
| `ScenarioName`, `PaperTable`, `FreeSpeakingAssignmentRule` | `StrEnum` | `config.py:29,34,39` | Closed string vocabularies |
| `PaperReference`, `ModelConfig`, `TrainingConfig`, `FreeSpeakingAssignmentConfig`, `DataConfig`, `ScenarioConfig`, `ExperimentConfig` | `BaseModel` | `config.py:45-152` | YAML-backed experiment spec |
| `TrainingRunSettings` | `BaseSettings` | `config.py:189-226` | Env/CLI runtime knobs |
| `ReadAloudAnnotation` | `@dataclass(frozen)` | `data.py:25` | Per-utterance gold annotation |
| `ReadAloudSample` | `@dataclass(frozen)` | `data.py:36` | Single featurised example |
| `ReadAloudBatch` | `@dataclass(frozen)` | `data.py:48` | Padded batch tensor bundle |
| `_GopCachePayload`, `_SslCachePayload` | `@dataclass(frozen)` | `data.py:525,531` | Loaded cache wrappers |
| `EvaluationMetrics` | `@dataclass(frozen)` | `metrics.py:11` | Eval result record |
| `ReadAloudPredictions` | `@dataclass(frozen)` | `model.py:12` | Forward-pass output |
| `AttentionPoolingConfig` | `@dataclass(frozen)` | `blocks.py:197` | Unused / dead? (never imported) |
| `TrialSummary`, `AggregateReport` | `@dataclass(frozen)` | `training.py:38,49` | Per-seed + aggregate results |
| `_TrackioHandle` | `@dataclass` (mutable) | `training.py:56` | Logging wrapper |
| `FeatureExtractionSettings` | `BaseSettings` | `features/ctc_gop.py:91` | Env-tunable extractor |
| `_GopCacheSidecar`, `_GopPartsManifest` | `@dataclass(frozen)` | `features/ctc_gop.py:102,126` | Cache manifest records |
| `_SslCacheSidecar` | `@dataclass(frozen)` | `features/ssl_utterance.py:47` | Cache sidecar |
| `TranscriptionResult` | `@dataclass(frozen)` | `freespeak/asr.py:41` | Whisper output |
| `AlignmentOpKind`, `AlignmentOp` | `Literal`, `@dataclass(frozen)` | `freespeak/align.py:21,24` | NW trace op |

## 3. Findings — numbered consolidation opportunities

### Finding 1 — Cache sidecar records should be `BaseModel`s in one module

- **Confidence: HIGH**
- **Duplicate/overlapping definitions**:
  - `src/p014/features/ctc_gop.py:102` (`_GopCacheSidecar` — version, model_id,
    split, dataset_id, num_examples, gop_dim, content_hash, phone_source)
  - `src/p014/features/ctc_gop.py:126` (`_GopPartsManifest` — same fields minus
    `content_hash`)
  - `src/p014/features/ssl_utterance.py:47` (`_SslCacheSidecar` — version,
    model_ids, split, dataset_id, num_examples, feature_dim, content_hash)
  - `src/p014/data.py:415-476` — word-embedding sidecar is a **bare dict**
    (`{"version", "model_id", "phone_source", "num_train", "num_test"}`) with
    no type at all; comparisons via `.get(...)` are error-prone.
- All four carry near-identical fields (version, model_id, dataset_id,
  num_examples, content hash, optional source key), each reimplements
  `to_dict()`, and each reads back as `dict[str, Any]` then pattern-matches
  field-by-field — this is exactly the freshness-check pattern in
  `_cache_is_fresh()` / `_parts_manifest_is_fresh()` / the inline loader in
  `load_or_build_word_embeddings`.
- **Proposed home**: new `src/p014/features/cache_schemas.py` with
  `CacheSidecarBase(BaseModel)` (version, model_id, dataset_id, split,
  num_examples, content_hash) and three subclasses (`GopCacheSidecar`,
  `GopPartsManifest`, `SslCacheSidecar`, `WordEmbeddingSidecar`). `BaseModel`
  gives `.model_dump()` and `.model_validate()` for free, eliminating every
  `to_dict()` method and every `.get(...) == expected` comparison.
- **Migration risk**: LOW. Types are underscore-private, imported nowhere
  outside their own modules. No circular risk: `features/` already imports
  nothing from `p014.data` or `p014.training`.

### Finding 2 — `ReadAloudSample` / `ReadAloudBatch` share a base shape

- **Confidence: MEDIUM**
- **Overlapping definitions**: `data.py:36` (`ReadAloudSample`, 8 fields) and
  `data.py:48` (`ReadAloudBatch`, 10 fields — same 8 plus `phone_mask`,
  `word_mask`). `move_batch_to_device` (`training.py:625`) hard-lists all 10
  fields; `collate_read_aloud_batch` (`data.py:613`) constructs all 10.
- **Proposed home**: keep in `data.py`, but extract a `ReadAloudTensors`
  mixin or `@dataclass` base holding the 8 common fields. Alternatively,
  give `ReadAloudBatch` a `to(device)` method that iterates `fields(self)`
  via `dataclasses.fields`. The latter is lower-ceremony and eliminates
  `move_batch_to_device` entirely.
- **Migration risk**: LOW. Single consumer (`training.py`), single
  constructor (`collate_read_aloud_batch`). `Dataset[ReadAloudSample]` and
  `DataLoader[ReadAloudBatch]` generic parameters update in-place.

### Finding 3 — Trackio / history records are typed as `dict[str, Any]`

- **Confidence: MEDIUM**
- **Sites**: `training.py:62` (`_TrackioHandle.local_log`), `training.py:64`
  (`log()` parameter), `training.py:100` (`init_kwargs`), `training.py:298`
  (`history: list[dict[str, Any]]`), `training.py:329` (`epoch_record`),
  `training.py:488` (`step_record`).
- `epoch_record` has a fixed, knowable shape: `epoch`, `train_loss`,
  `train_apa_loss`, `train_cono_loss`, `eval_loss`, `phone_mse`,
  `phone_pcc`, plus `word_pcc/{aspect}` × 3 and `utterance_pcc/{aspect}` × 5.
- **Proposed home**: new `src/p014/logging_schemas.py` (or inside
  `metrics.py`) with `EpochLog`, `StepLog`, `CurriculumLog` `TypedDict`s.
  `_TrackioHandle.log` stays loose (it's a pass-through) but
  `history: list[EpochLog]` is enforceable.
- **Migration risk**: LOW-MEDIUM. `TypedDict` assignments can surface real
  bugs (e.g. `word_pcc/{aspect}` keys are dynamically generated — these
  need a `Literal` union or a separate dict). Keep the loose signature on
  the trackio boundary and only tighten the in-process history.

### Finding 4 — `AttentionPoolingConfig` is dead code

- **Confidence: HIGH**
- **Definition**: `blocks.py:197` — declared but `AttentionPooling.__init__`
  takes keyword args directly; `AttentionPoolingConfig` is never instantiated
  or imported (confirmed via `rg`).
- **Proposed home**: delete, or actually thread it through `AttentionPooling`
  and `HiPPOReadAloud.__init__` so the pooling knobs come from one place.
- **Migration risk**: NONE (deletion) / LOW (adoption).

### Finding 5 — Config split is fine, but re-exports are thin

- **Confidence: LOW**
- `p014/__init__.py:3` exports only `ExperimentConfig` + two loaders.
  `TrainingRunSettings`, `ScenarioName`, and the enum set live in
  `config.py` but aren't surfaced. External callers (`cli.py`,
  `training.py`) import them via deep paths — fine internally, less
  friendly for notebooks or for a downstream P015 reuse.
- **Proposed home**: expand `p014/__init__.py` to re-export the enums and
  `TrainingRunSettings`. Optional; zero risk.

### Finding 6 — `UTTERANCE_SCORE_KEYS` / `WORD_SCORE_KEYS` are `tuple[str,...]` but semantically fixed-length

- **Confidence: LOW**
- `data.py:20-21` declares `UTTERANCE_SCORE_KEYS` (5 elements) and
  `WORD_SCORE_KEYS` (3 elements) as plain tuples. `ReadAloudAnnotation.word_scores`
  is `tuple[tuple[float, float, float], ...]` and `utterance_scores` is
  `tuple[float, float, float, float, float]` — these already encode the
  arity, but the key tuples don't.
- **Proposed home**: promote to `Literal`-typed enums or name the aspects
  as `StrEnum`s so downstream dicts (`word_pcc: dict[str, float]`) can be
  `dict[WordAspect, float]`. Ties into Finding 3.
- **Migration risk**: MEDIUM — touches `metrics.py`, `training.py`,
  `data.py`, and `EvaluationMetrics`.

### Finding 7 — `tuple[ReadAloudFeatureDataset, ReadAloudFeatureDataset, int, int]` appears twice

- **Confidence: MEDIUM**
- **Sites**: `data.py:70` (`load_read_aloud_resources`), `data.py:161`
  (`load_freespeak_resources`). Same 4-tuple; callers in `training.py:140`
  and `:153` unpack positionally.
- **Proposed home**: `@dataclass(frozen=True) class HippoResources` in
  `data.py` with `train_dataset`, `test_dataset`, `num_phone_tokens`,
  `gop_dim`. Makes the free-speak/read-aloud call sites in
  `_run_trial` self-documenting.
- **Migration risk**: LOW — two definers, one consumer.

## 4. Anti-patterns found

- **`dict[str, Any]` for structured records**: 27 occurrences. ~8 are
  legitimate (HF-dataset rows, torch checkpoints), but the rest
  (`epoch_record`, `step_record`, `history`, sidecar payloads,
  `overrides` in `cli.py:83`) have fixed shapes. Cross-reference agent
  5's weak-types audit.
- **Hand-rolled `to_dict()` on frozen dataclasses** (`ctc_gop.py:113`,
  `:136`; `ssl_utterance.py:57`) is a symptom of "should have been a
  `BaseModel`". Three separate copies.
- **Bare-dict sidecar in `data.py:464-475`** with no type at all —
  inconsistent with the equivalent records in `features/`.
- **Positional mega-tuple returns** (`tuple[Dataset, Dataset, int, int]`,
  Finding 7; `tuple[tuple[float, float, float], list[str], list[float]]`
  at `freespeak/annotation.py:50`) are hard to read — a named dataclass
  costs ~4 LOC and wins at every call site.
- **`_TrackioHandle` is mutable `@dataclass` with a mutable
  `list[dict[str, Any]]` default factory** — unusual in this otherwise
  frozen codebase; not wrong, but inconsistent.
- **No `Tuple`/`List`/`Optional` legacy forms found** — good.
- **No forward-reference strings found** — good; `from __future__ import
  annotations` is used everywhere.

## 5. Cross-cutting with the weak-types audit (agent 5)

Significant overlap with Finding 3 and the "Anti-patterns" section:

- The 27 `dict[str, Any]` occurrences should be the primary handoff list.
  Agent 5's replacement work should start from Findings 1 and 3 here
  (cache sidecars + trackio payloads) since those are the shapes most
  easily promoted to real types.
- The `cast(Any, ...)` calls at `data.py:453-455,488-489`,
  `features/ssl_utterance.py:*` for HuggingFace tokenizer/model objects
  are out of scope for consolidation but in scope for weak-types
  (consider `transformers.PreTrainedTokenizerBase` / `PreTrainedModel`).
- `AlignmentOpKind` (`freespeak/align.py:21`) is already a `Literal` —
  the pattern to emulate elsewhere (e.g. `ScenarioName` could be narrowed
  from `str` to `Literal["read_aloud", "free_speaking"]` for the
  `TrainingRunSettings.scenario` field, which already uses `Literal`).

---

**Net recommendation**: three new files —
`src/p014/features/cache_schemas.py` (Finding 1),
`src/p014/logging_schemas.py` (Finding 3), and a small `types` section added
to `data.py` (Findings 2, 7) — would absorb ~80% of the duplication without
any cross-module import risk. Deleting `AttentionPoolingConfig` is a
10-second win.
