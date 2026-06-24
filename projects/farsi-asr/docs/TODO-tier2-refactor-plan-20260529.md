# Persian ASR — Tier 2 Refactor Plan (2026-05-29)

Goal: collapse the tangled `src/` into one coherent spine so the project is
holdable-in-head, kill hardcoded paths, and make "what is the dataset" legible.
Plan only — review before executing. Git HEAD is committed, so every step is revertible.

## Findings that shape the plan

- **One dependency edge only:** `persian_omnilingual_asr.* → persian_asr_dataset` (paths + normalizer). Nothing imports the omni/parakeet packages externally. So `persian_asr_dataset` is the natural shared core.
- **Duplicates are asymmetric:** the omni `curation/ingest.py` and `curation/export_nemo_manifest.py` are **supersets** of the `persian_asr_dataset/cli` copies (they add Common Voice ingest + `--sample-id-file`). The omni copies are the keepers; the `cli/` ones are stale.
- The two `ledger.py` are byte-identical (keep the `paths`-importing form).
- `vendor/` is already gitignored; `vendor/omnilingual-asr` is unused in `src/`; `vendor/nemo` is used by parakeet via `DEFAULT_NEMO_ROOT`.
- `PROJECT_ROOT` is computed as `parents[2]` in some files and `parents[3]` in others — works by accident; one `paths.py` removes the footgun.

## Target layout (one spine + two tracks)

- `persian_asr/` — the spine (today `persian_asr_dataset`): `paths.py` (all roots, env-overridable), `text_normalization.py` (was `vendor/nvidia_...`; holds `maybe_normalize`), `ledger.py`, `registry.py` (new, see below), `canonical.py`, `ingest.py` (superset), `export_nemo_manifest.py` (superset), `scribe/` (the scribe_* tools), thin `cli/`.
- `persian_omnilingual_asr/` — Track A (fairseq2): `dataset_prep/`, `curation/` (score + candidate manifests), `benchmarks/`, `training/`. Its duplicate ledger/ingest/export REMOVED.
- `persian_parakeet_asr/` (→ optionally `finetune_parakeet`) — Track B (NeMo): keep as-is, depend on spine.
- **Rule:** spine depends on nothing internal; both tracks depend only on the spine; tracks never depend on each other.

## Hardcoded absolute paths to remove (route through `persian_asr/paths.py`, env-overridable)

1. `dataset_prep/curated.py:24` DEFAULT_DATASET_ROOT → `SELECTION_ROOT/"candidate-manifests"`
2. `dataset_prep/curated.py:28` DEFAULT_OUTPUT_ROOT → `TRAINING_ROOT/"omnilingual"`
3. `dataset_prep/text_audit.py:18` fairseq2 cache hash path → derive from `FAIRSEQ2_ASSET_DIR` / `--asset-dir` (likely stale hash; verify)
4. `dataset_prep/thomcles.py:35` → `RAW_ROOT/"thomcles_persian_omni"`
5. `dataset_prep/fleurs.py:34` → `RAW_ROOT/"fleurs_fa_ir_omni"`
6. 3× `FAIRSEQ2_ASSET_DIR` setdefault sites (`score_omni_manifest.py:84`, `training/train.py:71`, `benchmarks/asr.py:33`) → single `configure_fairseq2_assets()` in paths.py
Plus delete every local `parents[2]/parents[3]` recompute in favor of importing from the spine `paths`.

## Make "the dataset" legible (the owner's core wish)

Add `persian_asr/registry.py` + a single `data/REGISTRY.yaml` (or a `datasets` table in the existing ledger sqlite — keep "one place"). Each dataset records: `id` (stable hyphen-slug), `stage` (raw|canonical|curation|selection|training-export|benchmark), `inputs` (lineage), `filter_rule` (in words), `recipe` pointer, `produced_by` (command), `path`, `discardable` flag (encodes convert-and-discard-raw). CLI `persian-dataset-registry list|show|lineage|tree` so "what is the dataset" is one command.
**Naming scheme** to kill the `max-train-...-v3-full-*` sprawl: filenames become slugs `<corpus>-<intent>-<size>[-vN]`; the descriptive guts move into the registry `filter_rule`/`notes`. One casing convention everywhere (hyphen-case).

## Entry-point rationalization (~30 → ~22)

Merge the three duplicated pairs: `persian-dataset-ingest`+`persian-ingest-corpus` → `persian-ingest`; the two `*-export-nemo-manifest` → one; the two `*-run-nemo-curator` → one. Drop the redundant `-dataset-` infix on the survivors. Retarget `persian-scribe-*` to `persian_asr.scribe.*`.

## Parakeet + rename intent

Keep the package (it's tight). Honor "finetune-parakeet / no underscores" via the **console-script names** (already hyphenated: `persian-finetune-parakeet[-ctc]`). Optionally rename the import package `persian_parakeet_asr` → `finetune_parakeet` (valid identifier). Make its `paths.py` re-export core roots from the spine.

## Vendor

- Drop `vendor/omnilingual-asr` from disk (unused in src). Current Peacock dependency policy uses the sibling editable fork at `/home/simon/github/omnilingual-asr`, not the PyPI `omnilingual-asr==0.2.0` wheel.
- Keep `vendor/nemo` out of git (already gitignored); make `DEFAULT_NEMO_ROOT` env-overridable (`PERSIAN_NEMO_ROOT`); add `docs/vendor.md` with provenance.

## Safe ordered migration (verify after each step: `ruff check src`, imports resolve, touched `--help` runs; commit each)

1. **Hardcoded paths only** (no moves): add constants to `paths.py`, rewrite the 6 sites. Lowest risk.
2. **Move normalizer** → `text_normalization.py`; rewrite 16 importers; delete `src/.../vendor/`.
3. **Unify ledger**: delete omni copy; point 2 importers at the spine ledger.
4. **Collapse supersets**: promote omni `ingest`/`export_nemo_manifest` to spine; delete stale `cli/` subsets; merge entry points; move `run_nemo_curator` ruff ignores.
5. **One paths.py**: delete omni/parakeet local PROJECT_ROOT recomputes; import from spine; parakeet paths re-exports core.
6. **Registry/lineage** (additive): add `registry.py` + CLI + backfill `REGISTRY.yaml`; document naming scheme.
7. **Vendor**: drop `vendor/omnilingual-asr`; env-overridable `DEFAULT_NEMO_ROOT`; `docs/vendor.md`.
8. **(Last, optional, highest-churn) package renames**: `persian_asr_dataset`→`persian_asr`, `persian_parakeet_asr`→`finetune_parakeet`. Mechanical global import rewrite + hatch packages + script targets.

Steps 1–7 are each independently valuable even if the renames (8) are deferred.

## Risks

- Package rename touches every cross-package import — do as one mechanical verifiable step or defer (step 8).
- omni ingest/export reference `DATA_ROOT` from the omni ledger — must repoint to spine paths.
- `run_nemo_curator` E402/PLC0415 ruff ignores are path-keyed — update pyproject in the same step.
- fairseq2 asset-dir `setdefault` ordering is load-bearing — preserve, don't "tidy".
- data-dir hyphen-case renames could hit hardcoded paths in `configs/` — grep before renaming dirs.
