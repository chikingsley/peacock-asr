# Per-language ASR project standard (multi-model)

Status: approved 2026-06-16 (codex 5.5 xhigh, 3 rounds). Applies to every `projects/<lang>-asr`.

One uniform layout that supports **single-lane** (one model family) and **multi-model** (omni CTC +
Parakeet CTC/TDT + future) with the same skeleton — so a single-lane project is "set up to do both"
later by dropping in a family subpackage. The per-language project is **config only**; all logic lives
in the shared packages (`omni-curator`, `omni-finetune-core`, `parakeet-finetune-core`).

## src layout — shared top-level + thin per-family subpackages (no training logic in them)

```
src/<lang>_asr/
  __init__.py  sources.py  assets.py  curate.py      # shared / curation
  omni/        __init__.py  train.py  eval.py         # omni CTC family (always present)
  parakeet/    __init__.py(ParakeetProject)  train.py  eval.py  artifacts.json   # only where used
```

Package name is always `<lang>_asr` (no `_omnilingual_`).

## Console commands (`[project.scripts]` entry points — NOT a scripts/ folder)

```
<lang>-curate
<lang>-omni-train   <lang>-omni-eval
<lang>-parakeet-train-tokenizer  -train-ctc  -train-tdt  -train-nemo   <lang>-parakeet-eval
```

Bare `<lang>-train`/`<lang>-eval` aliases are retired. Use the explicit family commands above.

## data / artifacts

```
data/                          # gitignored (local cache; reproducibility lives in HF + tracked pointers)
  curator.sqlite  datasets/vN/  cache/      # shared curation store + omni export
  omni/final/                                # promoted omni checkpoints
  parakeet/manifests/  parakeet/tokenizers/  parakeet/final/
base_models/parakeet/            # ONE repo-level base-model/recipe cache, gitignored
  ctc.nemo  parakeet-tdt_ctc-110m-base-hybrid.nemo  nemo_recipes/
runs/ omni/<run>/  parakeet/<run>/           # live training output (ephemeral)
```

- **Eval defaults to the stable `data/<family>/final/<model>.nemo`** (or an HF revision) — never `runs/.../last.ckpt` (that's resume-only).
- **Tracking:** canonical manifests/tokenizer/final → push to Peacockery HF; track a small pointer file
  **outside** `data/` at `src/<lang>_asr/parakeet/artifacts.json` (local path, sha256, source command,
  row counts, HF repo id + revision). HF-push tooling may be staged as a follow-up; pointer files exist now.

## Rules

- Nothing in `experiments/` may be a default path in a `[project.scripts]` config object. Graduating an
  experiment moves its data → `data/<family>/` and code → `src/<lang>_asr/<family>/`.
- No cross-project path dependencies (a project must never import/point into another project's `src/`).

## Shared-core requirements (`parakeet-finetune-core`)

- Require an explicit `nemo_root` from the repo-level `base_models/parakeet/nemo_recipes`
  cache or env `PARAKEET_NEMO_ROOT`; **no `farsi-asr/src/finetune_parakeet` fallback.**
- Base model paths come from the repo-level `base_models/parakeet/` cache.
- Eval defaults to the stable final `.nemo`, not a `runs/.../last.ckpt`.
- Recipes are NOT packaged into the wheel (vendored NeMo tree is too heavy/churny) — they live in the
  shared repo-level dir above.

## Rollout order

0. Core fixes above + this doc.
1. **tajik-asr** (forces every requirement) — rename, subpackages, move canonical data out of
   `experiments/`, drop farsi dep, namespaced commands + aliases, import/`--help` smoke tests.
2. **georgian / russian / dari** (small clones) — `omni/` subpackage + `<lang>-omni-*` + aliases.
3. **farsi-asr LAST**, as its own compatibility project (5 wheels, 2 fairseq2 extensions, active
   dup-card avoidance) — preserve legacy; only align the modern `farsi_asr` package.

## Testing

Per-project import + `--help` smoke tests for each command; package-level tests for core changes.
(Full CI is out of scope for the refactor.)
