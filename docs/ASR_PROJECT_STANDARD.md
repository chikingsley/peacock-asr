# Per-language ASR project standard (multi-model)

Status: approved 2026-06-16 (codex 5.5 xhigh, 3 rounds). Applies to every `projects/<lang>-asr`.

One uniform layout that supports **single-lane** (one model family) and **multi-model** (omni CTC + Parakeet CTC/TDT + future) with the same skeleton — so a single-lane project is "set up to do both" later by dropping in a family subpackage. The per-language project is **config only**; all logic lives in the shared packages (`omni-curator`, `omni-finetune-core`, `parakeet-finetune-core`).

## src layout — shared top-level + thin per-family subpackages (no training logic in them)

```
src/<lang>_asr/
  __init__.py  sources.py  curate.py                   # shared / curation
  assets.py                                           # only when fairseq2/Omni cards are used
  omni/        __init__.py  train.py  eval.py         # only when the omni CTC family is used
  parakeet/    __init__.py(ParakeetProject)  train.py  eval.py  artifacts.json   # only where used
```

Package name is always `<lang>_asr` (no `_omnilingual_`).

## Console commands (`[project.scripts]` entry points — NOT a scripts/ folder)

```
<lang>-curate
<lang>-omni-train   <lang>-omni-eval
<lang>-parakeet-materialize  <lang>-parakeet-train-tdt  <lang>-parakeet-eval
```

Expose only commands for the families and workflows the project actually uses. Tokenizer, CTC, and generic NeMo-recipe commands are optional, not boilerplate. Bare `<lang>-train`/`<lang>-eval` aliases are retired; use explicit family commands.

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
- **Tracking:** canonical manifests/tokenizer/final → push to Peacockery HF; track a small pointer file **outside** `data/` at `src/<lang>_asr/parakeet/artifacts.json` (local path, sha256, source command, row counts, HF repo id + revision). HF-push tooling may be staged as a follow-up; pointer files exist now.

## Rules

- Nothing in `experiments/` may be a default path in a `[project.scripts]` config object. Graduating an experiment moves its data → `data/<family>/` and code → `src/<lang>_asr/<family>/`.
- No cross-project path dependencies (a project must never import/point into another project's `src/`).
- Each language project keeps two active operator documents: `README.md` for current commands/policy/state and `EXPERIMENTS.md` for measured results. Append a completed run to `EXPERIMENTS.md`; keep the executable recipe in config/code and use Git history for retired plans, per-run diaries, source-list drafts, and superseded decisions.
- A new experiment starts from a named executable preset plus a run id, seed, exact data manifest, and model identity. It earns an `EXPERIMENTS.md` entry after its result is measured; it does not earn another Markdown file.

## Shared-core requirements (`parakeet-finetune-core`)

- Require an explicit `nemo_root` from the repo-level `base_models/parakeet/nemo_recipes` cache or env `PARAKEET_NEMO_ROOT`; **no `farsi-asr/src/finetune_parakeet` fallback.**
- Base model paths come from the repo-level `base_models/parakeet/` cache.
- Eval defaults to the stable final `.nemo`, not a `runs/.../last.ckpt`.
- Recipes are NOT packaged into the wheel (vendored NeMo tree is too heavy/churny) — they live in the shared repo-level dir above.
- Same-language Parakeet fine-tuning leaves `default_tokenizer_dir` unset and preserves the tokenizer embedded in the base `.nemo`; vocabulary replacement is only for a real language/vocabulary change.
- `parakeet-finetune-core.materialize` is the shared omni-parquet -> deterministic FLAC + NeMo JSONL bridge. A language project supplies paths and a console entry point, not another exporter.

## Rollout order

0. Core fixes above + this doc.
1. **tajik-asr** (forces every requirement) — rename, subpackages, move canonical data out of `experiments/`, drop farsi dep, namespaced commands + aliases, import/`--help` smoke tests.
1. **georgian / russian / dari** (small clones) — `omni/` subpackage + `<lang>-omni-*` + aliases.
1. **farsi-asr LAST**, as its own compatibility project (5 wheels, 2 fairseq2 extensions, active dup-card avoidance) — preserve legacy; only align the modern `farsi_asr` package.

## Testing

Per-project import + `--help` smoke tests for each command; package-level tests for core changes. (Full CI is out of scope for the refactor.)
