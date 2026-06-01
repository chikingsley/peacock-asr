# How the curator handles data

omni-curator is a **library — code, zero data.** Every function takes an explicit output path; the
**data lives in the consuming project**, where it's used (e.g. `georgian-asr/data/`). You never
point the curator at "somewhere" — you point it at the target project's artifact folder.

## The two ways data comes in

- **`create/`** — raw audio with **no transcript** (YouTube, shows). We *generate* the labels:
  segment → Scribe ensemble → compile-down → (stitch) → polish. Clean source → `chunks_path`;
  messy/multi-speaker → `vad_path`.
- **`ingest/`** — datasets that **already have transcripts**: `huggingface.py` (FLEURS, …) and
  `commonvoice.py` (Common Voice, from the **Mozilla Data Collective**).

Both yield `Sample`s, so everything downstream is source-agnostic.

## Project data layout (gitignored)

```
<project>/data/
  raw/              ← downloads, TRANSIENT. Used to process, then disposable (re-download anytime).
  canonical_audio/  ← the canonical 16 kHz mono FLAC clips. The audio of record.
  curator.sqlite    ← the MASTER POOL: every Sample from every source, with metadata.
  datasets/         ← materialized training sets, one dir per ablation (v0/, v1/, …).
```

## The flow

```
ingest / create ─► process (16 kHz · normalize · tokenizer-coverage) ─► store        [ONCE]
                                                                          (curator.sqlite + canonical_audio/)
                                                                            │
                                                            ablation recipe (filter + mixture)
                                                                            ▼
                                                                  datasets/vN  ─►  omni-finetune-core
```

- **`raw/` is transient.** Download into it, process out of it, delete it whenever. Don't depend on it.
- **The store is the master pool.** You ingest/create **once**; every clip lands in `curator.sqlite`
  with its source / split / duration / text / speaker (+ quality scores later). Single source of truth.
- **A dataset version = a recipe, not a copy.** An ablation is a *filter over the store* (e.g.
  "FLEURS + CV-scripted, drop > 30 s, dev/test from FLEURS only") + a *mixture config*, materialized
  into `datasets/vN/` as omni-parquet. Clips are shared (hardlinked), so an ablation is cheap — no
  re-download, no re-process, just a different query.

So adding `v2 = v1 + World-Speech` or `v1 = v0 − spontaneous` is a recipe over the same pool, never a
new pipeline. The one piece still to build is the **`store → datasets/vN` export** — where an ablation
is defined.
