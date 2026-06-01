---
name: curate-verify-export
description: Use after ingesting/creating data, to score the store with Scribe verification, pick a WER quality tier, and export an omni-parquet training set (ablation). The final curate stage before fine-tuning.
---

# Curate: verify + export

The store (`data/curator.sqlite`) is the master pool. Before training you (1) score every clip with
Scribe verification, then (2) export a filtered ablation to `data/datasets/vN/` as omni-parquet.
A dataset version is a **recipe over the pool, not a copy** — the store is never mutated.

## 1. Verify (Scribe-v2) — library call, no CLI

`omni_curator.verify.verify_store` runs ONE Scribe-v2 pass per clip and scores the stored label
against it via jiwer, writing `scribe_wer` / `scribe_cer` columns (+ full breakdown in
`meta["scribe"]`). It is **idempotent** — only scores rows where `scribe_wer IS NULL` unless
`force=True`. No project CLI exists yet; run it from a short script in the project venv:

```python
from pathlib import Path
from omni_curator.store import CuratorStore
from omni_curator.verify import verify_store, scribe_summary

store = CuratorStore(Path("data/curator.sqlite"))
stats = verify_store(store)               # all un-scored clips; workers=100, model="scribe-v2"
print(stats.scored, stats.skipped, stats.failed, stats.wer)  # wer = mean/median/p90
print(scribe_summary(store))              # per-source mean/median WER+CER
store.close()
```

`scribe_language` defaults to `"auto"` (Scribe detects / code-switches) — keep it; the curator's
FLORES codes (e.g. `kat_Geor`) are not Scribe ISO codes. Per-clip failures are counted, never abort.

## 2. Pick a WER tier (see QUALITY.md)

`scribe_wer` is the store-level verification score (label vs fresh Scribe), **not** a model eval WER.
Pick the tier by **recording type**:

| recording type | excellent | good | acceptable |
|---|---|---|---|
| **broadcast** (scripted CV, FLEURS, audiobooks) | 0.05 | 0.15 | 0.25 |
| **conversational** (interviews, calls, drill/show audio) | 0.15 | 0.35 | 0.60 |

Map: `commonvoice-scripted-*`, `fleurs` → broadcast; `commonvoice-spontaneous-*`, YouTube → conversational.
Coarse fallback by resource level: high ≤0.20, medium ≤0.30, low ≤0.50.

## 3. Export the ablation

```bash
# from projects/georgian-asr
uv run georgian-export v0                  # raw baseline: everything <=40s, coverage-gated, no WER gate
uv run georgian-export v1 --max-wer 0.25   # + Scribe-WER gate at broadcast "acceptable"
```

Flags (`src/georgian_asr/export.py`): `name` (dir under `data/datasets/`), `--max-wer`
(reads `scribe_wer`; **run verify first**; omit for raw baseline), `--max-duration` (default
`OMNI_MAX_DURATION_S` = 40 s, Omni's hard input ceiling — never export above it), `--no-strict`.

Prints rows / hours / by-corpus / by-split / dropped-by-quality / coverage-gate `<unk>` rows.

## Coverage gate (export-blocking)

Every export runs the omni char-tokenizer coverage audit: each normalized label must encode with
**zero `<unk>`**, or the export **fails**. `--no-strict` downgrades the failure to a warning. This
catches normalization/script mismatches before they reach training. (Tokenizer model:
`src/<project>/models/omniASR_tokenizer_written_v2.model`.)

Output `data/datasets/vN/` is omni-parquet, shared clips hardlinked (an ablation is cheap). Feed it
to `omni-finetune`.
