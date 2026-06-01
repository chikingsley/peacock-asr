---
name: curate-ingest
description: Use when pulling an already-labeled ASR dataset (FLEURS via HuggingFace, or Common Voice via the Mozilla Data Collective) into a project's curator store. The first stage of the curate pipeline for sources that already have transcripts.
---

# Curate: ingest an existing-labeled dataset

Pull a dataset that **already has transcripts** into the project's master pool
(`data/curator.sqlite` + `data/canonical_audio/`). For raw audio with no transcript, use
`curate-create` instead.

`omni-curator` is a library (zero data). The ingest entry points live in the consuming **project**.
The cleanest reference is `projects/georgian-asr/` — `georgian-ingest` (defined in
`src/georgian_asr/ingest.py`).

## Commands (from the project dir, e.g. projects/georgian-asr)

```bash
uv run georgian-ingest fleurs        # google/fleurs ka_ge -> canonical_audio/fleurs + curator.sqlite
uv run georgian-ingest commonvoice   # Common Voice ka via Mozilla Data Collective
```

Each prints the ingested count and the resulting store totals + hours.

## Keys / env (root .env)

The CLI auto-loads `KEY=VALUE` lines from the **monorepo-root `.env`**.

- **`HF_TOKEN`** — for FLEURS (HuggingFace).
- **`MDC_API_KEY`** — Mozilla Data Collective API key, **required** for `commonvoice`. Missing it
  raises `set MDC_API_KEY in the root .env`.

## What happens under the hood

- **FLEURS** — `omni_curator.ingest.huggingface.load_fleurs(<flores_code>, language=..., audio_dir=..., streaming=True)`,
  upserted in batches. FLEURS already ships 16 kHz, so no resample.
- **Common Voice** — per MDC dataset id: `download_commonvoice(id, dest=raw/...)` →
  `load_commonvoice(...)` → `resample_samples(...)` (mp3 → 16 kHz mono FLAC into `canonical_audio/`)
  → `store.upsert(...)`. The MDC dataset ids are recorded in the project (see `COMMONVOICE_KA` in
  georgian's `ingest.py`) so they are never looked up again.

## Adapting to a new language project

Copy `ingest.py`, set `LANGUAGE`, the FLORES code (e.g. `fa_ir`, `tg_tj`), and the project's MDC
dataset ids. `data/raw/` is transient (re-downloadable); `data/canonical_audio/` + `curator.sqlite`
are the artifacts of record.

Flow position: **ingest** → process (built into the ingest path) → store → `curate-verify-export`.
