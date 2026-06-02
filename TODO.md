# TODO / deferred decisions

## omni-curator
- [x] **Language gate moved to export** (was label-time). `to_samples` no longer drops; the store
  keeps every clip (Scribe auto-detect gives each its own correct transcript). `Selection.language_gate`
  filters at export. ⚠️ *Remaining:* the `keep_for_language` heuristic itself is still lossy — it
  drops valid Tajik with no Tajik-only letters (e.g. `ман китоб хондам`). Now reversible (export-only),
  but consider a better gate: trusted-source mode, dictionary/stopword signal, or "reject obvious
  Russian, keep ambiguous Cyrillic."
- [ ] **HF `streaming=True` shutdown crash.** `load_hf_audio` defaults to `streaming=True`, but
  `datasets` streaming crashes at interpreter shutdown in this env (valid rows emit first). Validated
  with `streaming=False`. Investigate / default to non-streaming if it bites.
- [ ] **Dedup (audit #5/#6).** `curate.py` is near-copied across georgian/tajik — move the shared
  CLI/workflow builder into omni-curator (projects pass only language config + paths). And the
  omni-parquet schema/mixture writer is duplicated between omni-curator `export.py` and
  omni-finetune-core `parquet.py`/`mixture.py` — own it in one place or add a compat test.
- [ ] **Mechanical audit fixes.** eval `--limit` reads whole parquet first (hung on 1.2 GB — use
  `ParquetFile.iter_batches`); recipe `__main__` modules run argparse on import (add a `main()` guard);
  stale docs (tajik README documents deleted `dataset_prep`; omni-curator README/CURATING say
  export/normalize are "not yet"; metrics docstring points normalization at projects).

## tajik-asr
- [ ] **Re-ingest the legacy HF datasets.** The old `tajik_asr_combined_v0` had 3 corpora, all from
  HuggingFace (NOT Mozilla Data Collective — there is no Tajik MDC id; CV came from HF):
  - `common_voice_25_tg` — Common Voice 25, Tajik (config `tg`)
  - `fleurs_tg_tj` — FLEURS (✅ already re-ingested in the new store)
  - `muhtasham/tajik-asr-augmented-test` — small augmented set (~200 rows; historically dragged the
    Scribe WER macro — verify before trusting)
  More vetted Tajik HF candidates (not used in v0): `shunyalabs/tajik-speech-dataset` (audio+transcript),
  `WueNLP/sib-fleurs` (config `tgk_Cyrl`), `WueNLP/belebele-fleurs` (config `tgk_Cyrl`),
  `abduaziz-fleurs-cleaned`, `2m-belebele`.
  The **generic HF-audio loader now exists** (`omni_curator.ingest.huggingface.load_hf_audio`). Next:
  list these in `tajik sources.py` + wire a `curate ingest` path → store.
- [ ] **Wire the new export to train/eval (audit HIGH#2).** `tajik-curate export` writes
  `data/datasets/vN`, but `assets.py`/`eval.py`/the YAML configs still read the old
  `dataset_prep/artifacts/...`. Do this **when the first tajik export lands**: add a
  `data/datasets/v0` dataset card (like georgian's `georgian_asr_corpus`) + repoint eval. Keep the
  legacy artifacts as archived provenance.
- [ ] **Language-learning channels.** Two bilingual "teach Tajik speakers X" channels were excluded
  from `sources.py`. Now that we keep all languages, they're worth pulling — the other-language
  segments (English/Russian/etc) come in correctly transcribed by auto-Scribe and are useful data.
- [ ] **Converge `train.py` to the preset.** tajik's `train.py` loads tuned YAML configs
  (`configs/*.yaml`) while georgian builds the config in Python from `gpu_max_finetune`. The preset
  was derived from these runs, so switching tajik to it (and deleting `configs/`) would make the two
  projects byte-for-byte identical. Behavioral-equivalence check needed first.
