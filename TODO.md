# TODO / deferred decisions

## omni-curator
- [ ] **Language gate placement.** The Tajik-vs-Russian content gate (`keep_for_language` in
  `omni_curator/process/language.py`) currently runs at **label time** — called in `to_samples`
  (`create/pipeline.py`), so non-Tajik clips never enter the store. This is inconsistent with the
  curator's "store everything, filter at the export recipe" model (how the WER tiers work).
  Consider moving it to **export** so the store keeps everything (with a correct per-clip detected
  language tag) and the dataset recipe filters by language — more reversible/inspectable. Outcome
  is identical either way; this is about where the decision lives. (Left at label time for now.)

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
  Ingesting these needs a **generic HF-audio loader** in omni-curator (today only `load_fleurs` +
  the MDC `load_commonvoice` exist). Then list them in `tajik sources.py` and ingest → store.
- [ ] **Converge `train.py` to the preset.** tajik's `train.py` loads tuned YAML configs
  (`configs/*.yaml`) while georgian builds the config in Python from `gpu_max_finetune`. The preset
  was derived from these runs, so switching tajik to it (and deleting `configs/`) would make the two
  projects byte-for-byte identical. Behavioral-equivalence check needed first.
