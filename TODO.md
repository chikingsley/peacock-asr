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
- [ ] **Common Voice not re-derived.** `sources.COMMONVOICE` is empty — no Mozilla Data Collective
  dataset ids for Tajik yet, so the new store has FLEURS + YouTube but no CV. Add the MDC ids to
  re-ingest CV through the new pipeline.
- [ ] **Converge `train.py` to the preset.** tajik's `train.py` loads tuned YAML configs
  (`configs/*.yaml`) while georgian builds the config in Python from `gpu_max_finetune`. The preset
  was derived from these runs, so switching tajik to it (and deleting `configs/`) would make the two
  projects byte-for-byte identical. Behavioral-equivalence check needed first.
