# TODO — peacock-asr

Active work only, grouped by area. Completed items are in `CHANGELOG.md`.

## Curator pipeline

- [ ] Tajik v4 scale run: finish download → enqueue → segment → labelq → harvest → merge → verify → export v4 (`--max-wer 0.35`) → train (3–5 epochs / best-WER ckpt) → eval all models on v4 test + KenLM α=0.5/β=0. Dedup `tv_tajikistan`/`tvt_tojikiston`; gate-fix +1,563 clips already in store.
- [ ] Dari v0: refresh cookies → download remaining channels → stage Pimsleur Dari → create pipeline → export v0 → train cold-base vs Farsi-warm (`--regime warm_restart --lr 2e-6`), compare. Optional: Pashto language gate for bilingual channels.
- [ ] Georgian v1: wire the 28 researched channels (`docs/georgian_youtube_channels.md`) into `sources.py` → enqueue downloaded channels → first YouTube scrape → KenLM-fused eval.
- [ ] YouTube source quality: channel prescan gate (resolve handles, kill 404s, lane-routed, `data/prescan.sqlite`); category taxonomy (17-genre `category` alongside `tier`, capture title/desc/upload_date, flow to `samples`); category-stratified video-disjoint train/dev/test splits.

## Factory

- [ ] Build the polling supervisor (P1–P4 prereqs done): auto-(re)launch segment/archive/labelq per DB state, per-(project,stage) flock single-writer, auto-restart on death.
- [ ] Clip-tiering decision so overflow can't fill (factory §4; ties to Storage).

## Data lifecycle / HuggingFace

- [ ] Export final YouTube datasets → push/append to HF → delete local clips (the only durable overflow relief; segmenting nets ~0).
- [ ] `tajik-asr-youtube` community exporter (store → HF audio dataset, no WER gate, rich schema) + upload. Card drafted (`docs/hf-cards/ds-tajik-youtube.md`). (verify on HF)
- [ ] Finish in-flight uploads — models `omni-ctc-300m-tajik` (v3 step_20000), `omni-ctc-300m-persian` (scribe-v4-rewarm step_7000), `parakeet-ctc-109m-persian`; datasets `georgian-asr-corpus-v0`, `persian-asr-corpus-v4`, `tajik-asr-corpus-v3`. (verify on HF)
- [ ] Delete superseded persian model repos after uploads land (`omni-ctc-300m-v2-fleurs-fa-ir`, `...-thomcles-continue`); decide local deletion of persian scribe-v4 60G + tajik v3 71G parquets once HF copy is canonical. (verify on HF)

## Storage

- [ ] Decide: bigger overflow drive vs deliberate tiering — 1.8T overflow can't hold all live data; media holds symlinked spillover.
- [ ] Transparent re-segment-from-archive: segment reads `/mnt/storage` archive via path/symlink, not manual path-surgery.

## Cleanup / typing

- [ ] Delete deprecated shims: bare `<lang>-train`/`-eval` aliases (pyproject), `segment --procs` alias, `ScribeError.generation` field; update tests/docs to current surface.
- [ ] Source-provenance + fine-tune-policy typed models (origin/authority/tool, transform history, license gate; recipe/head/metric/eval contracts as Pydantic + validation tests).
- [ ] omni-curator carry-overs: HF `streaming=True` shutdown crash (default non-streaming if it bites); omni-parquet schema duplication between `export.py` and `parquet.py`/`mixture.py` (own in one place or compat test); mechanical fixes (eval `--limit` reads whole parquet — use `ParquetFile.iter_batches`; recipe `__main__` runs argparse on import; stale docs); language-gate heuristic lossiness (trusted-source mode / stopword signal).

## Training

- [ ] LLM ceiling bench (`omni-bench-llm`) on Tajik/Georgian/Farsi — decides whether LLM-in-the-loop (GER/NAR) work is worth it. Weights pre-cached.
- [ ] Farsi (parked): re-upload corrected card; dedup Thomcles vs `asr_farsi_youtube` on the next export.

## Someday (blocked on missing inputs)

- [ ] Native-speaker ground-truth check — the only true conversational-accuracy measure vs the Scribe-label ceiling. Not actionable until a Tajik speaker is available.

## Reference

- Factory design: `packages/omni-curator/docs/factory_plan.md`. New-language recipe + per-language data flow: `packages/omni-curator/docs/NEW_LANGUAGE.md`.
- Live per-language pipeline state is the auto-generated `STATUS.md` (`tools/status.py`) — not tracked here.
