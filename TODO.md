# TODO — peacock-asr

One file, all open work, by area.

## Status snapshot (2026-06-08)

| What | Where it stands |
|---|---|
| Tajik pipeline | **DONE end-to-end.** 41 channels → 342,845 clips / 1,826 h → Scribe-verified (script-aware) → export `data/datasets/v3` (1,070.8 h, held-out conversational test carved out). |
| Tajik models | **v3 SHIPS** (`omni_ctc_300m_v2_tajik_v3_step_20000`). FLEURS test 17.2 WER; **conversational held-out: v0 49.9 → v3 37.7 WER** (the data lever, proven). v2 = the contaminated full-data run (kept for reference). |
| Circuit breaker + key renewal | **DONE, proven live**, both layers: transport-level 401→renew→retry (superwhisper-api `0dbb542`) + run-level breaker/renewal in curator (`584e2006`, `c5b4cedf`) |
| Persian | Done/parked. Final model: **300M scribe-v4-rewarm** (1B lost on every benchmark). Migration into the template structure is the only remaining Persian work (last). |
| Georgian | On the split pipeline (committed). No model side yet (no assets/train/eval). |
| GPU | Free. Nothing running. |

## The data flow (reference — how it all fits)

```text
per language project:  ONE master store: data/curator.sqlite
  ingest  (FLEURS, HF Common Voice, any HF audio dataset)  ──►  rows (split preserved: train/dev/test)
  create  (YouTube: enqueue → segment → labelq → harvest)  ──►  per-channel staging stores
          merge  (per-channel staging → master)            ──►  rows (split='train')
  verify  (one fresh Scribe pass per row → scribe_wer/cer on the row)
  export  (filter: WER gate, duration, junk → ONE omni-parquet dataset, all sources together,
           partitioned corpus=<source>/split=<split>/language=<lang>) → data/datasets/<name>
  train   (assets.py dataset card → omni-finetune-core preset)
```

- **One SQLite per language.** Per-channel stores are parallel-write staging only; merge folds them in. Everything — HF datasets and YouTube — lands in the same master table and exports together.
- **Splits:** ingest preserves the source's dev/test (FLEURS dev/test = the fixed benchmarks). Created (YouTube) data is all `train` today — see "dev split" task below.
- **Naming:** exports are ablations `data/datasets/v0, v1, ...`; the version is the dataset axis, benchmarks stay fixed.

## Now / next (ordered)

- [x] Codex review of the breaker/renewal work — all 7 findings fixed (`c5b4cedf`)
- [x] Script-aware verify scoring (Sonnet transliteration won the bake-off vs uroman skeleton; hypothesis-only prompt, garbage control clean) + `rescore` CLI
- [x] Kill the `Any` types (ProcessFn/Mapping/SuperwhisperClient throughout)
- [x] Full verify + 150k-row rescore — every scoreable row has an honest score
- [x] Dev split: FLEURS dev/test are the benchmarks (complete in export; gates are train-only by design — `Selection.gated_splits`)
- [x] Export v2: WER ≤ 0.35 + descriptor-junk filter + language gate; 0 unk
- [x] Train tajik 300M launched (v2 preset, 20k steps)
- [x] Train tajik v2 + eval: best **step_19500, FLEURS test WER 17.17** (base 19.74, v0 17.34). Recorded in EXPERIMENTS.md. v2 wins but margin over v0 is thin — FLEURS can't show the conversational gain.
- [x] **Conversational test set — DONE, the data lever is proven.** Held-out 157 whole videos (frozen manifest, leakage-safe carve at export). On the conversational held-out (1,625 clips): v0 49.89 → **v3 37.65 WER** (−12.2 pts / −24.5% rel from 1,070h), and v3≈v2-contaminated (37.65 vs 37.40) so it's real generalization not memorization. **Shipping model: `omni_ctc_300m_v2_tajik_v3_step_20000`.** Recorded in EXPERIMENTS.md.
- [ ] **Scale conversational data — THE next lever (not started).** Everything so far trained on the
  ~1,070 h we already had. The held-out proved more conversational data → lower conversational WER, so:
  queue more channels and/or more videos per existing channel → `download → enqueue → segment → labelq
  → harvest → merge → verify → export v4 → train`. Same recipe, bigger corpus. Biggest expected win.
- [ ] **(Optional, before scaling) native-speaker spot-check** of ~100 machine-labeled conversational
  clips — the only way to know our *true* accuracy vs the Scribe-label ceiling (we can't beat the
  teacher on a benchmark the teacher wrote). Tells us if label quality, not data volume, is the cap.
- [ ] Tidy: delete the 43 MB empty `runs/.../v2-r2/ws_1.c4af3cd1` stub (crash-recovery leftover; the
  shipped model is in `ws_1.a8b9ba67`). Note the gotcha: editing a training config changes fairseq2's
  `ws_<hash>` so a re-launch starts fresh — to resume, hand-move checkpoints into the new ws dir.
  (The "269 unscored rows" are `♪`/`.`/`…` non-speech markers verify correctly skips and export drops —
  nothing to retry.)

## One pipeline, one structure (meta — after the above is moving)

- [ ] **Delete the fused path.** `create/run.py` (`label_to_store`, `label_youtube`), `pipeline.py` (`vad_path`, `chunks_path`, `_label_spans`), tajik `cmd_label`, package `cli.py` fused commands. Prereq: relocate `cut_audio`/`audio_duration` (used by the split's `segment.py`).
- [ ] **Delete chunks/align.** (`align.py`, `fuse/stitch.py`, `chunks_path`.) Built for scripted/clean audio, never used by any project — VAD handled everything including audiobooks. One pipeline. Git history keeps it if ever wanted.
- [ ] **`create/` reorg** by stage: no single-file folders, no folder-with-one-folder. Stage-ordered: sources → queue/segment → transcribe/labelq → fuse → harvest.
- [ ] **The language template.** "New language" = copy 5 files (`sources.py`, `curate.py`, `assets.py`, `train.py`, `eval.py`), fill `sources.py`, add ONE normalizer function in the package. Georgian + tajik conformed byte-for-byte; documented in omni-curator. Codex xhigh review before locking.
- [ ] **Dedup `curate.py`** across georgian/tajik — move the shared CLI/workflow builder into omni-curator (projects pass only language config + paths). Part of the template work.
- [ ] **Georgian model side:** `assets.py`, `train.py`, `eval.py` (copy from the template once locked).
- [ ] **Migrate persian-asr** into the template structure (LAST — preserve all artifacts/checkpoints as legacy cards; the 300M rewarm is the production model).

## tajik-asr (data backlog)

- [ ] **Re-ingest the legacy HF datasets** (the "old tajik data" — all HuggingFace, no Tajik MDC):
  - `common_voice_25_tg` — Common Voice 25, Tajik (HF config `tg`)
  - `fleurs_tg_tj` — ✅ already re-ingested
  - `muhtasham/tajik-asr-augmented-test` (~200 rows; historically dragged Scribe WER macro — verify before trusting)
  - candidates: `shunyalabs/tajik-speech-dataset`, `WueNLP/sib-fleurs` (`tgk_Cyrl`), `WueNLP/belebele-fleurs` (`tgk_Cyrl`)
  The generic loader exists (`omni_curator.ingest.huggingface.load_hf_audio`); wire ids into `sources.py` + `curate ingest`.
- [ ] **Wire export → train/eval:** `assets.py`/`eval.py`/YAMLs still point at legacy `dataset_prep/artifacts/...`; add the `data/datasets/v0` card when the first export lands.
- [ ] **Language-learning channels** ("teach Tajik speakers X") + extra Achilov channels → enqueue. Channel policy: meaningful-%-Tajik gets downloaded; only pure music/song channels skipped.
- [ ] **Converge tajik `train.py` to the preset** (georgian builds config in Python from `gpu_max_finetune`); behavioral-equivalence check first, then delete `configs/`.
- [ ] Mark zero-span (no-speech) videos in the queue so future enqueues skip them instead of re-segmenting.

## omni-curator (carried over)

- [ ] **HF `streaming=True` shutdown crash** — `datasets` streaming crashes at interpreter shutdown here; validated with `streaming=False`. Default to non-streaming if it bites again.
- [ ] **Omni-parquet schema duplication** between omni-curator `export.py` and omni-finetune-core `parquet.py`/`mixture.py` — own it in one place or add a compat test.
- [ ] **Mechanical audit fixes:** eval `--limit` reads whole parquet first (use `ParquetFile.iter_batches`); recipe `__main__` modules run argparse on import; stale docs (tajik README, omni-curator README/CURATING, metrics docstring).
- [ ] **Language gate heuristic** (`keep_for_language`) is lossy — drops valid Tajik with no Tajik-only letters. Export-only now (reversible); consider trusted-source mode / stopword signal.
