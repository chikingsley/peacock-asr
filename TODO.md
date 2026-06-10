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
- [x] **Recover wrongly-dropped Tajik (gate fix) — DONE (`6259a355`).** Audit of the 38,133 drops:
  ~95% genuinely non-Tajik (Russian/English/Persian/Chinese, correctly dropped). Added a function-word
  vocabulary tiebreak to `keep_for_language` for clips with no exclusive letters either way. Measured:
  **+1,563 Tajik recovered, 0 regressions, 0 Russian re-admitted.** Applies to the next export (v4) —
  the shipped v3 predates it (worth folding into the v4 scale run, not a re-export on its own).
- [x] Tidy: deleted the 43 MB `ws_1.c4af3cd1` stub + the empty root `src/` skeleton. Gotcha noted:
  editing a training config changes fairseq2's `ws_<hash>` so a re-launch starts fresh — to resume,
  hand-move checkpoints into the new ws dir. (`runs/.../-v2/ws_1.f42fe811` (7.4 G) = the abandoned
  run-1 drift, reclaimable if space is needed — no card references it.)
- [x] ~~Retry failed verify/rescore rows~~ — non-issue: the 269 unscored are `♪`/`.`/`…` non-speech
  markers verify correctly skips and export drops.

## HuggingFace publishing (policy set 2026-06-10)

**Policy: HF (Peacockery org, public) is the archive; local is working state.** When a dataset
version or model ships (has a result in EXPERIMENTS.md), push it with a plain-prose card
(provenance, gates, the machine-label caveat, benchmark numbers); superseded versions get their
numbers recorded and are deleted locally, never pushed. Naming: models `<family>-<size>-<language>`
(one repo per family+language holding the current BEST — recipe jargon stays in the card);
datasets `<language>-asr-corpus-vN` (immutable snapshots) + `<language>-asr-<scope>` for
community sets.

- [x] Fleurs mirror dupes deleted from HF (`fleurs-parquet`, `google-fleurs`).
- [ ] **In flight (tmux `hf-upload`, resumable):** models `omni-ctc-300m-tajik` (v3 step_20000),
  `omni-ctc-300m-persian` (scribe-v4-rewarm step_7000), `parakeet-ctc-109m-persian` (the
  scribe-classified BEST — the later exact-match run regressed and stays retired); datasets
  `georgian-asr-corpus-v0` (7.8G), `persian-asr-corpus-v4` (60G), `tajik-asr-corpus-v3` (71G).
- [ ] **`tajik-asr-youtube`** — the community set: all Tajik-language-gated YouTube clips, NO WER
  gate, rich schema (audio, text, channel, video_id, duration, scribe_wer/cer) so users set their
  own quality threshold. Needs a small HF-format exporter (store -> HF audio dataset), then upload.
- [ ] After uploads verify: delete the two superseded persian model repos on HF
  (`omni-ctc-300m-v2-fleurs-fa-ir`, `...-thomcles-continue`); decide local deletion of the
  persian scribe-v4 60G parquet + tajik v3 71G parquet (HF copy becomes canonical).

## Someday / not now (needs things we don't have yet)

- **Native-speaker ground-truth check.** The one thing that would tell us our *true* conversational
  accuracy (vs the Scribe-label ceiling — we can't beat the teacher on a benchmark the teacher wrote).
  Not actionable now (no Tajik speaker available); revisit if/when one is. Until then, the held-out
  conversational WER is a rigorous *relative* benchmark, not an absolute truth.

## One pipeline, one structure (meta — after the above is moving)

- [x] **Delete the fused path + chunks/align — DONE (`a4d2f067`).** Removed ~850 lines: `create/run.py`,
  `pipeline.py`, `align.py`, `fuse/stitch.py`, `fuse/polish.py`, `segmenters/chunks.py`, the package
  `cli.py` + its `omni-curator` entry point, and tajik `cmd_label`. `cut_audio` (the one survivor the
  split needs) moved to `create/audio.py`. Package + both projects pass ruff + ty.
- [x] **`create/` reorg — DONE (`6c70295b`).** Flat, one module per stage, in pipeline order: `youtube → queue → vad → segment → transcribe → fuse → labelq`. Single-file folders gone; fuse/ package merged to fuse.py (same import path); cut_audio folded into segment.py.
- [x] **The language template (curate side) — DONE (`8823c481`), Codex xhigh reviewed.** The entire
  12-command CLI lives in `omni_curator/project.py`, parameterized by a frozen `CuratorProject`;
  tajik + georgian `curate.py` are ~45 lines of pure config, identical in shape (georgian gained
  rescore/heldout/mixture-weights for free). Ingest sources are a registry (`IngestFn`) so Persian's
  seven corpora become entries, not API changes; coverage gate stays injected via
  `omni_curator/coverage.py`; fail-fast config validation. Recipe: `docs/NEW_LANGUAGE.md`.
- [x] **The language template (model side) — DONE (`18248691`).** `omni_finetune_core/project.py`
  owns train + eval, parameterized by `FinetuneProject`: pinned `TrainingPreset`s (typed configs —
  tajik v3 proven field-equivalent to its YAML, `configs/` deleted) + georgian's generic `--regime`
  path (step budget from TRUE export hours, never the weighted TSV). `fragment_cache_dir` now
  expressible in the typed config (the /tmp-crash fix can't regress). Eval ported to core with the
  normalizer injected; 7 core tests seeded. Both projects' train/eval are thin config.
- [x] **Georgian model side** — already existed (145.3 h v0 export!); now on the template
  (`georgian-train --regime gpu_max` ready to run; eval defaults to the v0 test split).
- [x] **Migrate persian-asr — phase 1 DONE (`38b9779f`, agent-run, additive).** New
  `src/persian_asr/` template package: `persian-curate` (full 12 commands), `persian-train-v2`
  (the production scribe-v4-rewarm recipe PINNED as a typed preset — 0 field diffs vs its YAML),
  `persian-eval-v2`; production checkpoint registered as
  `omni_ctc_300m_v2_persian_scribe_v4_rewarm_production` (step_7000, dev WER 11.15). Legacy cards
  imported (not redefined — fairseq2 raises on duplicate names), zero legacy removals.
  **Phase 2 (deletion pass) follow-ups:** port the 5 legacy corpora as IngestFns
  (`persian_asr_dataset.canonical.{mana_tts,neyshekar,worldspeech,cv25,omni(thomcles)}_samples`),
  then retire `persian_asr_dataset` + `persian_omnilingual_asr` CLIs and rename `-v2` entry
  points; `finetune_parakeet` is a separate decision; freeze a Persian heldout manifest + first
  `persian-curate export v0` before any new training.

## tajik-asr (data backlog) — audited 2026-06-09; 3 of 5 items were stale/done

- [x] **Re-ingest the legacy HF datasets — DONE (`887854fe`), ingested + verified LIVE:**
  - `hf-muhtasham` (1,300 rows / 2.9 h, forced split=train — augmented data never enters the
    benchmark partition) and `hf-commonvoice22` (282 gold rows / 0.4 h, splits preserved —
    human-validated CV dev/test are legitimate benchmark corpora). All 1,582 Scribe-scored,
    0 failures. NOTE: "common_voice_25_tg" never existed on HF — Mozilla stopped publishing at
    v17 (no Tajik); CV Tajik comes from the `fsicoli/common_voice_22_0` mirror via the new
    `commonvoice_hf_mirror_source`.
  - Candidates still unwired (small, optional): `shunyalabs/tajik-speech-dataset`,
    `WueNLP/sib-fleurs` + `WueNLP/belebele-fleurs` (`tgk_Cyrl`) — one `HUGGINGFACE` registry
    line each, when wanted.
- [x] ~~Wire export → train/eval~~ — stale: done by the v3 cards + model-side template
  (`assets.py` keeps `dataset_prep/` paths only as legacy PROVENANCE cards, deliberate).
- [x] **Language-learning channels — sources wired (`887854fe`):** `learning_tajik_achilovs` +
  the three "with Chris" channels added (the vocabulary language gate makes their Tajik
  harvestable; foreign clips drop at export). `zabon_omuzishi`/`intellect_online` were already
  in. Download + label happens as part of the v4 scale run.
- [x] ~~Converge tajik `train.py` to the preset~~ — done as part of the model-side template.
- [x] ~~Mark zero-span videos~~ — non-issue while `queue.sqlite` persists: the 257 zero-span
  videos are status=segmented there, and enqueue's INSERT OR IGNORE on the video-id PK means
  they never re-segment. Only relevant if the queue db is ever wiped.

## omni-curator (carried over)

- [ ] **HF `streaming=True` shutdown crash** — `datasets` streaming crashes at interpreter shutdown here; validated with `streaming=False`. Default to non-streaming if it bites again.
- [ ] **Omni-parquet schema duplication** between omni-curator `export.py` and omni-finetune-core `parquet.py`/`mixture.py` — own it in one place or add a compat test.
- [ ] **Mechanical audit fixes:** eval `--limit` reads whole parquet first (use `ParquetFile.iter_batches`); recipe `__main__` modules run argparse on import; stale docs (tajik README, omni-curator README/CURATING, metrics docstring).
- [ ] **Language gate heuristic** (`keep_for_language`) is lossy — drops valid Tajik with no Tajik-only letters. Export-only now (reversible); consider trusted-source mode / stopword signal.
