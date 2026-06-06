# TODO — peacock-asr

One file, all open work, by area. Status lines reflect 2026-06-05.

## Status snapshot (2026-06-05)

| What | Where it stands |
|---|---|
| Tajik YouTube curation | **DONE.** 41 channels, 342,845 clips / 1,826 h, all labeled + harvested + merged into `data/curator.sqlite` (346,025 rows with FLEURS) |
| Tajik verify (Scribe scoring) | 247,478/346,025 scored, **but the scores need a redo**: `auto` language let Scribe return Persian script for Cyrillic-labeled clips → fake WER 1.0 on ~half the channels. Full `--force` re-verify queued behind the script-aware scoring fix |
| Circuit breaker + key renewal | **DONE, proven live** (tafakkur_tv run started on a dead key, self-renewed, scored 367/367). Commits: monorepo `584e2006`, superwhisper-api `73db273` |
| Tajik training | Not started — blocked on re-verify → export |
| Persian | Nothing running. 1B eval (June 2) **lost to the 300M on every benchmark** (FLEURS 10.6 vs 8.5, CV 24.3 vs 19.4, mana 10.8 vs 6.6, youtube 26.2 vs 20.2). Final Persian model: **300M scribe-v4-rewarm** (`benchmarks/suites/canonical-tests-scribe-v4-rewarm-20260530`) |
| Georgian | On the split pipeline (committed). No model side yet (no assets/train/eval) |
| GPU | Idle (2 MiB used) — free for the tajik run once export lands |

## The data flow (reference — how it all fits)

```
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

- [ ] **Codex review** of the breaker/renewal work + structural critique (launched 2026-06-05)
- [ ] **Script-aware verify scoring.** Scribe (even forced `tgk`) deterministically returns Perso-Arabic script for some Tajik clips; same words, wrong alphabet → WER 1.0. Fix: detect hypothesis script ≠ label script → transliterate hypothesis to the label's script (free text LLM, script conversion only, raw kept in meta) → then jiwer.
- [ ] **Kill the `Any` types** in `transcribe.make_scribe_fns` / labelq `_client` / fuse client fns — real protocol types or nothing. (No `Any` where the type is known.)
- [ ] **Full re-verify**, all 346k rows, `--force`, one consistent method (overnight; free).
- [ ] **Tajik dev split decision:** keep FLEURS dev/test as the benchmarks (same as Persian); optionally carve a small YouTube dev slice **by video** (never split a video across train/dev). Then set `split` on those rows before export.
- [ ] **Export v0:** WER ≤ 0.35 + duration bounds + **junk filter**: bracket-descriptor-only labels (`[outro jingle]`, `[музыка]`) pass the WER gate (label == hypothesis) and must be dropped at export.
- [ ] **Train tajik 300M** on the export (new dataset card in `assets.py`, same benchmark suite pattern as Persian).

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
