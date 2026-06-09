# Tajik ASR — Experiment Log

Lab notebook of every fine-tuning experiment for the Tajik (`tgk_Cyrl`) ASR project,
in chronological order. Factual only; numbers are grounded in repo files
(`docs/`, `src/.../training/configs/*.yaml`, `runs/`, `README.md`). Where a number
is not recorded anywhere, it says "not recorded".

## Project at a glance

- **Model:** OmniASR CTC 300M (Facebook/Meta, via fairseq2 0.8.1; in-housed wav2vec2-asr recipe).
  Base card `omni_ctc_300m_v2_base`; tokenizer `omni_asr_tokenizer_written_v2_local`.
- **Data:** ~5.8 h real Tajik. 3 real sources after dedupe (9719 → 2589 rows):
  `fleurs_tg_tj` (1,815), `common_voice_25_tg` (572), `muhtasham_tajik_asr_augmented_test` (200).
  Splits: train 1,884 / dev 263 / test 440. Artifact `tajik_asr_combined_v0`.
- **Benchmark:** jiwer corpus-level WER/CER, whitespace-normalized. Two reference points:
  (a) **Scribe v2** (a commercial ASR used as the verifier/baseline), and
  (b) the fine-tune's own **dev** and held-out **test** splits via `tajik-eval-test`.
- **Optimizer regime (both fine-tunes):** lr 1e-5, bf16, grad-accum 2, layerwise activation
  checkpointing, no encoder freeze, `max_num_elements` 2.0M, num_steps 4,000 ceiling,
  validate every 200, keep best-WER-3 (ship best, not last).

## Summary table

| Date | Experiment | Headline result | Verdict |
|---|---|---|---|
| 2026-05-29 | Scribe v2 baseline (verifier, not a fine-tune) | dev WER 13.1% / CER 5.5% (corpus-level); macro WER 16.74% | reference baseline |
| 2026-05-29 | v0 fine-tune (real Tajik only, ~5.8 h) | dev WER **17.11%** @ step 1800; test WER 18.30% / CER 5.30% | **kept** — current best Tajik model |
| 2026-05-30 | v1 fine-tune (+9.5 h transliterated Persian FLEURS) | test WER 18.26% / CER 5.16% (Δ −0.04 / −0.14 = noise); dev WER 17.69% (worse) | **dead end** — augmentation is a wash |
| 2026-05-30 | YouTube Tajik labeling pipeline (data tooling, not a fine-tune) | 72-min video → 195 segs, 99.4% Cyrillic, 4m23s | infrastructure for the real lever (real Tajik audio) |

---

## 2026-05-29 — Scribe v2 baseline (`scribe-20260529T140514Z-444a2c81`)

- **Hypothesis / goal:** Establish a reference WER/CER for the curated data using the
  commercial Scribe ASR as a verifier, to decide what (if anything) to filter before training.
- **Changed:** No training. Ran Scribe v2 over all 2,587 retained rows; stored in the
  combined SQLite (`scribe_runs` / `scribe_transcripts` / `scribe_curation`).
- **Result (per README):**
  - Macro (per-source average): WER 16.74% / CER 8.19%.
  - By source: `fleurs_tg_tj` WER 8.95% / CER 3.03%; `common_voice_25_tg` WER 26.50% / CER 14.08%;
    `muhtasham...` WER 59.56% / CER 38.13%.
  - Corpus-level by split: train 11.7% · **dev 13.1%** · test 14.7%.
- **Verdict:** Reference baseline kept. Decided NOT to filter `muhtasham` despite its ~60% WER —
  judged to be Scribe mis-scripting (wrong alphabet), not bad data. Plan: train on v0 as-is and
  judge from real model results.

## 2026-05-29 — v0 fine-tune: real Tajik only (`tajik-asr-corpus-v0-ctc-300m-v2`)

- **Hypothesis / goal:** Fine-tune omniASR CTC 300M on the ~5.8 h real Tajik corpus as-is;
  see how close a small-data fine-tune gets to commercial Scribe, and whether data or training
  length is the lever.
- **Changed:**
  - Data: `tajik_asr_combined_v0` (train 1,884 rows / ~5.8 h), all 3 real sources, `tgk_Cyrl`.
    (Prerequisite fix: all audio resampled to 16 kHz mono — the original mixed sample rates
    [CV @ 32 kHz] had caused an OOM via fairseq2's length-batcher; see memory/README.)
  - Hyperparams: base card `omni_ctc_300m_v2_base`, lr 1e-5, bf16, grad-accum 2, layerwise
    act-ckpt, `max_num_elements` 2.0M, beta_corpus/beta_language 0.5, num_steps 4,000 ceiling,
    validate/checkpoint every 200, keep best-WER-3 + last-1.
- **Result:** Dev WER trend 19.5% (s200) → 18.3 (s1000) → **17.11% best @ step 1800** →
  plateaued/oscillating 17.1–17.4; stopped early ~step 2030. UER (~CER) ~1.6–1.9% throughout.
  Best-3 checkpoints by dev score: step_1800 (−17.106), step_1200 (−17.361), step_2000 (−17.387).
  Held-out **test** (439 rows, jiwer, measured 2026-05-30): **WER 18.30% / CER 5.30%**
  (sub/del/ins 1168/139/105).
  Head-to-head vs Scribe v2 on the same 263 dev rows: fine-tune WER 17.1% / **CER 4.3%** vs
  Scribe WER **13.1%** / CER 5.5% (they split the win: model gets characters right, word
  boundaries wrong).
- **Verdict:** **Kept — current best Tajik model.** Confirms the data is fine (model's ~1.7%
  dev CER ≪ Scribe's inflated 59% on muhtasham → Scribe was mis-scripting, not bad labels) and
  that **the lever is DATA** (5.8 h train is the ceiling), not training length.
  Best ckpt: `runs/omni-ctc-300m-tajik-asr-corpus-v0/ws_1.3dcb9e0b/checkpoints/step_1800`.

## 2026-05-30 — v1 fine-tune: + transliterated-Persian augmentation (`tajik-asr-corpus-v1-ctc-300m-v2`)

- **Hypothesis / goal:** Tajik is Persian in Cyrillic; transliterate abundant Persian (FA) audio
  transcripts FA→TG and add the Persian audio + transliterated text as "extra Tajik-like" data
  to break the 5.8 h ceiling.
- **Changed (vs v0 — only the dataset; identical hyperparameters for a clean A/B):**
  - Added corpus `persian_translit_fleurs`: Persian FLEURS train → ParsTranslit FA→TG
    (chrF++ 87.9, with the می/نمی imperfective-prefix fix) → Tajik `normalize_text` →
    ≥0.55 Tajik-vocab coverage filter → drop residual Perso-Arabic → 16 kHz FLAC → omni-parquet.
  - Mix (aggressive, user choice — "see if it works"): no hours cap. Kept **2,694 rows / 9.57 h**,
    tokenizer gate 0 unk. With `beta_corpus=0.5` this corpus ≈ 43.5% of sampled batches
    (real Tajik ≈ 56.5%). Artifact `tajik_asr_combined_v1`. Best dev ckpt = step_4000.
- **Result (439-row real Tajik test, jiwer, A/B same rows as v0):**
  - test: v1 WER **18.26%** / CER **5.16%** vs v0 WER 18.30% / CER 5.30% → **Δ −0.04 WER /
    −0.14 CER (noise)**.
  - dev: v1 WER **17.69%** vs v0 17.11% → v1 is actually **worse** on dev (+0.58).
  - Per corpus (test): common_voice_25_tg 21.88% → 22.42% (↑ worse); fleurs_tg_tj 17.91% → 17.81% (↓).
- **Verdict:** **Dead end** — fully documented in `docs/persian-augmentation-experiment-20260530.md`.
  Data verified real (not a pipeline bug). Why flat: the TEXT was transliterated but the AUDIO is
  still Iranian Persian (wrong-dialect acoustics — Iranian vs Tajik phonology diverged); the base
  Omni model already knows both languages' acoustics so nothing new is learned; FA→TG labels are
  subtly off-distribution Tajik (label noise); `beta_corpus=0.5` tempering is why it didn't HURT.
  Only micro-signal: within-register transfer (fleurs improved, conversational common_voice got
  worse). **Lever for Tajik = real Tajik audio, not transliterated Persian.** Builder/dataset/model
  kept for provenance (builder archived to `dataset_prep/archive/`); the approach is dead, not the code.

## 2026-05-30 — YouTube Tajik labeling pipeline (data infrastructure, not a fine-tune)

- **Hypothesis / goal:** Build the real lever — automatic, free labeling of real Tajik YouTube
  audio at hour scale to grow the training set beyond 5.8 h.
- **Changed:** No training. Built/proved the pipeline: NeMo frame-VAD → en+tgk Scribe ensemble →
  `compile_down` (claude-sonnet-4-6 via SuperWhisper, free) → Cyrillic label. (Decision: VAD beats
  overlapping-chunks because chunk overlap is transcribed differently by adjacent chunks → dedup
  fails; VAD has no overlap.)
- **Result:** Full 72-min "TAJIK SHOW" (`5OtJQ9d5SFw`) labeled end-to-end in **4m23s** wall-clock
  (10 workers): 195 segments, 67.4 min speech, 195/195 labeled (0 empty), median seg 25.1 s,
  **99.4% Cyrillic** (237 Latin chars = correct English code-switches; 0 Arabic, 0 romanized leakage).
- **Verdict:** Kept as the path forward. Labels are machine-generated; a native-speaker check is the
  gate before training on them. This is the lever the v1 post-mortem points to. (Labeling engine was
  later moved to the shared `omni-curator` package — commit `ee6988ae`.)

---

## 2026-06-07 — v2: the full new-pipeline corpus (~1,070 h), 300M fine-tune

- **Data:** first export off the rebuilt omni-curator split pipeline. 41 Tajik YouTube channels
  (VAD-segmented, Scribe-ensemble labeled, compile-down to Cyrillic) + FLEURS, Scribe-verified with
  **script-aware scoring** (cross-script hypotheses transliterated before WER so Perso-Arabic Scribe
  output doesn't fake WER 1.0 on correct Cyrillic labels), gated at WER ≤ 0.35 + a descriptor-junk
  filter + the language gate (37 k Russian-content clips dropped). Export `data/datasets/v2`:
  **183,140 rows / 1,070.8 h** (1,067 h train + FLEURS dev 240 / test 599 complete — curation gates
  are train-only by design, `Selection.gated_splits`). Coverage gate 0 `<unk>`.
- **Training:** two runs. Run 1 (true-hours mixture TSV, lr 1e-5) **drifted** — FLEURS dev WER rose
  20.35 → 21.68 over 4 k steps while UER improved: pure domain migration toward the ~99 %-YouTube
  mix (FLEURS was 2 % of sampled batches under sqrt tempering). Stopped, rebalanced. **Run 2**:
  hand-weighted TSV lifting FLEURS to ~12 % of sampling, lr 5e-6 + explicit tri-stage. Dev WER
  20.15 → 19.07 (8 k) → 18.01 (12 k) → 17.06 best @ **step 19500**. (Survived a /tmp-tmpfs fragment-
  cache overflow at step ~15 k — `cache_dir` moved to real disk, resumed in place from step_14500.)
  Card `omni_ctc_300m_v2_tajik_v2_step_19500`; preset `tajik-corpus-v2-300m`.
- **Result (anchored 3-way, FLEURS test, 599 rows, identical splits):**

  | model | data | WER | CER | MER | WIL |
  |---|---|---|---|---|---|
  | base (no FT) | — | 19.74 | 5.62 | 19.47 | 32.84 |
  | v0 | 5.8 h | 17.34 | 4.88 | 17.11 | 29.06 |
  | **v2** | **1,070 h** | **17.17** | 4.90 | **16.95** | **28.70** |

- **Verdict:** v2 is the **best model** (lowest WER/MER/WIL), beating base by 2.6 and edging v0. But
  **+0.17 WER over v0 from 184× the data is thin, and the benchmark is why**: FLEURS test is
  read-aloud speech, which v0's 5.8 h already adapted the model to; the v2 corpus is overwhelmingly
  *conversational* YouTube, whose value can't show on a read-aloud exam. The pipeline is proven
  end-to-end and produced our best checkpoint; **measuring what the 1,070 h actually bought needs a
  held-out conversational test set** (carve YouTube by video — never split a video across train/test)
  — that is the next experiment.

---

## 2026-06-08 — held-out conversational test set + v3 (the fair benchmark)

- **Why:** v2's win on FLEURS was thin (17.17 vs v0 17.34) because FLEURS is read-aloud and the
  v2 corpus is conversational — the benchmark couldn't see the conversational gain, and v2 had
  trained on every YouTube clip so no clean held-out existed.
- **Held-out set:** 157 **whole** conversational videos (no clip split across train/test),
  deterministic `sha1(video_id) % 50 == 0` over noisy-tier channels, frozen in
  `heldout_test_videos.json`. Carved at export (`Selection.heldout_test_videos`): held-out clips
  are gated as the train rows they are, the passing ones regroup to `split=test`, the rest of each
  held-out video drops — so no held-out video reaches train. v3 export: **1,625-clip conversational
  test** (18 channels) + FLEURS dev/test, train 180,683 (≈1,625 fewer than v2). References are
  machine-labeled (Scribe+compile-down, same ≤0.35 bar as training) — a rigorous *relative*
  benchmark, with FLEURS gold alongside as the absolute anchor.
- **Baseline (conversational held-out, 1,625 clips, pooled):**

  | model | data | WER | CER | MER |
  |---|---|---|---|---|
  | base (no FT) | — | 57.87 | 31.32 | 56.98 |
  | v0 | 5.8 h read | 49.89 | 18.88 | 48.89 |
  | v2 *(saw these clips — contaminated)* | 1,070 h | 37.40 | 13.91 | 36.32 |

- **Read:** the conversational benchmark spreads the models across **20 WER points** (58 → 50 → 37)
  where FLEURS jammed them into 3 (20 → 17 → 17) — it is the discriminating eval FLEURS could not be.

- **v3 result (the fair number, conversational held-out, 1,625 clips):** v3 (`step_20000`, FLEURS
  dev 17.41) trained on the corpus *minus* the held-out videos.

  | model | data | WER | CER | MER |
  |---|---|---|---|---|
  | v0 | 5.8 h read | 49.89 | 18.88 | 48.89 |
  | **v3** (clean) | 1,070 h | **37.65** | **14.04** | **36.59** |
  | v2 (contaminated) | 1,070 h | 37.40 | 13.91 | 36.32 |

- **Comparability rules for future versions (v4+):** (1) the test set = clips from the frozen 157
  videos *that pass the curation gates*, and gates evolve (the 2026-06-09 vocabulary-gate fix will
  admit more clips) — so **always compare models on the SAME export's test partition** (re-eval old
  models via `tajik-eval --dataset-root .../vN/version=0`), never across exports. (2) Known
  asymmetry, deliberate: FLEURS test is exported unfiltered (gold labels — never censor the exam),
  the conversational held-out IS WER-gated (machine labels — an unfiltered reference would grade
  against known-garbage labels). Consequence: the conversational number measures agreement on
  clips where Scribe agrees with itself; it is a *relative* benchmark.
- **CONCLUSION:** the 1,070 h bought a **12.24-point WER drop on real conversational Tajik**
  (49.89 → 37.65, **−24.5 % relative**; CER −25.6 %) — the data lever is real and large, and FLEURS
  hid all of it (v0→v2 there was 0.17). Critically, **v3 (37.65) ≈ v2-contaminated (37.40)**, a
  0.25-pt gap: v3 never saw these clips, v2 did, yet they match — so v2's number was *not* memorization,
  the model genuinely generalizes to conversational Tajik. **v3 is the shipping model**
  (`omni_ctc_300m_v2_tajik_v3_step_20000`) — the honest one, matching the contaminated ceiling.
  Next lever for conversational WER is *more* conversational data (the pipeline scales) and/or a
  native-speaker spot-check of the machine-labeled references.

---

## Gaps / not recorded

- v0 **test** WER/CER was first measured 2026-05-30 (the v0 entry above), not on the original
  2026-05-29 training day (GPU was on the Persian run); the README still notes it as "not yet
  measured" in its v0-vs-Scribe section.
- No fine-tune has yet been trained on the YouTube-labeled real Tajik audio — that experiment is
  pending a native-speaker label check.
- "Versions" higher than v1 (v2+) and a `tajik-derive-version` tool do not exist yet (README notes
  the gap).
