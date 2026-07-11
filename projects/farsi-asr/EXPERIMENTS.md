# Persian (Farsi) ASR — Experiment Log

Lab notebook of every fine-tuning experiment for the Persian (Farsi, `fa_ir`) ASR project, in chronological order. Factual only; numbers are grounded in repo files (`docs/`, `src/finetune_omni/training/configs/*.yaml`, `benchmarks/suites/*/summary.md`, `README.md`). Where a number is not recorded anywhere, it says "not recorded".

## Project at a glance

- **Two model tracks:**
  - **Parakeet CTC 109M** (NVIDIA NeMo): base `models/parakeet-ctc-109m/ctc.nemo`, custom Persian BPE tokenizer `fa_spe_bpe_v1024`. AdamW lr 1e-4, warmup 5000 (1500 for the smaller run), min_lr 5e-6, bf16, Lhotse dynamic bucketing (batch_duration 700, 30 buckets), early-stop on `val_wer` (patience 5).
  - **OmniASR CTC 300M** (Facebook/Meta via fairseq2): base card `omniASR_CTC_300M_v2`, tokenizer `omniASR_tokenizer_written_v2`. lr 1e-5, bf16, grad-accum 2 (8 for tiny sets), layerwise activation checkpointing, `max_num_elements` 3.84M, beta_corpus/language 0.5.
- **Benchmark = the six-split canonical suite**, jiwer WER/CER, run via `persian-benchmark-suite`, results under `benchmarks/suites/`. Splits: common_voice_25, fleurs, mana_tts, neyshekar, worldspeech, youtube (all `test`). Reference points: **Scribe v2** (commercial verifier) and the untrained **omni-base-ctc-300m-v2** (control, ~100% WER on Parakeet base; the Omni base is usable).
- **Note on dates:** the project was first git-tracked 2026-05-29 (commit `34a9013d`); many experiments predate tracking and are dated from their config or dataset names. The `-current-20260528` suite folders are **re-scores** of earlier-trained models on 2026-05-28 with the live normalizer (a normalizer fix cut ~6.5 WER points off every model without retraining), so those numbers are comparable to each other but newer than the original training date.

## Canonical benchmark provenance

The six benchmark splits are canonical test sets, not the same thing as the Scribe-v4 training surface. In training docs, **Scribe** means verifier/filter unless explicitly stated otherwise.

| Split             | Test rows / hours | Source config                                       | Label/reference provenance                                                                                                                                                                 |
| ----------------- | ----------------: | --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `common_voice_25` |   10,702 / 14.69h | `mozilla_data_collective/cv-corpus-25.0/fa`         | Upstream Mozilla Common Voice native test split; CC0; community/human reference text.                                                                                                      |
| `fleurs`          |       852 / 3.60h | `google/fleurs/fa_ir`                               | Upstream Google FLEURS native test split; CC-BY-4.0; official reference text.                                                                                                              |
| `mana_tts`        |     3,989 / 5.35h | `MahtaFetrat/Mana-TTS`                              | Synthetic/TTS source; derived canonical split; metadata carries ASR match quality/CER fields used during cleanup.                                                                          |
| `neyshekar`       |     1,331 / 2.05h | `Peacockery/neyshekar-v3-asr-aligned`               | External Neyshekar corpus, carried locally as the repaired/aligned `data/raw/neyshekar_v3_asr_aligned` mirror; source references after alignment/repair, not Scribe/YouTube pseudo labels. |
| `worldspeech`     |       359 / 1.33h | `disco-eth/WorldSpeech/fa_ir`                       | External WorldSpeech source; native test plus derived dev; CC-BY-NC-4.0; metadata includes ASR/CER/DNSMOS fields.                                                                          |
| `youtube`         |   13,899 / 33.35h | `pourmand1376/asr-farsi-youtube-chunked-10-seconds` | Legacy external chunked YouTube corpus; native split, source URLs retained, license unknown; separate from the new curated channel registry.                                               |

## Summary table (six-split macro WER / headline)

| Date (trained) | Experiment                                         | Headline (macro WER, key splits)                               | Verdict                    |
| -------------- | -------------------------------------------------- | -------------------------------------------------------------- | -------------------------- |
| ~early         | Parakeet base 109M (control)                       | 100% WER all splits                                            | control                    |
| ~early         | Parakeet broad-plus-mana 109M                      | CV 74.8% / neyshekar 99.4% (catastrophic)                      | dead end                   |
| ~early         | Parakeet broad-filtered 109M                       | all splits 46–100%                                             | dead end                   |
| ~2026-05       | Omni FLEURS-only 300M                              | FLEURS 14.70% but CV 41.9% / YT 52.8%                          | superseded (narrow)        |
| ~2026-05       | Omni FLEURS→Thomcles 300M                          | FLEURS 16.50%, CV 36.5%                                        | superseded                 |
| ~2026-05       | Omni clean-100h / balanced-100h / target-100h 300M | best ~CV 25–30%                                                | superseded by Scribe data  |
| ~2026-05       | Omni wer35-fastconformer-filtered 300M             | CV 23.8% / FLEURS 11.8%                                        | superseded                 |
| 2026-05-24     | **Parakeet scribe-classified 844h 109M**           | macro WER 22.56% (best broad parakeet)                         | kept (broad reference)     |
| 2026-05-25     | Parakeet scribe-exact-match 224h 109M              | macro WER 33.8% (regressed)                                    | dead end                   |
| 2026-05-25     | Omni scribe-exact-match 224h 300M                  | CV 21.9% / FLEURS 9.8%, but mana/neyshekar ↑                   | kept (clean-domain parent) |
| 2026-05-27     | Omni scribe-v3-max 1032h 300M (continue)           | CV 19.6% / FLEURS 8.85% (topped board) but ZWNJ-contaminated   | superseded by v4           |
| 2026-05-30     | **Omni scribe-v4 (clean re-export) 300M, 34k**     | CV 19.4% / FLEURS 8.7% / mana 6.6% / neyshekar 8.5% / YT 20.3% | **kept — current best**    |
| 2026-05-30     | Omni scribe-v4 re-warm 10k (lr 2e-6) 300M          | ≈ v4 baseline (tiny mixed deltas)                              | dead end (no further gain) |

______________________________________________________________________

## Parakeet 109M — early baselines (dates not recorded; pre-tracking)

### Parakeet base 109M — control (`parakeet-110m`)

- **Goal:** Sanity floor — the untuned NeMo Parakeet base on Persian.
- **Result:** 100.00% WER / 100.00% CER on every split (no Persian capability).
- **Verdict:** Control only — establishes that all Persian skill comes from fine-tuning.

### Parakeet broad-plus-mana 109M (`parakeet-broad-plus-mana`)

- **Changed:** Broad early Persian set + Mana TTS; recipe `batch_duration 700`, lr 1e-4, warmup 5000, the predecessor run `parakeet-ctc-109m-broad-plus-mana-20260515` (best epoch 12, val_wer 0.2926).
- **Result:** FLEURS 13.99% / mana 28.51% / youtube 33.32% but **common_voice 74.82%** and **neyshekar 99.42%** (CER ~98%) — catastrophic on those domains.
- **Verdict:** Dead end as a broad model — collapses on Common Voice / Neyshekar.

### Parakeet broad-filtered 109M (`parakeet-broad-filtered`)

- **Result:** Every split 46–100% WER (CV 46.4%, FLEURS 57.8%, neyshekar 99.9%).
- **Verdict:** Dead end — the filtering hurt rather than helped.

## Omni 300M — early data-recipe sweep (dates not recorded; pre-Scribe; re-scored 2026-05-28)

### Omni FLEURS-only 300M (`fleurs-fa-ir-ctc-300m-v2-finetune` → `omni-fleurs-fa-ir`)

- **Goal:** Fine-tune Omni 300M on FLEURS Persian alone (cleanest read speech) as a first target.
- **Changed:** Dataset `fleurs_fa_ir`; grad-accum 8, `max_num_elements` 960k, num_steps 5,000, validate/checkpoint every 500.
- **Result:** FLEURS 14.70% / 3.83% but CV **41.92%**, youtube **52.79%**, neyshekar 40.38%.
- **Verdict:** Superseded — strong on its own domain, useless broadly (too little / too narrow data).

### Omni FLEURS→Thomcles 300M (`thomcles-ctc-300m-v2-continue-from-fleurs` → `omni-fleurs-thomcles`)

- **Changed:** Continue from the FLEURS-final checkpoint (`omni_ctc_300m_v2_fleurs_fa_ir_final`) on `thomcles_persian_farsi_speech`; same 5,000-step regime.
- **Result:** FLEURS 16.50% (worse than FLEURS-only), CV 36.45%, youtube 36.41% (better than FLEURS-only).
- **Verdict:** Superseded — adding Thomcles broadened it slightly but still far from usable.

### Omni 100h-filter family: clean / balanced / target (300M, 15k steps)

- **Goal:** Compare three 100-hour selection strategies from the curation ledger (`persian_asr_clean_100h_filter`, `persian_asr_balanced_100h_filter`, `persian_asr_target_100h_filter`), each num_steps 15,000, grad-accum 2, `max_num_elements` 3.84M.
- **Result (six-split, key columns; CV / FLEURS / mana / neyshekar / youtube WER):**
  - **target-100h** (`omni-target-300m`): 29.89 / 12.17 / 31.42 / 30.74 / 34.73.
  - **balanced-100h**: 28.78 / 12.27 / 31.28 / 30.39 / 35.14.
  - **clean-100h**: 24.98 / 12.88 / 26.87 / 27.07 / 40.88.
- **Verdict:** Superseded by the Scribe-verified data tracks. clean-100h best on CV/mana/neyshekar, target/balanced best on youtube; none competitive with the later Scribe runs.

### Omni wer35-fastconformer-filtered 300M (`omni-wer35-fastconformer`)

- **Changed:** Data filtered by FastConformer-CTC WER ≤ 35% (NeMo Curator pass).
- **Result:** CV 23.79% / FLEURS 11.81% / mana 30.57% / neyshekar 25.76% / youtube 35.13%.
- **Verdict:** Superseded — best of the pre-Scribe Omni runs on several splits, but Scribe-verified data later beat it everywhere.

## 2026-05-24 — Parakeet scribe-classified 844h 109M (`parakeet-ctc-109m-scribe-classified-20260524-bd700-min10`)

- **Hypothesis / goal:** Use Scribe as a verifier/filter (not a label) to keep 571k diverse rows across 5 "close-enough" categories (exact_match, near_match, punctuation_or_orthography_only, number_or_symbol_mismatch, named_entity_mismatch) → broad, high-quality 844h training set.
- **Changed:** Data `train.filtered-maxchars400-cps60` = 520,812 rows / 844.35 h (dev 24,434 / 42.51 h). Recipe: Parakeet 109M, Lhotse bd700/qd15/30-buckets, lr 1e-4, warmup 5000, min_lr 5e-6, min_steps 43,420 / max_steps 130,260, val every ~4342, early-stop patience 5.
- **Result:** Stopped at the hard step cap (max_steps 130,260) at epoch 21, best `val_wer` 0.20084. Held-out Scribe test (23,910 rows): WER **20.69%** / CER 6.63%. Six-split canonical (CV/FLEURS/mana/neyshekar/worldspeech/youtube WER): 22.89 / 17.69 / 12.58 / 18.65 / 36.18 / 27.39 — **macro WER 22.56% / macro CER 8.58%**.
- **Verdict:** **Kept — the best broad six-split balance among Parakeet runs** (and the reference broad model). Strong on mana/CV; weaker than Omni on FLEURS.

## 2026-05-25 — Parakeet scribe-exact-match 224h 109M (`parakeet-ctc-109m-scribe-exact-match-20260525-bd700`)

- **Hypothesis / goal:** Train only on `exact_match` rows (Scribe == reference) — cleanest possible labels — to test whether exact-only beats the broader classified set.
- **Changed:** Data 227,474 rows / 246.40 h (CV 77.86%, mana 13.13%, neyshekar 5.23%, …), train 210,217 / 224.34 h. Same Parakeet recipe, warmup 1500, min/max steps 11,540 / 34,620.
- **Result (six-split WER):** CV 34.64 / FLEURS 27.15 / mana 20.64 / neyshekar 27.88 / worldspeech 46.44 / youtube 46.22 — **macro WER 33.83% / macro CER 13.03%** (re-scored `parakeet-scribe-exact`: 34.81 / 24.49 / 32.49 / 36.12 / 50.02 / 49.10).
- **Verdict:** **Dead end** — exact-only on the 109M regressed badly vs the 844h classified run. Confirms broad-but-verified > tiny-but-perfect for Parakeet.

## 2026-05-25 — Omni scribe-exact-match 224h 300M (`persian-asr-scribe-exact-match-20260525-ctc-300m-v2`)

- **Hypothesis / goal:** Same exact-match data, but on Omni 300M (which had been the stronger base on clean domains) — and as the parent checkpoint for later continued-training runs.
- **Changed:** Dataset `persian_asr_scribe_exact_match_20260525` (210,217 train / 9,429 dev rows; 224.34 h). Omni 300M, num_steps 34,000, validate/checkpoint every 1,000.
- **Result:** Finished at step 34,000; best validation at step 32,000 (card `omni_ctc_300m_v2_persian_scribe_exact_match_20260525_best`). Six-split WER: CV 21.87 / FLEURS 9.81 / mana 27.59 / neyshekar 28.32 / worldspeech 38.78 / youtube 36.31 — macro WER 27.11% / macro CER 8.60%.
- **Verdict:** **Kept as the clean-domain parent.** Strongest of the recent runs on CV and FLEURS, but WER much worse on mana/neyshekar/youtube (clean-domain win, not a broad recipe). Became the parent checkpoint for v3-max and v4.

## 2026-05-27 — Omni scribe-v3-max 1032h 300M, continue (`persian-asr-scribe-v3-max-20260527-ctc-300m-v2-continue`)

- **Hypothesis / goal:** Continue from the exact-match best checkpoint on the much larger "max defended surface" v3 (588K rows / 1032 h; all 5 keep-categories) to get a single strong broad model.
- **Changed:** Parent `omni_ctc_300m_v2_persian_scribe_exact_match_20260525_best`; dataset `persian_asr_scribe_v3_max_20260527`; num_steps 34,000. (Benchmarked at step 38,000.)
- **Result (six-split WER, step 38000):** CV 19.64 / FLEURS 8.85 / mana 35.72 / neyshekar 36.74 / worldspeech 36.63 / youtube 33.95. Topped the board on CV/FLEURS while it ran.
- **Verdict:** **Superseded by v4.** v3 was trained on ZWNJ-bearing labels but the Omni tokenizer has no ZWNJ piece → ~33% of its benchmark rows emit `⁇` (contaminated). The mana/neyshekar WER spikes reflect this. The locked decision is to map ZWNJ to a space and retrain on that clean surface rather than add ZWNJ to the tokenizer.

## 2026-05-30 — Omni scribe-v4 (clean re-export) 300M, 34k (`persian-asr-scribe-v4-ctc-300m-v2-continue`)

- **Hypothesis / goal:** v3's exact data, re-exported cleanly through the fixed normalizer (strips Cf+So categories incl. ZWNJ → space; validated **0 `<unk>`** over all 704,455 canonical rows) → remove the `⁇` contamination → best model.
- **Changed:** Dataset `scribe_v4` (563,749 train+dev rows; clean re-export of the v3 surface). Parent = the exact-match best checkpoint; same Omni regime, num_steps 34,000. Run dir `runs/scribe-v4/ws_1.760d57f2`.
- **Result (six-split WER / CER):** CV **19.37 / 4.72**, FLEURS **8.69 / 2.39**, mana **6.56 / 1.80**, neyshekar **8.49 / 1.91**, worldspeech 27.45 / 17.25, youtube **20.34 / 9.71**. Dev WER **11.27%** at step 34,000 (still best at the last step — no overfit).
- **Verdict:** **Kept — current best Persian model.** Massive drop vs v3 on mana (35.7→6.6) and neyshekar (36.7→8.5) — i.e. the v3 numbers there were the `⁇` contamination, now gone. Clean data + clean tokenizer is the decisive lever.

## 2026-05-30 — Omni scribe-v4 re-warm 10k, lr 2e-6 (`persian-asr-scribe-v4-rewarm10k-ctc-300m-v2`)

- **Hypothesis / goal:** Dev-WER was still improving at v4's last step. Test whether a warm restart (fresh optimizer + fresh tri_stage schedule peaking at 2e-6 = 20% of the original lr) squeezes more out past 11.27% dev-WER. (A companion config `...-continue-44k` planned a plain +10k resume; the re-warm is the run that was benchmarked.)
- **Changed:** Load v4 step_34000 weights via card `omni_ctc_300m_v2_scribe_v4_20260530_best`, fresh 10k-step run, lr 2e-6, explicit tri_stage [0.1/0.4/0.5], keep best-WER-3.
- **Result (six-split WER / CER):** CV 19.41 / 4.71, FLEURS **8.51 / 2.36**, mana 6.61 / 1.79, neyshekar **8.22 / 1.86**, worldspeech 27.61 / 17.30, youtube **20.19 / 9.66**.
- **Verdict:** **Dead end (no meaningful gain).** Mixed sub-0.2-point deltas vs the v4 baseline — marginally better on FLEURS/neyshekar/youtube, marginally worse on CV/worldspeech. Within noise; the v4 baseline remains the model of record.

## 2026-06-22 — KenLM fused CTC beam eval (`omni_ctc_300m_farsi_hf`, decode-only)

- **Goal:** Promote the LM decode experiment from loose scratch script to repeatable project command and check whether fixed KenLM fusion helps the current HF re-eval card.
- **Command:** `uv run --project projects/farsi-asr farsi-omni-eval-lm --benchmark <split> --alpha 0.3 --beta 0.0 --device cuda`
- **Setup:** Model card `omni_ctc_300m_farsi_hf`, checkpoint `data/benchmarks/model/model.pt` sha prefix `c5e0cff9d6a2`; KenLM `experiments/lm_decoding/lm4.bin`; corpus unigrams 78,971; tokenizer `omniASR_tokenizer_written_v2`. The promoted command verifies tokenizer identity, infers blank index 0, and aborts if active multi-character tokenizer pieces would corrupt `pyctcdecode`.
- **Logs:** `experiments/lm_decoding/official_fixed_a0.3_b0.0/*.log`; every full run ended with `LM_RUN_DONE`.

| Split             |   Rows | Greedy WER/CER |  Beam WER/CER | Beam+KenLM WER/CER | WER delta vs greedy |
| ----------------- | -----: | -------------: | ------------: | -----------------: | ------------------: |
| `common_voice_25` | 10,702 |   22.69 / 5.88 |  22.48 / 5.83 |   **13.34 / 3.77** |               -9.35 |
| `fleurs`          |    852 |   12.30 / 3.16 |  12.32 / 3.16 |   **10.66 / 2.85** |               -1.64 |
| `mana_tts`        |  3,989 |   14.26 / 3.38 |  14.14 / 3.37 |   **13.44 / 3.27** |               -0.82 |
| `neyshekar`       |  1,331 |   14.21 / 3.15 |  14.14 / 3.13 |   **11.68 / 2.61** |               -2.53 |
| `worldspeech`     |    359 |  33.54 / 18.55 | 33.47 / 18.58 |  **31.95 / 18.27** |               -1.59 |
| `youtube`         | 13,899 |  23.02 / 10.34 | 22.84 / 10.25 |   **19.60 / 9.66** |               -3.42 |

- **Macro six-split result:** greedy 20.00 WER / 7.41 CER; plain beam 19.90 / 7.39; beam+KenLM 16.78 / 6.74.
- **C1Tech comparison run:** 2,101 rows; greedy 18.46 / 5.64; plain beam 18.50 / 5.73; beam+KenLM 17.09 / 5.47. This matches the prior scratch result and confirms the promoted command is reproducing the old C1Tech baseline.
- **Verdict:** Keep `alpha=0.3 beta=0.0` as the current fixed KenLM setting for this card. Plain beam alone is nearly neutral; the LM carries the improvement, especially on Common Voice and YouTube. This is an inference/decoding result only and does not replace the fine-tuned model result of record above.

______________________________________________________________________

## 2026-06-24 — Normalizer regression found + eval ref-fix; FLEURS recovered

Auditing the promoted `farsi-omni-eval-lm` against the C1Tech benchmark surfaced two **measurement** bugs, neither of which is a model problem:

1. **Normalizer regression.** The standardization migration (`50fd3be6`, `0a7eb1bb`) replaced the proven NVIDIA fastconformer normalizer (`maybe_normalize` — ZWNJ → space, the surface every Omni CTC v2 model trained on) with a freshly-written hazm normalizer that **strips** ZWNJ, gluing morphemes (`می‌خوام` → `میخوام`). That changed word boundaries and inflated WER. Recovered the original from git (`1354ecc9`) and ported it back into `omni_curator.process.normalize.normalize_persian`, pinned to the upstream NVIDIA revision + README sha.
1. **Eval scored references raw.** `print_score` normalized the hypothesis but not the reference, so a raw FLEURS transcription (ZWNJ/punctuation intact) was compared against a normalized hypothesis. Fixed: reference and hypothesis now pass through the same normalizer (decision rule 2).

**FLEURS recovered** (300-row, fixed surface): greedy **16.51% → 8.46%** WER / 2.34% CER, matching the recorded `scribe-v4` FLEURS of 8.69% — the earlier 16.51% was purely the two eval bugs, not a model regression. KenLM still helps, more clearly on the clean surface: **8.46 → 6.91%** WER (−1.55 pts, −18% rel), α=0.3 / β=0.0; plain beam is neutral (8.40%).

**C1Tech corrected.** On the true (train-consistent) surface, C1Tech greedy is **20.46%** — the 18.46% in the KenLM section above is the superseded hazm surface (a coincidentally lower surface, not the one the model learned). The LM gain (~1.4 pts) carries over to the corrected surface.

**num2words:** confirmed *not* part of the trained surface — the NVIDIA normalizer keeps digits as digits, so expanding numbers would diverge from what v4 learned. Deferred to a deliberate retrain.

**Forcing layer.** Added `packages/omni-curator/tests/test_normalize.py`: pinned input→output cases per language + an invariant that no `Cf` (ZWNJ/bidi/BOM) character survives normalization. A refactor that changes normalization behavior now fails loudly instead of silently shifting every WER number — which is exactly how this regressed in the first place.

______________________________________________________________________

## 2026-07-10 — A0: N-best persistence + oracle WER gate (decode-only)

- **Goal:** Extend `farsi-omni-eval-lm` to retain top-N unique beam hypotheses with acoustic and KenLM scores, persist them in the shared benchmark SQLite store, and measure oracle WER to gate neural rescoring (A3 requires ≥2 WER points of top-16 oracle headroom on dev).
- **Tooling added:** `--nbest N` (top-N unique candidates via `pyctcdecode.decode_beams_batch`, oracle WER at cutoffs 1/4/8/16, duplicate-rate report, new `nbest_candidates` table in `asr-benchmark-core` keyed `(run_id, row_index, rank)`) and `--logits-dir` (fp16 log-prob cache, ~5.5 MB/row at vocab 10,288; a complete cache decodes on CPU with the GPU untouched — verified bit-identical WER against the live path on a 20-row smoke). Run IDs now embed the benchmark name (`omni-ctc-300m-farsi-<benchmark>-...`); prediction inserts are batched (per-row commits ran ~6 s/row under concurrent disk load). Rows over the pipeline's 40 s cap are dropped and counted (1 such row in fleurs dev).
- **Eval surface note:** the canonical `youtube` test symlink is dangling (see data-loss note below), so the conversational dev/test sets were restored from upstream `pourmand1376/asr-farsi-youtube-chunked-10-seconds` test shards 0–1 (test rows were export-policy-skipped from all training). The two shards share 694/804 videos, so the split is by `md5(video_id)` parity instead: `youtube_dev_conv` (3,873 rows) and `youtube_test_conv` (3,203 rows), zero video overlap, registered in `BENCHMARKS`.
- **Result (model `omni_ctc_300m_farsi_hf`, KenLM α=0.3 β=0.0, beam 64, top-16):**

| Benchmark          |  Rows | Greedy |  Beam | Beam+LM 1-best | LM oracle@4 | LM oracle@8 | LM oracle@16 | Headroom@16 |
| ------------------ | ----: | -----: | ----: | -------------: | ----------: | ----------: | -----------: | ----------: |
| `c1tech`           | 2,101 |  20.46 | 20.58 |          19.19 |       14.69 |       13.00 |        11.82 |        7.37 |
| `youtube_dev_conv` | 3,873 |  20.60 | 20.41 |          17.60 |       15.07 |       14.25 |        13.66 |        3.94 |
| `fleurs_dev`       |   361 |   9.23 |  9.09 |           7.21 |        5.15 |        4.59 |         4.17 |        3.04 |

- Duplicate rate is 0.0% everywhere (pyctcdecode returns unique texts); the LM decoder keeps fewer distinct candidates per row (12.9–14.7) than plain beam (14.4–15.6) but its candidate set has strictly better oracle WER — KenLM steers beams toward correct words rather than just re-ranking. Timing split on c1tech: logit generation 27.2 s for 3.19 h audio; each beam decode ~59 s (~196×RT on 8 workers). The c1tech beam+LM 1-best here (19.19) differs from the July shortlist number (19.31) by 0.12 — same config, `decode_beams_batch` vs `decode_batch` path.
- **Verdict:** **Gate PASSED on both dev splits** (3.94 and 3.04 points ≥ 2). Candidates + scores persisted for c1tech, youtube_dev_conv, and fleurs_dev (beam and kenlm variants, `-nb16` runs) in `data/benchmarks/results/persian-shortlist.sqlite3`; logit caches under `experiments/lm_decoding/logits/`.

## 2026-07-10 — A3: neural N-best rescoring with frozen causal LMs (decode-only)

- **Goal:** Convert the A0 oracle headroom with `final = beam_score + alpha * neural_lm_logprob + beta * word_count` over the persisted top-16 (`experiments/lm_decoding/rescore_neural.py`, scores cached per unique text). Stop condition from the plan: recovery below 25% of oracle headroom kills the arm.
- **Result (alpha/beta grid-tuned on each dev split, KenLM-fused beam score as anchor):**

| Scorer                        | youtube_dev_conv (base 17.60, oracle@16 13.66) | fleurs_dev (base 7.21, oracle@16 4.17) |
| ----------------------------- | ---------------------------------------------- | -------------------------------------- |
| `HooshvareLab/gpt2-fa` (124M) | 17.51 (−0.09, 2.3% recovery)                   | 6.67 (−0.54, 17.8% recovery)           |
| `Qwen/Qwen2.5-0.5B`           | 17.60 (−0.00, best alpha=0)                    | 6.98 (−0.22, 7.2% recovery)            |

- Anchoring on the acoustic-only score instead of the fused score is strictly worse (18.53 on youtube_dev_conv). Rescorer-only throughput: 307×RT (Qwen2.5-0.5B, fp16, batch 32, RTX 5070) — speed is not the blocker.
- **Verdict:** **Dead end (stop condition triggered).** Both frozen scorers recover far under 25% of the oracle headroom; on conversational speech the modern multilingual model contributes literally nothing (tuner selects alpha=0). The headroom is real but frozen small causal LMs cannot rank it out; per the plan, custom-LM training is not justified by this gate. The statistical-LM lever (A1/A2) is the productive path.

## 2026-07-10 — A2 (screening): domain-split LMs, interpolation, and pruning

- **Corpora:** `corpus.txt` (538,117 scribe_v4 train labels) partitioned by exact normalized-line match against the full upstream YouTube train transcription dump (113,204 lines column-projected from all 32 shards): 103,780 conversational + 434,337 read lines. Domain 4-grams built with the lm4 recipe; log-linear interpolation weights tuned on dev refs with kenlm `interpolate` (built in-tree with a local Eigen 3.4 after Arch's Eigen 5 config refused kenlm's version request).
- **Dev perplexity screen (word PPL):**

| LM                            | youtube_dev_conv | fleurs_dev |
| ----------------------------- | ---------------: | ---------: |
| lm4 (mixed, pruned `0 1 1 1`) |            125.2 |      919.2 |
| lm4_read                      |            173.7 |      887.0 |
| lm4_conv                      |            472.8 |    1,661.9 |
| lm6 (pruned)                  |            110.0 |      897.7 |
| **lm4np (unpruned)**          |         **43.4** |      925.9 |

- Tuned log-linear weights collapse onto the read LM (1.07 / −0.01) at PPL 199.7 on combined dev refs — worse than the natural mix. The conversational sub-corpus is too small (104k lines) to beat the mixed model it is already inside.
- **Verdict:** **Interpolation: dead end with current text** — the natural scribe_v4 mixture dominates any read/conv split-and-merge. **Pruning is the real lever:** removing lmplz pruning cuts conversational PPL by 2.9× (125 → 43). WER confirmation of lm4np/lm6np runs in the A1 sweep; new external text (not a re-weighting of existing text) is what would reopen A2.

## 2026-07-10 — A1 (partial, closed): KenLM alpha/beta sweep on cached logits

- **Setup:** the A0 logit caches make the sweep CPU-only (`--sweep` grid alpha {0.15,0.30,0.45,0.60} × beta {−0.25,0,0.25}, ~215×RT per decoder on 8 workers). Dev surfaces: `youtube_dev_conv` and `fleurs_dev`.
- **Result:** the surface is flat near the optimum and consistent across both devs — alpha 0.45 is the better center: youtube_dev_conv 17.47 (a0.45/b0.0) vs 17.66 at the current a0.3/b0.0; fleurs_dev 7.11 (a0.45/b0.25) vs 7.21. lm6 adds ~0.03 over lm4 on fleurs_dev. Beta is inert (±0.25 moves WER ≤0.1).
- **Verdict:** **Closed by direction mid-run** — remaining Omni beam-width/unpruned-LM WER arms were dropped when decode work was re-scoped to the Parakeet lane. Standing conclusions: alpha 0.45 is a free ~0.2 if the Omni KenLM path is ever revisited; the A2 perplexity result (unpruned order-6 = 3.7× better conversational PPL) transfers to NGPU-LM on the Parakeet checkpoint, which is the production lane (NVIDIA recommends order 6 for BPE models there). The lm6np/lm4np ARPA+binary artifacts are built and ready under `experiments/lm_decoding/`.

## 2026-07-10 — Track B Gate 0 + first data-mixture screen (110M hybrid TDT, 2k steps)

- **Setup:** first Parakeet TDT runs for Farsi through `farsi-parakeet-train-tdt` (base `parakeet-tdt_ctc-110m-base-hybrid.nemo`, fresh Persian BPE-1024 `fa_spe_bpe_v1024_scribe_v4` trained on the scribe_v4 label text, simple recipe, AdamW 3e-4, warmup 200, batch-dur 120, bf16, loss-init fix active). Data is the freshly materialized local surface (the scribe_v4 export and old NeMo manifests are off-disk): fleurs 12.5 h + neyshekar 36.8 h + worldspeech 24.5 h read, plus restored upstream YouTube train shards for conversational audio (normalized with the canonical Persian normalizer, transcoded to 16 kHz mono FLAC). Fixed val set `val_fixed` = fleurs_dev 1.4 h + conversational dev 2.0 h. **Gotcha found:** the local worldspeech export is entirely stereo (6,529/6,529 files) — the fairseq2 pipeline tolerated it, NeMo does not; all files downmixed in place. A second gotcha: `farsi-parakeet-eval --kind` selected default model paths but never switched the hybrid's decoding head, so ctc and tdt evals silently scored the same lane — fixed in `parakeet_finetune_core/eval.py` (hybrids now `change_decoding_strategy` per kind).
- **Gate 0 (mechanics, 32 fleurs rows, 1,500 steps, batch-dur 60):** loss 113 → 5.8, several exact reproductions in fp32 decode, 0 empty hypotheses, all-Persian output, clean `.nemo` export/restore, RTFx 572, peak 1.34 GiB. Residual 24.7% WER on the memorization set (final export is the last step, where tiny-batch loss bounced). **Mechanics pass.** Side note: with a 32-row dataset the Lightning `val_check_interval=250` never fires (epoch shorter than the interval), so Gate 0 ran without mid-run validation; the 20 h screens validate normally.
- **Gate 1 screen — read-only vs 50/50 mixture at matched 20 h / 2k steps** (one variable: data mixture; fp32 dev WER on the best-val_loss checkpoint):

| Arm          | Train surface (realized)                     |   val_loss best | TDT fleurs_dev | TDT conv dev | CTC fleurs_dev | CTC conv dev |
| ------------ | -------------------------------------------- | --------------: | -------------: | -----------: | -------------: | -----------: |
| `b0-read20h` | fleurs 7 h + neyshekar 9 h + worldspeech 4 h |     42.15 @1828 |          61.44 |        79.29 |          69.45 |        82.30 |
| `b1-mix50`   | read 10 h (3.5/4.5/2) + YouTube conv 10 h    | **35.35 @1965** |      **60.71** |    **70.08** |          71.56 |        76.43 |

- Both arms: 0–17 empty hypotheses out of 3,807 conversational rows, TDT RTFx 760–1,000 at batch 16. These are matched short screens for arm comparison, not deployment numbers (the ladder: short runs reject recipes, never promote models).
- **Verdict:** **Conversational data decisively earns its slot** — the 50/50 mixture wins the conversational dev by 9.2 TDT points with no read-domain tax (fleurs_dev even improves slightly) despite halving read hours. `mix50` is the data surface for the following knob arms; conversational pool since extended to ~100 h (shards 0–11 materialized, 1 undecodable row skipped) alongside 74 h read for the eventual 10k confirmation run.
- **LR arms on the mix50 surface (same 2k screen, one variable):** `b2-mix50-lr15` (1.5e-4) was 22 val_loss points behind at step 750 with bf16 WER still ~1.0 and was killed early; `b3-mix50-lr50` (5e-4) was stable throughout and won everything — val_loss **30.14** @1963 (vs 35.35 @3e-4), fp32 TDT dev WER **52.68 fleurs_dev / 62.61 conversational** (vs 60.71 / 70.08 at 3e-4). Fresh 1024-row decoder/joint want the higher LR at this budget; no instability at 5e-4 with warmup 200 and clip 1.0.
- **Promoted to Gate 2:** `gate2-full173h-lr50` — 10k steps on the full local surface (74,752 rows / 172.99 h realized: fleurs 11.5 + neyshekar 36.8 + worldspeech 24.5 + YouTube conversational 100.2 ≈ 42/58 read/conv), lr 5e-4, warmup 1,000, val every 500, same tokenizer/recipe. This run predates the trainer's `--seed` flag and is therefore a single-seed result; the later matched seed arms establish the short-screen noise envelope.

## 2026-07-10 — Gate 2: `gate2-full173h-lr50` (110M hybrid TDT, 10k steps, full local surface)

- **Setup:** winner recipe from the screens (simple replacement, BPE-1024 `fa_spe_bpe_v1024_scribe_v4`, AdamW lr 5e-4, warmup 1,000, batch-dur 120, bf16, val every 500) on the full local surface: 74,752 rows / 172.99 h (fleurs 11.5 + neyshekar 36.8 + worldspeech 24.5 read, YouTube conversational 100.2; 42/58 read/conv; raw upstream labels, no Scribe filtering available locally). This was a single-seed run before `--seed` was added. val_loss fell 79.8 → **15.19**, still edging down at the 10k cap; best checkpoint = step 10,000.
- **Result (fp32, best-val_loss checkpoint, `farsi-parakeet-eval`, batch 16):**

| Set               | TDT WER / CER | CTC WER / CER |
| ----------------- | ------------: | ------------: |
| fleurs_dev        |  26.17 / 8.94 | 31.70 / 10.12 |
| fleurs_test       |  24.64 / 8.85 | 31.38 / 10.00 |
| neyshekar_test    |  24.64 / 8.04 |  31.18 / 9.56 |
| worldspeech_test  | 41.14 / 23.75 | 45.87 / 23.83 |
| youtube_dev_conv  | 34.93 / 17.67 | 40.57 / 18.86 |
| youtube_test_conv | 34.38 / 16.74 | 39.76 / 17.94 |

- Dev and test agree within ~0.5–1.5 points on both domains (the video-disjoint conversational split behaves). The TDT head beats the auxiliary CTC head by 5–6 WER everywhere; CTC CER is closer. TDT RTFx 700–1,000 at fp32 batch 16 on the RTX 5070. Empty hypotheses ≤71/3,807 (conversational), 0 on read sets. The worldspeech test/dev audio was also stereo (the earlier downmix pass covered only train manifests) — fixed in place, evals re-run.
- **Context:** the historical broad Parakeet CTC-109M (`scribe-classified`, 844 h Scribe-filtered, ~130k steps) sits at fleurs 17.69 / neyshekar 18.65 / worldspeech 36.18 / youtube 27.39. This 10k-step run on 1/5 the hours of **unfiltered** labels lands within 6–7 points of it on read sets — with a TDT head the old model lacks, at a fraction of the budget.
- **Verdict:** **Recipe confirmed; ready for a full-budget decision.** The validated levers for the production run: this recipe + more steps (val_loss had not plateaued), the conversational mixture, label filtering when Scribe returns (the plan's B5 clean-vs-current ablation), and NGPU-LM fusion (A4) with the already-built unpruned order-6 ARPA on this checkpoint. Model artifacts: `runs/parakeet/gate2-full173h-lr50/gate2-full173h-lr50_{final,best-valloss}.nemo`.

## 2026-07-10 — Ablation round 2: seed noise and optimizer (110M hybrid, matched 2k screens)

- **Setup:** `--seed` added to the TDT trainer (`pl.seed_everything` + Lhotse sampler `seed`/`shard_seed`), then three valid matched arms of the b3 recipe (mix50 surface, lr 5e-4) with one variable each. Final comparisons use fp32 decoding from each best-`val_loss` `.nemo` checkpoint.

| Arm            | Change                       | val_loss @~1963 | fleurs_dev WER/CER | conv dev WER/CER | Verdict                                                                                       |
| -------------- | ---------------------------- | --------------: | -----------------: | ---------------: | --------------------------------------------------------------------------------------------- |
| `b4a-seed0`    | AdamW baseline, seed 0       |           29.35 |      52.54 / 26.94 |    62.34 / 38.64 | baseline                                                                                      |
| `b4b-seed1`    | AdamW baseline, seed 1       |           29.30 |      51.12 / 24.29 |    61.25 / 35.32 | seed spread is 1.42 read WER and 1.09 conversational WER                                      |
| `b5-adafactor` | Adafactor, seed 0, same 5e-4 |           29.92 |      52.97 / 25.29 |    62.99 / 36.35 | about 1.2 WER worse than the AdamW mean on both surfaces; overlaps the observed seed envelope |

- **Invalid arm:** `b6-extrestore` is excluded rather than scored. Extend/restore requires the old token IDs to retain their original meanings and new target-language pieces to be appended. This arm supplied an independently trained Persian BPE-1024 tokenizer as a same-size replacement for the English BPE-1024 vocabulary, then copied the English decoder rows by numeric ID. Its `val_loss` is therefore scientifically uninterpretable and cannot establish whether extend/restore helps.

- **Verdict:** recipe unchanged at this gate: simple replacement, AdamW, lr 5e-4. Adafactor remains a memory-enabling fallback for the 0.6B model because AdamW cannot fit that full fine-tune on the local GPU; this 110M test provides no evidence that Adafactor improves quality and does not prove quality parity at 0.6B. The hybrid-loss, tokenizer, SpecAugment, and first 0.6B gates are completed below; duration profile remains open.

## 2026-07-10 — A4: NGPU-LM greedy fusion on the Gate 2 Parakeet checkpoint (decode-only)

- **Setup:** the production NeMo LM path, now unblocked by the Gate 2 BPE checkpoint. Token-level unpruned 6-gram built from the scribe_v4 label text encoded with `fa_spe_bpe_v1024` at NeMo's token offset 100 (`lm6np_bpe1024.arpa` 602 MB → `.nemo` 551 MB via `NGramGPULanguageModel.from_arpa(vocab_size=1024)`). `farsi-parakeet-eval` gained `--ngram-lm` / `--ngram-lm-alpha` (sets `greedy.ngram_lm_model/alpha` through `change_decoding_strategy`; works per head on hybrids).
- **Dev alpha sweep (TDT greedy, baselines 26.17 / 34.93):** read speech peaks early, conversational keeps gaining — fleurs_dev best at α=0.2 (25.23), conversational at α=0.5–0.7 (31.95–31.99). Single frozen config by macro dev WER: **α=0.3** (25.57 / 32.64).
- **Frozen test pass (α=0.3, TDT greedy):**

| Set               | greedy |  +NGPU-LM |     delta |
| ----------------- | -----: | --------: | --------: |
| fleurs_test       |  24.64 |     24.75 |     +0.11 |
| neyshekar_test    |  24.64 |     23.64 |     −1.00 |
| worldspeech_test  |  41.14 |     42.13 |     +0.99 |
| youtube_test_conv |  34.38 | **31.63** | **−2.75** |

- RTFx 640–794 with fusion (vs 700–1,000 plain) — the LM is nearly free at decode time. Empty hypotheses unchanged.
- **Verdict:** **Promoted as the deployment decode for conversational-weighted use** — −2.75 conversational and −1.0 neyshekar for ~nothing, mirroring the Omni KenLM pattern (LM text matches conversational/CV-style domains; worldspeech regresses ~1 on its noisy surface). A per-domain alpha (0.5–0.7 conversational-only) buys another ~0.5. The same `.nemo` LM carries to any future Farsi Parakeet checkpoint that keeps the `fa_spe_bpe_v1024` tokenizer.

## 2026-07-10 — Ablation round 3: auxiliary CTC weight and tokenizer size (110M, matched 2k screens)

- **Auxiliary CTC setup:** same `mix50` data, BPE-1024, AdamW 5e-4, seed 0, and 2,000-step budget; only `ctc_loss_weight` changed. The raw auxiliary CTC loss was roughly 100 times the TDT loss, so even small nominal weights materially changed the shared-encoder gradient.

| CTC weight |      fleurs_dev WER / CER |        conv dev WER / CER |
| ---------: | ------------------------: | ------------------------: |
|       0.30 | 51.12–52.54 / 24.29–26.94 | 61.25–62.34 / 35.32–38.64 |
|       0.10 |             49.85 / 23.81 |             60.23 / 34.94 |
|       0.01 |             47.85 / 22.39 |             59.41 / 34.25 |
|      0.001 |             45.00 / 20.68 |             57.46 / 33.25 |
|       0.00 |         **42.98 / 19.15** |         **54.71 / 31.15** |

- **CTC verdict:** pure TDT wins monotonically across the entire screen. The promoted recipe sets `ctc_loss_weight=0.0`; the hybrid CTC head remains useful as an optional decoder but its training loss is removed from this recipe.
- **Tokenizer setup:** same pure-TDT 2,000-step screen and identical 61 MB / 538,117-line corpus for all tokenizer sizes (`sha256 16019dcabaa498b02dbbabcec6c6d8ccec6780c367aef4138c4efb5452a4d5f8`).

| Tokenizer | fleurs_dev WER / CER | conv dev WER / CER |
| --------: | -------------------: | -----------------: |
|   BPE-512 |    **40.56 / 14.80** |  **54.10 / 26.61** |
| BPE-1,024 |        42.98 / 19.15 |      54.71 / 31.15 |
| BPE-2,048 |        49.56 / 26.64 |      61.90 / 40.04 |

- **Tokenizer verdict:** BPE-512 wins both domains, with the largest gain in CER. BPE-512 carries into the full-data SpecAugment gate and the 0.6B transfer.

## 2026-07-10 — Ablation round 4: full-data SpecAugment and two-seed 110M confirmation

- **Setup:** full 74,752-row / 172.99 h training surface, BPE-512, pure TDT, AdamW 5e-4, warmup 1,000, batch-duration 120, 10,000 steps, seed 0. The trainer now controls NeMo's actual forward-path `spec_augmentation` module. `current` = 2 frequency + 10 time masks, `half` = 1 + 5, and `off` removes the module.

| SpecAugment | best val_loss | fleurs_dev WER / CER | conv dev WER / CER | Empty conv hypotheses |
| ----------- | ------------: | -------------------: | -----------------: | --------------------: |
| current     |        0.5561 |         22.66 / 7.43 |      33.07 / 16.06 |                    78 |
| half        |        0.5399 |         21.08 / 6.68 |      31.66 / 15.45 |                    76 |
| off         |    **0.5284** |         21.21 / 6.74 |  **31.24 / 15.25** |                    80 |

- **SpecAugment verdict:** `off` wins the two-surface average (26.23 WER versus 26.37 for `half`) and the conversational target by 0.42 WER; `half` retains a 0.13-WER FLEURS edge. The conversational target decides promotion, so `off` is the candidate recipe.
- **Seed-1 confirmation:** the exact `off` recipe reproduced the training trajectory (`val_loss 0.5305` versus seed 0 `0.5284`) and improved both fp32 dev gates: fleurs **20.70 / 6.69** and conversational **31.18 / 15.18**. Frozen test scores: fleurs **20.44 / 6.58**, neyshekar **18.97 / 5.48**, worldspeech **36.26 / 20.63**, and conversational **30.68 / 14.19**. The recipe is stable across two seeds.
- **Promoted 110M recipe:** simple tokenizer replacement, BPE-512, pure TDT, SpecAugment off, AdamW 5e-4, warmup 1,000, batch-duration 120, 10,000-step minimum, best checkpoint by `val_loss`, and promotion by fp32 fixed-surface WER. Artifact: `runs/parakeet/c3-bpe512-spec-off-s1/c3-bpe512-spec-off-s1_best-valloss.nemo`.

## 2026-07-10 — First controlled 0.6B Adafactor transfer

- **Setup:** `parakeet-tdt-0.6b-v3`, simple replacement with the promoted BPE-512 tokenizer, pure TDT, SpecAugment off, 5e-4, warmup 1,000, 10,000 steps, seed 0. Adafactor, batch-duration 60, and fused batch 2 are the validated memory-safe changes; full 617M-parameter fine-tuning peaked around 9.9 GB on the RTX 5070. The 0.6B base and 110M base both use `mean_volume` RNNT reduction. Best checkpoint = step 9,500 (`val_loss 0.5738`); step 10,000 rose to 0.5800.

| Set               | 110M seed 0 WER / CER | 110M seed 1 WER / CER |    0.6B WER / CER | 0.6B WER delta vs 110M seed 1 |
| ----------------- | --------------------: | --------------------: | ----------------: | ----------------------------: |
| fleurs_dev        |          21.21 / 6.74 |          20.70 / 6.69 |  **20.70 / 6.62** |                          0.00 |
| youtube_dev_conv  |         31.24 / 15.25 |     **31.18 / 15.18** | 31.75 / **15.13** |                         +0.57 |
| fleurs_test       |                     — |          20.44 / 6.58 |  **20.36 / 6.43** |                         −0.08 |
| neyshekar_test    |                     — |          18.97 / 5.48 |  **18.63 / 5.11** |                         −0.34 |
| worldspeech_test  |                     — |     **36.26 / 20.63** | 36.55 / **20.50** |                         +0.29 |
| youtube_test_conv |                     — |     **30.68 / 14.19** | 31.00 / **14.08** |                         +0.32 |

- **Aggregate:** four frozen-test macro WER is **26.64%** for 0.6B versus **26.59%** for 110M; macro CER is **11.53%** versus **11.72%**. This is WER parity with a 0.19-point CER gain, rather than a scale win. The larger model improves FLEURS and Neyshekar, regresses slightly on WorldSpeech and conversational WER, and improves CER on every frozen test.
- **Speed:** at matched fp32 batch 32, 0.6B runs at 339.6× RTFx on FLEURS and 459.7× on conversational dev versus 768.9× and 865.7× for 110M, roughly 1.9–2.3 times slower. For 0.6B, batch 8 is better on variable-duration FLEURS (396.5×), while batch 32 is better on conversational speech. The speed prize remains with 110M.
- **Exposure caveat:** this is matched at 10,000 optimizer steps, while memory forced batch-duration 60 for 0.6B versus 120 for 110M. The 0.6B run therefore saw roughly half the audio per optimizer step and still reached WER parity. A fair matched-audio follow-up should use two 60-second microbatches per optimizer step through gradient accumulation, or an equivalent 20,000-step budget, while keeping all other knobs frozen.
- **Operational note:** after both 2.4 GB `.nemo` artifacts were written and verified, the 0.6B trainer hung during post-export teardown on an autograd thread. The finished process was stopped, and the fixed fp32 suite ran from the verified best model (`sha256 633f6c73f7065fc764854ff91cebb0dcdb66f58db2d2df325b2185ecd81308bc`). Artifact: `runs/parakeet/d0-bpe512-spec-off-06b-ada-s0/d0-bpe512-spec-off-06b-ada-s0_best-valloss.nemo`.
- **Verdict:** **110M remains the deployment recipe.** It matches 0.6B macro WER, wins conversational WER and throughput, fits with AdamW, and trains with twice the audio per optimizer step. The next 0.6B experiment is a matched-audio confirmation, not another broad knob sweep.

## 2026-07-11 — BPE-512 NGPU-LM greedy and batched-beam promotion gate

- **Setup:** rebuilt the token-level unpruned order-6 LM for the promoted BPE-512 tokenizer from the same 538,117-line / 61 MB corpus used by the tokenizer experiments. The corpus became 17,992,934 offset-100 tokens and 15,442,806 n-grams. Artifacts: `lm6np_bpe512.arpa` 519.5 MB (`sha256 85ca870a4ce1afceda15d8519992686da412a32669cc6f68e909cb13444d034a`) and `lm6np_bpe512.nemo` 450.3 MB (`sha256 20823066e22cbba1d579a295a29ad6e98788581e18368e381c0b3ab72f82a20a`). A reusable `farsi-parakeet-build-ngram-lm` command now owns SentencePiece encoding, KenLM `lmplz`, and NeMo conversion.
- **Decoder correctness:** `farsi-parakeet-eval` now exposes batched GPU beam size and records the decode config in its summary. TDT uses NeMo `malsd_batch`; CTC uses `beam_batch`. A live smoke found that the NVIDIA checkpoint serialized `pruning_mode=LATE` and `blank_lm_score_mode=LM_WEIGHTED_FULL`, while current NeMo accepts the lowercase enum values. The evaluator normalizes those fields before changing strategy. A second live smoke found that standalone 0.6B TDT models require `change_decoding_strategy(config)` while the 110M hybrid requires `change_decoding_strategy(config, decoder_type=...)`; both paths now share the same config builder. The shared core has 47 passing tests and clean Ruff checks.

### Greedy alpha sweep on fixed dev surfaces

| Alpha | fleurs_dev WER / CER | conv dev WER / CER | Macro WER |
| ----: | -------------------: | -----------------: | --------: |
| plain |         20.70 / 6.69 |      31.18 / 15.18 |     25.94 |
|   0.1 |         20.07 / 6.54 |      30.13 / 14.92 |     25.10 |
|   0.2 |     **19.87** / 6.54 |      29.51 / 14.82 |     24.69 |
|   0.3 |         20.12 / 6.69 |      28.96 / 14.71 | **24.54** |
|   0.4 |         20.99 / 6.95 |      28.57 / 14.68 |     24.78 |
|   0.5 |         22.50 / 7.57 |      28.23 / 14.68 |     25.37 |
|   0.7 |         23.39 / 8.06 |  **27.99** / 14.86 |     25.69 |

- Clean speech peaks at α=0.2, conversational WER continues improving through α=0.7, and α=0.3 wins the two-surface macro. Greedy fusion runs at 699–855× RTFx on FLEURS and 862–941× on conversational dev at batch 32, versus 656× and 920× for the measured plain runs; the timing spread is dominated by run variance rather than a material fusion penalty.

### Batched-beam dev matrix

| Beam | LM alpha | fleurs_dev WER / CER | conv dev WER / CER | Macro WER | FLEURS / conv RTFx |
| ---: | -------: | -------------------: | -----------------: | --------: | -----------------: |
|    4 |    plain |         19.67 / 6.28 |      30.14 / 14.73 |     24.90 |          726 / 838 |
|    8 |    plain |         19.62 / 6.21 |      30.02 / 14.56 |     24.82 |          662 / 875 |
|   12 |    plain |         19.53 / 6.20 |      29.90 / 14.48 |     24.72 |          660 / 823 |
|    4 |      0.2 |         18.03 / 6.31 |      25.31 / 13.90 |     21.67 |          671 / 866 |
|    8 |      0.2 |         17.67 / 6.35 |      24.33 / 13.46 |     21.00 |          655 / 828 |
|   12 |      0.2 |     **17.58** / 6.34 |      24.07 / 13.36 |     20.82 |          656 / 829 |
|    4 |      0.3 |     **17.58** / 6.52 |      24.45 / 14.06 |     21.01 |          718 / 886 |
|    8 |      0.3 |         17.81 / 7.11 |      23.43 / 13.68 | **20.62** |          602 / 816 |
|   12 |      0.3 |         18.32 / 7.90 |  **23.09** / 13.63 |     20.71 |          621 / 797 |

- **Interpretation:** beam alone contributes about 1.0 clean WER and 1.3 conversational WER. Beam+LM is the large gain. More beam helps conversational speech, while FLEURS begins to overfit the LM at α=0.3 beyond beam 4. Beam 4 / α=0.3 is the speed/accuracy balance; beam 8 / α=0.3 is the lowest dev macro WER; beam 12 / α=0.3 is only the conversational specialist.

### Frozen held-out comparison

| Set               |      Greedy WER / CER | Balance: beam 4, α=0.3 | Lowest macro: beam 8, α=0.3 |
| ----------------- | --------------------: | ---------------------: | --------------------------: |
| fleurs_test       |          20.44 / 6.58 |       **17.67** / 6.72 |                17.86 / 7.33 |
| neyshekar_test    |          18.97 / 5.48 |           14.46 / 4.24 |        **13.46** / **3.89** |
| worldspeech_test  | **36.26** / **20.63** |          36.52 / 23.94 |               37.61 / 25.93 |
| youtube_test_conv |         30.68 / 14.19 |          23.52 / 13.09 |       **22.54** / **12.93** |
| four-set macro    |         26.59 / 11.72 |      **23.04** / 12.00 |           **22.87** / 12.52 |

- **Verdict:** promote **beam 4 / α=0.3** as the balanced clean/conversational decode: it cuts frozen macro WER by 3.55 absolute while retaining 357–727× RTFx across the four tests. Keep **beam 8 / α=0.3** as the lowest-macro and conversational/Neyshekar option: it cuts macro WER by 3.72 and conversational WER by 8.14 absolute. Keep greedy for WorldSpeech-like noisy audio; the LM corpus mismatch leaves WER flat-to-worse and substantially worsens CER there. A domain-aware deployment should use beam 4 for FLEURS-like clean speech, beam 8 for conversational/Neyshekar speech, and greedy for noisy WorldSpeech-like input.

## 2026-07-11 — Matched-audio 0.6B Adafactor confirmation

- **Setup:** exact first-0.6B recipe with one controlled change: `accumulate_grad_batches=2` over the memory-safe 60-second Lhotse microbatch budget. This gives roughly 120 seconds of audio per optimizer update, matching the promoted 110M exposure, while retaining 10,000 optimizer updates. Other fixed knobs: BPE-512, simple replacement, pure TDT, SpecAugment off, Adafactor 5e-4, warmup 1,000, fused batch 2, seed 0, and best checkpoint by `val_loss`. The shared trainer now preserves `--val-every` in optimizer-step units under accumulation and suppresses duplicate future microbatch log rows.
- **Training result:** every validation checkpoint through step 9,000 improved over the first 0.6B run at the same optimizer step. The new best is step 9,000 at `val_loss=0.4901`, 14.6% below the old best `0.5738`; step 9,500 rose to 0.4935 and step 10,000 ended at 0.4929, so this surface plateaus near 9k–10k. Peak active training memory stayed around 10.1 GB and the post-export teardown completed cleanly.

| Set               | 110M seed 1 WER / CER | First 0.6B WER / CER | Matched-audio 0.6B WER / CER |
| ----------------- | --------------------: | -------------------: | ---------------------------: |
| fleurs_dev        |          20.70 / 6.69 |         20.70 / 6.62 |             **17.84 / 5.48** |
| youtube_dev_conv  |         31.18 / 15.18 |        31.75 / 15.13 |            **27.86 / 13.55** |
| fleurs_test       |          20.44 / 6.58 |         20.36 / 6.43 |             **18.45 / 5.84** |
| neyshekar_test    |          18.97 / 5.48 |         18.63 / 5.11 |             **16.22 / 4.26** |
| worldspeech_test  |         36.26 / 20.63 |        36.55 / 20.50 |            **34.21 / 19.86** |
| youtube_test_conv |         30.68 / 14.19 |        31.00 / 14.08 |            **27.26 / 12.44** |
| four-test macro   |         26.59 / 11.72 |        26.64 / 11.53 |            **24.04 / 10.60** |

- **Speed:** architecture throughput remains in the first-0.6B envelope: batch-8 RTFx is 421 on FLEURS dev and 454 on conversational dev; batch 32 gives 357 and 454. Across frozen tests at batch 8, RTFx is 259–404. The 110M model remains about 1.9–2.3 times faster.
- **Artifact:** `runs/parakeet/d1-bpe512-spec-off-06b-ada-acc2-s0/d1-bpe512-spec-off-06b-ada-acc2-s0_best-valloss.nemo`, 2,469,642,240 bytes, `sha256 31693631d8d3eb015f6763b808e81d8b3f969ed557e6ce5d6b97487f4817056b`.
- **Verdict:** **promote this as the 0.6B accuracy recipe.** Gradient accumulation converts model scale into a 2.55-point macro-WER gain over the 110M winner and improves every fixed test surface. The 110M recipe becomes the speed lane; the 0.6B accumulation-2 recipe becomes the accuracy lane. The 9k–10k plateau closes further same-data step tuning; the next training gains should come from data quality or additional data.

## 2026-07-11 — Combined 0.6B + BPE-512 NGPU-LM endpoint

- **Setup:** transferred the two frozen 110M decoder candidates without retuning: beam 4 / α=0.3 and beam 8 / α=0.3, using the same BPE-512 order-6 NGPU-LM. The initial standalone smoke was discarded after proving the evaluator had ignored decode controls outside its hybrid branch; all results below come from the repaired standalone strategy path and logs explicitly report `malsd_batch`.
- **Dev:** beam 4 reaches FLEURS 15.13 / 5.01 and conversational 22.62 / 12.85; beam 8 reaches 15.25 / 5.37 and 21.99 / 12.90. Beam 4 is the clean balance, while beam 8 wins the two-surface macro and conversational WER.

| Set               | 0.6B greedy WER / CER | 0.6B beam 4, α=0.3 |    0.6B beam 8, α=0.3 |       Routed best |
| ----------------- | --------------------: | -----------------: | --------------------: | ----------------: |
| fleurs_test       |          18.45 / 5.84 |       15.69 / 5.45 |  **15.42** / **5.41** |            beam 8 |
| neyshekar_test    |          16.22 / 4.26 |       12.90 / 3.49 |  **12.21** / **3.31** |            beam 8 |
| worldspeech_test  | **34.21** / **19.86** |      36.74 / 25.09 |         39.52 / 28.88 |            greedy |
| youtube_test_conv |         27.26 / 12.44 |      21.72 / 11.80 | **20.95** / **11.83** |            beam 8 |
| four-test macro   |         24.04 / 10.60 |  **21.76** / 11.46 |         22.03 / 12.36 | **20.70 / 10.10** |

- **Speed:** beam 4 runs at 279–386× RTFx across the frozen sets; beam 8 runs at 275–394×. The larger beam has little measured throughput cost on this GPU, while the 0.6B acoustic encoder dominates latency.
- **Verdict:** the best fixed single decoder is **0.6B beam 4 / α=0.3** at 21.76 macro WER. The best accuracy policy is **0.6B beam 8 / α=0.3 for clean, Neyshekar, and conversational audio plus 0.6B greedy for WorldSpeech-like noisy audio**, reaching **20.70 macro WER / 10.10 macro CER**. The LM mismatch on WorldSpeech is severe enough that this routing decision belongs in the deployment/data policy rather than another global-alpha compromise.

## 2026-07-11 — Additive ASR-edge + CTC forced-alignment quality pilot

- **Goal:** prove a bounded, inspectable data-cleaning path on the exact training surface before promoting any rejection threshold. `omni-quality` now draws a deterministic sample, adds NeMo-SDP-style beginning/end ASR mismatch metrics plus WER/CER, runs the version-matched NeMo Forced Aligner, and imports word coverage and clip-margin metrics without deleting source rows.
- **Input:** seed-20260711 reservoir sample of 160 rows from `data/parakeet/manifests/gate2_full_train.jsonl` (74,752 rows): 90 YouTube, 49 Neyshekar, 14 WorldSpeech, and 7 FLEURS. Sample SHA256 `c74a74053536031d59eb394829f7ac413650e802a5012e9ac7f77a7e1d201e49`.
- **Draft ASR:** matched-audio 0.6B TDT best model SHA256 `31693631d8d3eb015f6763b808e81d8b3f969ed557e6ce5d6b97487f4817056b`, greedy batch 16: 24.71% WER / 11.68% CER, zero empty hypotheses, 298.5× RTFx, 5.32 GiB peak allocation.
- **ASR-edge result:** 45/160 rows had a nonzero beginning or ending mismatch, 23 exceeded 3 characters, 10 exceeded 10 characters, and 5 exceeded 20 characters. Nine of the ten rows over 10 characters also exceeded 35% WER. The largest cases visibly include missing beginnings or truncated transcript tails, so the signal is useful for review stratification; the one edge-only row and 23 high-WER rows without a large edge show that it cannot replace whole-utterance agreement.
- **CTC alignment:** exact `Peacockery/parakeet-ctc-109m-farsi` revision `27418c9ffa05a8a0fe66fc5900ca0181a3ad25a7`, cached at `base_models/parakeet/parakeet-ctc-109m-farsi/model.nemo`, SHA256 `b11fc64a92e4cc457c2693356e85d296143b702bc8e25e2f7fe8ff485a4afa72`; NeMo 2.7.3 NFA commit `1d4ee423806d461f9146ae982f9da8eb32495ae7`, CTC logits on CUDA and Viterbi on CPU. `nfa-prepare` normalized 29/160 labels onto the pinned Persian surface and left zero empty. All 160 rows aligned. Median leading/trailing margins were 0.320/0.474 seconds; median aligned-span ratio was 0.895. Eighteen rows had span ratio below 0.70 and one had a margin over 2 seconds; none of the low-span rows overlapped the edge-over-10 set and only one exceeded 35% WER, so alignment geometry provides a complementary boundary/silence signal rather than a proxy for transcript WER.
- **Artifacts:** gitignored under `data/audit/quality-pilot-20260711/`; `scored.jsonl` preserves the original row plus `quality.asr_edge`, `quality.asr_agreement`, and `quality.ctc_alignment`.
- **Verdict:** apparatus proven, automatic filtering deferred. The next gate is a human review stratified across signal intersections and score deciles, followed by a size-matched random-versus-cleaned training ablation. The published 985-hour v4 corpus can be re-audited, while the old WER35 Hub artifact cannot reconstruct its rejected population because it publishes only the 416,056 kept rows.

______________________________________________________________________

## Best result of record (current)

`scribe-v4-baseline` (Omni CTC 300M, 2026-05-30): CV 19.37% · FLEURS 8.69% · mana 6.56% · neyshekar 8.49% · worldspeech 27.45% · youtube 20.34% WER. For the current four-set Parakeet surface, the fixed result of record is matched-audio 0.6B + beam 4 / α=0.3 at 21.76 macro WER; the domain-routed result of record is beam 8 / α=0.3 on clean/conversational speech plus greedy on WorldSpeech-like noise at 20.70 macro WER. The historical broad Parakeet CTC-109M reference is 22.56% macro WER.

## Gaps / not recorded

- Exact training dates for the early Parakeet baselines (base / broad-plus-mana / broad-filtered) and the early Omni data-recipe sweep (FLEURS / Thomcles / 100h-clean/balanced/target / wer35-fastconformer) are **not recorded** in tracked files; their benchmark folders are dated 2026-05-28 because that is when they were re-scored with the live normalizer, not when trained.
- A predecessor Parakeet run `parakeet-ctc-109m-broad-plus-mana-20260515` (best epoch 12, val_wer 0.2926) has no canonical-suite card here.
- The `...-continue-44k` plain-resume variant has a config but no separate benchmark card found (only the re-warm run was scored).
- Scribe v2 (`scribe-v2`, the commercial verifier) is included as a baseline reference, not a fine-tune: six-split WER CV 31.28 / FLEURS 9.90 / mana 14.54 / neyshekar 15.17 / worldspeech 31.30 / youtube 29.23 (macro 21.90%).
