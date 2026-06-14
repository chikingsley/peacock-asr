# Persian (Farsi) ASR — Experiment Log

Lab notebook of every fine-tuning experiment for the Persian (Farsi, `fa_ir`) ASR project,
in chronological order. Factual only; numbers are grounded in repo files
(`docs/`, `src/finetune_omni/training/configs/*.yaml`, `benchmarks/suites/*/summary.md`,
`README.md`). Where a number is not recorded anywhere, it says "not recorded".

## Project at a glance

- **Two model tracks:**
  - **Parakeet CTC 109M** (NVIDIA NeMo): base `models/parakeet-ctc-109m/ctc.nemo`, custom Persian
    BPE tokenizer `fa_spe_bpe_v1024`. AdamW lr 1e-4, warmup 5000 (1500 for the smaller run),
    min_lr 5e-6, bf16, Lhotse dynamic bucketing (batch_duration 700, 30 buckets), early-stop on
    `val_wer` (patience 5).
  - **OmniASR CTC 300M** (Facebook/Meta via fairseq2): base card `omniASR_CTC_300M_v2`,
    tokenizer `omniASR_tokenizer_written_v2`. lr 1e-5, bf16, grad-accum 2 (8 for tiny sets),
    layerwise activation checkpointing, `max_num_elements` 3.84M, beta_corpus/language 0.5.
- **Benchmark = the six-split canonical suite**, jiwer WER/CER, run via `persian-benchmark-suite`,
  results under `benchmarks/suites/`. Splits: common_voice_25, fleurs, mana_tts, neyshekar,
  worldspeech, youtube (all `test`). Reference points: **Scribe v2** (commercial verifier) and the
  untrained **omni-base-ctc-300m-v2** (control, ~100% WER on Parakeet base; the Omni base is usable).
- **Note on dates:** the project was first git-tracked 2026-05-29 (commit `34a9013d`); many
  experiments predate tracking and are dated from their config / run-doc / dataset names. The
  `-current-20260528` suite folders are **re-scores** of earlier-trained models on 2026-05-28 with
  the live normalizer (a normalizer fix cut ~6.5 WER points off every model without retraining; see
  `docs/zwnj-normalization-decision-20260529.md`), so those numbers are comparable to each other but
  newer than the original training date.

## Summary table (six-split macro WER / headline)

| Date (trained) | Experiment | Headline (macro WER, key splits) | Verdict |
|---|---|---|---|
| ~early | Parakeet base 109M (control) | 100% WER all splits | control |
| ~early | Parakeet broad-plus-mana 109M | CV 74.8% / neyshekar 99.4% (catastrophic) | dead end |
| ~early | Parakeet broad-filtered 109M | all splits 46–100% | dead end |
| ~2026-05 | Omni FLEURS-only 300M | FLEURS 14.70% but CV 41.9% / YT 52.8% | superseded (narrow) |
| ~2026-05 | Omni FLEURS→Thomcles 300M | FLEURS 16.50%, CV 36.5% | superseded |
| ~2026-05 | Omni clean-100h / balanced-100h / target-100h 300M | best ~CV 25–30% | superseded by Scribe data |
| ~2026-05 | Omni wer35-fastconformer-filtered 300M | CV 23.8% / FLEURS 11.8% | superseded |
| 2026-05-24 | **Parakeet scribe-classified 844h 109M** | macro WER 22.56% (best broad parakeet) | kept (broad reference) |
| 2026-05-25 | Parakeet scribe-exact-match 224h 109M | macro WER 33.8% (regressed) | dead end |
| 2026-05-25 | Omni scribe-exact-match 224h 300M | CV 21.9% / FLEURS 9.8%, but mana/neyshekar ↑ | kept (clean-domain parent) |
| 2026-05-27 | Omni scribe-v3-max 1032h 300M (continue) | CV 19.6% / FLEURS 8.85% (topped board) but ZWNJ-contaminated | superseded by v4 |
| 2026-05-30 | **Omni scribe-v4 (clean re-export) 300M, 34k** | CV 19.4% / FLEURS 8.7% / mana 6.6% / neyshekar 8.5% / YT 20.3% | **kept — current best** |
| 2026-05-30 | Omni scribe-v4 re-warm 10k (lr 2e-6) 300M | ≈ v4 baseline (tiny mixed deltas) | dead end (no further gain) |

---

## Parakeet 109M — early baselines (dates not recorded; pre-tracking)

### Parakeet base 109M — control (`parakeet-110m`)

- **Goal:** Sanity floor — the untuned NeMo Parakeet base on Persian.
- **Result:** 100.00% WER / 100.00% CER on every split (no Persian capability).
- **Verdict:** Control only — establishes that all Persian skill comes from fine-tuning.

### Parakeet broad-plus-mana 109M (`parakeet-broad-plus-mana`)

- **Changed:** Broad early Persian set + Mana TTS; recipe `batch_duration 700`, lr 1e-4,
  warmup 5000, the predecessor run `parakeet-ctc-109m-broad-plus-mana-20260515` (best epoch 12,
  val_wer 0.2926).
- **Result:** FLEURS 13.99% / mana 28.51% / youtube 33.32% but **common_voice 74.82%** and
  **neyshekar 99.42%** (CER ~98%) — catastrophic on those domains.
- **Verdict:** Dead end as a broad model — collapses on Common Voice / Neyshekar.

### Parakeet broad-filtered 109M (`parakeet-broad-filtered`)

- **Result:** Every split 46–100% WER (CV 46.4%, FLEURS 57.8%, neyshekar 99.9%).
- **Verdict:** Dead end — the filtering hurt rather than helped.

## Omni 300M — early data-recipe sweep (dates not recorded; pre-Scribe; re-scored 2026-05-28)

### Omni FLEURS-only 300M (`fleurs-fa-ir-ctc-300m-v2-finetune` → `omni-fleurs-fa-ir`)

- **Goal:** Fine-tune Omni 300M on FLEURS Persian alone (cleanest read speech) as a first target.
- **Changed:** Dataset `fleurs_fa_ir`; grad-accum 8, `max_num_elements` 960k, num_steps 5,000,
  validate/checkpoint every 500.
- **Result:** FLEURS 14.70% / 3.83% but CV **41.92%**, youtube **52.79%**, neyshekar 40.38%.
- **Verdict:** Superseded — strong on its own domain, useless broadly (too little / too narrow data).

### Omni FLEURS→Thomcles 300M (`thomcles-ctc-300m-v2-continue-from-fleurs` → `omni-fleurs-thomcles`)

- **Changed:** Continue from the FLEURS-final checkpoint (`omni_ctc_300m_v2_fleurs_fa_ir_final`) on
  `thomcles_persian_farsi_speech`; same 5,000-step regime.
- **Result:** FLEURS 16.50% (worse than FLEURS-only), CV 36.45%, youtube 36.41% (better than FLEURS-only).
- **Verdict:** Superseded — adding Thomcles broadened it slightly but still far from usable.

### Omni 100h-filter family: clean / balanced / target (300M, 15k steps)

- **Goal:** Compare three 100-hour selection strategies from the curation ledger
  (`persian_asr_clean_100h_filter`, `persian_asr_balanced_100h_filter`, `persian_asr_target_100h_filter`),
  each num_steps 15,000, grad-accum 2, `max_num_elements` 3.84M.
- **Result (six-split, key columns; CV / FLEURS / mana / neyshekar / youtube WER):**
  - **target-100h** (`omni-target-300m`): 29.89 / 12.17 / 31.42 / 30.74 / 34.73.
  - **balanced-100h**: 28.78 / 12.27 / 31.28 / 30.39 / 35.14.
  - **clean-100h**: 24.98 / 12.88 / 26.87 / 27.07 / 40.88.
- **Verdict:** Superseded by the Scribe-verified data tracks. clean-100h best on CV/mana/neyshekar,
  target/balanced best on youtube; none competitive with the later Scribe runs.

### Omni wer35-fastconformer-filtered 300M (`omni-wer35-fastconformer`)

- **Changed:** Data filtered by FastConformer-CTC WER ≤ 35% (NeMo Curator pass).
- **Result:** CV 23.79% / FLEURS 11.81% / mana 30.57% / neyshekar 25.76% / youtube 35.13%.
- **Verdict:** Superseded — best of the pre-Scribe Omni runs on several splits, but Scribe-verified
  data later beat it everywhere.

## 2026-05-24 — Parakeet scribe-classified 844h 109M (`parakeet-ctc-109m-scribe-classified-20260524-bd700-min10`)

- **Hypothesis / goal:** Use Scribe as a verifier/filter (not a label) to keep 571k diverse rows
  across 5 "close-enough" categories (exact_match, near_match, punctuation_or_orthography_only,
  number_or_symbol_mismatch, named_entity_mismatch) → broad, high-quality 844h training set.
- **Changed:** Data `train.filtered-maxchars400-cps60` = 520,812 rows / 844.35 h (dev 24,434 / 42.51 h).
  Recipe: Parakeet 109M, Lhotse bd700/qd15/30-buckets, lr 1e-4, warmup 5000, min_lr 5e-6,
  min_steps 43,420 / max_steps 130,260, val every ~4342, early-stop patience 5.
- **Result:** Stopped at the hard step cap (max_steps 130,260) at epoch 21, best `val_wer` 0.20084.
  Held-out Scribe test (23,910 rows): WER **20.69%** / CER 6.63%. Six-split canonical
  (CV/FLEURS/mana/neyshekar/worldspeech/youtube WER):
  22.89 / 17.69 / 12.58 / 18.65 / 36.18 / 27.39 — **macro WER 22.56% / macro CER 8.58%**.
- **Verdict:** **Kept — the best broad six-split balance among Parakeet runs** (and the reference
  broad model). Strong on mana/CV; weaker than Omni on FLEURS.

## 2026-05-25 — Parakeet scribe-exact-match 224h 109M (`parakeet-ctc-109m-scribe-exact-match-20260525-bd700`)

- **Hypothesis / goal:** Train only on `exact_match` rows (Scribe == reference) — cleanest possible
  labels — to test whether exact-only beats the broader classified set.
- **Changed:** Data 227,474 rows / 246.40 h (CV 77.86%, mana 13.13%, neyshekar 5.23%, …),
  train 210,217 / 224.34 h. Same Parakeet recipe, warmup 1500, min/max steps 11,540 / 34,620.
- **Result (six-split WER):** CV 34.64 / FLEURS 27.15 / mana 20.64 / neyshekar 27.88 /
  worldspeech 46.44 / youtube 46.22 — **macro WER 33.83% / macro CER 13.03%** (re-scored
  `parakeet-scribe-exact`: 34.81 / 24.49 / 32.49 / 36.12 / 50.02 / 49.10).
- **Verdict:** **Dead end** — exact-only on the 109M regressed badly vs the 844h classified run.
  Confirms broad-but-verified > tiny-but-perfect for Parakeet.

## 2026-05-25 — Omni scribe-exact-match 224h 300M (`persian-asr-scribe-exact-match-20260525-ctc-300m-v2`)

- **Hypothesis / goal:** Same exact-match data, but on Omni 300M (which had been the stronger base
  on clean domains) — and as the parent checkpoint for later continued-training runs.
- **Changed:** Dataset `persian_asr_scribe_exact_match_20260525` (210,217 train / 9,429 dev rows;
  224.34 h). Omni 300M, num_steps 34,000, validate/checkpoint every 1,000.
- **Result:** Finished at step 34,000; best validation at step 32,000 (card
  `omni_ctc_300m_v2_persian_scribe_exact_match_20260525_best`). Six-split WER:
  CV 21.87 / FLEURS 9.81 / mana 27.59 / neyshekar 28.32 / worldspeech 38.78 / youtube 36.31
  — macro WER 27.11% / macro CER 8.60%.
- **Verdict:** **Kept as the clean-domain parent.** Strongest of the recent runs on CV and FLEURS,
  but WER much worse on mana/neyshekar/youtube (clean-domain win, not a broad recipe). Became the
  parent checkpoint for v3-max and v4.

## 2026-05-27 — Omni scribe-v3-max 1032h 300M, continue (`persian-asr-scribe-v3-max-20260527-ctc-300m-v2-continue`)

- **Hypothesis / goal:** Continue from the exact-match best checkpoint on the much larger
  "max defended surface" v3 (588K rows / 1032 h; all 5 keep-categories) to get a single strong
  broad model.
- **Changed:** Parent `omni_ctc_300m_v2_persian_scribe_exact_match_20260525_best`; dataset
  `persian_asr_scribe_v3_max_20260527`; num_steps 34,000. (Benchmarked at step 38,000.)
- **Result (six-split WER, step 38000):** CV 19.64 / FLEURS 8.85 / mana 35.72 / neyshekar 36.74 /
  worldspeech 36.63 / youtube 33.95. Topped the board on CV/FLEURS while it ran.
- **Verdict:** **Superseded by v4.** v3 was trained on ZWNJ-bearing labels but the Omni tokenizer
  has no ZWNJ piece → ~33% of its benchmark rows emit `⁇` (contaminated). The mana/neyshekar WER
  spikes reflect this. Decision (`docs/zwnj-normalization-decision-20260529.md`): retrain on a
  ZWNJ-free surface rather than add ZWNJ to the tokenizer.

## 2026-05-30 — Omni scribe-v4 (clean re-export) 300M, 34k (`persian-asr-scribe-v4-ctc-300m-v2-continue`)

- **Hypothesis / goal:** v3's exact data, re-exported cleanly through the fixed normalizer
  (strips Cf+So categories incl. ZWNJ → space; validated **0 `<unk>`** over all 704,455 canonical
  rows) → remove the `⁇` contamination → best model.
- **Changed:** Dataset `scribe_v4` (563,749 train+dev rows; clean re-export of the v3 surface).
  Parent = the exact-match best checkpoint; same Omni regime, num_steps 34,000. Run dir
  `runs/scribe-v4/ws_1.760d57f2`.
- **Result (six-split WER / CER):** CV **19.37 / 4.72**, FLEURS **8.69 / 2.39**, mana **6.56 / 1.80**,
  neyshekar **8.49 / 1.91**, worldspeech 27.45 / 17.25, youtube **20.34 / 9.71**. Dev WER **11.27%**
  at step 34,000 (still best at the last step — no overfit).
- **Verdict:** **Kept — current best Persian model.** Massive drop vs v3 on mana (35.7→6.6) and
  neyshekar (36.7→8.5) — i.e. the v3 numbers there were the `⁇` contamination, now gone. Clean
  data + clean tokenizer is the decisive lever.

## 2026-05-30 — Omni scribe-v4 re-warm 10k, lr 2e-6 (`persian-asr-scribe-v4-rewarm10k-ctc-300m-v2`)

- **Hypothesis / goal:** Dev-WER was still improving at v4's last step. Test whether a warm restart
  (fresh optimizer + fresh tri_stage schedule peaking at 2e-6 = 20% of the original lr) squeezes
  more out past 11.27% dev-WER. (A companion config `...-continue-44k` planned a plain +10k resume;
  the re-warm is the run that was benchmarked.)
- **Changed:** Load v4 step_34000 weights via card `omni_ctc_300m_v2_scribe_v4_20260530_best`,
  fresh 10k-step run, lr 2e-6, explicit tri_stage [0.1/0.4/0.5], keep best-WER-3.
- **Result (six-split WER / CER):** CV 19.41 / 4.71, FLEURS **8.51 / 2.36**, mana 6.61 / 1.79,
  neyshekar **8.22 / 1.86**, worldspeech 27.61 / 17.30, youtube **20.19 / 9.66**.
- **Verdict:** **Dead end (no meaningful gain).** Mixed sub-0.2-point deltas vs the v4 baseline —
  marginally better on FLEURS/neyshekar/youtube, marginally worse on CV/worldspeech. Within noise;
  the v4 baseline remains the model of record.

---

## Best result of record (current)

`scribe-v4-baseline` (Omni CTC 300M, 2026-05-30): CV 19.37% · FLEURS 8.69% · mana 6.56% ·
neyshekar 8.49% · worldspeech 27.45% · youtube 20.34% WER. For broad-domain Parakeet,
`parakeet-ctc-109m-scribe-classified-20260524` (macro WER 22.56%) is the reference.

## Gaps / not recorded

- Exact training dates for the early Parakeet baselines (base / broad-plus-mana / broad-filtered)
  and the early Omni data-recipe sweep (FLEURS / Thomcles / 100h-clean/balanced/target /
  wer35-fastconformer) are **not recorded** in tracked files; their benchmark folders are dated
  2026-05-28 because that is when they were re-scored with the live normalizer, not when trained.
- A predecessor Parakeet run `parakeet-ctc-109m-broad-plus-mana-20260515` (best epoch 12,
  val_wer 0.2926) is referenced in the run docs but has no canonical-suite card here.
- The `...-continue-44k` plain-resume variant has a config but no separate benchmark card found
  (only the re-warm run was scored).
- Scribe v2 (`scribe-v2`, the commercial verifier) is included as a baseline reference, not a
  fine-tune: six-split WER CV 31.28 / FLEURS 9.90 / mana 14.54 / neyshekar 15.17 /
  worldspeech 31.30 / youtube 29.23 (macro 21.90%).
