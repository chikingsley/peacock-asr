# Georgian ASR — experiment log

Newest at the bottom. Numbers come from `georgian-omni-eval` (corpus-level jiwer on the export's test split) unless noted.

## 2026-06-12 — v0: first Georgian fine-tune (145.3 h gold)

- **Setup:** preset `georgian-corpus-v0-300m` — omni CTC 300M v2 base, `georgian_asr_corpus` (FLEURS + Common Voice scripted-25 + spontaneous-3, 145.3 h, all human-labeled), 30k steps (~9.6 epochs), lr 1e-5, validate every 1k (dev = 13,456 clips). Run survived a mid-run driver-update crash at step ~22k (resumed from `step_22000`, nothing lost).

- **Dev curve:** 30.82 (step 3k) → 18.92 (step 29k), monotonic; step 30k ticked up to 18.93, so best-WER selection ships `step_29000` (`omni_ctc_300m_v2_georgian_v0_step_29000`).

- **Test split (14,096 rows):**

  | model             | pooled WER | pooled CER | CV-scripted (13,117) | FLEURS (979) |
  | ----------------- | ---------- | ---------- | -------------------- | ------------ |
  | base (no FT)      | 45.01      | 7.87       | 46.67 / 7.94         | 35.53 / 7.45 |
  | **v0 step_29000** | **20.73**  | **3.30**   | 20.09 / 3.09         | 24.39 / 4.58 |

- **Read:** −54 % relative pooled — the recipe transfers to a third language without changes. But it lands above the roadmap's expectation band (~10–16 FLEURS) and far from the published near-replica's ~5.7 on Common Voice, so v0 is a working baseline, not a ceiling. Ranked next levers: (1) KenLM fusion — `experiments/lm_decoding/lm4.bin` is already built from the v0 train labels (Tajik gained −16 % rel for free); (2) more epochs (dev was still creeping at 30k); (3) the real one: data — 145 h gold vs the 1,070 h that moved Tajik; the Georgian YouTube scrape hasn't started.

## 2026-06-12 — KenLM fusion on v0 (inference-time, no training)

- **Setup:** same harness as the Tajik experiment (`experiments/lm_decoding/run.py`); word 4-gram from the v0 train labels (64,633 lines → 7 MB binary). FLEURS test, 979 rows, full α/β grid (so mildly optimistic — no held-out exists for Georgian yet).

  | readout                              | WER             | CER  | decode   |
  | ------------------------------------ | --------------- | ---- | -------- |
  | greedy (production)                  | 24.71           | 4.62 | ~free    |
  | beam, no LM                          | 24.67           | 4.60 | 287× RT  |
  | **beam + KenLM (α=0.3–0.7 plateau)** | **18.90–18.97** | 3.84 | ~290× RT |

- **CONCLUSION: −5.8 WER (−23.5 % rel) — larger than Tajik's −16 %,** consistent with a small scripted corpus the LM covers well. Flat across α=0.3–0.7 and insensitive to β (unlike Tajik, where β hurt). Pending: blind apply to the CV-scripted test partition (13k rows, ~7 h CPU forward) and an LM rebuilt from a larger Georgian text corpus.
