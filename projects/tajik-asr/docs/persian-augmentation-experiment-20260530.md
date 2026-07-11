# Persian→Tajik transliteration augmentation — experiment post-mortem (2026-05-30)

**Verdict: dead end for acoustic augmentation.** Adding 9.5 h of transliterated Persian audio to the 5.8 h real-Tajik fine-tune moved the real Tajik **test** WER by **−0.04%** (18.30% → 18.26%) and CER by **−0.14%** (5.30% → 5.16%) — i.e. noise. It did not help. This documents what we did, the numbers, and — more importantly — **why** it came out flat, so we don't repeat it and so the reasoning is reusable for other languages.

## Hypothesis

Tajik is Persian written in Cyrillic; the two are closely related. We have only ~5.8 h of real Tajik, which is the WER ceiling. Idea: transliterate abundant Persian (Farsi) audio transcripts FA→TG (Perso-Arabic → Cyrillic) and add the Persian audio + transliterated text as "extra Tajik-like" training data — cheap data to break the 5.8 h ceiling.

## Method

- **Transliteration:** ParsTranslit (char-level CTranslate2 FA→TG, chrF++ 87.9 / CER 0.05), vendored, with the می/نمی imperfective-prefix fix.
- **Source:** Persian FLEURS train (formal read speech), the cleanest available register.
- **Pipeline:** `fa_to_tajik` → Tajik `normalize_text` → Tajik-vocab coverage filter (≥0.55) → drop residual Perso-Arabic → 16 kHz FLAC → omni-parquet.
- **Mix (aggressive, by user choice — "see if it works"):** no hours cap. Kept **2,694 rows / 9.57 h**, tokenizer gate **0 unk**. With `beta_corpus=0.5` this corpus is ~43.5% of sampled batches (real Tajik ~56.5%). Artifact `tajik_asr_combined_v1`.
- **Training:** identical recipe to v0 (Omni CTC 300M, same hyperparameters). v1 best dev checkpoint = step_4000.
- **Eval:** 439-row real Tajik test split (1 clip >40 s excluded), jiwer corpus-level, CPU, via `tajik-eval-test`. Same rows for both models = fair A/B.

The data was verified real (valid Tajik Cyrillic text, 16 kHz audio matching `audio_size`, tokenizer-clean) — the flat result is **not** a pipeline bug.

## Results

| split    | model                |        WER |   CER |    MER | sub / del / ins    |
| -------- | -------------------- | ---------: | ----: | -----: | ------------------ |
| **test** | v0 (real only)       |     18.30% | 5.30% | 18.05% | 1168 / 139 / 105   |
| **test** | v1 (+9.5 h translit) |     18.26% | 5.16% | 18.02% | 1162 / 147 / 100   |
| **dev**  | v0                   | **17.11%** |     — |      — | (best @ step 1800) |
| **dev**  | v1                   | **17.69%** |     — |      — | (best @ step 4000) |

Per corpus (test):

| corpus                     | v0 WER |       v1 WER | v0 CER |  v1 CER |
| -------------------------- | -----: | -----------: | -----: | ------: |
| common_voice_25_tg (n=121) | 21.88% | **22.42%** ↑ |  4.17% | 3.98% ↓ |
| fleurs_tg_tj (n=318)       | 17.91% | **17.81%** ↓ |  5.41% | 5.28% ↓ |

Note v1 is actually **worse on dev** (+0.58%) and flat-to-marginally-better on test.

## Why it came out flat — the analysis

1. **We transliterated the TEXT, but the AUDIO is still Iranian Persian.** ASR learns audio→text. We taught the model *Iranian-accented audio → Tajik spelling*. Iranian Persian and Tajik have **diverged phonology** (vowel system — Tajik preserves the majhul vowels ē/ō that merged in Iranian; different vowel realizations; Iranian /ɒː/). So the acoustic distribution of the augmentation is the **wrong dialect**. This is the "Dialect Matters" caution made concrete: related-variety *audio* can mislead even when the *text* is correct.

1. **The base Omni model already knows both Persian and Tajik.** Omni is massively multilingual; the 300M base already has strong Persian *and* Tajik acoustic representations. Adding more Persian-derived audio teaches it little **new** — the fine-tune's real job is adapting the *output* to Tajik orthography/vocabulary/register, which real Tajik data does and transliterated Persian cannot.

1. **Transliteration is "Persian in Cyrillic," not natural Tajik.** Even at chrF++ 87.9 the FA→TG mapping isn't 1:1: Iranian vocabulary/idioms, izafe constructions, and loanword inventories differ (Iranian Persian = French/Arabic loans; Tajik = Russian loans). So the labels are subtly off-distribution Tajik — added **label noise**, not signal.

1. **The tempering is why it didn't HURT.** `beta_corpus=0.5` held the augmentation to ~43.5% of batches, so real Tajik still dominated the gradient. A more aggressive, untempered mix would likely have *regressed* (pulling the model toward Iranian acoustics). Flat is the expected outcome of "wrong-dialect data, tempered."

1. **The one real micro-signal: within-register transfer.** CER improved slightly everywhere (−0.14%), and the only WER improvement was on **fleurs** (formal read speech — the *same register* as the FLEURS Persian we transliterated), while **common_voice** (more conversational) got slightly **worse**. So the tiny benefit that exists transfers *within matched register/acoustics* and evaporates (or inverts) across register. Consistent with (1).

## Conclusion

Text-only transliteration of cross-dialect audio is ~useless for ASR fine-tuning **when the base model already knows both languages** — the bottleneck is Tajik orthography/vocabulary/register, and the augmentation's audio is the wrong dialect while its text is noisy. The lever for Tajik remains **real Tajik audio**, not transliterated Persian. Do not invest further in this direction.

## What would actually be worth trying instead

- **Real Tajik audio** — the obvious lever. Tajik YouTube (lectures, news, podcasts), segmented with VAD; Tajik radio. (There is already a `youtube_learning_tajik_v0` artifact to build from.)
- **TTS-generated Tajik** — synthesize audio from Tajik *text* with a *Tajik* TTS voice, so the phonology is actually Tajik (unlike here, where the audio was Iranian).
- **Persian as TEXT only** — if used at all, for an external LM / rescoring layer, never as acoustic training data (CTC has no built-in LM, so this needs added decoding infra).
- **Continued SSL pretraining on unlabeled Tajik audio** before the CTC fine-tune, to adapt the encoder's acoustics to Tajik specifically.

## Artifacts (kept for provenance)

- Builder + transliterator: archived to `dataset_prep/archive/build_persian_augmentation.py` and `dataset_prep/archive/parstranslit/` (2026-05-30). Code still valid; the *approach* is the dead end, not the code. Run via `python -m ...archive.build_persian_augmentation`.
- Dataset: `dataset_prep/artifacts/tajik_asr_combined_v1/` (+ its `dataset_summary.json`).
- Model: `runs/omni-ctc-300m-tajik-asr-corpus-v1/` (card `omni_ctc_300m_v2_tajik_v1_step_4000`).
- Eval tool: `tajik-eval-test` (`src/tajik_asr/eval/test_split.py`).
