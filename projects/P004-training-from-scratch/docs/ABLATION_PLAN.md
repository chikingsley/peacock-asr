# Track 07 Ablation Plan: Training from Scratch

## Research Question

Can a Conformer model trained from scratch on LibriSpeech produce GOP posteriors
that are competitive with fine-tuned w2v-BERT 2.0 for pronunciation scoring?

Specifically:

- Does encoder-CTC trained from scratch generalize as well as fine-tuned SSL encoders
  when used as a posterior source for GOP features?
- Which architecture (Conformer vs Zipformer) gives the best phone posteriors per GPU-hour?
- What is the minimum viable dataset size for competitive phone posteriors?

## Frozen Evaluation Protocol (inherited from Track 05)

- Dataset: SpeechOcean762 (2500 train / 2500 test, pinned revision)
- Evaluation: phone-level PCC with 95% CI, minimum 3 seeds on the GOPT scorer
- Feature extraction: GOP-SF from CTC posteriors (same pipeline as Track 05)
- Track 05 baseline (w2v-BERT 2.0 fine-tuned): PCC 0.648

Posterior quality proxy: Phone Error Rate (PER) on TIMIT or LibriSpeech test-clean,
measured before applying to the GOP pipeline.

---

## Phase 1: Reproduce icefall TIMIT TDNN-LSTM Recipe (Learning Sandbox)

Goal: Learn the icefall training pipeline before committing to expensive runs.
This phase has no paper value — it is infrastructure and process validation only.

Recipe: `github.com/k2-fsa/icefall egs/timit/ASR`

| Run ID | Architecture | Dataset | Expected PER | Purpose |
|--------|-------------|---------|-------------|---------|
| P1-A | TDNN-LSTM (icefall default) | TIMIT | ~17.66% | Reproduce paper number |
| P1-B | TDNN-LSTM + minor tuning | TIMIT | < 17% | Verify we can modify training |

Implementation needed:

- Install k2, lhotse, icefall dependencies
- Run data prep for TIMIT
- Verify k2 graph compilation works on our GPU (RTX 5070)
- Confirm CTC posteriors can be extracted in the format GOP pipeline expects

Expected effort: 2-3 days
Decision gate: If we cannot reproduce TIMIT PER within 1% relative of paper, stop and diagnose.

---

## Phase 2: Adapt icefall Conformer Recipe for Phoneme CTC on LibriSpeech

Goal: Replace TDNN-LSTM with Conformer and scale to LibriSpeech 960h with phoneme labels.

Starting point: `github.com/k2-fsa/icefall egs/librispeech/ASR/conformer_ctc/`

The existing recipe uses BPE/word-level output. We need to adapt it for:

1. Phoneme-level output vocabulary (ARPABET, ~40 phones)
2. Phone-labeled training data (gilkeyio/librispeech-alignments or G2P-derived)
3. CTC posterior extraction in GOP format

| Run ID | Architecture | Dataset | Output | Purpose |
|--------|-------------|---------|--------|---------|
| P2-A | Conformer-S (small) | LS-100h clean | ARPABET CTC | Feasibility check, fast |
| P2-B | Conformer-M (medium) | LS-960h | ARPABET CTC | Full LibriSpeech |
| P2-C | Conformer-M | LS-960h | ARPABET CTC + phone-aligned | GOP posterior extraction |

For P2-C, posterior extraction must match the format expected by `gop.py`:

- Shape: `(T, num_phones)` log-probabilities from CTC output layer
- No beam search — raw softmax over ARPABET vocabulary

Implementation needed:

- Adapt BPE tokenizer to ARPABET phone set
- Write phone-level label preparation script from LibriSpeech alignments
- Write posterior extraction script compatible with `_compute_lpr_features_batched`
- Run GOPT scorer on extracted posteriors to get PCC on SpeechOcean762

Expected effort: 1-2 weeks
Expected PER: 8-15% on TIMIT (Conformer-M trained on 960h, evaluated on TIMIT)
Expected PCC: Unknown — this is the key experiment

Decision gate:

- If PCC >= 0.60 on SpeechOcean762: from-scratch is viable, continue to Phase 3.
- If PCC < 0.55: fine-tuned SSL is clearly better, document and stop.
- If PCC 0.55-0.60: ambiguous, run Phase 3 before deciding.

---

## Phase 3: Head-to-Head vs Fine-Tuned w2v-BERT 2.0

Goal: Direct comparison with Track 05 best result under identical scoring conditions.

Identical setup: same GOPT scorer, same SpeechOcean762 split, same GOP-SF pipeline.
Only variable: the acoustic model providing posteriors.

| Run ID | Acoustic Model | Training | PCC | GPU-hours | Notes |
|--------|---------------|----------|-----|-----------|-------|
| P3-A | w2v-BERT 2.0 (fine-tuned) | Track 05 | 0.648 | ~4h | Baseline |
| P3-B | Conformer-M (from scratch) | Phase 2 best | TBD | TBD | Our from-scratch |
| P3-C | Conformer-L (large) | LS-960h | TBD | TBD | Scale test |
| P3-D | Conformer-M + CommonVoice | 960+1000h | TBD | TBD | More data |

Comparisons to report:

- PCC delta (from-scratch vs fine-tuned)
- PER delta (acoustic quality proxy)
- Total training GPU-hours (from-scratch is much more expensive)
- Posterior distribution quality (entropy, calibration)

Decision gate:

- If from-scratch PCC >= 0.64 (within 1% of baseline): architecturally competitive.
- If from-scratch PCC >= 0.648: from-scratch matches or beats fine-tuning.
- Either outcome is publishable with the framing: "how much data/compute does from-scratch need?"

---

## Phase 4: Zipformer Phoneme CTC (If Phase 2-3 Show Promise)

Goal: Test whether Zipformer (ZIPA's architecture) improves over Conformer.

Zipformer is k2-fsa's improved architecture. ZIPA used it with 17K hours.
We would use the same 960h LibriSpeech training set for a fair comparison.

| Run ID | Architecture | Dataset | Purpose |
|--------|-------------|---------|---------|
| P4-A | Zipformer-S | LS-960h | Architecture comparison vs Conformer-S |
| P4-B | Zipformer-M | LS-960h | Fair match to Conformer-M |

This phase is gated on Phase 2-3 showing Conformer is at least competitive with
fine-tuned SSL. If fine-tuning wins decisively, Zipformer is not worth the engineering effort.

Expected effort: 3-5 days (recipe adaptation, same data pipeline as Phase 2)

---

## Decision Rules Summary

| Gate | Condition | Action |
|------|-----------|--------|
| Phase 1 done | Reproduce TIMIT PER within 1% of paper | Proceed to Phase 2 |
| Phase 2 done | PCC >= 0.55 on SpeechOcean762 | Proceed to Phase 3 |
| Phase 2 done | PCC < 0.55 | Write negative result, stop |
| Phase 3 done | PCC >= 0.64 | Proceed to Phase 4, write full paper |
| Phase 3 done | PCC 0.55-0.64 | Write data efficiency paper, stop |
| Phase 3 done | PCC < 0.55 | Write negative result, fine-tuning wins |

---

## Paper Structure Preview (if Phase 3 positive)

- **Section 1**: Introduction — from-scratch vs fine-tuning for pronunciation scoring
- **Section 2**: Related Work — ZIPA, POWSM, PRiSM, Conformer, icefall
- **Section 3**: Method — icefall Conformer adapted for phoneme CTC, GOP extraction
- **Table 1**: Phase 2 results — Conformer PER vs training data size
- **Table 2**: Phase 3 results — from-scratch vs fine-tuned, PCC on SpeechOcean762
- **Table 3**: Phase 4 results (if run) — Conformer vs Zipformer
- **Discussion**: Data efficiency, compute cost, practical recommendation
- **Conclusion**: Under what conditions does from-scratch beat fine-tuning?

## Deliverables per Run

- Config snapshot (architecture, dataset, phoneme vocab, CTC settings)
- MLflow run ID + metrics JSON
- PER on TIMIT test set (acoustic quality proxy)
- PCC on SpeechOcean762 (end-task quality)
- Wall-clock training time, GPU-hours
- Checkpoint location
- Random seeds used (minimum 3 seeds for scoring runs)
