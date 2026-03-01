# Experiment Log

Chronological record of benchmark runs on SpeechOcean762.
All results use the test split (2500 utterances, ~47K phones).

Base method: GOP-SF-SD-Norm (segmentation-free goodness of pronunciation),
adapted from [CTC-based-GOP](https://github.com/YuanGongND/gopt) (Gong et al.).

Paper target: PCC = 0.648 (GOPT transformer on feature vectors).

---

### Run 6 — 2026-03-01: Occupancy feature + SVR GridSearchCV

- **Changed**: Appended CTC expected count (occupancy) as 42nd feature dimension.
  Replaced default SVR with GridSearchCV over C=[0.1, 1.0, 10.0],
  epsilon=[0.01, 0.1, 0.5], dynamic fold count min(5, n_samples).
- **Backend**: original (checkpoint-8000)
- **Eval**: Per-phone SVR on 42-dim features (LPP + LPR + occupancy)
- **Result**: PCC = 0.548, MSE = 0.2136, 47,369 phones
- **Baseline**: PCC was 0.539 (run 5)
- **Takeaway**: Marginal gain (+0.009). SVR is near its ceiling on these features.
  The remaining gap to 0.648 needs the GOPT transformer (contextual model
  that sees full utterances, not isolated phones).

---

### Run 5 — 2026-02-28: SVR + feature vectors (full dataset)

- **Changed**: Extracted LPP + LPR feature vectors (41-dim) per phone using
  GPU-accelerated batched `torch.nn.functional.ctc_loss`. Trained per-phone
  SVR instead of polynomial regression.
- **Backend**: original (checkpoint-8000)
- **Eval**: Per-phone SVR on 41-dim features
- **Result**: PCC = 0.539, MSE = 0.2168, 47,369 phones
- **Baseline**: PCC was 0.320 (run 4, scalar scores)
- **Takeaway**: Big jump from scalar to feature vectors (+0.219 PCC).
  Validates the feature extraction pipeline. GPU batching (620x speedup)
  made full dataset runs practical (~65 min vs estimated ~11 hours serial).

---

### Run 4 — 2026-02-27: Scalar GOP, all three backends (full dataset)

- **Changed**: Full dataset run with all three backends using scalar GOP scores.
- **Eval**: Per-phone polynomial regression (order 2) on scalar scores
- **Results**:

| Backend | Vocab | PCC | 95% CI | MSE | Phones |
|---------|-------|-----|--------|-----|--------|
| xlsr-espeak | 387 IPA | **0.3197** | [0.3115, 0.3278] | 0.6656 | 39/39 |
| original | 39 ARPABET | 0.3104 | [0.3022, 0.3185] | 0.6872 | 39/39 |
| zipa | 127 IPA chars | 0.0749 | [0.0656, 0.0842] | 0.9253 | 32/39 |

- **Takeaway**: xlsr-espeak edges out original slightly. Both are in the
  expected range for scalar GOP (paper baseline: 0.361). ZIPA is not viable
  for GOP without fine-tuning — see DECISIONS.md for full analysis.

---

### Run 3 — 2026-02-27: ZIPA with unicode fixes (run 2)

- **Changed**: Fixed ER mapping (U+025D → U+025C), G mapping (U+0261 → ASCII 'g').
  7 diphthong/affricate phones excluded (32/39 phones scored).
- **Backend**: zipa
- **Eval**: Per-phone polynomial regression on scalar scores
- **Result**: PCC = 0.0749, MSE = 0.9253, 43,827 phones (32/39)
- **Baseline**: PCC was 0.0668 (run 2, broken mappings)
- **Takeaway**: Unicode fixes helped marginally. Fundamental issue is ZIPA's
  character-level CTC head — posteriors reflect transcription confidence,
  not pronunciation quality.

---

### Run 2 — 2026-02-26: ZIPA initial (broken ER/G mappings)

- **Backend**: zipa (broken unicode mappings for ER and G)
- **Eval**: Per-phone polynomial regression on scalar scores
- **Result**: PCC = 0.0668, MSE = 0.9291
- **Takeaway**: Near-zero correlation. Diagnosed as vocab incompatibility
  (character-level IPA, not phoneme-level).

---

### Run 1 — 2026-02-26: First scalar GOP benchmark

- **Changed**: Initial implementation of GOP-SF pipeline with original backend.
- **Backend**: original (checkpoint-8000)
- **Eval**: Per-phone polynomial regression on scalar scores
- **Result**: PCC ~0.31 (see run 4 for precise numbers on full dataset)
- **Takeaway**: Pipeline works. In expected range for scalar GOP.

---

## Score Progression (original backend)

```text
0.31  ██████████░░░░░░░░░░  Scalar GOP + poly regression
0.539 █████████████████░░░  SVR + 41-dim features
0.548 █████████████████░░░  SVR + 42-dim features + GridSearchCV
0.648 ████████████████████  Paper target (GOPT transformer)  ← next
```
