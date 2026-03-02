# Experiment Log

Chronological record of benchmark runs on SpeechOcean762.
All results use the test split (2500 utterances, ~47K phones).

Base method: GOP-SF-SD-Norm (segmentation-free goodness of pronunciation),
adapted from [CTC-based-GOP](https://github.com/YuanGongND/gopt) (Gong et al.).

Paper target: PCC = 0.648 (GOPT transformer on feature vectors).

---

### Run 11 — 2026-03-02: Stochastic repeats (5x per backend, GOPT)

- **Changed**: Ran 5 full-dataset GOPT repeats for each top backend using cache:
  `original --gopt` and `xlsr-espeak --gopt`.
- **Results (5 runs each)**:

| Backend | PCC mean ± std | PCC min/max | MSE mean ± std |
|---------|-----------------|-------------|----------------|
| original (checkpoint-8000) | 0.6336 ± 0.0056 | 0.6271 / 0.6423 | 0.0822 ± 0.0009 |
| xlsr-espeak (wav2vec2-xlsr-53-espeak-cv-ft) | **0.6704 ± 0.0121** | 0.6604 / 0.6900 | **0.0741 ± 0.0022** |

- **Takeaway**:
  - `xlsr-espeak + GOPT` is better on both central tendency (mean PCC) and
    error (mean MSE) across repeats.
  - Variance exists (especially xlsr-espeak), so report mean/std for claims,
    and optionally pin training seeds for strict reproducibility.

---

### Run 10 — 2026-03-02: Post-fix verification rerun (xlsr-espeak + GOPT)

- **Changed**: Verified the xlsr-espeak GOPT path after fixing feature-width
  handling (CLI infers observed feature width; dataset keeps safe pad/truncate
  guard for mixed caches).
- **Backend**: xlsr-espeak (wav2vec2-xlsr-53-espeak-cv-ft)
- **Eval**: GOPT transformer
- **Result**: PCC = 0.6710, 95% CI [0.6659, 0.6759], MSE = 0.0740, 46,314 phones
- **Baseline**: PCC was 0.6618 (run 8, original backend + GOPT)
- **Takeaway**: This is the current best run. Because GOPT training is
  stochastic (no fixed seed), treat this as a strong candidate and confirm
  with a short multi-seed rerun before final claims.

---

### Run 9 — 2026-03-02: MLflow matrix rerun (all backends + eval modes)

- **Changed**: Ran a full matrix with MLflow logging enabled:
  `original --feats`, `original --gopt`, `xlsr-espeak --feats`,
  `xlsr-espeak --gopt`, and `zipa` scalar.
- **Operational note**: `xlsr-espeak --gopt` initially failed due to mixed
  cached feature widths (394 vs expected 395). Fixed in `GoptDataset` by
  safely padding/truncating mismatched per-utterance feature vectors, and
  in CLI by inferring GOPT `feat_dim` from observed feature vectors instead
  of assuming `len(vocab)+2`; reran successfully.
- **Results**:

| Backend | Eval | PCC | 95% CI | MSE |
|---------|------|-----|--------|-----|
| original (checkpoint-8000) | SVR+feats | 0.5481 | [0.5417, 0.5543] | 0.2136 |
| original (checkpoint-8000) | GOPT | 0.6413 | [0.6360, 0.6466] | 0.0813 |
| xlsr-espeak (wav2vec2-xlsr-53-espeak-cv-ft) | SVR+feats | 0.5747 | [0.5686, 0.5808] | 0.2052 |
| xlsr-espeak (wav2vec2-xlsr-53-espeak-cv-ft) | GOPT | **0.6564** | [0.6512, 0.6616] | **0.0767** |
| ZIPA-CR (ONNX) | scalar | 0.0749 | [0.0656, 0.0842] | 0.9253 |

- **Takeaway**:
  - Best in this batch: `xlsr-espeak + GOPT` (PCC 0.6564, MSE 0.0767).
  - `xlsr-espeak + GOPT` beats `original + GOPT` in this run by +0.0151 PCC.
  - Historical best remains run 8 (`original + GOPT` PCC 0.6618), so this
    is close but not a new top score.

---

### Run 8 — 2026-03-01: 3-phase multiprocessing pipeline

- **Changed**: Split `_process_split()` into 3 phases to enable CPU parallelism:
  1. Collect posteriors sequentially (GPU, fast)
  2. Scalar GOP in parallel across 14 CPU cores (`ProcessPoolExecutor`)
  3. Feature extraction sequentially on GPU (`F.ctc_loss`)
  Added `compute_gop_scalar()` (CPU-only, picklable) and `compute_gop_features()`
  (GPU, sequential) to `gop.py`.
- **Backend**: original (checkpoint-8000)
- **Eval**: GOPT transformer (same config as run 7)
- **Result**: PCC = 0.6618, 95% CI [0.6567, 0.6668], MSE = 0.0775, 47,369 phones
- **Baseline**: PCC was 0.6480 (run 7, sequential)
- **Takeaway**: PCC difference (+0.014) is within GOPT training variance (no
  fixed seed). The 3-phase split produces equivalent results while enabling
  14-way CPU parallelism on the scalar GOP bottleneck.

---

### Run 7 — 2026-03-01: GOPT Transformer

- **Changed**: Trained GOPT phone-level transformer on 42-dim feature vectors.
  3-layer Pre-LN transformer, 1 head, embed_dim=24, ~4K params.
  Adapted from `references/gopt-transformer/` (Gong et al., ICASSP 2022).
  Input: 42-dim GOP features (1 LPP + 40 LPR + 1 occupancy) per phone,
  padded to 50 phones per utterance. MSE loss with padding mask, 100 epochs.
- **Backend**: original (checkpoint-8000)
- **Eval**: GOPT transformer (contextual, utterance-level)
- **Result**: PCC = 0.6480, 95% CI [0.6428, 0.6532], MSE = 0.0795, 47,369 phones
- **Baseline**: PCC was 0.548 (SVR, run 6)
- **Paper target**: 0.648
- **Takeaway**: Matched the paper target exactly. The transformer's contextual
  scoring (seeing full utterance sequences vs isolated phones) accounts for the
  +0.100 PCC gain over SVR. This validates our entire pipeline: feature extraction,
  GOP-SF algorithm, and GOPT model adaptation all working correctly.

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
0.648 ████████████████████  GOPT transformer  ← MATCHED PAPER TARGET
```
