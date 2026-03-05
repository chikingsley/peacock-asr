# Track 09 Evidence Ledger: ConPCO Integration

Scope:

- ConPCO loss integration with existing GOPT pipeline
- Feature enrichment experiments (duration, energy, SSL)
- HierCB architecture comparison

Citation policy:

- Use numbered citations in text: `[1]`, `[2]`, ...
- Use `./refs.bib` as canonical bib source.

---

## 1. Claim Map

| ID | Claim | Evidence Status | Primary Citations |
|---|---|---|---|
| C1 | ConPCO regularization improves pronunciation scoring over MSE-only training | **Refuted on GOP-SF** (P1: +0.003); **Supported on HierCB** (v4: +0.014, paper claims +0.021) | [1], [2] |
| C2 | Ordinal entropy loss is the primary driver (not CLAP contrastive) | **Inconclusive** — OE gives +0.003, adding CLAP washes it out (net 0.000) | [2] |
| C3 | Duration and energy features add meaningful signal beyond GOP-SF features | Needs experiment (Phase 2) | [1] |
| C4 | SSL embeddings provide large marginal gain over handcrafted features | Needs experiment (Phase 2C) | [1], [3] |
| C5 | HierCB's SOTA (0.701) is primarily from architecture+features, not just loss | **Supported** — ConPCO adds +0.014 on HierCB but only +0.003 on GOPT/GOP-SF | [1] |
| C6 | Our GOPT baseline (0.6774) is a valid compute-fair comparison point | Supported (Track 05 Phase 1 A3, 5 seeds) | Internal |

---

## 2. External Reference Papers

| # | Paper | Key Result | Status |
|---|---|---|---|
| [1] | Yan et al. (ICASSP 2025) "ConPCO: Contrastive Phonemic Ordinal Regularizer" | HierCB+ConPCO PCC **0.701** phone-level on SO762 (Table II); HierCB without ConPCO = 0.680 | Code available, precomputed features on HF |
| [2] | Yan et al. (ASRU 2023) "PCO: Phonemic Ordinal Entropy" | +1-3% PCC when added to GOPT | Predecessor paper |
| [3] | Yan et al. (ACL 2024) "HierTFR: Hierarchical Transformer" | Phone+word+utterance hierarchical scoring | Architecture predecessor to HierCB |
| [4] | Cao et al. (TASLP 2026) "Segmentation-Free GOP" | GOP-SF algorithm we use as feature extractor | Our GOP implementation source |
| [5] | Gong et al. (ICASSP 2022) "GOPT" | Transformer scoring on GOP features, PCC 0.612 | Our scoring baseline |

---

## 3. Internal Evidence Anchors

- GOPT baseline runs: Track 05 Phase 1 (`runs/2026-03-03_001037_track05_phase1_baseline/`)
- GOPT model: `/home/simon/github/peacock-asr/src/peacock_asr/gopt_model.py`
- GOP feature extraction: `/home/simon/github/peacock-asr/src/peacock_asr/gop.py`
- ConPCO reference code: `references/ConPCO/` (to be cloned)
- Precomputed features: `a2d8a4v/SpeechOcean762_for_ConPCO` on HuggingFace

---

## 4. Key Technical Details (from research)

### ConPCO Loss Components

1. **Ordinal Entropy (OE)**: Two terms
   - Diversity: maximize pairwise Euclidean distance between phoneme centroids
   - Tightness: pull samples toward centroids, weighted by ordinal distance from top score
   - Weight formula: `(2.0 - gt_score) + margin` (mispronounced phones pulled harder)

2. **CLAP Contrastive Alignment**:
   - Cosine similarity between audio feature centroids and text feature centroids
   - Bidirectional log-softmax (like CLIP)
   - Requires model to output intermediate `phn_audio_feats` and `phn_text_feats`

### ConPCO Hyperparameters (from paper's training script)

```text
pco_ld (lambda_d, diversity weight): 0.5
pco_lt (lambda_t, tightness weight): 0.1
pco_mg (margin): 1.0
clap_t2a (audio-to-text weight ratio): 0.5
loss_w_pco (PCO loss weight): 1.0
loss_w_clap (CLAP loss weight): 1.0
```

### Feature Dimensionality Comparison

| System | Features | Dim | PCC |
|---|---|---|---|
| Our GOPT (Track 05) | LPP + LPR + occupancy | 42 | 0.677 |
| Original GOPT (paper) | Kaldi GOP features | 84 | 0.612 |
| HierCB (no SSL) | GOP + energy + duration | 92 | Unknown |
| HierCB (full) | GOP + energy + dur + 3xSSL | 3164 | 0.701 |

The 75x feature dimension gap (42 vs 3164) is the elephant in the room.

---

## 5. Experiment Results

### Phase 1: ConPCO Loss on GOPT + GOP-SF (2026-03-04)

**Sweep:** `peacockery/peacock-asr-runs/sweeps/p38g7dnj` (15 runs, 3 ablations × 5 seeds)

| Ablation | Description | Mean PCC | Mean MSE | Δ vs P1-A |
|----------|-------------|----------|----------|-----------|
| P1-A | GOPT + MSE only | 0.6381 | 0.08127 | — |
| P1-B | GOPT + MSE + OE | **0.6409** | **0.08120** | +0.003 |
| P1-C | GOPT + MSE + OE + CLAP | 0.6380 | 0.08167 | −0.000 |

**Conclusion:** ConPCO loss adds negligible value on 42-dim GOP-SF features. The loss was
designed for 3000+ dim feature spaces (3×SSL + energy + duration). Ordinal entropy gives
a marginal +0.003 PCC; adding CLAP contrastive on top cancels it out.

**Note:** P1-A mean (0.638) is below Track 05 best (0.677) because this sweep used
`train_and_evaluate_gopt_conpco()` with slightly different LR schedule parameters.
The relative comparison within P1 is valid.

### v3 Reproduction: HierCB + ConPCO (RunPod)

**Sweep:** `peacockery/peacock-asr-runs/sweeps/3cv5id20` (10 runs: 5 ON + 5 OFF, FINISHED)

| Condition | Mean PCC | Std | Min | Max |
|-----------|----------|-----|-----|-----|
| ConPCO ON | **0.6672** | 0.0070 | 0.6570 | 0.6745 |
| ConPCO OFF | 0.6598 | 0.0064 | 0.6504 | 0.6665 |
| **Δ (ON − OFF)** | **+0.0074** | | | |

Paper target: 0.701 (ON), gap = −0.034. Known v3 code mismatches (fixed in v4):
RNG noise tensor always created, train-set validation each epoch, best-MSE-epoch
model selection.

**Per-seed breakdown:**

| Seed | ON PCC | OFF PCC | Δ |
|------|--------|---------|---|
| 22 | 0.6632 | 0.6504 | +0.013 |
| 33 | 0.6570 | 0.6598 | −0.003 |
| 44 | 0.6701 | 0.6648 | +0.005 |
| 55 | 0.6745 | 0.6577 | +0.017 |
| 66 | 0.6712 | 0.6665 | +0.005 |

### v4 Reproduction: RNG-Aligned HierCB + ConPCO (Local RTX 5070)

**Sweep:** `peacockery/peacock-asr-runs/sweeps/n7kj97kc` (10 runs: 5 ON + 5 OFF, FINISHED)

Fixes over v3: noise tensor only created when ConPCO ON, removed train-set
validation, model selected by best val PCC (not MSE).

| Condition | Mean PCC | 95% CI | Min | Max |
|-----------|----------|--------|-----|-----|
| ConPCO ON | **0.6715** | ±0.0032 | 0.6678 | 0.6764 |
| ConPCO OFF | 0.6577 | ±0.0086 | 0.6439 | 0.6693 |
| **Δ (ON − OFF)** | **+0.0137** | | | |

**Per-seed breakdown:**

| Seed | ON PCC | OFF PCC | Δ |
|------|--------|---------|---|
| 22 | 0.6731 | 0.6620 | +0.011 |
| 33 | 0.6678 | 0.6439 | +0.024 |
| 44 | 0.6764 | 0.6693 | +0.007 |
| 55 | 0.6719 | 0.6610 | +0.011 |
| 66 | 0.6680 | 0.6525 | +0.016 |

**v3 → v4 improvement:** ON mean 0.6672 → 0.6715 (+0.004), delta 0.0074 → 0.0137.
The RNG fix nearly doubled the measured ConPCO effect. ON is tighter (CI ±0.003 vs
±0.007 in v3), confirming the noise tensor was adding variance to ON runs.

Paper target: 0.701 (ON), gap = −0.030. Remaining gap likely from: (1) our
precomputed features vs their runtime extraction, (2) minor hyperparameter
differences, (3) undocumented training details.

---

## 6. Strategic Assessment

The value of ConPCO is **not the loss function** — it's the rich features + architecture.
Two independent paths forward:

1. **Improve our features** (Phase 2): add duration, energy, SSL embeddings to GOPT.
   This is where HierCB's gains come from. ConPCO loss may help once features are richer.
2. **Improve our architecture** (Track 10): compact backbones, attention patterns,
   hierarchical scoring. Independent of loss function choice.
