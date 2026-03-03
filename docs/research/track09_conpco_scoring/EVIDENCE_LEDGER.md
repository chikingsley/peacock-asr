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
| C1 | ConPCO regularization improves pronunciation scoring over MSE-only training | Needs experiment (Phase 1) | [1], [2] |
| C2 | Ordinal entropy loss is the primary driver (not CLAP contrastive) | Needs experiment (Phase 1 B vs C) | [2] |
| C3 | Duration and energy features add meaningful signal beyond GOP-SF features | Needs experiment (Phase 2) | [1] |
| C4 | SSL embeddings provide large marginal gain over handcrafted features | Needs experiment (Phase 2C) | [1], [3] |
| C5 | HierCB's SOTA (0.743) is primarily from architecture+features, not just loss | Hypothesis from research agent analysis | [1] |
| C6 | Our GOPT baseline (0.6774) is a valid compute-fair comparison point | Supported (Track 05 Phase 1 A3, 5 seeds) | Internal |

---

## 2. External Reference Papers

| # | Paper | Key Result | Status |
|---|---|---|---|
| [1] | Yan et al. (ICASSP 2025) "ConPCO: Contrastive Phonemic Ordinal Regularizer" | HierCB+ConPCO PCC 0.743 phone-level on SO762 | Code available, precomputed features on HF |
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
| HierCB (full) | GOP + energy + dur + 3xSSL | 3164 | 0.743 |

The 75x feature dimension gap (42 vs 3164) is the elephant in the room.
