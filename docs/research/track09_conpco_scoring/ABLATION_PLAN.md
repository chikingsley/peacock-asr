# Track 09 Ablation Plan: ConPCO Integration

## Research Question

How much of the ConPCO+HierCB SOTA performance (PCC 0.743) comes from:

1. The ConPCO loss function alone (added to our existing GOPT)?
2. Additional input features (energy, duration, SSL embeddings)?
3. The HierCB architecture itself (BlockCNN, hierarchical levels)?

## Goal

Isolate each factor's contribution to PCC on SpeechOcean762 through incremental additions, following lab methodology (one change at a time, controlled compute).

## Frozen Setup (inherited from Track 05)

- Dataset: SpeechOcean762 (2500 train / 2500 test, pinned revision)
- Evaluation: PCC with 95% CI, minimum 3 seeds
- Feature extraction: GOP-SF from CTC posteriors
- Baseline: GOPT with MSE loss (Track 05 Phase 1 A3: PCC 0.6774 +/- 0.0127)

## Phase 1: ConPCO Loss on Existing GOPT (Minimal Change)

Keep architecture identical. Only change the training loss.

| Run ID | Model | Loss | Features | Purpose |
|---|---|---|---|---|
| P1-A | GOPT (ours) | MSE only | LPP+LPR (42-dim) | Baseline (rerun from Track 05 A3) |
| P1-B | GOPT (ours) | MSE + Ordinal Entropy | LPP+LPR (42-dim) | Isolate ordinal loss effect |
| P1-C | GOPT (ours) | MSE + OE + CLAP contrastive | LPP+LPR (42-dim) | Full ConPCO loss effect |

Implementation needed:

- Port `ContrastivePhonemicOrdinalRegularizer` from ConPCO repo
- Add two projection heads to GOPTModel: `phn_audio_proj`, `phn_text_proj`
- Combined loss: `MSE + w_pco * OE + w_clap * CLAP`
- Hyperparameters from ConPCO paper: pco_ld=0.5, pco_lt=0.1, pco_mg=1.0, clap_t2a=0.5

Expected effort: 1-2 days
Expected gain: +1-3% PCC (based on PCO predecessor paper, ASRU 2023)

## Phase 2: Feature Enrichment (Cheap Features First)

Keep GOPT architecture. Add features incrementally to the input.

| Run ID | Model | Loss | Features | Purpose |
|---|---|---|---|---|
| P2-A | GOPT | MSE + ConPCO (best from P1) | LPP+LPR + duration (43-dim) | Duration feature effect |
| P2-B | GOPT | MSE + ConPCO | LPP+LPR + duration + energy (50-dim) | Energy feature effect |
| P2-C | GOPT | MSE + ConPCO | LPP+LPR + dur + energy + wav2vec2 SSL (1074-dim) | Single SSL model effect |

Implementation needed:

- Duration extraction: from CTC forced alignment (we already have occupancy, convert to duration)
- Energy extraction: 7 RMS energy stats per phone segment (mean, std, median, MAD, max, min, sum)
- SSL extraction: wav2vec2-xlsr-53 mean-pooled embeddings per phone (1024-dim, we already load this model)

Expected effort: 3-5 days (Phase 2A/B easy, 2C needs phone-segmented SSL pooling)

## Phase 3: Architecture Ablation (If Phase 1-2 Show Promise)

Replace GOPT blocks with HierCB-style components.

| Run ID | Model | Loss | Features | Purpose |
|---|---|---|---|---|
| P3-A | GOPT + BlockCNN | Best from P2 | Best from P2 | CNN branch effect |
| P3-B | HierCB (phone only) | Best from P2 | Best from P2 | Full HierCB phone-level |
| P3-C | HierCB (phone + word) | Best from P2 | Best from P2 | Hierarchical word aggregation |

Expected effort: 1-2 weeks
Expected gain: Unknown, this is where the bulk of HierCB's advantage may lie

## Phase 4: Full Replication (If Worth It)

Full HierCB + ConPCO with all 3 SSL models (HuBERT + wav2vec2 + WavLM = 3072-dim).

| Run ID | Model | Loss | Features | Purpose |
|---|---|---|---|---|
| P4-A | HierCB (full) | Full ConPCO | All (3164-dim) | Full replication target |

Target: PCC >= 0.743 (match paper)

## Decision Rules

- `Phase 1 -> Phase 2`: If ConPCO loss gives any gain (>= 0.005 PCC), proceed. Even null result is publishable.
- `Phase 2 -> Phase 3`: If feature enrichment gives meaningful gain (>= 0.015 PCC), architecture changes are worth exploring.
- `Phase 3 -> Phase 4`: If phone-level HierCB shows promise, add full hierarchy.
- At any point: If cumulative gain exceeds 0.03 PCC over baseline, we have a paper.

## Paper Structure Preview

This ablation naturally generates the paper:

- **Table 1**: Phase 1 (loss function ablation)
- **Table 2**: Phase 2 (feature enrichment)
- **Table 3**: Phase 3 (architecture, if applicable)
- **Discussion**: Where does the gain actually come from? Is the loss, the features, or the architecture?
- **Practical takeaway**: What's the minimum-complexity path to meaningful improvement?

## Deliverables per Run

- Config snapshot (model, scorer, loss, feature dims)
- MLflow run ID + metrics JSON
- Wall-clock time, GPU hours
- One-line summary for results table
- Random seeds used (minimum 3)
