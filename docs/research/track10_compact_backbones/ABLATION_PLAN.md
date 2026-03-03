# Track 10 Ablation Plan: Compact Backbones

## Research Question

Can we match or exceed our current phone-level PCC (~0.68 with xlsr-53 300M) using
significantly smaller CTC backbones? What is the compute-accuracy Pareto frontier?

## Goal

Systematic comparison of CTC backbone size vs pronunciation scoring quality,
with two axes:

1. **Backbone axis**: Which model generates the CTC posteriors for GOP?
2. **Scorer axis**: GOPT transformer vs HMamba (Mamba blocks) as scoring head.

## Frozen Setup (inherited from Track 05)

- Dataset: SpeechOcean762 (2500 train / 2500 test, pinned revision)
- Evaluation: PCC with 95% CI, minimum 3 seeds
- Feature extraction: GOP-SF from CTC posteriors
- Scorer: GOPT with MSE loss (unless comparing HMamba)
- Baseline: xlsr-53 (300M) + GOPT, PCC 0.6774 +/- 0.0127

## Phase 1: Drop-In Backbone Swap (Same GOP Pipeline)

Keep GOP-SF algorithm and GOPT scorer identical. Only change the CTC backbone.
Each backbone needs CTC fine-tuning on LibriSpeech with 41-token ARPABET vocab.

| Run ID | Backbone | Params | Pre-training | Fine-tune Data | Purpose |
|---|---|---|---|---|---|
| P1-A | xlsr-53 (ours) | 300M | Multilingual | LibriSpeech 100h | Baseline (Track 05 A3) |
| P1-B | wav2vec2-base | 95M | English 960h | LibriSpeech 100h | 3x smaller, same family |
| P1-C | HuBERT-base | 95M | English 960h | LibriSpeech 100h | Different pre-training |
| P1-D | wav2vec2-large | 317M | English 960h | LibriSpeech 100h | Size control (bigger ≠ better?) |

Implementation needed:

- Fine-tune wav2vec2-base with CTC on LibriSpeech (41 ARPABET tokens)
- Fine-tune HuBERT-base with CTC on LibriSpeech (41 ARPABET tokens)
- Create backend adapters for each (same interface as `original` backend)
- Feature dim remains 42 (LPP 20 + LPR 20 + occupancy 2) for all

Expected effort: 3-5 days (fine-tuning 3 models + backend integration)
Expected result: wav2vec2-base may lose 2-5% PCC; HuBERT-base is the wildcard.

## Phase 2: Citrinet-256 (Extreme Compression)

Citrinet-256 is 10M params with 8x time reduction. Requires vocabulary change.

| Run ID | Backbone | Params | Vocab | Challenge | Purpose |
|---|---|---|---|---|---|
| P2-A | Citrinet-256 (stock) | 10M | 256 SentencePiece | Subword → phoneme mapping | Feasibility test |
| P2-B | Citrinet-256 (fine-tuned) | 10M | 41 ARPABET | Replace CTC head + fine-tune | Direct phoneme CTC |

Implementation needed:

- P2-A: Investigate SentencePiece → ARPABET mapping via lexicon
- P2-B: Replace Citrinet CTC head (256 → 41 tokens), fine-tune on LibriSpeech
- Handle 8x time reduction: GOP-SF expects ~20ms frames, Citrinet uses ~80ms
- NeMo ↔ HuggingFace model conversion

Expected effort: 5-7 days (NeMo ecosystem integration is non-trivial)
Expected result: Unknown. 30x smaller model may lose 5-15% PCC.

## Phase 3: Alternative Scoring Heads

Keep the best backbone(s) from Phase 1. Swap the scoring head.

| Run ID | Backbone | Scorer | Params (head) | Purpose |
|---|---|---|---|---|
| P3-A | Best from P1 | GOPT transformer | ~31K | Baseline scorer |
| P3-B | Best from P1 | HMamba (Mamba blocks) | ~31K (est.) | Mamba vs transformer |
| P3-C | Best from P1 | HiPAMA (multi-aspect attention) | ~32K | Published alternative |

Implementation needed:

- Port HMamba Mamba blocks from <https://github.com/Fuann/hmamba>
- Port HiPAMA multi-aspect attention from <https://github.com/doheejin/HiPAMA>
- Keep input features identical (42-dim GOP-SF)

Expected effort: 3-5 days
Expected result: Scoring head may matter less than backbone quality.

## Phase 4: Pareto Analysis

Combine results into a compute-accuracy tradeoff plot.

| Configuration | Backbone Params | Head Params | Total FLOPs | PCC |
|---|---|---|---|---|
| P1-A (baseline) | 300M | ~31K | TBD | 0.677 |
| P1-B (small) | 95M | ~31K | TBD | TBD |
| P1-C (HuBERT) | 95M | ~31K | TBD | TBD |
| P2-B (tiny) | 10M | ~31K | TBD | TBD |
| P3-B (Mamba) | best | ~31K | TBD | TBD |

This table becomes Figure 1 of the paper: PCC vs backbone parameters, with
each point labeled by model name.

## Decision Rules

- `Phase 1 → Phase 2`: Always proceed (Phase 2 is independent).
- `Phase 1 → Phase 3`: If any 95M backbone achieves >= 0.65 PCC, scoring head matters.
- `Phase 2 → paper`: Even negative Citrinet results are publishable (first attempt).
- At any point: If a 95M backbone matches the 300M baseline within CI, that's the headline.

## Paper Structure Preview

- **Table 1**: Phase 1 (backbone comparison, same scorer)
- **Table 2**: Phase 2 (Citrinet feasibility)
- **Table 3**: Phase 3 (scoring head comparison)
- **Figure 1**: Pareto frontier (PCC vs params)
- **Discussion**: Is the 300M backbone justified? Practical deployment implications.

## Deliverables per Run

- Config snapshot (backbone, scorer, loss, feature dims)
- MLflow run ID + metrics JSON
- Wall-clock time, GPU hours, backbone inference time per utterance
- One-line summary for results table
- Random seeds used (minimum 3)
