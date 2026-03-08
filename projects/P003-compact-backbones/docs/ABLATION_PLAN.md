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

## Phase 0: Backbone Preparation (CTC Fine-Tuning Recipe)

Before swapping backbones, we need to validate a CTC fine-tuning recipe that works
for any HuggingFace SSL model with a 41-token ARPABET vocabulary on LibriSpeech.

| Run ID | Backbone | Params | Script | Status | Purpose |
|---|---|---|---|---|---|
| P0-A | w2v-bert-2.0 | 600M | `projects/P003-compact-backbones/code/training/train_phoneme_head.py` | **Training + eval complete** | Validate recipe, produce 600M Pareto point |

Details:

- Training on LibriSpeech (train_clean_100 + train_clean_360 + train_other_500)
- 41-token ARPABET CTC head, BF16, 3 epochs
- Hub push to `Peacockery/w2v-bert-phoneme-en`
- MLflow tracking at `mlflow.peacockery.studio`
- Same recipe will be reused for wav2vec2-base and HuBERT-base in Phase 1

The w2v-bert-2.0 result is useful in two ways: (1) it validates the fine-tuning
pipeline end-to-end, and (2) it adds a 600M data point to the Pareto plot,
establishing the upper bound for backbone size. Final `P003` eval result:
`0.6755 +/- 0.0066 PCC`, effectively matching the `xlsr-53 + GOPT` baseline.

## Phase 1: Drop-In Backbone Swap (Same GOP Pipeline)

Keep GOP-SF algorithm and GOPT scorer identical. Only change the CTC backbone.
Each backbone needs CTC fine-tuning on LibriSpeech with 41-token ARPABET vocab.

| Run ID | Backbone | Params | Pre-training | Fine-tune Data | Purpose |
|---|---|---|---|---|---|
| P1-A | xlsr-53 (ours) | 300M | Multilingual | LibriSpeech 100h | Baseline (Track 05 A3) |
| P1-B | wav2vec2-base | 95M | English 960h | LibriSpeech 100h | 3x smaller, same family |
| P1-C | HuBERT-base | 95M | English 960h | LibriSpeech 100h | Different pre-training |
| P1-D | wav2vec2-large | 317M | English 960h | LibriSpeech 100h | Size control (bigger ≠ better?) |

Current Phase 1 status:

- `P1-B` complete: `wav2vec2-base + GOPT = 0.640 +/- 0.009 PCC`
- `P1-C` complete: `HuBERT-base + GOPT = 0.6489 +/- 0.0093 PCC`
- `P1-D` still pending

Implementation still needed:

- Fine-tune wav2vec2-large with CTC on LibriSpeech (41 ARPABET tokens)
- Create backend adapter for wav2vec2-large (same interface as `original` backend)
- Feature dim remains 42 (LPP 20 + LPR 20 + occupancy 2) for all

Interpretation so far:

- HuBERT-base modestly improves on wav2vec2-base at the same 95M scale
- Neither 95M model matches the `xlsr-53 + GOPT` baseline
- `wav2vec2-large` is now the cleanest remaining Phase 1 backbone question

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
| P1-B (small) | 95M | ~31K | TBD | 0.640 |
| P1-C (HuBERT) | 95M | ~31K | TBD | 0.649 |
| P0-A (w2v-BERT) | 600M | ~31K | TBD | 0.676 |
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
