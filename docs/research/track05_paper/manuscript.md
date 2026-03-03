# CTC Phoneme Posteriors for Segmentation-Free Pronunciation Assessment

## Abstract

We study pronunciation assessment in a constrained but practical setting: fixed
CTC phoneme posteriors, fixed data protocol, and controlled changes only in the
scoring layer. Our pipeline uses segmentation-free goodness of pronunciation
(GOP) derived from CTC posteriors, then compares three downstream scoring
heads: per-phone polynomial regression on scalar GOP, per-phone SVR on GOP
feature vectors, and a phone-level GOPT transformer on the same feature vectors.
Experiments are run on SpeechOcean762 with a frozen backend
(`xlsr-espeak`) and a fixed evaluation protocol. The completed Phase-1
ablation shows a large gap between scalar and feature-based scoring:
PCC improves from 0.3195 (scalar+poly) to 0.5747 (SVR+features), and to
0.6774 mean PCC with GOPT over five seeds (std 0.0127), with corresponding MSE
drop from 0.6655 to 0.0729. These results indicate that, in this stack, most
performance gain comes from richer GOP representations and contextual
downstream scoring rather than from changing the upstream posterior model.
This draft is intentionally scoped as a reproducible systems-and-ablation study
built on segmentation-free GOP, not as a new GOP objective.

## 1. Introduction

Automatic pronunciation assessment is commonly framed as a posterior scoring
problem: given speech and a canonical phone sequence, produce phone-level (or
utterance-level) scores that correlate with expert ratings. In classic GOP
pipelines, this often depends on explicit alignment, which introduces failure
modes when timestamps or phone boundaries are noisy.

Segmentation-free GOP avoids forced alignment by marginalizing over CTC paths
[@cao2026segmentation_free_gop]. That makes it attractive as a robust scoring
core, but practical performance still depends on how those scores are consumed:
scalar per-phone scoring, richer feature vectors, or contextual models such as
GOPT [@gong2022gopt_transformer_pronunciation_assessment].

This paper isolates that downstream question. We fix the upstream posterior
generator and dataset protocol, then run controlled ablations over scoring
heads. The goal is to answer a narrow but actionable question: where does most
quality gain come from in a modern GOP-SF stack?

Contributions:
- A reproducible Track-05 scoring-layer benchmark protocol on SpeechOcean762.
- A controlled ablation of scalar GOP, feature-vector GOP+SVR, and feature-vector GOP+GOPT.
- Empirical evidence that feature richness plus contextual scoring dominates scalar GOP in this setup.

## 2. Related Work

### 2.1 Pronunciation Scoring and GOP

Segmentation-free GOP formulates pronunciation scoring directly from CTC
posteriors without forced alignment [@cao2026segmentation_free_gop]. Recent
extensions include phonologically constrained substitution spaces
[@parikh2025enhancing_gop_ctc_mdd] and logit-oriented scoring variants
[@parikh2025logit_based_gop]. Our work does not propose a new GOP objective in
this draft; we evaluate how existing GOP-derived signals behave under different
downstream heads.

### 2.2 Downstream Scoring Models

GOPT-style transformer models consume per-phone GOP features and predict human
scores with contextual phone-sequence modeling
[@gong2022gopt_transformer_pronunciation_assessment]. We adapt this idea as one
of the compared heads, alongside non-contextual baselines.

### 2.3 Phoneme Modeling Context

The broader space includes SSL-based pronunciation assessment
[@kim2022ssl_pronunciation_assessment], cross-lingual phoneme recognition
[@xu2021zeroshot_crosslingual_phoneme_recognition], large multilingual speech
models [@pratap2023mms_scaling_1000_languages], and universal phone recognition
systems [@li2020allosaurus_universal_phone_recognition]. Recent multilingual
phone-centric models (ZIPA/POWSM/PRiSM) further motivate robust phone-level
representations [@zhu2025zipa; @li2026powsm; @bharadwaj2026prism]. In this
paper we freeze backend choice to isolate scoring effects.

## 3. Methods

### 3.1 Pipeline Overview

Pipeline:
- Audio -> CTC phone posterior matrix `P(t, v)`.
- Canonical phone sequence `y` from dataset annotations.
- Segmentation-free GOP computations from CTC forward/denominator passes.
- Downstream score prediction via one of three heads (Poly, SVR, GOPT).

Implementation anchors:
- GOP core: `src/peacock_asr/gop.py`.
- Evaluation heads: `src/peacock_asr/evaluate.py`.
- GOPT adaptation: `src/peacock_asr/gopt_model.py`.

For phone position `i`, scalar GOP is computed as a log-likelihood ratio:
`GOP_i = -L_self + L_denom(i)`, where `L_self` is CTC NLL of canonical `y` and
`L_denom(i)` is CTC NLL when position `i` is replaced by an arbitrary state.

### 3.2 GOP-SF Feature Extraction

For each phone position, we extract:
- `LPP`: canonical CTC loss term.
- `LPR[k]`: substitution/deletion log-likelihood-ratio features over vocabulary.
- `occupancy`: expected count proxy from denominator forward pass.

This yields a per-phone feature vector of dimension `1 + |V| + 1`.
For the current backend in this study, that is 394 dimensions.
Feature extraction follows the batched CTC-loss implementation in `gop.py`
(`_compute_lpr_features_batched`), matching the reference formulation while
remaining practical on full splits.

### 3.3 Downstream Prediction Heads

We compare three heads:

1. Scalar + Poly:
- Input: scalar GOP only.
- Model: per-phone polynomial regression (order 2).

2. Features + SVR:
- Input: per-phone GOP feature vectors.
- Model: per-phone SVR with class balancing and cross-phone negatives.

3. Features + GOPT:
- Input: per-phone GOP feature vectors + phone IDs.
- Model: phone-level transformer (embed=24, heads=1, depth=3, seq_len=50),
  trained with masked MSE for 100 epochs.

## 4. Experimental Setup

### 4.1 Dataset and Splits

We use SpeechOcean762 [@zhang2021speechocean762] with its standard split
protocol as implemented in this repo (2,500 train / 2,500 test utterances).
All reported runs evaluate on the full test split.

### 4.2 Metrics

Primary metric is Pearson correlation coefficient (PCC) between predicted and
human phone-level scores. We report 95% confidence intervals from SciPy
`pearsonr(...).confidence_interval(...)`. Secondary metric is MSE.
Each completed run in Phase-1 evaluates on 46,314 phones.

### 4.3 Reproducibility

We run a fixed Phase-1 batch config:
`runs/track05_phase1_baseline.yaml`.

Jobs:
- `A1`: `scalar` (1 run).
- `A2`: `feats` (1 run).
- `A3`: `gopt` (5 runs; seeds 501-505).

Execution command:
`uv run peacock-asr batch --config runs/track05_phase1_baseline.yaml --output-dir runs`

Artifacts:
- `runs/2026-03-03_001037_track05_phase1_baseline/summary.tsv`
- `runs/2026-03-03_001037_track05_phase1_baseline/aggregates.tsv`

## 5. Results

### 5.1 Main Ablation Table (Frozen Backend: xlsr-espeak)

| ID | Mode | PCC | 95% CI | MSE | Notes |
|---|---|---:|---|---:|---|
| A1 | Scalar + Poly | 0.3195 | [0.3113, 0.3277] | 0.6655 | single run |
| A2 | Features + SVR | 0.5747 | [0.5686, 0.5808] | 0.2052 | single run |
| A3 | Features + GOPT | 0.6774 ± 0.0127 | per-run CIs in summary | 0.0729 ± 0.0024 | mean/std over 5 seeds |

### 5.2 GOPT Seeded Runs (A3)

| Seed | PCC | 95% CI | MSE |
|---:|---:|---|---:|
| 501 | 0.6958 | [0.6911, 0.7005] | 0.0694 |
| 502 | 0.6655 | [0.6604, 0.6706] | 0.0750 |
| 503 | 0.6782 | [0.6732, 0.6831] | 0.0727 |
| 504 | 0.6820 | [0.6771, 0.6868] | 0.0721 |
| 505 | 0.6653 | [0.6602, 0.6704] | 0.0751 |

### 5.3 Findings

- Moving from scalar GOP to feature-vector GOP with SVR increases PCC from
  `0.3195 -> 0.5747` (+0.2552 absolute).
- Replacing SVR with GOPT on the same feature family yields
  `0.5747 -> 0.6774` (+0.1027 mean absolute).
- Relative to scalar baseline, GOPT mean PCC is ~112% higher.

In this constrained experiment, downstream representation/scorer choice is the
dominant factor.

## 6. Discussion

The main empirical signal is clear: scalar GOP is not enough for strong
correlation in this setup, while richer GOP-derived vectors plus contextual
modeling provide substantial gains. This aligns with prior intuition that
downstream scoring capacity can dominate once posterior quality is adequate.

Scope boundaries:
- This is not evidence that backend quality is unimportant.
- This does not establish cross-dataset or cross-language generalization.
- This draft does not yet test algorithmic GOP variants (logit/constrained).

## 7. Limitations and Threats to Validity

- Single benchmark dataset (SpeechOcean762).
- Single frozen backend in this phase (`xlsr-espeak`).
- GOPT has non-trivial seed variance; we report mean/std but wider stability
  analysis is still limited.
- Results are phone-level correlation oriented; we do not evaluate end-user CAPT
  outcomes here.

## 8. Conclusion

This paper presents a narrow, reproducible ablation of scoring-layer design in
a segmentation-free GOP pipeline. Under a frozen backend and fixed protocol,
feature-vector GOP plus contextual GOPT scoring substantially outperforms
scalar GOP baselines. The immediate next step is to add algorithmic GOP
variants (e.g., logit-based and constrained substitutions) under the same
evaluation contract to test whether further gains come from the GOP objective
itself or mostly from downstream modeling.

## Reproducibility Appendix (Draft)

Run folder:
- `runs/2026-03-03_001037_track05_phase1_baseline/`

Primary artifacts:
- `batch_spec.yaml`
- `summary.tsv`
- `aggregates.tsv`
- per-run logs (`a1_scalar_r1.log`, `a2_feats_r1.log`, `a3_gopt_r1..r5.log`)

## Bibliography

Use `./refs.bib`.
