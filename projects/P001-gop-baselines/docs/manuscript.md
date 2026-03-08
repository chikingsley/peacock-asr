# CTC Phoneme Posteriors for Segmentation-Free Pronunciation Assessment

## Abstract

We study pronunciation assessment in a constrained but practical setting:
canonical text is known, phone-level human scores are available, and the main
question is how to score pronunciation once a phoneme posterior model is fixed.
Our pipeline uses a frozen CTC phoneme backend to produce phone posteriors over
time, computes segmentation-free goodness of pronunciation (GOP-SF) features
from those posteriors, and compares three downstream scoring heads: scalar GOP
with polynomial regression, GOP feature vectors with per-phone SVR, and GOP
feature vectors with a phone-level transformer scorer (GOPT)
[@cao2026segmentation_free_gop; @gong2022gopt_transformer_pronunciation_assessment].
The goal of this paper is narrow by design: isolate where the gain comes from
in a modern GOP-SF stack. In the completed `P001` paper-close campaign,
feature-vector scoring substantially outperformed scalar GOP, and GOPT reached
the best phone-level PCC on both frozen backends (`0.6368 ± 0.0115` on
`original`, `0.6774 ± 0.0127` on `xlsr-espeak`).

## 1. Introduction

Automatic pronunciation assessment is often framed as a constrained prediction
task: given a spoken utterance and the canonical text, estimate how well each
expected phone was produced. The central difficulty is that phone boundaries in
real speech are uncertain, especially for non-native speech. Traditional
alignment-heavy pipelines therefore inherit failure modes from forced alignment.

Segmentation-free GOP addresses that problem by computing pronunciation
evidence directly from CTC posteriors without requiring explicit phone
boundaries [@cao2026segmentation_free_gop]. That gives us a robust scoring
core, but it does not settle the next question: once those GOP signals exist,
what is the right downstream scorer? A single scalar per phone may be too weak.
Richer feature vectors may help. Contextual sequence modeling may help more.

This paper isolates that downstream question. We keep the dataset protocol
fixed, use frozen phoneme-posterior backends, and vary only the scoring head
and scalar score definition. The contribution is not a new speech foundation
model. It is a controlled paper-grade ablation over the scoring layer.

Our working contributions are:

- A reproducible pronunciation-scoring benchmark built around SpeechOcean762
  [@zhang2021speechocean762].
- A controlled comparison of scalar GOP, GOP feature vectors with SVR, and GOP
  feature vectors with GOPT.
- A scalar score-variant study comparing baseline GOP-SF, logit margin, and a
  convex mixture of the two.
- A project structure that separates backend quality questions (`P003`) from
  scoring-layer questions (`P001`).

## 2. Task Definition

The task in this paper is not open-ended ASR. We assume the expected text is
known. That text is converted to a canonical phone sequence, and the system
predicts phone-level pronunciation scores that should correlate with human
ratings.

The full pipeline is:

1. Audio waveform -> frozen phoneme-posterior backend.
2. Backend output -> phone posterior matrix `P(t, v)` over time `t` and phone
   vocabulary `v`.
3. Posterior matrix + canonical phone sequence `y` -> segmentation-free GOP
   features.
4. GOP features -> downstream scorer.
5. Downstream scorer -> predicted phone-level scores.

This task framing matters. It means:

- we are not solving unrestricted transcription here;
- we do not require forced phone boundaries;
- we care about correlation with human pronunciation scores, not only token
  accuracy.

## 3. Related Work

### 3.1 GOP and Mispronunciation Detection

Segmentation-free GOP computes pronunciation evidence directly from CTC
posteriors, avoiding forced alignment and making the scoring core more robust
to boundary noise [@cao2026segmentation_free_gop]. Recent extensions explore
phonological constraints and logit-based variants
[@parikh2025enhancing_gop_ctc_mdd; @parikh2025logit_based_gop].

### 3.2 Downstream Scoring

GOPT-style models show that phone-level transformer scoring can improve
pronunciation assessment by modeling contextual information across phone
positions [@gong2022gopt_transformer_pronunciation_assessment]. That is the
main contextual scorer baseline in this paper.

### 3.3 SSL and Phoneme Modeling Context

Self-supervised speech models such as wav2vec 2.0 and related encoders are now
standard backbones for phoneme-centric speech tasks, including pronunciation
assessment and cross-lingual phoneme recognition
[@kim2022ssl_pronunciation_assessment; @xu2021zeroshot_crosslingual_phoneme_recognition].
Large multilingual speech models and modern phone-recognition systems further
motivate strong phone-level representations
[@pratap2023mms_scaling_1000_languages; @li2020allosaurus_universal_phone_recognition; @zhu2025zipa; @li2026powsm; @bharadwaj2026prism].

Our scope is intentionally narrower. We use this broader literature as backend
context, but the paper's main claim is about scoring-layer design, not about
foundation-model pretraining itself.

## 4. Method

### 4.1 Frozen Phoneme-Posterior Backends

The backend takes raw audio and produces a posterior distribution over phones
at each time step. Architecturally, this is easiest to think about as:

- a speech encoder that maps waveform segments to contextual embeddings;
- an output head that maps those embeddings to phone logits;
- a softmax that converts logits into phone probabilities.

In this paper, the backend is frozen during scoring experiments. That allows us
to isolate the scoring layer. Backend training and phoneme-head adaptation live
in `P003`, not here.

### 4.2 Segmentation-Free GOP Core

Given a posterior matrix `P(t, v)` and canonical phone sequence `y`, we compute
per-phone pronunciation evidence using the segmentation-free GOP formulation
[@cao2026segmentation_free_gop]. The scalar score for phone position `i` is:

`GOP_i = -L_self + L_denom(i)`

where:

- `L_self` is the CTC negative log-likelihood of the canonical sequence;
- `L_denom(i)` is the CTC negative log-likelihood when phone position `i` is
  replaced by an arbitrary phone.

This paper keeps only the core equation in the main text. Low-level dynamic
programming details are implementation material and can stay in code and the
appendix.

Implementation anchors:

- GOP core: `projects/P001-gop-baselines/code/p001_gop/gop.py`
- evaluation/scoring glue: `projects/P001-gop-baselines/code/p001_gop/evaluate.py`
- GOPT scorer: `projects/P001-gop-baselines/code/p001_gop/gopt_model.py`

### 4.3 GOP Feature Extraction

For each phone position, we derive a feature vector from the GOP computation.
The exact feature family depends on the backend and scoring mode, but the main
ingredients are:

- canonical loss evidence;
- substitution-oriented likelihood-ratio features;
- occupancy-style signals from the denominator computation.

These features are richer than a single scalar GOP score and are the input to
both the SVR baseline and the GOPT scorer.

### 4.4 Downstream Scoring Heads

We compare three downstream scorers:

1. Scalar + Poly

- Input: one scalar GOP value per phone.
- Scorer: per-phone polynomial regression.

2. Features + SVR

- Input: GOP feature vector per phone.
- Scorer: per-phone SVR.

3. Features + GOPT

- Input: GOP feature vectors plus phone identities across the sequence.
- Scorer: a lightweight phone-level transformer in the style of GOPT
  [@gong2022gopt_transformer_pronunciation_assessment].

The key point is that GOPT does not replace GOP. It sits after GOP feature
extraction as a learned contextual scorer.

### 4.5 Scalar Score Variants

Phase 2 studies whether the scalar score itself can be improved without moving
to full feature-vector scoring. We compare:

- `gop_sf`: baseline scalar GOP-SF;
- `logit_margin`: a margin-style score derived from target vs alternative
  logits;
- `logit_combined`: a convex mixture of the two.

The mixture is:

`score = alpha * logit_margin + (1 - alpha) * gop_sf`

We evaluate three coarse settings in the seeded matrix
`alpha in {0.25, 0.50, 0.75}`, then run a dense cached alpha sweep over
`0.00..1.00` in steps of `0.05`.

## 5. Experimental Setup

### 5.1 Dataset

We use SpeechOcean762 [@zhang2021speechocean762], a standard benchmark for
non-native English pronunciation assessment with phone-level supervision.

### 5.2 Metrics

Primary metric is Pearson correlation coefficient (PCC) between predicted and
human phone-level scores. We also report 95% confidence intervals and mean
squared error (MSE).

### 5.3 Canonical `P001` Matrix

The canonical paper-close campaign contains two phases plus a dense analysis
pass:

Phase 1:

- `A1`: scalar + polynomial regression
- `A2`: features + SVR
- `A3`: features + GOPT, five seeds (`501..505`)

Phase 2:

- `B1`: `gop_sf`
- `B2`: `logit_margin`
- `B3`: `logit_combined`, `alpha=0.25`
- `B4`: `logit_combined`, `alpha=0.50`
- `B5`: `logit_combined`, `alpha=0.75`

Phase 2b:

- dense alpha sweep for `original`
- dense alpha sweep for `xlsr-espeak`

Backends:

- `original`
- `xlsr-espeak`

The campaign spec and naming contract are defined in
`docs/FINAL_CAMPAIGN_SPEC.md`.

### 5.4 Reproducibility Contract

The canonical evidence lives in:

- project-local batch artifacts under
  `projects/P001-gop-baselines/experiments/final/batches/`
- dense alpha sweep artifacts under
  `projects/P001-gop-baselines/experiments/final/alpha_sweeps/`
- machine and run manifests under
  `projects/P001-gop-baselines/experiments/final/manifests/`
- grouped W&B runs under
  `peacockery/peacock-asr-p001-gop-baselines`

This is the evidence source for the final tables, not older ad hoc `runs/`
folders.

## 6. Results

### 6.1 Phase 1: Downstream Scorer Comparison

Phase 1 answers the main paper question directly: richer GOP-derived features
help a great deal, and contextual sequence scoring helps again on top of that.
The effect is consistent across both frozen backends.

| Backend | A1 Scalar | A2 Features+SVR | A3 Features+GOPT |
|---|---:|---:|---:|
| original | 0.3104 | 0.5481 | 0.6368 ± 0.0115 |
| xlsr-espeak | 0.3195 | 0.5747 | 0.6774 ± 0.0127 |

Per-seed GOPT results are recorded in
`projects/P001-gop-baselines/experiments/final/results/per_run_summary.tsv`.

Interpretation:

- moving from scalar GOP to SVR on GOP features gives the largest jump;
- GOPT improves again over SVR on the same feature family;
- `xlsr-espeak` is the stronger frozen backend, but the ranking of
  `A1 < A2 < A3` holds on both backends.

### 6.2 Phase 2: Scalar Variant Ablation

This section is the scalar-only follow-up question:

- if we stay in the scalar regime, is plain GOP-SF already the best scalar?
- does logit information help?
- if it helps, where is the useful mixing range?

Observed scalar results:

| Backend | B1 gop_sf | B2 logit_margin | B3 alpha=.25 | B4 alpha=.50 | B5 alpha=.75 |
|---|---:|---:|---:|---:|---:|
| original | 0.3104 | 0.2206 | 0.3338 | 0.3022 | 0.2561 |
| xlsr-espeak | 0.3195 | 0.1849 | 0.3452 | 0.3222 | 0.2664 |

Interpretation:

- pure `logit_margin` is worse than baseline `gop_sf` on both backends;
- a low-weight mixture improves scalar PCC on both backends;
- even the best scalar mixture remains far below `A2` and `A3`.

### 6.3 Phase 2b: Dense Alpha Sweep

The dense sweep confirms that the useful scalar-mixing region is low-alpha for
both backends, but the exact optimum shifts slightly by backend:

| Backend | Best alpha | Best PCC | Best MSE |
|---|---:|---:|---:|
| original | 0.20 | 0.3361 | 0.6147 |
| xlsr-espeak | 0.25 | 0.3452 | 0.5981 |

The dense sweep is not another training phase. It is an analysis pass that
reuses cached scalar outputs to trace the full `alpha` response curve.

## 7. Discussion

The discussion should stay narrow and avoid overselling:

- If GOPT wins, the paper's main conclusion is that downstream representation
  and contextual scoring matter more than a scalar-only formulation under a
  fixed posterior backend.
- If the scalar mixture helps only slightly, the conclusion is not that scalar
  GOP is solved. It is that scalar score shaping has some upside, but does not
  close the gap to richer downstream scorers.
- If backend effects differ between `original` and `xlsr-espeak`, we should say
  that directly instead of collapsing them into one narrative.

## 8. Limitations

- Single benchmark dataset.
- Controlled, transcript-known evaluation setting.
- Frozen-backend framing means the paper does not address whether stronger
  phoneme backends alone would change the ordering.
- We evaluate phone-score correlation, not full end-user CAPT workflow quality.

## 9. Conclusion

This paper is a controlled study of pronunciation scoring once phoneme
posteriors are already available. The core question is not how to build the
largest speech model. It is how to turn phone posteriors into pronunciation
scores that align with human judgments. In this completed study, richer
GOP-derived features and contextual scoring dominate scalar-only scoring on
both frozen backends, while scalar logit mixing offers only a secondary gain.

## Appendix A. Figure Plan

Recommended main figure:

- Figure 1: `audio -> phoneme posterior matrix -> GOP feature extraction -> scorer -> phone-level scores`

Recommended appendix figure:

- Figure 2: dense `alpha` sweep curves for `original` and `xlsr-espeak`

## Appendix B. Writing Notes

- Keep only the two core equations in the main text.
- Leave low-level DP derivations in the appendix or code references.
- Use citekeys from `./refs.bib`.
- Populate final tables from the canonical campaign artifacts, not stale
  historical run folders.
