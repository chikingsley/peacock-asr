---
title: "Developing an Automatic Pronunciation Scorer: Aligning Speech Evaluation Models and Applied Linguistics Constructs"
authors:
  - "Danwei Cai"
  - "Ben Naismith"
  - "Maria Kostromitina"
  - "Zhongwei Teng"
  - "Kevin P. Yancey"
  - "Geoffrey T. LaFlair"
citation_author: "Cai et al."
year: 2025
doi: "10.1111/lang.70000"
pages: "170-196"
source_pdf: "paper.pdf"
extraction_method: "manual-curated from local paper PDF (PDF text extraction + method reconstruction)"
extracted_at: "2026-03-23"
llm_friendly: true
---

## Metadata

- Title: Developing an Automatic Pronunciation Scorer: Aligning Speech Evaluation Models and Applied Linguistics Constructs
- Authors: Danwei Cai, Ben Naismith, Maria Kostromitina, Zhongwei Teng, Kevin P. Yancey, Geoffrey T. LaFlair
- Year: 2025
- Journal: Language Learning 75(S1):170-196
- DOI: 10.1111/lang.70000
- Source PDF: `paper.pdf`

## Abstract (Paraphrase)

The study proposes a new automatic pronunciation scorer for Duolingo English Test (DET)-type high-stakes speaking responses. It argues that pronunciation scoring should be designed around current applied-linguistics definitions (intelligibility, comprehensibility, segmental and suprasegmental control) and aligned human judgments, not solely ASR proxy measures. They adapt an existing hierarchical speech model to open-response inputs, validate against expert human ratings, benchmark against four baselines, and run subgroup bias analysis with differential feature functioning (DFF).

## Research Questions

1. Can a state-of-the-art hierarchical pronunciation model be adapted to predict construct-aligned human pronunciation ratings on open-response L2 speech?
2. Does this adapted scorer outperform existing automatic baselines when predicting those ratings?
3. Does the adapted scorer show bias toward subgroup variables not intended as part of the construct (e.g., gender, device, language family)?

## Introduction and Problem Setting

The paper starts from a validity perspective in language assessment:

- Human scoring is costly, variable, and prone to fatigue or bias.
- Automatic scorers are increasingly used, but some still operate as black-box feature stacks with weak construct-level interpretability.
- For pronunciation, they argue high-stakes systems should target measurable constructs from applied linguistics and assessment standards, rather than only proxy features from transcript confidence or phoneme matching.
- They position their contribution as an operational, construct-aware system design rather than a pure CAPT (computer-assisted pronunciation training) feature benchmark.

## Background Literature

### 1) Pronunciation construct framing

The paper links pronunciation scoring to intelligibility/comprehensibility frameworks instead of strict native-like accent modeling. It discusses:

- Segmental relevance through functional load and intelligibility literature (e.g., high-load substitutions matter more than low-load variants).
- Prosodic and fluency dimensions (stress, rhythm, pauses, rate, timing, repair fluency, etc.).
- CEFR descriptors as a pragmatic scoring scaffold (overall phonological control, sound articulation, prosody), with emphasis on communication success over inner-circle accent ideals.

### 2) Existing automatic pronunciation and speech scorers

They contrast major systems used in English testing ecosystems:

- TOEFL iBT SpeechRater: broad feature stack but limited explicit segmental coverage according to the authors' review.
- Pearson CASE: high-level claims with less public feature transparency.
- Cambridge Linguaskill CASE.
- Duolingo's own existing DUO speaking scoring logic, where pronunciation is closely tied to ASR confidence in many operational uses.

The takeaway is that many production systems include pronunciation-like proxies but often lack explicit articulation of construct alignment.

### 3) CAPT-related automatic methods

They discuss GOP-style methods and more recent DNN-driven extensions:

- Traditional GOP is phoneme-level pronunciation confidence style scoring.
- They position it as useful but not sufficient for broad spoken proficiency constructs.

### 4) Fairness and bias

The authors motivate subgroup fairness by highlighting known bias risks in automatic scoring:

- L1/accents and dialect interactions.
- Gender effects.
- Device and recording-quality artifacts.

In high-stakes contexts, they treat fairness diagnostics as core validity evidence.

## The Proposed Scoring Framework

The model is adapted from a hierarchical read-aloud pronunciation scorer and extended to open-response.

### 1) Architecture overview

The hierarchy is:

- Frame-level features (~20 ms acoustic slices).
- Phoneme-level features (aggregated from frame features).
- Word-level features (aggregate phoneme outputs).
- Utterance-level features (aggregate word outputs into a single response representation).

The final pronunciation score is produced from utterance-level embeddings through a projection layer.

### 2) Feature details

- Frame features come from a phoneme recognition network over raw waveform.
- Phoneme embeddings are aligned to reference phoneme sequences derived from forced/transcribed text and then combined with acoustic outputs.
- Linguistic-acoustic similarity between observed phoneme embeddings and reference targets is computed to detect deviations likely linked to pronunciation quality.
- Transformer blocks are used at word and utterance aggregation stages.

### 3) Adapting from read-aloud to open-response

For open-ended responses (variable scripts and multiple utterances), the paper's pipeline is:

1. Transcribe with Whisper (large ASR model).
2. Segment transcript using punctuation boundaries and length constraints (rough target 10-35 words).
3. Force-align each segment with audio.
4. Score each segment with the adapted hierarchical model.
5. Compute response-level score as duration-weighted mean of segment scores.

The motivation is to keep computational load manageable while preserving temporal structure for long, unscripted responses.

## Dataset Construction

### 1) Human-rated pronunciation dataset

Total samples: 2,624 responses from 1,312 test takers.

| Subset | Samples | Double-rated |
|---|---:|---:|
| Pilot | 312 | 312 (100%) |
| Main | 2,312 | 469 (20.3%) |
| L2 English | 2,060 | 427 (20.7%) |
| L1 English | 252 | 42 (16.7%) |

Sampling constraints emphasized:

- Balanced by gender and provisional proficiency (stratified by an ML predictor of CEFR speaking level).
- L2 sample diversity by L1 and language-family grouping.
- L1-English group filtered for test takers from English-speaking countries (with minor exceptions noted).

### 2) Text and segment extraction

To preserve rating feasibility, they target ~30 second clips from longer task responses.

- They selected utterance-bounded segments based on punctuation and combined utterances until ~30 sec.
- Chosen clips had no mid-word trimming (selection at utterance boundaries).
- No aggressive denoising/VAD preprocessing was applied before scoring to preserve raw signal realism.
- Sampling source period: DET sessions from Feb 2022 to Mar 2023 across one image-description and three prompt-based task types.

### 3) Rubrics and rating process

Ratings use a 1-6 CEFR-style scale (A1-C2 mapping):

- Overall phonological control
- Sound articulation
- Prosodic features

These were applied holistically into one pronunciation score (not three separate sub-scores in production output for this study).

Raters:

- 4 expert raters: 2 in-house raters for pilot + 2 contractor raters for main.
- Contractor raters received calibration/training and independent rating; issues resolved by oversight from in-house lead.

Quality control:

- Invalid responses were removed if audio quality/pronunciation evidence was insufficient.
- Approximately 20% of the main set was double-rated.

### 4) Inter-rater agreement

Table 4 values:

- Quadratic weighted kappa: 0.85
- Spearman correlation: 0.84
- ICC: 0.85

Interpretation: very high agreement according to their stated benchmarks, giving confidence in target definition and label quality.

## Experiments

### 1) Evaluation protocol

- Five-fold cross-validation.
- Each fold: one split as test, four splits as train.

### 2) Baselines

Baseline systems compared:

- GOP scorer (phoneme confidence-based metric averaged over response).
- ASR confidence (Whisper medium ASR score as pronunciation proxy).
- Microsoft Pronunciation Assessment (MPA), including prosody and pronunciation outputs.
- Proposed hierarchical scorer trained on Speechocean762 as a transfer/data-compatibility comparison.

### 3) Scoring outputs and reporting

Primary metrics:

- Spearman's rho between automatic score and human rating.
- QWK (quadratic weighted kappa), with continuous model scores isotonic-projected back into 1-6 then rounded.

## Results

### 1) Main comparison with human judgments

| Method | Spearman's rho (95% CI) | QWK (95% CI) |
|---|---|---|
| Human interrater | 0.86 [0.84, 0.88] | 0.87 [0.84, 0.88] |
| GOP | 0.66 [0.64, 0.68] | 0.60 [0.58, 0.65] |
| Whisper (medium ASR) | 0.72 [0.71, 0.74] | 0.69 [0.67, 0.72] |
| MPA prosody | 0.77 [0.75, 0.78] | 0.71 [0.69, 0.74] |
| MPA pronunciation | 0.75 [0.74, 0.77] | 0.70 [0.68, 0.73] |
| Proposed scorer trained on Speechocean762 | 0.71 [0.69, 0.73] | 0.65 [0.63, 0.68] |
| Proposed scorer trained on DET pronunciation dataset | 0.82 (+/- 0.02) [0.81, 0.84] | 0.81 (+/- 0.01) [0.80, 0.83] |

Key finding:

- Dataset alignment (construct-aligned, L2 plus balanced representation) appears more important than simply reusing a standard pronunciation corpus in this domain.
- The proposed DET-trained scorer is close to human interrater agreement.

### 2) Significance

Steiger tests (pairwise r-comparisons) reported all p-values < .001 versus baselines, indicating statistically reliable gains over each compared method in this setting.

### 3) DFF bias analysis

Model:
Fi = b0 + b1 *Gi + b2* theta_i + epsilon

- Fi: pronunciation feature/score
- Gi: group indicator (gender, OS, language family binary flags)
- theta_i: gold-standard proficiency

Interpretation:

- b1 significant and non-zero indicates differential behavior for subgroup beyond proficiency.

Findings:

- Gender: no meaningful DFF signal.
- Operating system: consistent negative coefficients for Windows across methods; likely recording-quality confound (SNR noted as ~29 dB vs ~31 dB on Mac in internal analysis).
- Language family: negative bias for at least two families (paper text shows Indo-Aryan clearly and one other family appears in their table/analysis but is not clearly recoverable from extracted text).

Mitigation actions they propose:

- Add noisy and heterogeneous recording augmentation.
- Improve preprocessing for poorer channels.
- Better balance dataset composition by OS and affected language families.

## Discussion

- The strongest system-level message is that high interrater human standards plus construct-aligned labels can outperform strong commercial/baseline solutions.
- For this task, the proposed scoring model outperforms GOP, ASR confidence, and MPA by notable margins while remaining close to expert agreement.
- Remaining limitations are mostly operational and fairness-focused: representation imbalance and recording variability.

The paper explicitly notes that this version requires additional operational hardening before full deployment in production settings.

## Limitations and Inference Notes

- The exact identity of the second strongly flagged language-family group in DFF visuals is partially obscured in the local text extraction; the extracted paper still shows Indo-Aryan plus one additional impacted group in regression summaries.
- The study is high-stakes and specific to DET; external validity to other assessment products requires re-calibration and separate fairness checks.
- The pipeline includes extra cost: trained raters, larger annotated dataset, and significant compute.

## Relevance to Peacock (Pronunciation/APR Work)

This is directly adjacent to current work:

- It validates the core principle that open-response scoring needs ASR transcript mediation, but scoring robustness comes from construct alignment and rich segmental-to-utterance modeling.
- It supports using construct-aligned human data over raw benchmark read-aloud corpora for production pronunciation tasks.
- It provides a concrete blueprint for segment weighting and subgroup fairness diagnostics if we scale pronunciation scoring into a high-stakes style deployment.

## Quick Practical Mapping to Existing Work

- If your architecture already has frame/phone-level representation learning, this paper provides a tested reason to keep the hierarchy and add robust transcript-alignment for open-response.
- If you are evaluating by mean correlation only, the inclusion of DFF-style bias regression is the minimum responsible extension for subgroup trustworthiness.
