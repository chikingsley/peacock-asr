---
title: "Expert and Crowdsourced Annotation of Pronunciation Errors for Automatic Scoring Systems"
authors:
  - "Anastassia Loukina"
  - "Melissa Lopez"
  - "Keelan Evanini"
  - "David Suendermann-Oeft"
  - "Klaus Zechner"
citation_author: "Loukina et al."
year: 2015
doi: "10.21437/Interspeech.2015-591"
pages: "2809-2813"
source_pdf: "paper.pdf"
extraction_method: "manual-curated from local paper PDF"
extracted_at: "2026-03-23"
llm_friendly: true
---

## Metadata

- Venue: Interspeech 2015
- Authors: Anastassia Loukina, Melissa Lopez, Keelan Evanini, David Suendermann-Oeft, Klaus Zechner
- DOI: 10.21437/Interspeech.2015-591
- Pages: 2809-2813
- Source PDF: `paper.pdf`

## Abstract (Paraphrase)

The paper compares two annotation strategies for collecting labels used in pronunciation-focused automatic scoring on non-native spontaneous English speech:

1. expert annotation of pronunciation errors, and
2. crowdsourced annotation.

They evaluate crowdsourcing with two tasks: (a) direct pronunciation error detection and (b) orthographic transcription of short fragments. Their core finding is that simple transcription-style crowdsourcing aligns better with proficiency targets and reliability than direct error marking, especially for assessment goals tied to intelligibility/communication.

## Research Questions

The study compares annotation regimes on three axes:

- Inter-annotator agreement.
- Predictive validity relative to expert pronunciation proficiency scores.
- Robustness to annotation conditions (audio quality and annotator diligence).

## 1. Introduction

Automatic spoken scoring needs large labeled data. Traditional pronunciation error corpus building is expensive and subjective:

- experts provide phonetic detail, but tasks are slow and sometimes inconsistent;
- pronunciation is construct-complex because "error" itself depends on target accent, segmental focus, and listener goals.

The authors stress construct implications:

- human raters and assessments often evaluate communicative success (intelligibility/comprehensibility) rather than strict native-likeness.
- Some non-native pronunciation deviations affect perception less than prosody or larger-level features.

This motivates testing whether crowdsourcing can approximate useful training signal with lower cost and acceptable psychometric properties.

## 2. Data and Methodology

### 2.1 Corpus of non-native speech

Corpus details:

- 143 responses from 140 non-native English test takers.
- 7 native-language groups.
- Responses are up to ~1 minute and unscripted (spontaneous).
- 143 responses include scores from expert human judges in a 4-point proficiency rubric (overall pronunciation, fluency, intelligibility, listener effort).
- Average response segmentation yields 1,767 fragments initially; 1,752 fragments retained after quality filtering.
- Total post-filter word count: 14,374 words.

### 2.2 Annotation design

#### 2.2.1 Crowdsourced annotation

Set up on Amazon Mechanical Turk:

- each fragment presented in randomized order.
- each fragment annotated by multiple workers (5 judgments in final retained set).
- two tasks:
  - Error Detection (ED): mark words that are "noticeably mispronounced" using the reference text as support.
  - Transcription (TR): transcribe what they hear using standard English spelling.
- To control for possible task contamination, transcription tasks were posted before error detection tasks.
- MTurk workers filtered:
  - U.S.-based location
  - qualification test (sample transcription + error detection + demographics)
- Workers with outlying responses were excluded after quality analyses.

The ED and TR tasks were designed to test two different target constructs:

- ED approximates direct segmental accuracy.
- TR approximates intelligibility/understandability of what listeners actually perceive.

#### 2.2.2 Expert annotation

- Subset: 75 responses (12 double-annotated for agreement checks).
- Experts followed "serious errors only" guidelines adapted from prior work.
- Experts had full response access with waveform/spectrogram and could replay segments.
- Unlike crowdsourcers, expert labels were not tied to strict segment constraints from ASR transcript corrections.

## 3. Results

### 3.1 Inter-annotator agreement

All agreement uses localization kappa (Cohen for expert pairs, Fleiss for crowd with 5 annotators) and correlation r on number of errors per response.

Table 1 summary:

| Task | Annotation Set | Nw | Nr | kappa | r |
|---|---|---:|---:|---:|---:|
| ED | Crowd | 14,374 | 143 | 0.297 | 0.71 |
| ED | Expert | 1,443 | 12 | 0.492 | 0.53 |
| TR | Crowd | 14,374 | 143 | 0.429 | 0.82 |

Interpretation:

- Crowdsourcing improves count-level agreement relative to localization, but localization agreement remains modest.
- Transcription localization agreement is higher than ED crowd localization.
- Expert localization agreement is highest among the three cells.

### 3.2 Validity against proficiency scores

They computed per-annotator word-level error probabilities:

- Ppron: fraction of annotators marking a word as pronunciation error (ED).
- Ptr: fraction of annotators failing to transcribe a word (TR).
- Averaged per response to Pbarpron and Pbartr.

Correlations with expert proficiency:

- Pbarpron (Crowd): rho = -0.70 (all 143 responses), -0.72 (overlap subset of 75 responses).
- Pbarpron (Expert): rho = -0.48.
- Pbartr (Crowd): rho = -0.56 (all 143), -0.58 (overlap 75).

Negative values are expected since more errors mean lower proficiency scores.

### 3.3 Expert-crowd annotation overlap

On overlapping words with 75 shared responses (5,155 words):

- Agreement between crowd ED and first expert: kappa = 0.33
- Agreement between crowd ED and second expert: kappa = 0.28 (or 0.27 depending on count split reported)
- TR-to-expert localization agreement was not highlighted as strong in the text extract and is interpreted as not superior to expert ED overlap for exact localization.

Conclusion from this section:

- ED crowd is not a close proxy for expert ED labels at word level.
- Crowd-derived global indicators are more stable and more useful for downstream scoring than exact error localization.

### 3.4 External factor analysis

#### 3.4.1 Audio quality effects

After excluding fragments with clearly poor audio:

- For ED task, annotators were more lenient and less consistent under lower audio quality.
- For TR task, lower quality similarly related to more transcription errors, but:
  - ED quality judgments were less consistent.
  - TR quality judgments were more strongly tied to actual pronunciation difficulty.

Quantitative notes:

- ED quality judgments likely conflated audio and accent quality effects because annotators heard only transcript-coupled playback.
- TR quality judgments were more "interpretable" for intelligibility.

#### 3.4.2 Number of playbacks

Each crowd worker had to play at least once; extra playbacks were tracked:

- average playbacks varied between 1 and 8 across workers.
- ED task: more playbacks correlated with more marked pronunciation errors (partial r = 0.39, p = 0.01).
- TR task: no such correlation between extra playbacks and transcription error rate.

Interpretation:

- ED appears more sensitive to worker diligence/effort.
- TR task is more robust to diligence variance.

### 3.5 Discussion (within results section)

The authors interpret the patterns as:

- Direct error marking is highly subjective even between experts.
- Crowdsourcing can be more useful if the target is communicative/assessment-aligned pronunciation constructs.
- Transcription is less burdensome, has better relative agreement among workers, and is better aligned with holistic proficiency constructs where intelligibility matters.

## 4. Conclusion

Primary conclusions:

- Crowdsourced transcription and crowdsourced error detection are both noisier than ideal expert annotations at exact word-level localization.
- For predicting proficiency and supporting scoring targets, crowdsourced transcription provides better signal than crowdsourced error detection and, surprisingly, outperforms expert error-marking signal in this study's setup.
- If pronunciation assessment is meant to support real communication outcomes, simpler crowd tasks (like transcription) can be preferable.
- Expert annotations are not unnecessary in all settings, but they should be used with explicit construct-aware guidelines and realistic expectations about subjectivity.

## Practical Interpretation for Dataset Design

- If you need large-scale labels for pronunciation scoring, a transcription-first strategy can be cost-effective and construct-aligned with "understandable speech" outcomes.
- Exact phone-level correction labels may be too brittle for noisy, spontaneous corpora without very high-quality annotation processes and larger redundancy.
- This paper supports hybrid annotation strategies: inexpensive broad crowd annotations for scaling, plus smaller expert slices for construct calibration.

## Notes and extraction caveats

- The PDF text extraction was complete enough for section content and major statistics but some figure-level/table labels in external sections are noisy.
- Reported results are robustly supported in text by key values above; where ambiguous, I kept only values explicit in the extracted body.
