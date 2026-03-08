---
title: "Expert and Crowdsourced Annotation of Pronunciation Errors for Automatic Scoring Systems"
authors:
  - "Anastassia Loukina"
  - "Melissa Lopez"
  - "Keelan Evanini"
  - "David Suendermann-Oeft"
  - "Klaus Zechner"
citation_author: "Loukina et al"
year: 2015
doi: "10.21437/Interspeech.2015-591"
pages: "2809-2813"
source_pdf: "paper.pdf"
extraction_method: "Manual rewrite from the local paper.pdf; the existing local paper.md extraction was used only as a noisy cross-check."
extracted_at: "2026-03-07T20:03:41-08:00"
llm_friendly: true
---

# Title

Expert and Crowdsourced Annotation of Pronunciation Errors for Automatic Scoring Systems

## Metadata

- Authors: Anastassia Loukina, Melissa Lopez, Keelan Evanini, David Suendermann-Oeft, Klaus Zechner
- Venue: Interspeech 2015
- Pages: 2809-2813
- DOI: 10.21437/Interspeech.2015-591
- Task: collecting labels for pronunciation-error detection in non-native spontaneous English speech

## TL;DR

The paper compares expert pronunciation-error annotation with two crowd tasks: direct word-level error marking and plain orthographic transcription. All three approaches show disagreement, but the transcription task is more consistent and scales better.

The most important finding is practical: simple crowd transcription correlates with proficiency better than expert error annotation, and it is better aligned with intelligibility-focused assessment than detailed phonetic correction.

## Abstract

The paper argues that collecting training labels for pronunciation assessment is not just an annotation-cost problem, but also a construct problem. Expert phonetic annotation is expensive and subjective, and it may over-label deviations that do not matter for intelligibility. The authors test whether crowdsourced judgments, especially transcription-style judgments, provide a more reliable and more assessment-relevant signal for automatic scoring systems.

## Research Question

Which labeling strategy is most useful for training or evaluating pronunciation-related scoring in spontaneous non-native English speech:

- expert annotation of serious pronunciation errors
- crowd word-level error marking
- crowd orthographic transcription as an indirect measure of intelligibility

## Method

- The authors start from orthographic transcripts, then use forced alignment to obtain word and phone boundaries.
- Crowd workers on Amazon Mechanical Turk annotate short fragments in two separate tasks:
- error detection, where workers mark words they judge noticeably mispronounced
- transcription, where workers simply type what they heard in standard English spelling
- Each crowd fragment receives 5 judgments per task.
- Expert annotators label a subset of responses by marking serious pronunciation errors that should be corrected.
- The comparison focuses on three criteria:
- inter-annotator agreement
- validity relative to expert proficiency scores
- robustness to external factors such as playback behavior and perceived audio quality

## Data

- Speech: 143 responses from 140 non-native English speakers with 7 different native-language backgrounds
- Task type: unscripted English proficiency-test responses, up to 1 minute each
- Scoring: expert proficiency scores on a 1-4 scale focused on pronunciation, fluency, intelligibility, and listener effort
- Fragmentation: 1,767 short fragments initially, reduced to 1,752 after filtering
- Final crowd-analysis set: 14,374 words across 143 responses
- Crowd annotation volume: 17,670 judgments total across transcription and error-detection tasks
- Crowd annotators: 57 U.S.-based annotators, almost all self-reporting North American English as L1
- Expert annotation subset: 75 responses, with 12 double-annotated for agreement analysis

## Results

- Localization agreement is modest in every setting:
- crowd error detection: Fleiss' kappa `0.297`
- expert error detection: Cohen's kappa `0.492`
- crowd transcription-derived error labels: Fleiss' kappa `0.429`
- Agreement on the number of errors per response is stronger than exact localization:
- crowd error detection: `r = 0.71`
- expert annotation: `r = 0.53`
- crowd transcription: `r = 0.82`
- Correlation with expert proficiency scores favors crowd labels over expert error marking:
- crowd pronunciation-error probability: Spearman `rho = -0.70` on all 143 responses, `-0.72` on the 75-response overlap
- expert error annotation: `rho = -0.48`
- crowd transcription-error probability: `rho = -0.56` on all 143 responses, `-0.58` on the 75-response overlap
- On the overlapping 75 responses, crowd error labels and expert labels match only moderately:
- kappa `0.33` versus one expert
- kappa `0.27` versus the other
- The crowd transcription signal is less sensitive to annotator diligence effects than direct error marking.

## Limitations / Notes

- The expert-versus-crowd comparison is not fully matched because expert annotation is available only for a subset of the corpus.
- The paper studies spontaneous speech, which is more realistic than read speech but also harder to annotate consistently.
- The transcription task is not the same construct as phone-level pronunciation accuracy; it is closer to intelligibility.
- The conclusion is not that experts are unnecessary in all settings, but that fine-grained pronunciation-error labels are unusually subjective and expensive for this application.

## Relevance To Peacock

This paper is directly useful for dataset design. It supports collecting multiple cheap crowd judgments around intelligibility, especially transcription-like judgments, instead of assuming that expert phone-level correction is the only valid supervision.

For Peacock, the main takeaway is that if the target is learner-facing assessment or feedback tied to comprehensibility, transcription-derived labels may be more scalable and more faithful to the real construct than dense manual pronunciation-error markup.
