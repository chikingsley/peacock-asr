---
title: "Assessment of Non-Native Speech Intelligibility using Wav2vec2-based Mispronunciation Detection and Multi-level Goodness of Pronunciation Transformer"
authors: "Ram C.M.C Shekar; Mu Yang; Kevin Hirschi; Stephen Looney; Okim Kang; John Hansen"
citation_author: "Shekar et al"
year: 2023
doi: "10.21437/Interspeech.2023-2371"
pages: 5
source_pdf: "paper.pdf"
extraction_method: "manual-curated from paper.pdf"
extracted_at: "2026-03-07"
llm_friendly: true
---

# Assessment of Non-Native Speech Intelligibility using Wav2vec2-based Mispronunciation Detection and Multi-level Goodness of Pronunciation Transformer

## Metadata

- Authors: Ram C.M.C Shekar; Mu Yang; Kevin Hirschi; Stephen Looney; Okim Kang; John Hansen
- Citation author: Shekar et al
- Year: 2023
- DOI: 10.21437/Interspeech.2023-2371
- arXiv: N/A
- Pages: 5
- Venue: Interspeech 2023
- Source PDF: `paper.pdf`

## TL;DR

This paper studies whether pronunciation-assessment signals can help explain intelligibility in L2 English speech. The authors combine utterance-, word-, and phone-level scores from GOPT with wav2vec2-based mispronunciation features and hand-labeled prosodic features on an International Teaching Assistant (ITA) speech dataset. Their main conclusion is that human-rated prosody is the strongest predictor of intelligibility, while phoneme-level MDD features are more informative than multilevel GOPT scores when prosodic labels are absent.

## Abstract

Automatic pronunciation assessment (APA) is useful for CAPT, but intelligibility assessment remains difficult, especially when labeled non-native speech is scarce and phone-level accuracy alone is insufficient. This paper combines wav2vec 2.0-based mispronunciation detection and diagnosis (MDD) with Goodness of Pronunciation Transformer (GOPT) scores to characterize L2 intelligibility. Using an L2 speech dataset with human-annotated prosodic labels, the study compares multigranular pronunciation scores, ASR transcripts, and phoneme-level diagnostic features as predictors of intelligibility. The authors argue that this combination can support more practical, near-instant intelligibility assessment tools for L2 learners.

## Research Question

The paper asks which automatically derived pronunciation signals best predict perceived intelligibility in non-native English speech. More concretely, it compares:

- multilevel GOPT scores
- wav2vec2-based MDD diagnostics
- human versus ASR transcripts
- human-rated prosodic annotations such as articulation rate, lexical stress, and pause duration

## Method

The system has two main modeling components.

- GOPT is used as a multigranular pronunciation scorer. It produces phone-, word-, and utterance-level scores, including utterance-level accuracy, fluency, completeness, prosody, and total score.
- A wav2vec2-based MDD model provides phoneme-diagnostic features such as phoneme error rate (PER), match error rate (MER), and information loss (IL).

The authors then run downstream regression analyses to predict intelligibility from different combinations of these features. They compare simple linear regression, random forest regression, and XGBoost, and use feature-importance analysis to understand which inputs matter most.

## Data

The study uses an ITA dataset built from conversational classroom speech produced by 54 adult learners with diverse L1 backgrounds. The recordings are split into lower-intelligibility `L2` and higher-intelligibility `L2-High` portions.

The paper reports several annotation conditions:

- `L2` with human speech recognition (HSR): 15 speakers, 887 utterances
- `L2` with ASR transcripts: 15 speakers, 5012 utterances
- `L2` with HSR plus prosody labels: 57 speakers, 3000 utterances
- `L2` with ASR plus prosody labels: 57 speakers, 4839 utterances
- `L2-High` with HSR: 19 speakers, 438 utterances

Human prosody annotations include articulation rate, lexical stress, and silent pause duration. ASR transcripts are produced with Whisper.

## Results

The empirical pattern is fairly consistent across the paper:

- Random forest regression performs best among the tested regressors.
- When prosodic labels are available, human-rated prosodic features dominate intelligibility prediction.
- Among GOPT features, utterance-level scores matter more than word- or phone-level scores.
- In settings without human prosody labels, MDD features such as information loss, phoneme error rate, and match error rate become strong predictors.
- The paper claims that phoneme-level MDD signals are more useful for intelligibility prediction than the multilevel GOPT scores alone.
- Human versus ASR transcription appears less important than the presence or absence of prosodic and diagnostic features.

The authors especially emphasize that prosody and low-level phonetic diagnostics carry more signal for intelligibility than a generic multiaspect pronunciation score.

## Limitations / Notes

- The paper is not introducing a new end-to-end intelligibility model; it is mainly a feature-comparison study built on existing GOPT and wav2vec2-style components.
- A large part of the reported signal comes from human-labeled prosodic annotations, which may be expensive to reproduce at scale in a production setting.
- The dataset composition is somewhat heterogeneous, with different subsets having different combinations of human and machine annotations.
- Some regression results in the paper appear almost perfectly fit under random forest, so they should be read cautiously as feature-exploration results rather than strong evidence of robust generalization.

## Relevance To Peacock

This paper is useful because it pushes against a narrow "phone accuracy is enough" view of pronunciation assessment.

- It supports treating intelligibility as related to, but not reducible to, phone-level correctness.
- It reinforces the importance of prosodic supervision or prosody-aware proxy features.
- It suggests that a practical Peacock-style assessment stack may need separate channels for:
  - low-level phoneme diagnostics
  - utterance-level assessment
  - explicit prosody features
- It is also a useful citation when motivating why MDD outputs should be part of a broader intelligibility model rather than the whole model.
