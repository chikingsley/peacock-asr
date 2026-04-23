---
title: "Explore wav2vec 2.0 for Mispronunciation Detection"
authors:
  - "Xiaoshuo Xu"
  - "Yueteng Kang"
  - "Songjun Cao"
  - "Binghuai Lin"
  - "Long Ma"
citation_author: "Xu et al"
year: 2021
venue: "Interspeech 2021"
doi: "10.21437/Interspeech.2021-777"
pages: "4428-4432"
source_pdf: "paper.pdf"
extraction_method: "Manually summarized from the published PDF; no public LaTeX source was located."
extracted_at: "2026-03-22"
llm_friendly: true
---

## Metadata

- Authors: Xiaoshuo Xu, Yueteng Kang, Songjun Cao, Binghuai Lin, Long Ma
- Venue: Interspeech 2021
- DOI: 10.21437/Interspeech.2021-777
- Pages: 4428-4432
- Task: phone-level mispronunciation detection for prompted L2 speech

## TL;DR

This paper is an early demonstration that wav2vec 2.0 pretraining transfers well to mispronunciation detection. The model uses unlabeled speech for self-supervised pretraining, then fine-tunes with a small amount of pronunciation-labeled data using a lightweight convolution + pooling head over aligned phone segments. On L2-ARCTIC, it reaches an F1 of 0.610 and beats prior detection baselines.

## Abstract

The paper treats mispronunciation detection as a binary classification problem over prompted speech segments. It uses wav2vec 2.0 to pretrain on unlabeled audio, then fine-tunes on a small non-native dataset with a simple classifier on top of the speech representation. The core claim is that self-supervised pretraining can replace ASR-style pretraining and still provide strong representations for MDD.

## Method

- Pretrain wav2vec 2.0 on Librispeech.
- Fine-tune on L2-ARCTIC with a small labeled set.
- Add a pointwise convolution layer and adaptive max pooling on top of the encoder.
- Use the canonical phone and alignment range as inputs to classify whether a phone segment is mispronounced.
- Compare against an ASR-pretrained variant, a direct training baseline, and earlier likelihood-based methods.

## Results

- Proposed-SS reaches `0.610` F1 on the L2-ARCTIC test set.
- Proposed-ASR reaches `0.602` F1.
- Direct training without pretraining is much weaker at `0.381` F1 on validation.
- The paper reports that wav2vec 2.0 pretraining is competitive with ASR pretraining and that the method is less sensitive to imperfect alignment than likelihood-based approaches.

## Relevance To Peacock

This is a useful reference for low-resource MDD pipelines because it shows that SSL backbones can replace explicit ASR pretraining without losing much performance. The model is simple, which makes it a good baseline for future layer-selection or diagnostic-feature experiments.
