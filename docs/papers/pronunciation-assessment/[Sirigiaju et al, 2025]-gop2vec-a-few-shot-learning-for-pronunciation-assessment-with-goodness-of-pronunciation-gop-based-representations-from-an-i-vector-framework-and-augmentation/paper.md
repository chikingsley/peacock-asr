---
title: "GoP2Vec: A few shot learning for pronunciation assessment with goodness of pronunciation (GoP) based representations from an i-vector framework and augmentation"
authors:
  - "Meenakshi Sirigiaju"
  - "Chiranjeevi Yarra"
citation_author: "Sirigiaju et al"
year: 2025
doi: "10.21437/Interspeech.2025-2359"
pages: "5063-5067"
source_pdf: "paper.pdf"
extraction_method: "Manual rewrite from the local paper.pdf; the existing local paper.md extraction was used only as a noisy cross-check."
extracted_at: "2026-03-07T20:03:41-08:00"
llm_friendly: true
---

# Title

GoP2Vec: A few shot learning for pronunciation assessment with goodness of pronunciation (GoP) based representations from an i-vector framework and augmentation

## Metadata

- Authors: Meenakshi Sirigiaju, Chiranjeevi Yarra
- Venue: Interspeech 2025
- Pages: 5063-5067
- DOI: 10.21437/Interspeech.2025-2359
- Task: few-shot utterance-level pronunciation assessment from GoP-derived fixed vectors

## TL;DR

The paper proposes turning variable-length phoneme-level GoP sequences into fixed-length `GoP2Vec` embeddings using an i-vector pipeline, after first lengthening each utterance with controlled speech augmentation. A small MLP then predicts pronunciation quality from these vectors.

With only a few labeled training samples, the method beats unsupervised GoP baselines and comes close to a supervised transformer baseline on SpeechOcean762. The best variant uses pitch-scale augmentation rather than time-scale augmentation.

## Abstract

The authors target pronunciation assessment in the low-label regime. Instead of training a large end-to-end model directly on learner audio, they extract phoneme-level GoP scores, make the sequences longer with pronunciation-preserving augmentation, compress them into fixed-dimensional i-vectors, and train a lightweight classifier. The claim is that this preserves useful pronunciation information while reducing the amount of labeled data needed for scoring.

## Research Question

Can a few-shot pronunciation-assessment system built from augmented GoP sequences and i-vector-style embeddings match or approach stronger supervised systems while using very little annotated training data?

## Method

- Compute phoneme-level GoP scores using a Kaldi ASR pipeline trained on LibriSpeech, following the cited GoP method from prior work.
- Lengthen short utterances by concatenating augmented versions of the same speech:
- time-scale modification (`TSM`, scales `0.8` to `1.2`)
- pitch-scale modification (`PSM`, `-20Hz` to `20Hz`)
- combined `TSM + PSM`
- Train a background GMM and total-variability matrix for each augmentation condition.
- Convert each augmented GoP sequence into a fixed-dimensional `GoP2Vec` i-vector.
- Train a simple MLP with two hidden layers to predict pronunciation score classes.

## Data

- `voisTUTOR`:
- 1,676 unique stimuli from 16 Indian L2 English learners
- 6 L1 backgrounds, balanced `8` male / `8` female
- expert utterance-level scores on a `0-4` scale
- about `14` hours total
- `SpeechOcean762`:
- 5,000 English utterances from 250 Mandarin L2 learners
- sentence-level total scores on a `0-10` scale, using the median of 5 expert ratings
- about `6` hours total
- Few-shot training setup:
- `voisTUTOR`: 150 training samples, remainder used for testing
- `SpeechOcean762`: 241 training samples from the train split, all 2,500 test utterances used for evaluation
- Note: the paper also states `voisTUTOR` has `12,535` samples in the experimental setup, which does not cleanly match the earlier description of `1,676` unique stimuli. The source of that larger count is not fully explained in the paper.

## Results

- Best augmentation choice is consistently `PSM`.
- `voisTUTOR` correlations:
- `TSM = 0.66`
- `PSM = 0.69`
- `TSM + PSM = 0.67`
- unsupervised baseline (`USV`) = `0.61`
- `SpeechOcean762` correlations:
- `TSM = 0.68`
- `PSM = 0.71`
- `TSM + PSM = 0.69`
- unsupervised baseline (`USV`) = `0.62`
- supervised baseline (`SV`) = `0.74`
- Longer utterances score better than single-word utterances.
- Reported cross-corpus trends are also positive:
- train on `voisTUTOR`, test on `SpeechOcean762`: from `0.65` on 1-word items to `0.77` on items longer than 7 words
- train on `SpeechOcean762`, test on `voisTUTOR`: from `0.63` on 1-word items to `0.75` on items longer than 7 words
- Best reported GoP2Vec hyperparameters:
- `voisTUTOR`: best around `k = 16` GMM components and `d = 5`
- `SpeechOcean762`: best around `k = 16` and `d = 8`

## Limitations / Notes

- The method still depends on a decent GoP extraction pipeline and forced alignment, so it is not truly annotation-free or ASR-free.
- The paper compares favorably to unsupervised baselines, but it remains slightly below the supervised baseline on SpeechOcean762 (`0.71` vs `0.74`).
- The training-sample accounting for `voisTUTOR` is not fully clear from the text.
- Because the model uses utterance-level ratings with different scales across datasets, some of the cross-corpus interpretation should be taken cautiously.

## Relevance To Peacock

This paper is useful if Peacock wants a low-label scoring pipeline that is lighter than a full end-to-end acoustic model. The core idea is to treat GoP sequences as variable-length evidence and compress them into stable fixed representations before a small supervised head.

For Peacock, the most reusable idea is not specifically i-vectors, but the strategy of combining weakly structured pronunciation features, augmentation, and a small scorer when labeled ratings are scarce. The main downside is that the whole stack still inherits the strengths and weaknesses of the underlying GoP extractor.
