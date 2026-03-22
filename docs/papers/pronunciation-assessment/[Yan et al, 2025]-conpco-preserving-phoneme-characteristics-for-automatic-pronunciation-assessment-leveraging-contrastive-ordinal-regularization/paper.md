---
title: "ConPCO: Preserving Phoneme Characteristics For Automatic Pronunciation Assessment Leveraging Contrastive Ordinal Regularization"
authors:
  - "Bi-Cheng Yan"
  - "Yi-Cheng Wang"
  - "Jiun-Ting Li"
  - "Meng-Shin Lin"
  - "Hsin-Wei Wang"
  - "Wei-Cheng Chao"
  - "Berlin Chen"
citation_author: "Yan et al."
year: 2025
doi: "10.1109/ICASSP49660.2025.10890778"
pages: 5
source_pdf: "paper.pdf"
extraction_method: "manual-curated from local paper.pdf and nearby repository markdown"
extracted_at: "2026-03-07"
llm_friendly: true
---

# ConPCO: Preserving Phoneme Characteristics For Automatic Pronunciation Assessment Leveraging Contrastive Ordinal Regularization

## Metadata

- Authors: Bi-Cheng Yan, Yi-Cheng Wang, Jiun-Ting Li, Meng-Shin Lin, Hsin-Wei Wang, Wei-Cheng Chao, Berlin Chen
- Year: 2025
- DOI: 10.1109/ICASSP49660.2025.10890778
- Pages: 5
- Source PDF: `paper.pdf`
- Venue clue from PDF metadata: ICASSP 2025

## TL;DR

- Adds a phoneme-aware regularizer to multi-granular pronunciation scoring, instead of treating APA as plain regression.
- The regularizer jointly enforces speech-text phoneme alignment, inter-phoneme separation, and score-aware compactness.
- On Speechocean762, `HierCB + ConPCO` achieves the paper’s best reported phone-level MSE/PCC (0.071 / 0.701) and strong gains at word and utterance levels.

## Abstract

The paper targets a weakness in regression-based APA: models can fit human scores without learning features that cleanly preserve phoneme identity. ConPCO addresses this by aligning phoneme embeddings from speech with phoneme-text embeddings, then regularizing the feature space so phoneme categories stay distinct while score ordinality is reflected in within-category geometry. The method is evaluated in a hierarchical APA model and is reported to outperform competitive baselines on Speechocean762.

## Research Question

How can an APA model preserve phoneme-specific structure in its latent space while still predicting ordinal pronunciation scores across phone, word, and utterance levels?

## Method

- Base model: `HierCB`, a hierarchical APA architecture with convolution-augmented Branchformer blocks.
- Feature inputs: GOP-style features, duration statistics, energy statistics, and SSL features from Wav2Vec 2.0, WavLM, and HuBERT.
- ConPCO regularizer:
- `L_con`: contrastive alignment between phoneme encoder outputs and phoneme-text embeddings
- `L_pc`: increases separation among phoneme categories
- `L_o`: score-weighted compactness term that reflects ordinal phone-level accuracy labels
- Training objective combines the multi-granular MSE loss with the ConPCO regularizer.

## Data

- Dataset: `Speechocean762`
- 5,000 English recordings spoken by 250 Mandarin L2 learners
- Split: 2,500 train / 2,500 test utterances
- Score coverage:
- Phone-level accuracy: 47,076 train / 47,369 test
- Word-level accuracy/stress/total: 15,849 train / 15,967 test
- Utterance-level accuracy/completeness/fluency/prosody/total: 2,500 train / 2,500 test
- Scenario: read-aloud APA

## Results

- `HierCB` alone: phone MSE 0.076, PCC 0.680; word-total PCC 0.645; utterance-total PCC 0.796
- `HierCB + ConPCO`: phone MSE 0.071, PCC 0.701; word accuracy 0.669, stress 0.437, total 0.682; utterance accuracy 0.780, completeness 0.749, fluency 0.830, prosody 0.823, total 0.803
- Relative to `HierCB`, ConPCO gives:
- Better phone accuracy (+0.021 PCC, lower MSE)
- Stronger word-level stress (+0.082 PCC)
- Better utterance completeness (+0.072 PCC)
- The paper also reports that `HierCB` outperforms earlier baselines such as GOPT-SSL, 3M, and HiPAMA, and that ConPCO further improves on top of `HierCB`.

## Limitations / Notes

- Read-aloud only; no direct evaluation on spontaneous or open-response speech.
- The paper explicitly notes limited explainability of the assessment outputs.
- The gains are strongest for pronunciation-quality dimensions tied to phoneme structure; that is useful, but still benchmark-bound to a single dataset and learner population.

## Relevance To Peacock

- Useful for any Peacock pipeline that needs phoneme-aware supervision rather than only holistic regression targets.
- The contrastive speech-text phoneme alignment is directly applicable to canonical-prompt settings.
- Less directly applicable to free speech, but the representation-learning idea could transfer to forced-alignment or phoneme-diagnostic tasks.
