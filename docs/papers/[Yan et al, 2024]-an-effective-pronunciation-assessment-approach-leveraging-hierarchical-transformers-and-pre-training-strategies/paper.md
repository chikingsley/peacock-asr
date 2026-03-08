---
title: "An Effective Pronunciation Assessment Approach Leveraging Hierarchical Transformers and Pre-training Strategies"
authors:
  - "Bi-Cheng Yan"
  - "Jiun-Ting Li"
  - "Yi-Cheng Wang"
  - "Hsin-Wei Wang"
  - "Tien-Hong Lo"
  - "Yung-Chang Hsu"
  - "Wei-Cheng Chao"
  - "Berlin Chen"
citation_author: "Yan"
year: 2024
doi: "10.18653/v1/2024.acl-long.95"
pages: "1737-1747"
source_pdf: "paper.pdf"
extraction_method: "Manually curated from the local PDF, using nearby OCR text only to recover tables and ablation results in a clean, structured form."
extracted_at: "2026-03-07T20:02:23-08:00"
llm_friendly: true
---

# An Effective Pronunciation Assessment Approach Leveraging Hierarchical Transformers and Pre-training Strategies

## Metadata
- Authors: Bi-Cheng Yan, Jiun-Ting Li, Yi-Cheng Wang, Hsin-Wei Wang, Tien-Hong Lo, Yung-Chang Hsu, Wei-Cheng Chao, Berlin Chen
- Year: 2024
- Venue: ACL 2024
- DOI: 10.18653/v1/2024.acl-long.95
- Pages: 1737-1747
- Task: Multi-aspect, multi-granular automatic pronunciation assessment

## TL;DR
This paper proposes HierTFR, a hierarchical Transformer APA model that explicitly models phone-, word-, and utterance-level structure, adds aspect-attention and selective fusion across levels, and introduces both a correlation-aware regularizer and level-specific pretraining.

On Speechocean762, HierTFR is the strongest model in the readable comparison table. The ablations show that the custom pretraining contributes a large share of the gain.

## Abstract
The paper argues that flat parallel APA models underuse linguistic hierarchy and aspect relatedness. HierTFR addresses this with hierarchical phone-to-word-to-utterance modeling, explicit aspect interaction modules, a regularizer that encourages predicted aspect correlations to resemble human annotation correlations, and pretraining objectives tailored to different linguistic levels.

## Research Question
Can a hierarchical Transformer APA model with relation-aware optimization and custom pretraining outperform flat multi-task baselines and prior hierarchical systems?

## Method
- Representation flow:
- Phone-level contextual modeling over pronunciation features and prompt-aligned text embeddings.
- Word-level modeling built from phone representations plus word-level attention.
- Utterance-level pooling over phone, word, and utterance streams.
- Aspect-related modules:
- Aspect attention for word and utterance scores.
- Selective fusion gates to combine phone-, word-, and utterance-level evidence when predicting utterance-level aspects.
- Losses:
- Standard multi-granular APA MSE loss.
- Correlation-aware regularization that matches the correlation matrix of predicted aspect scores to the correlation matrix of target labels.
- Pretraining:
- Mask-predict objectives at phone and word levels.
- Pairwise utterance ranking objective for utterance-level accuracy representations.

## Data
- Dataset: Speechocean762.
- Contents: 5,000 recordings from 250 Mandarin L2 English learners.
- Split: 2,500 training and 2,500 test utterances.
- Labels: five expert annotations per item, with the median score used.
- Scores are normalized to the 0-2 scale used by prior APA work.
- Features include the same pronunciation-oriented feature family used by earlier GOPT-style systems; the appendix also spells out GOP, energy, and duration feature extraction.

## Results
- Main comparison on Speechocean762:
- HierTFR phone level: MSE 0.081, PCC 0.644.
- HierTFR word level PCC: accuracy 0.622, stress 0.325, total 0.634.
- HierTFR utterance level PCC: accuracy 0.735, completeness 0.513, fluency 0.801, prosody 0.795, total 0.764.
- Strong baseline comparison:
- GOPT phone PCC 0.612 vs HierTFR 0.644.
- GOPT utterance total 0.742 vs HierTFR 0.764.
- HiPAMA utterance total 0.754 vs HierTFR 0.764.
- The completeness gain is especially large:
- GOPT 0.155.
- HiPAMA 0.276.
- HierTFR 0.513.
- Ablations:
- Without correlation-aware loss: most metrics dip slightly; completeness rises a bit, but overall performance is lower.
- Without pretraining: phone PCC falls from 0.644 to 0.621, word accuracy from 0.622 to 0.545, and utterance total from 0.764 to 0.739.
- Without selective fusion: utterance-level metrics drop notably.
- Without aspect attention: word-level metrics drop the most.

## Limitations / Notes
- The strongest evidence is on one benchmark only.
- Stress remains much weaker than most other aspects, even in the best model.
- The model is not ASR-free; it still relies on prompt-aligned pronunciation features rather than pure end-to-end raw-audio modeling.
- The correlation-aware loss is helpful overall, but the ablation table shows the tradeoff is not uniform for every aspect.

## Relevance To Peacock
- This is one of the more directly reusable APA papers in the batch because it combines hierarchical structure, cross-aspect interaction, and explicit ablation evidence.
- The pretraining results are especially relevant if Peacock wants stronger supervision efficiency from limited annotated pronunciation data.
- The selective fusion idea is a practical pattern for merging phone-, word-, and utterance-level evidence in a single scorer.
