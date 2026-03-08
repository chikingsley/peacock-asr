---
title: "MuFFIN: Multifaceted Pronunciation Feedback Model with Interactive Hierarchical Neural Modeling"
authors:
  - "Bi-Cheng Yan"
  - "Ming-Kang Tsai"
  - "Berlin Chen"
citation_author: "Yan et al."
year: 2025
doi: null
pages: 16
source_pdf: "paper.pdf"
extraction_method: "manual-curated from local paper.pdf and nearby repository markdown"
extracted_at: "2026-03-07"
llm_friendly: true
---

# MuFFIN: Multifaceted Pronunciation Feedback Model with Interactive Hierarchical Neural Modeling

## Metadata

- Authors: Bi-Cheng Yan, Ming-Kang Tsai, Berlin Chen
- Year: 2025
- DOI: Not found in the local PDF
- Pages: 16
- Source PDF: `paper.pdf`
- Note: The PDF looks like a pre-publication IEEE manuscript draft; it still contains a manuscript-ID placeholder line.

## TL;DR

- MuFFIN jointly handles automatic pronunciation assessment (APA) and mispronunciation detection/diagnosis (MDD) in one model instead of treating them as separate systems.
- It combines a hierarchical neural architecture, ConPCO-style phoneme regularization, and a phoneme-specific variation (PhnVar) strategy for MDD imbalance.
- On Speechocean762 it reports strong APA results and improves MDD from F1 65.99 to 67.98 and PER from 2.36 to 2.33 when PhnVar is added.

## Abstract

The paper frames CAPT feedback as two connected problems: scoring pronunciation quality and identifying phone-level pronunciation errors. MuFFIN unifies them in a shared hierarchical model so that phoneme-discriminative representations learned for one task can help the other. To strengthen the phoneme space, the paper adds a contrastive phonemic ordinal regularizer; to reduce MDD imbalance, it perturbs phoneme logits using phoneme-specific variation that depends on both class frequency and pronunciation difficulty.

## Research Question

Can a single hierarchical model jointly solve multi-granular pronunciation scoring and phone-level mispronunciation detection/diagnosis better than separate-task systems, and how should it handle severe class imbalance in MDD?

## Method

- MuFFIN uses a hierarchical architecture over phoneme, word, and utterance levels with convolution-augmented Branchformer blocks.
- It jointly optimizes:
- APA losses across multiple granularities and aspects
- MDD detection loss
- MDD diagnosis loss
- It adds a ConPCO-style regularizer to align speech phoneme representations with phoneme-text embeddings while preserving score ordinality.
- It adds `PhnVar`, which perturbs diagnosis logits with Gaussian noise whose scale depends on:
- a quantity factor for class frequency
- a pronunciation-difficulty factor based on mispronunciation rates

## Data

- Dataset: `Speechocean762`
- 5,000 English recordings from 250 Mandarin L2 learners
- Split: 2,500 train / 2,500 test utterances
- APA labels:
- Phone-level accuracy: 47,076 train / 47,369 test
- Word-level scores: 15,849 train / 15,967 test
- Utterance-level scores: 2,500 train / 2,500 test
- MDD labels:
- Correct pronunciations: 45,088 train / 45,959 test
- Deletions: 450 / 396
- Substitutions: 914 / 593
- Non-categorical errors: 488 / 332
- Accented errors: 136 / 89
- Scenario: read-aloud CAPT

## Results

- APA results for `MuFFIN`:
- Phone-level: MSE 0.063, PCC 0.742
- Word-level: accuracy 0.705, stress 0.315, total 0.714
- Utterance-level: accuracy 0.807, completeness 0.768, fluency 0.841, prosody 0.832, total 0.830
- Compared with the strong APA baseline `3MH`, MuFFIN improves phone PCC from 0.693 to 0.742 and utterance-total PCC from 0.811 to 0.830, but it is weaker on word-level stress (0.315 vs. 0.361).
- MDD results:
- `MuFFIN`: recall 64.33, precision 66.89, F1 65.99, PER 2.36
- `MuFFIN + PhnVar`: recall 68.37, precision 67.60, F1 67.98, PER 2.33
- The paper reports the MDD detection gain from PhnVar as statistically significant (`p < 0.001`).
- Compared baselines on MDD are much weaker overall, e.g. `JAM` at F1 45.01 / PER 2.81 and `Ryu2023` at F1 41.50 / PER 9.93.

## Limitations / Notes

- Limited to read-aloud learning scenarios with known canonical text.
- Accent coverage is narrow because the dataset is Mandarin-only.
- The paper notes that explainability remains limited because the model mainly learns to mimic expert annotations.
- The local PDF is clearly not the final typeset version, so missing DOI/publication metadata is likely due to draft status rather than the work being unpublished.

## Relevance To Peacock

- Strong fit for Peacock if the goal is to combine scoring and phone-level error analysis in one training setup.
- The joint APA+MDD framing is useful for datasets where error labels and proficiency labels coexist.
- PhnVar is especially relevant if Peacock has long-tail phoneme or error distributions and wants a simple imbalance-handling idea that does more than class-frequency correction alone.
