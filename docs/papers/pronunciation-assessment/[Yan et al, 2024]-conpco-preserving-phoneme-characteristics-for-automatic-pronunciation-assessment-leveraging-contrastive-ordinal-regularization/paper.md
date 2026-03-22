---
title: "ConPCO: Preserving Phoneme Characteristics for Automatic Pronunciation Assessment Leveraging Contrastive Ordinal Regularization"
authors:
  - "Bi-Cheng Yan"
  - "Wei-Cheng Chao"
  - "Jiun-Ting Li"
  - "Yi-Cheng Wang"
  - "Hsin-Wei Wang"
  - "Meng-Shin Lin"
  - "Berlin Chen"
citation_author: "Yan et al."
year: 2024
doi: null
pages: 5
source_pdf: "paper.pdf"
extraction_method: "manual-curated from local paper.pdf and nearby repository markdown"
extracted_at: "2026-03-07"
llm_friendly: true
---

# ConPCO: Preserving Phoneme Characteristics for Automatic Pronunciation Assessment Leveraging Contrastive Ordinal Regularization

## Metadata

- Authors: Bi-Cheng Yan, Wei-Cheng Chao, Jiun-Ting Li, Yi-Cheng Wang, Hsin-Wei Wang, Meng-Shin Lin, Berlin Chen
- Year: 2024
- DOI: Not found in the local PDF
- Pages: 5
- Source PDF: `paper.pdf`
- Note: This local 2024 file appears to be an earlier/precursor version of the 2025 ICASSP paper with the same core idea.

## TL;DR

- Proposes ConPCO, a phoneme-aware regularizer for regression-based automatic pronunciation assessment (APA).
- Combines contrastive alignment between speech-derived phoneme embeddings and phonetic-text embeddings with an ordinal compactness term.
- On Speechocean762, the best reported system (`HierCB + ConPCO`) reaches phone-level MSE 0.071 / PCC 0.701, word-total PCC 0.682, and utterance-total PCC 0.803.

## Abstract

The paper argues that standard regression losses for APA do not explicitly preserve phoneme identity in the learned feature space. To address this, it introduces ConPCO, which aligns phoneme representations from speech with phonetic text embeddings and regularizes them so that phoneme categories stay distinct while ordinal score structure is retained. The method is evaluated with a hierarchical APA model on the Speechocean762 benchmark and is reported to outperform several prior APA baselines.

## Research Question

Can a regression-based APA model be trained so that its internal representations remain phoneme-discriminative while still respecting the ordinal nature of pronunciation scores?

## Method

- ConPCO has three pieces:
- A contrastive term aligns speech-derived phoneme representations with phoneme-text embeddings in a shared space.
- A phonemic-characteristic term pushes different phoneme categories farther apart.
- An ordinal term pulls samples toward their phoneme centroid with strength tied to the phone-level score.
- The regularizer is tested with `HierCB`, a hierarchical APA architecture with convolution-augmented Branchformer blocks over phoneme, word, and utterance levels.
- Input features include GOP-style features, duration and energy statistics, and SSL acoustic features.

## Data

- Dataset: `Speechocean762`
- 5,000 English recordings from 250 Mandarin L2 learners
- Split: 2,500 train / 2,500 test utterances
- Reported label counts:
- Phone-level accuracy: 47,076 train / 47,369 test
- Word-level scores: 15,849 train / 15,967 test
- Utterance-level scores: 2,500 train / 2,500 test
- Scenario: read-aloud pronunciation assessment

## Results

- Best reported model: `HierCB + ConPCO`
- Phone-level: MSE 0.071, PCC 0.701
- Word-level: accuracy 0.669, stress 0.437, total 0.682
- Utterance-level: accuracy 0.780, completeness 0.749, fluency 0.830, prosody 0.823, total 0.803
- The paper states that `HierCB` improves over earlier APA baselines, and adding ConPCO further improves phone accuracy and word-level scoring.
- The authors also report qualitative t-SNE evidence that ConPCO better aligns speech and text phoneme embeddings and clusters them by phoneme category.

## Limitations / Notes

- Limited to read-aloud APA; not evaluated on open-response or spontaneous speech.
- The paper notes limited explainability of the produced scores.
- The local PDF does not expose a DOI or clear venue metadata.
- The 2024 local version substantially overlaps with the 2025 ICASSP version; where the 2024 PDF text extraction was noisier, I relied on clearly recoverable table values and the paper’s written claims.

## Relevance To Peacock

- Directly relevant if Peacock needs phoneme-aware scoring features instead of only utterance-level regression.
- The contrastive alignment idea could help tie acoustic units to canonical phoneme targets during dataset building or supervision design.
- The paper is especially useful for multi-granular pronunciation feedback, but less useful for spontaneous speech because it assumes a read-aloud setup.
