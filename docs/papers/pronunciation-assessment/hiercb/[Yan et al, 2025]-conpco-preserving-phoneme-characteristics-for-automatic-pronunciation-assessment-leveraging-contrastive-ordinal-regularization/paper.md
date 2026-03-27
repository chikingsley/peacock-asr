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
extraction_method: "pdftotext section-by-section reconstruction"
extracted_at: "2026-03-23"
llm_friendly: true
---

# ConPCO: Preserving Phoneme Characteristics for Automatic Pronunciation Assessment Leveraging Contrastive Ordinal Regularization

## Metadata

- Authors: Bi-Cheng Yan, Yi-Cheng Wang, Jiun-Ting Li, Meng-Shin Lin, Hsin-Wei Wang, Wei-Cheng Chao, Berlin Chen
- Year: 2025
- DOI: 10.1109/ICASSP49660.2025.10890778
- Pages: 5
- Venue: IEEE ICASSP 2025 (manuscript in PDF)
- Source PDF: `paper.pdf`
- Keywords: computer-assisted language learning, automatic pronunciation assessment, contrastive learning

## Abstract

Existing APA systems are largely trained with regression losses and can underutilize phoneme-level structure.

ConPCO is introduced as a regularization framework for regression-based APA.
- aligns phoneme representations from the APA encoder with phoneme-level text prompt embeddings
- increases separability across phoneme categories
- enforces ordinal behavior using pronunciation-score proximity in the feature space

It is evaluated in a hierarchical model called `HierCB` and tested on SpeechOcean762 with stronger phone/word/utterance outcomes than baselines.

## I. Introduction

The paper frames CAPT as read-aloud pronunciation feedback in which learners are given a text prompt and pronunciation is evaluated against it.

Two practical observations motivate the method:

- Regression-only training often optimizes score prediction while weakening explicit phoneme structure.
- In standard APA setups, phoneme audio features and text-prompt phoneme embeddings live in separate spaces.
- Intra-class collapse can happen, where same-score phonemes collapse without preserving phoneme identity.
- Ordinal ordering of scores (e.g., phone accuracy labels are ordered) is not directly reflected in latent geometry.

ConPCO is proposed to correct all three.

## II. Methodology

### A. Contrastive Phonemic Ordinal Regularizer (ConPCO)

ConPCO has three additive terms:
- contrastive term `L_con`
- phonemic characteristic term `L_pc`
- ordinal term `L_o`

#### 1) Contrastive term

Let:
- `H_p = (h_1^p, ..., h_N^p)` be phoneme representations from the APA model
- `E_p = (e_1^p, ..., e_N^p)` be phoneme-text prompt embeddings

After projections and per-class centroiding, a pair set is formed:
- `M = {(z_i^p, z_i^t), i = 1...M}`

The contrastive objective is:

- `L_con = L_p2t + L_t2p`

It maximizes similarity for paired items and minimizes similarity for non-paired items with temperature `τ`.

#### 2) Phonemic characteristic term

This term manages inter-phoneme structure by encouraging meaningful distances between different phoneme centroids:

- `L_pc = - (1 / [M(M-1)]) * sum_{i!=j} || z_i^p - z_j^p ||_2`

It operationally increases phoneme category separation.

#### 3) Ordinal term

This term preserves score order (ordinality) in representation space:

- `L_o = (1/N) * sum_i w_i * || h_i^p - z_{c_i}^p ||_2`
- `w_i = |C - y_i^p|`
- `C = 3` (set to highest accuracy score + margin)

Larger mismatch from target score gets higher penalty, tightening score-aware compactness.

### B. ConPCO total objective

- `L_ConPCO = λ_con * L_con + λ_pc * L_pc + λ_o * L_o`

During training ConPCO is added to the multi-granularity multi-aspect pronunciation objective.

### C. Hierarchical APA model: HierCB

To validate ConPCO, authors build `HierCB`, a hierarchical architecture with three stages:
- phoneme-level
- word-level
- utterance-level

All stages use convolution-augmented Branchformer encoders.

#### Feature extraction

Audio features (concatenate then project):
- `X_p = Linear_p([E_GOP; E_Dur; E_Eng; E_SSL])`
- GOP features: likelihood/probability-based pronunciation proxies
- `E_Dur`: duration statistics
- `E_Eng`: energy statistics
- `E_SSL`: SSL features from Wav2Vec2.0, WavLM, HuBERT

#### Phoneme level

- Add phoneme-level textual embedding `E_p` to `X_p`
- `H_p^0 = X_p + E_p`
- `H_p = PhnEnc(H_p^0)` with 3 convolution-augmented Branchformer blocks
- Regress phoneme accuracy from `H_p`

#### Word level

- Attention pooling over phonemes to words
- Apply word embedding `E_w`
- Word encoder: 2 convolution-augmented Branchformer blocks
- Separate depthwise branches for word-level accuracy/stress/total scores

#### Utterance level

- Merge word-level aspect representations by weighted average
- Combine with projection of depthwise-convolved phoneme/audio/word features
- Single utterance encoder (1 convolution-augmented Branchformer)
- 5 aspect regressors for utterance (accuracy/fluency/completeness/prosody/total)

## III. Experiments

### A. Experimental setup

Dataset: `Speechocean762`

- 5,000 utterances from 250 Mandarin L2 speakers
- Split: 2,500 train / 2,500 test

Pronunciation labels (counts):
- Phoneme accuracy: 47,076 train / 47,369 test
- Word-level accuracy/stress/total (each): 15,849 / 15,967
- Utterance-level accuracy/completeness/fluency/prosody/total (each): 2,500 / 2,500

Features:
- GOP + duration + energy + SSL (`E_SSL` from last layer of Wav2vec2.0/WavLM/HuBERT)
- Frame-level SSL pooled to phoneme level via forced alignments

Optimization:
- 5 independent trials
- 100 epochs each
- Adam, initial LR `1e-3`, batch size `25`
- LR decay by `0.1` when validation loss stalls 10 epochs
- Metric: PCC for levels/aspects; MSE for phoneme accuracy
- code repo: `https://github.com/bicheng1225/ConPCO`

### B. Qualitative analysis

The paper includes t-SNE visualizations showing:
- plain hierarchical model: less aligned phoneme-text features
- +ConPCO: better alignment between text and acoustic phoneme representations
- improved phoneme category structure after contrastive + regularizer terms

### C. Pronunciation assessment results

#### Table II (phoneme/word)
- HierCB outperforms baselines in phone/word levels.
- `+ConPCO` gives further improvements over `HierCB` and `+PCO`.
- Reported highlights:
  - Phoneme MSE/PCC: `0.071 / 0.701` for `HierCB + ConPCO`
  - Word-level stress is notably improved with ConPCO (`0.437`).

#### Table III (utterance level)
- HierCB competitive with strong baselines and further improved by ConPCO.
- `HierCB + ConPCO` utterance total PCC reaches `0.803` and gains over both `3M` and `GOPT-SSL` in several metrics.

Model family comparisons include:
- GOP-only baselines: LSTM/GOPT/GFR/HiPAMA
- SSL-enhanced baseline: GOPT-SSL, 3M
- hierarchical baselines: HierBFR/HierCB/HiPAMA
- phone-level regularizer comparisons: `PCO` vs `ConPCO`

### D. Reported takeaway

- ConPCO is stronger than plain MSE-style training in preserving phoneme structure.
- ConPCO is especially useful for pronunciation clarity-related outcomes and utterance total gains.

## IV. Conclusion

ConPCO is introduced as the first contrastive-based approach in this context for APA:
- preserve phoneme characteristics,
- retain ordinal target geometry,
- work inside multi-level hierarchical pronunciation models.

SpeechOcean762 experiments support practical gains across phoneme, word, and utterance metrics.

## Limitations and Future Work

- Read-aloud only; not yet validated for open-response speech.
- Explainability remains limited.
- The authors explicitly point to spontaneous speech and explainable feedback as next steps.

## Acknowledgements

- Supported by E.SUN bank, grant number `202408-NTU-02`.
- Standard caveat: findings do not necessarily reflect sponsor views.

## Relevance

ConPCO is the direct precursor to the ConPCO-like module used in later MuFFIN work:
- it is lightweight at the objective level,
- compatible with hierarchical APA stacks,
- and directly targets the phoneme structure issue that weakens multilingual pronunciation transfer in regression-only pipelines.

