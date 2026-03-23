---
arxiv: 2106.07447
title: "HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units"
authors:
  - "Wei-Ning Hsu"
  - "Benjamin Bolte"
  - "Yao-Hung Hubert Tsai"
  - "Kushal Lakhotia"
  - "Ruslan Salakhutdinov"
  - "Abdelrahman Mohamed"
citation_author: "Hsu et al"
year: 2021
venue: "IEEE/ACM TASLP 2021"
doi: "10.1109/TASLP.2021.3122291"
pages: "3451-3460"
source_pdf: "paper.pdf"
extraction_method: "Manually summarized from the published PDF; no local LaTeX source was added in this pass."
extracted_at: "2026-03-22"
llm_friendly: true
---

# HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units

## Metadata

- Authors: Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai, Kushal Lakhotia, Ruslan Salakhutdinov, Abdelrahman Mohamed
- Venue: IEEE/ACM Transactions on Audio, Speech, and Language Processing 2021
- DOI: 10.1109/TASLP.2021.3122291
- arXiv: 2106.07447
- Pages: 3451-3460
- Task: self-supervised speech representation learning with masked hidden-unit prediction

## TL;DR

HuBERT replaces wav2vec 2.0's contrastive quantized-target setup with masked prediction of offline cluster labels. The key idea is that the pseudo-labels only need to be consistent, not intrinsically perfect. In practice that simpler masked-prediction objective becomes one of the core speech SSL backbones used throughout later pronunciation, probing, and fusion work.

## Abstract

The paper frames speech SSL as a masked prediction problem over hidden units obtained from offline clustering. A BERT-style model predicts cluster assignments only on masked regions, forcing the network to model both local acoustics and broader sequence context. The authors argue that consistency of the cluster labels matters more than their absolute quality, and they use iterative clustering to improve the targets over multiple rounds. The model matches or exceeds wav2vec 2.0 performance across LibriSpeech and Libri-light fine-tuning settings.

## Method

- Extract speech features and generate pseudo-labels with offline clustering.
- Mask regions of the continuous speech sequence.
- Predict hidden-unit cluster IDs only on masked regions.
- Use iterative re-clustering to refine teacher labels across rounds.
- Scale the model up to a `1B` parameter configuration for stronger transfer.

## Results

- The paper reports that HuBERT matches or improves upon wav2vec 2.0 across `10 min`, `1 h`, `10 h`, `100 h`, and `960 h` fine-tuning subsets on LibriSpeech / Libri-light benchmarks.
- With the `1B` parameter model, it reports up to `19%` and `13%` relative WER reduction on `dev-other` and `test-other`.
- The paper's main empirical claim is that masked hidden-unit prediction is competitive even when the clustering labels are noisy, provided they are consistent enough.

## Relevance To Peacock

HuBERT is the other canonical SSL backbone the vault needed. A large share of the pronunciation-assessment papers in this repo either use HuBERT directly, compare against it, or probe its intermediate layers. Missing the source paper made the `Kim`, `Shih`, `Chiu`, and `ConPCO` threads harder to read coherently.
