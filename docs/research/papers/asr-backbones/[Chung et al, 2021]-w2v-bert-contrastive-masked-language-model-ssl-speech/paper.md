---
arxiv: 2108.06209
title: "W2V-BERT: Combining Contrastive Learning and Masked Language Modeling for Self-Supervised Speech Pre-Training"
authors:
  - "Yu-An Chung"
  - "Yu Zhang"
  - "Wei Han"
  - "Chung-Cheng Chiu"
  - "James Qin"
  - "Ruoming Pang"
  - "Yonghui Wu"
citation_author: "Chung et al"
year: 2021
venue: "ASRU 2021"
source_pdf: "paper.pdf"
extraction_method: "Manual summary from arXiv PDF."
extracted_at: "2026-03-22"
llm_friendly: true
---

## Metadata

- Authors: Yu-An Chung (MIT CSAIL), Yu Zhang, Wei Han, Chung-Cheng Chiu, James Qin, Ruoming Pang, Yonghui Wu (Google Brain)
- arXiv: 2108.06209
- Venue: ASRU 2021
- Task: Self-supervised speech pre-training for ASR

## TL;DR

W2V-BERT joins two previously separate SSL objectives — wav2vec 2.0's contrastive discretization and BERT's masked language modeling — into a single end-to-end model. The contrastive module produces a codebook of discrete speech tokens; the MLM module uses those tokens as prediction targets. The critical finding: without the contrastive module, the codebook collapses immediately, making MLM impossible. With it, W2V-BERT XL (0.6B params) outperforms HuBERT Large (1.0B) on LibriSpeech. W2V-BERT 2.0, described in the Seamless papers (2308.11596, 2312.05187), scales this to 4.5M hours and powers SeamlessM4T.

## Abstract

Motivated by the success of masked language modeling (MLM) in NLP, w2v-BERT explores MLM for self-supervised speech representation learning. It combines contrastive learning (from wav2vec 2.0) with MLM (from BERT), where the former discretizes continuous speech into tokens and the latter learns contextualized representations by predicting those tokens. Both objectives are optimized simultaneously in an end-to-end fashion. Pre-trained on Libri-Light 60k hours, w2v-BERT XL achieves 5-10% relative WER reduction over HuBERT and wav2vec 2.0 on LibriSpeech test-clean/test-other without self-training or LM.

## Problem Statement

Prior SSL speech methods fall into two camps:

1. **Contrastive only** (wav2vec 2.0): trains the contrastive task using a separately obtained codebook; strong ASR but no MLM.
2. **MLM with clustering** (HuBERT): alternates between k-means clustering and masked prediction in two separate stages; strong but requires coordination between the two.

The question: can we combine both objectives end-to-end without the two-stage coordination overhead, and does MLM add anything beyond contrastive learning alone?

## Architecture

```text
Input speech
     │
     ▼
[Feature Encoder]           ← 2D conv subsampling (4× time reduction)
     │
     ├──────────────────────────────────────────┐
     │                                          │
     ▼                                          ▼
[Contrastive Module]                    (without masking)
 N × Conformer blocks                          │
     │ (with masking)                          ▼
     ├── Context vectors             [Quantizer (codebook)]
     │   (used for contrastive loss)      → discrete token IDs
     │                                         │
     └────────────────► target IDs ────────────┘
     │
     ▼
[MLM Module]
 M × Conformer blocks
     │
     ▼
 Softmax prediction of token ID at masked positions
```

- **Contrastive loss** L_c = L_w + α·L_d (diversity penalty, α=0.1 following wav2vec 2.0)
- **MLM loss** L_m = cross-entropy at masked positions
- **Combined**: L_p = β·L_c + γ·L_m (β=γ=1 in all experiments)

## Model Sizes (Table 1)

| Model | Params (B) | Contrastive Layers (N) | MLM Layers (M) | Model Dim | Attn Heads | Conv Kernel |
|-------|-----------|----------------------|----------------|-----------|------------|-------------|
| w2v-BERT XL | 0.6 | 12 | 12 | 1024 | 8 | 5 |
| w2v-BERT XXL | 1.0 | 12 | 30 | 1024 | 8 | 5 |

Codebook size: 1024. Code dim: 1024.

## Pre-training Setup

- Data: Libri-Light unlab-60k subset (~60,000 hours of English audiobooks)
- Input: 80-dim log-mel filterbank coefficients
- Masking: randomly mask with probability 0.065, extend for 10 time steps (same as wav2vec 2.0)
- Optimizer (XL): Adam, peak LR 2e-3, warmup 25k steps (transformer schedule)
- Optimizer (XXL): Adafactor, β1=0.9, β2=0.98
- Batch size: 2048

## Key Finding: Contrastive Module is Essential for MLM

Without the contrastive module, the codebook collapses immediately (diversity loss → 1.0 within the first 50k steps, MLM accuracy → 100% trivially). The model "cheats" by assigning all time steps to the same code, making MLM trivially solvable with no learned representation.

With the contrastive module, the diversity loss stabilizes and MLM accuracy converges to a meaningful level. The contrastive objective forces discriminative codebook entries, giving the MLM module useful prediction targets.

## Results (Table 2, LibriSpeech 960h supervised)

Pre-training Only (no self-training, no LM):

| Method | Unlabeled (hrs) | test-clean | test-other |
|--------|----------------|------------|------------|
| wav2vec 2.0 | 60k | 2.2 | 4.5 |
| HuBERT Large | 60k | — | — |
| w2v-Conformer XXL | 60k | 1.7 | 3.5 |
| **w2v-BERT XL (Ours)** | 60k | **1.5** | **2.9** |
| **w2v-BERT XXL (Ours)** | 60k | **1.4** | **2.5** |

Pre-training + Self-training + LM:

| Method | test-clean | test-other |
|--------|------------|------------|
| w2v-Conformer XXL+ | 1.5 | 2.7 |
| **w2v-BERT XL** | **1.5** | **2.6** |
| **w2v-BERT XXL** | **1.4** | **2.4** |

## Results on Voice Search (Table 4)

Evaluated on Google's internal Voice Search dataset (34.3k hours unlabeled domain data, 1k supervised hours fine-tuning):

| Method | Test WER |
|--------|----------|
| Conformer (baseline) | 10.7 |
| w2v-Conformer-XL | 10.8 |
| w2v-Conformer-XL-tuned | 8.9 |
| **w2v-BERT XL** | **6.2** |

w2v-BERT XL improves over the tuned contrastive baseline by 30% relatively on Voice Search — a harder, noisier domain than LibriSpeech.

## Ablation: Contrastive Module Capacity (Table 3)

Tested w2v-BERT with varying N (contrastive layers) keeping total at 24 layers:

| Config | N (contrastive) | M (MLM) | test-clean | test-other |
|--------|----------------|---------|------------|------------|
| C2 | 2 | 22 | 2.5 | 5.1 |
| C4 | 4 | 20 | 2.5 | 5.1 |
| C6 | 6 | 18 | 2.5 | 4.7 |
| C8 | 8 | 16 | 2.3 | 4.6 |
| C10 | 10 | 14 | 2.4 | 4.5 |
| **C12 (w2v-BERT XL)** | **12** | **12** | **2.4** | **4.4** |
| C24 | 24 | 0 | 2.4 | 4.9 |

Sweet spot at C8–C12: too little contrastive → MLM underpowered; too much contrastive → MLM module too small to learn useful representations. C24 (contrastive only, same as wav2vec 2.0) is worse than any C8+ split.

## Relation to Other Methods

```text
wav2vec 2.0      ──► contrastive only (quantizer jointly trained)
HuBERT           ──► MLM only (k-means labels, 2-stage alternation)
vq-wav2vec       ──► contrastive → quantizer → downstream MLM (2-stage, frozen quantizer)
DiscreteBERT     ──► same 2-stage problem
w2v-Conformer    ──► wav2vec 2.0 contrastive but with conformer layers, no quantizer
w2v-BERT         ──► contrastive + MLM simultaneously, end-to-end, conformer layers
```

## W2V-BERT 2.0

W2V-BERT 2.0 is not described in this paper — it is described in the SeamlessM4T (2308.11596) and Seamless (2312.05187) papers. Key differences:

- Pre-trained on 4.5M hours (vs 60k here)
- Used as the speech encoder in SeamlessM4T v1 and v2
- Enables 100-language ASR and speech translation

## Relevance to Peacock

Indirect but important. W2V-BERT is the architectural precursor to W2V-BERT 2.0, which powers SeamlessM4T. Understanding the contrastive + MLM interaction is useful for interpreting why W2V-BERT 2.0 features are more multilingual and robust than a pure contrastive model. If we ever consider adding a SeamlessM4T SSL encoder to the CHConv pool alongside wav2vec2/HuBERT/WavLM, knowing W2V-BERT's design helps assess what layer to extract from and what information each layer encodes.
