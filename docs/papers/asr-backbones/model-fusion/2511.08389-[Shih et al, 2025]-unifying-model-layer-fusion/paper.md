---
arxiv: 2511.08389
title: "Unifying Model and Layer Fusion for Speech Foundation Models"
authors:
  - "Yi-Jen Shih"
  - "David Harwath"
citation_author: "Shih et al."
year: 2025
venue: "arXiv:2511.08389 (ASRU-2025 preprint style)"
source_pdf: "paper.pdf"
extraction_method: "Manual section-by-section rewrite from local LaTeX sources and pdf text on 2026-03-23."
extracted_at: "2026-03-23"
llm_friendly: true
---

# Unifying Model and Layer Fusion for Speech Foundation Models

## Metadata

- Title: Unifying Model and Layer Fusion for Speech Foundation Models
- Authors: Yi-Jen Shih, David Harwath
- Venue/ID: arXiv:2511.08389, IEEE 2025 ASR preprint style
- Task scope: ASR and non-ASR (speaker verification, emotion recognition) using speech foundation models (SFMs)

## Abstract

The paper proposes a single module that fuses both (1) multiple upstream models and (2) layers inside each upstream model at once. It is based on the “Interface” abstraction used in prior layer-fusion work, but now generalized to handle multiple upstreams with mixed dimensions and lengths. The authors run evaluations on LibriSpeech ASR and ML-SUPERB ASR variants plus SUPERB speaker verification and emotion recognition, and report consistent gains. They also test scaling by both model count and model size. Main claim: this unified module improves performance over separate layer/fusion approaches, especially when supervised and self-supervised models are fused together.

## 1. Introduction

Speech Foundation Models (SFMs) are usually pretrained with either self-supervised learning (SSL) or supervised learning (SL) and then fine-tuned on downstream tasks. Prior work has shown that:

- Combining multiple encoders can improve robustness and accuracy.
- Combining multiple internal layers (instead of using only the final layer) can improve downstream transfer.
- A learnable interface module generally works better than static weighted sums, especially in several ASR and non-ASR tasks.

The authors position this paper as a direct extension:

1) Prior fusion papers often optimize layer and model fusion separately.
2) Earlier evaluation was narrow in task coverage (often only LibriSpeech ASR).
3) Earlier methods were mostly SSL-only fusion in some cases.

The contributions they list are:

- A unified framework that optimizes layer and model fusion jointly.
- A broader evaluation that includes ASR and non-ASR tasks.
- Empirical evidence that SL+SSL fusion is generally stronger than SSL-only fusion when upstreams are complementary.

## 2. Related Work

The related work section groups prior efforts into several lines:

- Ensemble-like fusion of multiple encoders and weighted combinations.
- Layer-wise fusion methods for SSL models.
- Hierarchical or attentional layer fusion alternatives beyond weighted sum.
- Multi-stage and transfer-based methods that are avoided here because of extra cost (e.g., Opt/effuse-style distillation-like pipelines).
- Limitation of prior work that evaluated mostly ASR and did not systematically include speaker/emo paralinguistic tasks together with layer+model fusion.

## 3. Unifying Model and Layer Fusion

### 3.1 Background on Interfaces for SSL Models

The prior interface abstraction uses three modules:

1) Upstream U
2) Interface I
3) Downstream D

The mapping definitions are:

```text
U(·) : R^{T′} -> R^{L × T × D}
I(·) : R^{L × T × D} -> R^{T × D}
```

`T′` is input utterance length, `T` is frame length after downsampling, `L` is number of layers, and `D` is hidden size.  
Upstream is frozen and interface + downstream are trained in this setup.

Weighted sum is treated as a trainable baseline but is considered suboptimal due to “information collision” between layers, which motivates Hierarchical Convolution (HConv). The same paper that introduced HConv is cited as a stronger layer-fusion alternative.

### 3.2 Extending Interfaces to Model and Layer Fusion

The generalized setting accepts `N` upstream models:

```text
I(·) : R^{L1 × T1 × D1}, R^{L2 × T2 × D2}, ..., R^{LN × TN × DN} -> R^{T × D}
```

To combine heterogeneous models:

- first align layer counts `L` and frame counts `T` across models by upsampling and linear interpolation when needed.
- merge state tensors either by additive summation (HConv) or concatenation followed by projection (CHConv).

HConv uses addition in the layer- and temporal-aligned stack.
CHConv concatenates feature dimensions across models and projects back to a shared dimension.

## 4. Experimental Setup

### 4.1 Tasks

ASR tasks:

- SUPERB LibriSpeech ASR (WER)
- ML-SUPERB Mono-1h (13 languages, average CER)
- ML-SUPERB Multi-1h (143 languages, one hour each, average CER)

Non-ASR tasks:

- Speaker Verification (SV, EER)
- Emotion Recognition (ER, accuracy)

### 4.2 Upstream Models

Base models used:

- HuBERT Base
- WavLM Base+
- Data2Vec Base
- Whisper Small

All were set to 13 layers, 768 dimension in evaluation.

For SSL-only experiments, the paper evaluates all 2-model and 3-model combinations.
For SL+SSL cases, only best SSL candidates are fused with Whisper.

There are two Whisper versions; English Whisper Small is used for LibriSpeech and multilingual version for other tasks.

### 4.3 Baselines and Proposed Interfaces

Baselines:

- WS: weighted sum fusion
- GumD: dimension-wise Gumbel selection + projection baseline

GumD’s temporal interleaving is replaced with concatenation+projection in this study to match interface-time alignment and avoid downstream time-dimension inflation.

Proposed:

- HConv (addition-style fusion + hierarchical convolutions)
- CHConv (concatenation-style fusion + projection + HConv)

HConv is treated as the primary comparison point for fair parameter counts; CHConv is included mainly for 2-model settings to show the upper-bound effect of extra parameters.

## 5. Results and Discussion

### 5.1 Comparison with and without fusion

From Table I, fused configurations usually beat single-model counterparts for the same interface. The paper highlights two representative comparisons:

- Whisper Small + Data2Vec on SV: improvements over best single under WS/HConv/CHConv of `14.74%`, `12.36%`, and `8.35%`.
- Data2Vec + WavLM Base+ on SV: deltas are `-1.2%`, `-4.17%`, and `-9.49%` for WS/HConv/CHConv.

Inference from these: model-pair selection is often more important than interface choice.

### 5.2 Comparison between interfaces

Main trend from the paper:

- HConv and CHConv are generally strongest, especially for single models.
- WS is more stable than GumD, particularly on SV.
- CHConv gives additional gains over HConv for more demanding setups but with higher capacity/parameter load.
- On ML-SUPERB multilingual ASR, proposed methods can approximately match baselines, likely due to language mismatch with SSL pretraining conditions.

### 5.3 Fusion among SSL and SL

SL+SSL fusion (Whisper with SSL upstreams) is usually strongest.
The paper hypothesizes this comes from different objective families creating more heterogeneous representations, while SSL models are all closer to each other (all masked-prediction based).

An exception is LibriSpeech where Whisper single model can be weaker than SSL backbones in this setup due to training-domain overlap differences.

### 5.4 Scalability for number of upstream models

Moving from 2-model to 3-model fusion does not produce consistent gains in this study.

### 5.5 Scalability for large model fusion

Large model tests combine:

- WavLM Large
- Data2Vec Large
- Whisper Large/Medium

Observed trend: larger models widen HConv’s gain over WS on some tasks (e.g., Mono-1h and ER), with example gaps in the paper:

- Mono-1h: 3.61% gap WS vs HConv
- ER: 4.97% gap WS vs HConv

But this gap is not always maintained on LibriSpeech and SV.

## 6. Detailed Results for Different Languages on ML-SUPERB Mono-1h

The paper includes a per-language table for base and large models (Table III). Key extracted findings:

- Whisper is strongest on non-English languages in general but weaker than SSL baselines on English sets.
- HConv delivers more consistent improvements for SSL models across languages.
- Two-model fusion `WavLM + Whisper` is often strongest due to complementary English vs non-English strengths.
- Example from base models: `fra2` HConv WavLM+Whisper is `29.7` CER vs `33.0` (Whisper) and `42.8` (WavLM) with HConv; in that row HConv yields the largest drop.
- 3-model fusion adds little incremental gain over best 2-model pair.

### 6.1 Base Models

Selected model-level contrasts from the base-table slice:

| Model/Pair | Interface | Avg |
| --- | --- | --- |
| WavLM | WS | 32.01 |
| WavLM | HConv | 30.74 |
| Whisper | WS | 26.27 |
| Whisper | HConv | 26.23 |
| Data2Vec | WS | 35.77 |
| Data2Vec | HConv | 34.22 |
| WavLM + Data2Vec | WS | 32.02 |
| WavLM + Data2Vec | HConv | 30.62 |
| WavLM + Whisper | WS | 25.89 |
| WavLM + Whisper | HConv | 24.88 |

### 6.2 Large Models

| Model/Pair | Interface | Avg |
| --- | --- | --- |
| WavLM Large | WS | 29.45 |
| WavLM Large | HConv | 28.26 |
| Whisper Medium | WS | 24.78 |
| Whisper Medium | HConv | 23.36 |
| WavLM Large + Whisper Medium | WS | 26.48 |
| WavLM Large + Whisper Medium | HConv | 22.87 |

## 7. Conclusion

The paper concludes:

- unified fusion improves results across tasks with consistent trends,
- model-set choice is critical,
- SL+SSL combinations are often superior to SSL-only combinations,
- a practical limitation remains in inference cost as fusion size grows,
- promising extension is joint distillation to reduce runtime overhead.

## 8. Acknowledgements

- This work is supported by NSF grant `2238605`.
- Standard disclaimer is included that interpretations are those of the authors, not NSF.

## 9. Reproducibility and Result Tables

### Table I (from paper)

Evaluation results for interfaces across ASR and non-ASR tasks (`LS`, `Mono`, `Multi`, `SV`, `ER` metrics).

| Model | Interface | LS ↓ | Mono-1h ↓ | Multi-1h ↓ | SV ↓ | ER ↑ |
| --- | --- | --- | --- | --- | --- | --- |
| HuBERT | WS | 6.32 | 34.66 | 32.6 | 3.93 | 65.71 |
| HuBERT | HConv | 5.80 | 33.66 | 32.9 | 3.63 | 69.78 |
| WavLM | WS | 5.40 | 32.01 | 29.2 | 3.52 | 66.95 |
| WavLM | HConv | 4.78 | 30.74 | 27.6 | 2.79 | 70.76 |
| Data2Vec | WS | 4.83 | 35.77 | 34.2 | 4.87 | 65.60 |
| Data2Vec | HConv | 4.53 | 34.22 | 33.5 | 4.20 | 69.33 |
| Whisper | WS | 6.73 | 26.27 | 23.8 | 4.64 | 69.33 |
| Whisper | HConv | 6.40 | 26.23 | 21.7 | 3.30 | 70.36 |
| HuBERT + WavLM | WS | 5.42 | 31.89 | 29.4 | 3.59 | 68.29 |
| HuBERT + WavLM | GumD | 5.89 | 32.83 | 30.3 | 5.66 | 61.83 |
| HuBERT + WavLM | HConv | 4.86 | 30.65 | 31.8 | 2.74 | 70.49 |
| HuBERT + WavLM | CHConv | 4.82 | 30.40 | 29.8 | 2.88 | 70.55 |
| Data2Vec + WavLM | WS | 4.76 | 32.02 | 29.1 | 3.56 | 68.29 |
| Data2Vec + WavLM | GumD | 4.72 | 32.93 | 30.2 | 6.37 | 62.07 |
| Data2Vec + WavLM | HConv | 4.30 | 30.62 | 30.5 | 2.91 | 71.95 |
| Data2Vec + WavLM | CHConv | 4.39 | 30.82 | 31.1 | 3.06 | 70.93 |
| Data2Vec + HuBERT | WS | 4.87 | 33.80 | 32.1 | 4.01 | 66.98 |
| Data2Vec + HuBERT | GumD | 4.94 | 34.84 | 32.8 | 6.25 | 61.88 |
| Data2Vec + HuBERT | HConv | 4.60 | 32.29 | 33.3 | 3.92 | 70.80 |
| Data2Vec + HuBERT | CHConv | 4.47 | 32.40 | 32.5 | 3.72 | 69.94 |
| Whisper + Best SSL | WS | 4.77 | 25.89 | 22.9 | 3.96 | 70.58 |
| Whisper + Best SSL | GumD | 4.70 | 27.37 | 23.3 | 5.95 | 65.24 |
| Whisper + Best SSL | HConv | 4.93 | 24.88 | 21.1 | 2.90 | 71.29 |
| Whisper + Best SSL | CHConv | 4.52 | 23.54 | 20.2 | 3.03 | 74.86 |
| Data2Vec + HuBERT + WavLM | WS | 4.75 | 31.95 | 31.4 | 3.25 | 67.09 |
| Data2Vec + HuBERT + WavLM | GumD | 5.08 | 32.52 | 30.3 | 5.64 | 62.59 |
| Data2Vec + HuBERT + WavLM | HConv | 4.47 | 30.74 | 36.5 | 2.88 | 71.15 |
| Whisper + Best 2 SSLs | WS | 4.64 | 25.75 | 22.7 | 3.23 | 70.01 |
| Whisper + Best 2 SSLs | GumD | 4.80 | 27.54 | 23.4 | 5.52 | 63.71 |
| Whisper + Best 2 SSLs | HConv | 4.74 | 24.75 | 22.4 | 2.82 | 71.71 |

### Table II (large model scaling)

| Model/Fusion | Interface | LS ↓ | Mono-1h ↓ | SV ↓ | ER ↑ |
| --- | --- | --- | --- | --- | --- |
| Best SSL | WS | 3.22 | 29.45 | 2.80 | 68.67 |
| Best SSL | HConv | 3.10 | 28.26 | 2.21 | 72.49 |
| Whisper | WS | 5.59 | 24.78 | 3.76 | 71.48 |
| Whisper | HConv | 5.25 | 23.36 | 3.24 | 72.96 |
| Whisper + Best SSL | WS | 3.12 | 26.48 | 2.71 | 68.55 |
| Whisper + Best SSL | HConv | 3.55 | 22.87 | 2.19 | 73.52 |
