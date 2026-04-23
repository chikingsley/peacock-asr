---
arxiv: 2508.08962
title: "Selection of Layers from Self-supervised Learning Models for Predicting Mean-Opinion-Score of Speech"
authors:
  - "Xinyu Liang"
  - "Fredrik Cumlin"
  - "Victor Ungureanu"
  - "Chandan K. A. Reddy"
  - "Christian Schuldt"
  - "Saikat Chatterjee"
citation_author: "Liang et al."
year: 2025
venue: "arXiv:2508.08962"
source_pdf: "paper.pdf"
extraction_method: "Manual section-by-section rewrite from local PDF on 2026-03-23."
extracted_at: "2026-03-23"
llm_friendly: true
---

## Metadata

- Authors: Xinyu Liang, Fredrik Cumlin, Victor Ungureanu, Chandan K. A. Reddy, Christian Schuldt, Saikat Chatterjee
- arXiv: 2508.08962
- Year: 2025
- Task: non-intrusive MOS prediction / speech quality assessment
- Main question: should MOS systems keep using the last SSL layer, or are earlier layers better?

## TL;DR

- The paper tests layer-by-layer MOS prediction using multiple speech SSL backbones.
- Across Wav2Vec2, HuBERT, and WavLM families, the best MOS layer is usually **early**, roughly in the first quarter of the network.
- That early-layer preference remains even after downstream fine-tuning.
- A simple projection head on the right early layer beats or matches more complicated MOS systems that rely on last-layer SSL features plus extra information.

## Abstract

Speech quality assessment systems often take the final hidden layer from an SSL backbone and feed it into a lightweight regressor. This paper argues that this is an untested convention rather than a justified choice. It systematically evaluates individual layers from several SSL models for MOS prediction and finds that early layers consistently outperform or match the final layer. The result is both better performance and simpler systems.

## 1. Introduction

The paper starts from a broader observation from SSL analysis:

- early layers look more acoustic,
- later layers look more linguistic / semantic,
- downstream tasks should therefore care about layer choice.

This has already been studied for ASR-like tasks and representation analysis, but not much for speech quality assessment (SQA / MOS prediction), where a common practice is still:

```text
take final SSL layer
  -> shallow network
  -> MOS
```

The authors want to test whether that convention is actually wrong.

Their contribution is a careful layer-by-layer study of MOS prediction across:

- multiple SSL families,
- multiple SSL scales,
- multiple languages and datasets,
- direct frozen-feature use and downstream fine-tuning.

## 2. Selection of Layers in SSL Models for MOS Prediction

### 2.1 System Design

The experimental setup is intentionally simple.

For a chosen SSL model:

1. extract hidden states from one layer at a time;
2. feed that single-layer feature sequence to a lightweight projection head;
3. train the projection head to predict utterance-level MOS;
4. compare layers directly.

The projection head is inspired by DNSMOS-style design:

- 1D convolutions,
- flattening,
- scalar MOS output.

The reason for this simple head is important: the paper wants the comparison to reflect **layer quality**, not the capacity of a huge downstream network.

ASCII view:

```text
audio
  -> SSL backbone
      -> choose one layer l
          -> projection head
              -> MOS
```

The paper explicitly does **not** fuse layers together. It isolates them to see which one is best.

### 2.2 SSL Models

The paper evaluates six pre-trained speech SSL models:

| Model | Layers | Dim | Params | Training data |
| --- | ---: | ---: | ---: | ---: |
| w2v2 base | 12 | 768 | 94.4M | 960h |
| HuBERT base | 12 | 768 | 94.4M | 960h |
| WavLM base | 12 | 768 | 94.4M | 960h |
| w2v2 xlsr 300m | 24 | 1024 | 315M | 436,000h |
| w2v2 xlsr 1b | 48 | 1280 | 962M | 436,000h |
| w2v2 xlsr 2b | 48 | 1920 | 2.16B | 436,000h |

So the study covers both:

- different SSL objectives and architectures;
- different model scales.

### 2.3 MOS Datasets

Three benchmark datasets are used:

| Dataset | Language | Train | Val | Test |
| --- | --- | ---: | ---: | ---: |
| BVCC | English | 4,974 | 1,066 | 1,066 |
| Tencent | Chinese | 8,000 | 2,000 | 1,563 |
| NISQA | English (test set uses German LiveTalk subset) | 11,020 | 2,700 | 232 |

Why this matters:

- BVCC and Tencent test in-domain MOS prediction;
- NISQA LiveTalk adds an out-of-domain / cross-language generalization check.

### 2.4 Training and Metrics

Input audio is standardized to 16 kHz and fixed to 8 seconds by padding or random cropping.

Metrics:

- MSE
- LCC
- SRCC

The paper treats **LCC** as the main metric.

Projection-head training:

- 30 epochs
- Adam
- learning rate `1e-4`
- batch size `64`
- MSE training loss
- report average over 5 random runs per scenario

### 2.5 Fine-Tuning Setting

The paper also studies whether the same layer preference remains after fine-tuning.

Procedure:

- fine-tune `wav2vec2 base` on quantized MOS prediction as a 9-way classification problem;
- save checkpoints across epochs;
- inspect which internal layer becomes best after fine-tuning.

Important result: fine-tuning improves performance a bit, but **does not move the optimal layer very much**.

## 3. Experiments

### 3.1 Base Models: Main Layer Trend

For the base 12-layer models, the best-performing layers are typically around layers `3` to `5`.

This is the first major result:

- the optimal MOS layer is not late;
- it sits in the shallow-to-middle region.

Best layers from Table III:

| Model | BVCC | Tencent | NISQA LiveTalk |
| --- | ---: | ---: | ---: |
| w2v2 base | 3 | 3 | 4 |
| HuBERT base | 3 | 5 | 4 |
| WavLM base | 3 | 4 | 4 |

This consistency across models is one of the strongest parts of the paper.

### 3.2 Larger Models

For larger multilingual Wav2Vec2 XLSR models, the optimal layer moves deeper in absolute index, but still stays roughly in the **first quarter** of the stack.

Best layers:

| Model | BVCC | Tencent | NISQA LiveTalk |
| --- | ---: | ---: | ---: |
| w2v2 xlsr 300m (24L) | 5 | 7 | 5 |
| w2v2 xlsr 1b (48L) | 7 | 15 | 43 |
| w2v2 xlsr 2b (48L) | 7 | 13 | 39 |

The paper notes that BVCC clearly favors early layers. Tencent and NISQA become flatter for some large models, but the overall trend still supports "use an early layer, not the last one."

### 3.3 Fine-Tuning Results

The authors fine-tune Wav2Vec2 base on MOS prediction and then probe layers again.

Result:

- performance improves slightly after fine-tuning;
- the identity of the best layer stays almost unchanged.

So the paper’s claim is not just about frozen SSL usage. The layer preference appears structurally tied to the task.

### 3.4 Best Numbers vs Prior Systems

The paper compares its best layer-selected systems with prior SSL-based MOS predictors such as UTMOS and SSL-MOS.

Selected best results from Table III:

| Model | Best Layer | BVCC LCC | Tencent LCC | NISQA LiveTalk LCC |
| --- | ---: | ---: | ---: | ---: |
| w2v2 base | 3 | 0.867 | 0.962 | 0.802 |
| HuBERT base | 3/5/4 | 0.866 | 0.964 | 0.825 |
| WavLM base | 3/4/4 | 0.864 | 0.963 | 0.863 |
| w2v2 xlsr 300m | 5/7/5 | 0.884 | 0.972 | 0.920 |
| w2v2 xlsr 1b | 7/15/43 | 0.885 | 0.974 | 0.912 |
| w2v2 xlsr 2b | 7/13/39 | 0.886 | 0.974 | 0.922 |

The authors call out one especially important comparison:

- a simple `w2v2 xlsr 300m` with the right selected layer plus a tiny projection head already beats `UTMOS`, despite UTMOS relying on extra inputs and a more complicated system design.

## 4. Interpretation

The paper’s interpretation is:

- MOS prediction depends more on acoustic / low-to-mid-level signal properties than on the more abstract linguistic content emphasized by late layers.
- Early layers preserve the information that matters for human-perceived speech quality.
- Final layers may be too specialized toward phonetic / semantic abstractions learned during SSL pretraining, which is not the same target as MOS.

They connect this to earlier layer-analysis work showing that:

- similarity to spectrogram features decreases with depth,
- similarity to phoneme and word labels increases with depth.

MOS seems to want the former more than the latter.

## 5. Conclusion

The paper’s bottom line is very direct:

- stop assuming the last SSL layer is best for MOS;
- an early layer, usually around the first quarter of the network, is a better default;
- this improves both quality and efficiency;
- the trend is robust across SSL families, model sizes, datasets, and even downstream fine-tuning.

## Relevance to Peacock

This paper matters because it gives a clean negative result against the lazy default of "just take the last SSL layer."

For Peacock, that has two immediate consequences:

1. If you are extracting SSL features for pronunciation scoring, intelligibility, or open-response scoring, final-layer-only extraction is not a safe assumption.
2. If Shih-style `HConv` or `CHConv` is on the table, this paper is part of the case for it: layer choice materially changes downstream behavior, so learning or at least testing better layer aggregation is worth doing.

It is not a model-fusion paper, but it is directly adjacent to that line: it strengthens the argument that **where** you read from the SSL backbone matters just as much as **which** backbone you picked.
