---
arxiv: 2406.12209
title: "Interface Design for Self-Supervised Speech Models"
authors:
  - "Yi-Jen Shih"
  - "David Harwath"
citation_author: "Shih et al."
year: 2024
venue: "arXiv:2406.12209"
pages: 5
source_pdf: "paper.pdf"
extraction_method: "Manual section-by-section rewrite from local PDF on 2026-03-23."
extracted_at: "2026-03-23"
llm_friendly: true
---

# Interface Design for Self-Supervised Speech Models

## Metadata

- Authors: Yi-Jen Shih, David Harwath
- arXiv: 2406.12209
- Year: 2024
- Task scope: how to connect a frozen SSL speech encoder to downstream speech tasks
- Main idea: treat the layer-aggregation module itself as a first-class design object, not just an implementation detail

## TL;DR

- The paper argues that the standard layerwise weighted sum used with frozen SSL speech models is not the best interface.
- It formalizes the pipeline as `Upstream -> Interface -> Downstream`.
- It compares several interfaces and shows that a hierarchical 1D convolution over layers (`HConv`) is usually better than weighted sum.
- The gain is not just "more parameters": even when the downstream model is enlarged, weighted sum still loses to the hierarchical convolution interface.
- The same pattern mostly remains when the upstream model is end-to-end fine-tuned.

## Abstract

The paper studies how to aggregate hidden states from different layers of self-supervised speech models before handing them to downstream task heads. Instead of treating layer fusion as a fixed weighted sum, the authors define an explicit interface module between the upstream SSL encoder and the downstream predictor. They test several interface designs and report that a convolutional interface over the layer dimension consistently outperforms weighted sum across multiple tasks and multiple upstream models.

## 1. Introduction

The motivation starts from a familiar usage pattern for speech SSL:

1. pre-train a large upstream SSL model on unlabeled speech;
2. use it as a feature extractor for downstream tasks;
3. either fine-tune end-to-end, choose one layer, or compute a weighted sum over layers.

The authors argue that these three choices hide an important design question: **how should information across layers actually be aggregated?**

Their key claim is that the default weighted sum is attractive because it is cheap, but it may be suboptimal because the same feature dimension across different transformer layers does not necessarily encode compatible information. Naively summing those dimensions can cause what they call **information collision**.

This leads to their reframing:

```text
speech waveform
  -> upstream SSL model
  -> interface
  -> downstream task head
```

The paper’s contribution is to explicitly define and test that middle piece.

## 2. Proposed Interface Methods

### 2.1 Interface Definition

The framework is:

- Upstream `U`: maps raw waveform to all hidden states, shape `L x T x D`
- Interface `I`: aggregates across the layer dimension, output shape `T x D`
- Downstream `D`: solves the task using the interface output

Formally:

```text
U(.) : R^T' -> R^(L x T x D)
I(.) : R^(L x T x D) -> R^(T x D)
```

Under this view, the standard weighted sum is just one possible interface:

```text
I_WS(h) = sum_l w_l * h_l
```

where `w_l` are learned layer weights.

### 2.2 Why Weighted Sum May Be Weak

The paper’s intuition is simple:

- different layers encode different kinds of information;
- the same channel index in different layers is not guaranteed to mean the same thing;
- summing them directly may blur or destroy useful structure.

The problem should get worse as models get deeper, because more layers means more chances for incompatible features to be averaged together.

### 2.3 Proposed Interfaces

The paper tests several alternatives.

#### Grouped Weighted Sums

Instead of one global weighted sum over all layers:

- split layers into groups,
- compute a weighted sum inside each group,
- concatenate group outputs,
- project back to the downstream dimension.

This is meant to reduce global information collision while still being simple.

#### Concatenation + Learnable Projection

Take all layer outputs, concatenate them along feature dimension, then learn a projection:

```text
(L, T, D) -> (T, L*D) -> (T, D)
```

This lets the model learn which parts of which layers matter.

#### Hierarchical Convolution over Layers

This is the main proposal.

The argument is that neighboring transformer layers should still have some local similarity because of residual connections, so layer aggregation can exploit locality. The interface:

- applies 1D convolutions over the **layer dimension**;
- uses kernel size `5`, stride `3`;
- stacks about `floor(log_3 L)` convolution layers so that all upstream layers are progressively collapsed to one output.

ASCII view:

```text
layer stack
  l1
  l2
  l3
  ...
  lL
   |
   v
[Conv over layers]
   |
   v
[Conv over layers]
   |
   v
single fused representation per time step
```

This is the `Hierarchical Conv.` or `HConv` interface.

#### CLS Pooling over Layer Dimension

At each time step:

- treat the layer axis as a token sequence,
- prepend a learnable CLS token,
- run a transformer layer,
- use the CLS token as the fused representation.

Unlike the simpler methods, this interface is data-dependent, not just task-dependent.

#### PCA + Concatenation

This is the non-parametric baseline:

- reduce each layer with PCA,
- concatenate reduced features from all layers,
- keep total dimension near the original upstream size.

The paper includes it mainly as a comparison point.

## 3. Experiments

### 3.1 Setup

Upstream SSL models:

- HuBERT Base
- HuBERT Large
- WavLM Base
- WavLM Large
- XLSR-53

Benchmarks:

- ML-SUPERB monolingual ASR
- ML-SUPERB multilingual ASR
- ML-SUPERB LID
- SUPERB emotion recognition
- SUPERB intent classification
- SUPERB speaker verification
- SUPERB phoneme recognition

The paper first runs a pilot on ML-SUPERB monolingual ASR with HuBERT Base to compare all proposed interfaces. Then it keeps the best ones for the full evaluation.

### 3.2 Pilot: HuBERT Base on ML-SUPERB Monolingual ASR

Table 1 compares:

- weighted sum
- grouped weighted sums
- concat + projection
- PCA + concat
- hierarchical convolution
- CLS pooling

Main outcomes:

- `Hierarchical Conv.` is best on both the `10min` and `1hr` settings.
- `CLS Pooling` is competitive on `1hr` but very poor on `10min`, suggesting it needs more supervised data.
- `PCA + Concat` is clearly weak.

Reported values from the table:

| Interface | Mono-10min | Mono-1hr |
| --- | ---: | ---: |
| Weighted Sum | 42.85 | 35.15 |
| GroupWS (#2) | 41.84 | 34.47 |
| GroupWS (#3) | 43.08 | 33.96 |
| GroupWS (#4) | 42.52 | 33.99 |
| Concat + Proj | 43.20 | 34.26 |
| PCA + Concat | 45.16 | 36.58 |
| Hierarchical Conv. | **41.51** | **33.88** |
| CLS Pooling | 48.36 | 33.92 |

This is why the rest of the paper focuses mainly on `HConv` and `CLS Pooling`.

### 3.3 Full Benchmark Comparison

Across ML-SUPERB and SUPERB, the general pattern is:

- `Hierarchical Conv.` beats weighted sum on most tasks and most upstreams.
- It is especially strong for phoneme recognition and ASR.
- The gains often become larger when the upstream model is larger.

One of the strongest examples in the paper:

- HuBERT Base + weighted sum on SUPERB phoneme recognition: `5.41` PER
- HuBERT Base + HConv: `3.07` PER

That is large enough that **HuBERT Base + HConv beats HuBERT Large + weighted sum** (`3.53` PER).

Other notable results from Table 2:

| Upstream | Interface | Mono-1hr | Multi-1hr | ER | IC | SV | PR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HuBERT Base | Weighted Sum | 35.1 | 24.4 | 64.92 | 98.34 | 5.11 | 5.41 |
| HuBERT Base | HConv | **33.9** | **24.0** | **68.49** | **99.45** | 5.62 | **3.07** |
| HuBERT Large | Weighted Sum | 32.3 | 22.3 | 67.62 | 98.76 | **5.98** | 3.53 |
| HuBERT Large | HConv | **30.0** | **21.4** | **72.44** | **99.53** | 6.03 | **1.76** |
| WavLM Base | Weighted Sum | 34.2 | 24.3 | 65.94 | 98.63 | **4.69** | 4.84 |
| WavLM Base | HConv | **32.4** | **23.6** | **68.57** | **99.53** | 5.48 | **3.06** |
| WavLM Large | Weighted Sum | 30.1 | 20.8 | 70.62 | 99.31 | **3.77** | 3.06 |
| WavLM Large | HConv | **28.0** | **19.4** | **74.95** | **99.71** | 5.20 | **1.72** |
| XLSR-53 | Weighted Sum | 35.1 | 20.2 | 66.34 | 95.62 | 6.45 | 4.50 |
| XLSR-53 | HConv | **30.6** | **19.8** | **72.01** | **99.55** | **5.63** | **2.69** |

The main exception is speaker verification with HuBERT/WavLM, where weighted sum is sometimes better.

### 3.4 Is It Just More Parameters?

The authors test this directly.

They enlarge the downstream model while keeping weighted sum as the interface (`WS w/ Large DS`) and compare it against `HConv`.

Result:

- weighted sum with a larger downstream still does **not** match hierarchical convolution.

Examples from Table 3:

| Upstream | HConv Mono-1hr | WS + Large DS Mono-1hr |
| --- | ---: | ---: |
| HuBERT Base | **33.9** | 35.2 |
| HuBERT Large | **30.0** | 32.9 |
| WavLM Base | **32.4** | 34.4 |
| WavLM Large | **28.0** | 30.5 |
| XLSR-53 | **30.6** | 35.4 |

So the gain is not just "bigger head = better result." The layer-fusion mechanism itself matters.

### 3.5 Does Interface Choice Still Matter Under End-to-End Fine-Tuning?

The paper also drops the "frozen upstream" assumption and fine-tunes the upstream model end-to-end on ML-SUPERB Monolingual 1h with HuBERT Base.

Results from Table 4:

| Interface | Freeze Upstream | Fine-tuned |
| --- | ---: | ---: |
| Weighted Sum | 35.1 | 31.5 |
| Hierarchical Conv. | **33.9** | **31.1** |
| WS + Large DS | 35.2 | 31.6 |

The gap becomes smaller under full fine-tuning, but it does not disappear.

That is important: the interface question is not only a frozen-feature problem.

## 4. Conclusion

The paper’s final position is:

- interface design should be treated as a distinct design axis in SSL speech transfer;
- hierarchical convolution is a strong default interface for aggregating across SSL layers;
- the benefit comes from better information aggregation, not just parameter count;
- the effect persists, though more weakly, under end-to-end fine-tuning.

## Relevance to Peacock

This paper is the immediate ancestor of the later Shih `model + layer fusion` paper. For Peacock, the core takeaway is not just "use HConv somewhere." It is more specific:

- if you already know multiple SSL layers matter, **do not collapse them with a plain weighted sum unless you have to**;
- if you already know multiple SSL models matter, this paper suggests the fusion point should be a real module, not just a fixed averaging trick;
- if you are inheriting a `3M`/`MuFFIN`-style feature stack built from selected SSL outputs, this paper is the clean justification for inserting a learned interface between upstream SSL features and downstream pronunciation scoring.

In that sense, this paper is the conceptual bridge between:

- `3M` / `MuFFIN` style multi-SSL feature concatenation, and
- the later `Shih 2025` paper that explicitly unifies **model fusion + layer fusion**.
