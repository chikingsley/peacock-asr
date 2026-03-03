# Contrastive Ordinal Regularization for CTC-Based Pronunciation Scoring

## Abstract

*TODO: Write after Phase 1 experiments complete.*

We investigate whether contrastive phonemic ordinal (ConPCO) regularization
[@yan2025conpco], the loss function behind the current SOTA on SpeechOcean762,
can improve pronunciation scoring when applied to an existing GOPT pipeline
[@gong2022gopt_transformer_pronunciation_assessment] without changing the
model architecture or input features. Our baseline uses segmentation-free GOP
features (42 dimensions) and a 3-layer transformer scorer, achieving PCC 0.677
on SpeechOcean762. We incrementally add: (1) ordinal entropy loss, (2) CLAP
contrastive alignment, (3) duration and energy features, and (4) SSL
embeddings. This ablation isolates how much of the reported ConPCO+HierCB
SOTA (PCC 0.743) comes from the loss function versus richer features and
architecture.

## 1. Introduction

Automatic pronunciation assessment systems typically extract frame-level
features from a CTC-based acoustic model, compute goodness of pronunciation
(GOP) scores, and pass those scores to a downstream predictor. The choice
of downstream scorer — polynomial regression, SVR, or a transformer like
GOPT — has a large effect on final quality [@cao2026segmentation_free_gop;
@gong2022gopt_transformer_pronunciation_assessment].

Recent work by Yan and Chen introduces ConPCO [@yan2025conpco], a
regularization objective that adds two terms to the standard MSE training
loss: an ordinal entropy term that enforces structured spacing between
phoneme score levels, and a contrastive alignment term that matches audio
and text phoneme representations. When combined with HierCB (a hierarchical
convolutional-block architecture using 3072 dimensions of SSL features),
ConPCO achieves PCC 0.743 on SpeechOcean762, the current best reported
result.

However, it remains unclear how much of this improvement comes from:

- The ConPCO loss function itself
- The 75x increase in input feature dimensionality (42 → 3164)
- The HierCB architecture (BlockCNN, hierarchical word/utterance levels)

This paper answers that question through controlled ablation, following our
lab's methodology of one change at a time with compute-fair comparisons.

## 2. Related Work

### 2.1 Pronunciation Scoring and GOP

*Inherit from Track 05 manuscript, Section 2.1.*

### 2.2 Contrastive and Ordinal Methods

ConPCO builds on ordinal entropy [@yan2023pco], which itself adapts ideas
from the ordinal entropy loss in representation learning (Liang et al.,
ICLR 2023). The key insight: pronunciation scores are ordinal (0 < 1 < 2),
not categorical. Standard MSE treats all errors equally, but ordinal entropy
encourages the model to learn representations where score levels are
meaningfully separated.

The CLAP contrastive term aligns audio features with text (phoneme identity)
features, similar to CLIP/CLAP objectives in multimodal learning. This
encourages the model to form clean phoneme clusters in feature space.

### 2.3 HierCB Architecture

HierCB [@yan2024hiertfr] uses a three-level hierarchy (phone → word →
utterance) with BlockCNN blocks that combine self-attention and depth-wise
convolution via learned branch merging. It requires SSL features from three
models (HuBERT, wav2vec2-300M, WavLM), energy statistics, and phone
durations — totaling 3164 input dimensions versus our 42.

## 3. Method

### 3.1 Baseline Pipeline

*Describe our GOP-SF + GOPT pipeline. Inherited from Track 05.*

### 3.2 ConPCO Loss Integration

*Describe the two loss components added to MSE:*

- Ordinal entropy (diversity + tightness)
- CLAP contrastive alignment

### 3.3 Feature Enrichment (Phases 2-3)

*Describe incremental feature additions: duration, energy, SSL.*

## 4. Experimental Setup

### 4.1 Dataset and Protocol

SpeechOcean762 [@speechocean762]. 2500 train / 2500 test. Phone-level PCC
as primary metric. Minimum 3 seeds per configuration.

### 4.2 Implementation Details

*TODO after implementation.*

## 5. Results

### 5.1 Phase 1: Loss Function Ablation

*TODO: Table 1 from ABLATION_PLAN.md Phase 1.*

### 5.2 Phase 2: Feature Enrichment

*TODO: Table 2 from ABLATION_PLAN.md Phase 2.*

### 5.3 Phase 3: Architecture Comparison

*TODO if applicable.*

## 6. Discussion

*Where does the gain come from? What's the minimum-complexity improvement path?*

## 7. Conclusion

*TODO after experiments.*

## References
