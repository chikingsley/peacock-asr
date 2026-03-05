# Do You Need 300M Parameters? Compact CTC Backbones for GOP-Based Pronunciation Scoring

## Abstract

*TODO: Write after Phase 1 experiments complete.*

Current CTC-based pronunciation assessment systems rely on large self-supervised
models (300M+ parameters) as feature extractors, but the relationship between
backbone size and phone-level scoring quality has never been systematically studied.
We compare wav2vec2-base (95M), HuBERT-base (95M), and Citrinet-256 (10M) against
wav2vec2-xlsr-53 (300M) as GOP feature extractors, using identical segmentation-free
GOP features and GOPT transformer scoring. We additionally test HMamba
[@chao2025hmamba] and HiPAMA [@do2023hipama] scoring heads as alternatives to the
GOPT transformer. On SpeechOcean762, we find [TODO: results].

## 1. Introduction

Automatic pronunciation assessment typically follows a two-stage pipeline:
(1) a CTC-based acoustic model generates phoneme posteriors, from which
Goodness of Pronunciation (GOP) features are computed, and (2) a downstream
scorer (polynomial regression, SVR, or transformer) maps GOP features to
pronunciation quality scores.

Recent work has progressively increased the backbone model size — from Kaldi
GMM-HMM systems to wav2vec2-xlsr-53 (300M) [@baevski2020wav2vec2] — achieving
phone-level PCC improvements from 0.612 [@gong2022gopt_transformer_pronunciation_assessment]
to 0.677 (our baseline, using GOP-SF [@cao2026segmentation_free_gop]).

However, a fundamental question remains unexplored: **how much of this improvement
comes from the backbone's 300M parameters, and how much from the superior
GOP-SF algorithm?** If a 95M-parameter backbone achieves comparable PCC when
paired with GOP-SF features, the 300M model is computationally wasteful.

No published paper has tested wav2vec2-base or HuBERT-base as CTC backbones
for GOP-based phone-level pronunciation scoring on SpeechOcean762. All prior
GOP work uses Kaldi models [@gong2022gopt_transformer_pronunciation_assessment;
@do2023hipama; @han2026hia], while SSL studies use embeddings directly without
GOP [@kim2022ssl_pronunciation]. This paper fills that gap.

## 2. Related Work

### 2.1 GOP-Based Pronunciation Assessment

*Inherit from Track 05 manuscript, Section 2.1.*

### 2.2 Self-Supervised Backbones for Pronunciation

Kim et al. [@kim2022ssl_pronunciation] compared SSL models for utterance-level
pronunciation scoring (not phone-level GOP), finding that HuBERT-base (95M)
achieves holistic PCC 0.75, substantially outperforming wav2vec2-base (0.65)
despite identical parameter count. This suggests pre-training objective matters
more than size for pronunciation tasks.

### 2.3 Compact CTC Models

Citrinet [@li2021citrinet] achieves competitive ASR (WER 3.8%) with only 10M
parameters through 1D time-channel separable convolutions and squeeze-and-excitation
blocks. However, its 256-token SentencePiece vocabulary and 8x time reduction
create challenges for phoneme-level GOP computation.

### 2.4 Alternative Scoring Heads

HMamba [@chao2025hmamba] replaces GOPT's transformer blocks with Mamba selective
state space blocks, adding hierarchical phone → word → utterance aggregation.
HiPAMA [@do2023hipama] uses multi-aspect attention for joint phone/word/utterance
scoring. Both report improvements over GOPT but use Kaldi GOP features.

## 3. Method

### 3.1 GOP-SF Pipeline

*Describe our GOP-SF + GOPT pipeline. Inherited from Track 05.*

### 3.2 Backbone Fine-Tuning Protocol

*Describe CTC fine-tuning on LibriSpeech with 41-token ARPABET vocab.*

### 3.3 Citrinet Adaptation

*Describe vocabulary replacement and time reduction handling.*

## 4. Experimental Setup

### 4.1 Dataset and Protocol

SpeechOcean762 [@speechocean762]. 2500 train / 2500 test. Phone-level PCC
as primary metric. Minimum 3 seeds per configuration.

### 4.2 Implementation Details

*TODO after implementation.*

## 5. Results

### 5.1 Backbone Size vs PCC

*TODO: Table 1 from ABLATION_PLAN.md Phase 1.*

### 5.2 Citrinet-256 Feasibility

*TODO: Table 2 from ABLATION_PLAN.md Phase 2.*

### 5.3 Scoring Head Comparison

*TODO: Table 3 from ABLATION_PLAN.md Phase 3.*

### 5.4 Pareto Analysis

*TODO: Figure 1 from Phase 4.*

## 6. Discussion

*Is the 300M backbone justified? What's the minimum viable backbone?
Practical deployment implications for edge devices.*

## 7. Conclusion

*TODO after experiments.*

## References
