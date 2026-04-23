---
title: "MuFFIN: Multifaceted Pronunciation Feedback Model With Interactive Hierarchical Neural Modeling"
authors:
  - "Bi-Cheng Yan"
  - "Ming-Kang Tsai"
  - "Berlin Chen"
citation_author: "Yan et al."
year: 2025
doi: "10.1109/TASLPRO.2025.3619765"
pages: 16
source_pdf: "../papers/pronunciation-assessment/hiercb/[Yan et al, 2025]-muffin-multifaceted-pronunciation-feedback-model-with-interactive-hierarchical-neural-modeling/paper.pdf"
extraction_method: "Near-verbatim pdftotext transcription from the local PDF with light cleanup of layout artifacts."
extracted_at: "2026-03-23"
llm_friendly: true
---

## Abstract

Computer-assisted pronunciation training (CAPT) manages to facilitate second-language (L2) learners to practice pronunciation skills by offering timely and instructive feedback. To examine pronunciation proficiency from multiple facets, existing methods for CAPT broadly fall into two categories: mispronunciation detection and diagnosis (MDD) as well as automatic pronunciation assessment (APA). The former aims to pinpoint phonetic pronunciation errors and provide diagnostic feedback, while the latter seeks instead to quantify pronunciation proficiency pertaining to various aspects. Despite the natural complementarity between MDD and APA, researchers and practitioners, however, often treat them as independent tasks with disparate modeling paradigms. In light of this, we in this paper first introduce MuFFIN, a Multi-Faceted pronunciation Feedback model with an Interactive hierarchical Neural architecture, to jointly address the tasks of MDD and APA. To better capture the nuanced distinctions between phonemes in the feature space, a novel phoneme-contrastive ordinal regularization mechanism is then put forward to optimize the proposed model to generate more phoneme-discriminative features while factoring in the ordinality of the aspect scores. In addition, to address the intricate data imbalance problem in MDD, we design a simple yet effective training objective, which is specifically tailored to perturb the outputs of a phoneme classifier with the phoneme-specific variations, so as to better render the distribution of predicted phonemes meanwhile considering their mispronunciation characteristics. A series of experiments conducted on the Speechocean762 benchmark dataset demonstrates the efficacy of our method in relation to several cutting-edge baselines, showing state-of-the-art performance on both the APA and MDD tasks.

## Index Terms

Computer-assisted pronunciation training (CAPT), automatic pronunciation assessment (APA), mispronunciation detection and diagnosis (MDD), multi-aspect and multi-granular pronunciation assessments, contrastive learning.

## Publication Details

Received 18 March 2025; revised 8 July 2025 and 12 September 2025; accepted 21 September 2025. Date of publication 9 October 2025; date of current version 24 October 2025. The associate editor coordinating the review of this article and approving it for publication was Dr. Ricardo Gutierrez-Osuna. (Corresponding author: Berlin Chen.)

Bi-Cheng Yan and Berlin Chen are with the Department of Computer Science and Information Engineering, National Taiwan Normal University, Taipei 11677, Taiwan (e-mail: <80847001s@ntnu.edu.tw>; <berlin@ntnu.edu.tw>).

Ming-Kang Tsai is with the Department of Chemistry, National Taiwan Normal University, Taipei 11677, Taiwan (e-mail: <mktsai@ntnu.edu.tw>).

Digital Object Identifier 10.1109/TASLPRO.2025.3619765

## I. Introduction

Fueled by the amplified demand for foreign language acquisition, research on computer-assisted pronunciation training (CAPT) has aroused significant attention amidst the tide of globalization, figuring prominently in the field of computer-assisted language learning (CALL) [1], [2]. To bridge the gap between insufficient supplies and pressing needs from language teachers and learners, CAPT systems have emerged as appealing learning tools ubiquitously, shifting the conventional pedagogical paradigm from teacher-led to self-directed learning. Beyond their critical roles in education and language learning, CAPT systems also serve as a handy reference for professionals (e.g., interviewers and examiners) in high-stakes assessments, with the goals of reducing the workload [3], [4], alleviating the burdens of recruiting new human experts, and achieving consistent and objective assessment results [5], [6], [7].

A de-facto archetype system for CAPT is normally instantiated in a read-aloud scenario, where an L2 learner is provided with a reference text and instructed to pronounce it correctly. By taking the learner's speech paired with the reference text as input, CAPT systems are anticipated to assess the learner's oral competence from multiple facets, providing detailed and potentially diagnostic performance feedback with a near-instant turnaround. To this end, mispronunciation detection and diagnosis (MDD) and automatic pronunciation assessment (APA) are two active strands of research in developing pronunciation feedback modules for CAPT. The former seeks to pinpoint phonetic pronunciation errors and provides L2 learners with the corresponding diagnostic feedback [8], [9]. The latter, in contrast, concentrates more on assessing the learner's pronunciation quality through multi-faceted pronunciation scores, reflecting his/her proficiency pertaining to specific aspects or some extent of spoken language usage [10], [11]. One time-tested approach for MDD is goodness of pronunciation (GOP) and its derivatives [12], [13], which calculate the ratio between the likelihoods of the canonical and most likely pronounced phonemes. Phoneme-level erroneous pronunciations are subsequently detected if the likelihood ratios of certain phoneme segments fail below predetermined thresholds. On a separate front, the models of iconic APA methods are typically trained to mimic human ratings based on surface features (viz. a set of hand-crafted features). These models either employ a classifier to predict a holistic score representing learners' oral proficiency [10] or use regressors to estimate continuous analytic scores for specific pronunciation aspects, such as phoneme-level accuracy [14], word-level lexical stress [15], and utterance-level pronunciation quality [16], [17].

In spite of the complementary nature of MDD and APA, most existing efforts treat them as independent tasks, thereby developing two disparate feedback modules for use in CAPT. However, some prior studies reveal that an L2 English learner tends to have lower utterance-level assessment scores of intelligibility and fluency [18] whenever his or her utterances frequently contain

## II. Related Work

### A. MDD

Survey of three MDD families:

- pronunciation scoring-based methods
- dictation-based methods
- prompt-based end-to-end methods

Examples include GOP/likelihood-ratio based scoring, CTC dictation style systems, anti-phone modeling, and attention-based prompt alignment methods.

### B. APA

Earlier APA relies on handcrafted features and single-aspect modeling.

Recent methods moved toward:

- multi-aspect and multi-granularity scoring
- combining GOP with SSL and prosodic cues
- hierarchical architectures to align phoneme, word, and utterance levels

Most prior work still treats APA and MDD separately or lacks deep diagnostic support at scale.

## III. Multi-Faceted Pronunciation Feedback Model with Interactive Hierarchical Neural Modeling

### A. Model architecture overview

MuFFIN has three hierarchical levels:

- Phoneme-level modeling
- Word-level modeling
- Utterance-level modeling

All levels use convolution-augmented Branchformer blocks:

- one branch: multi-head self-attention for suprasegmental/global context
- other branch: depthwise/pointwise convolutions for fine-grained local cues

The architecture includes:

- a phoneme-level pronunciation feedback module (detect, diagnose, score)
- word-level attention pooling and regressors
- utterance-level modules with residual SSL path and attention pooling
- a single integrated training objective combining APA and MDD

### B. Problem formulation

For utterance `U` represented by audio `X` and prompt text `T`:

- prompt converts to `M` words and canonical phoneme sequence `q = (q1...qN)`.
- model predicts aspect score sequences at each granularity:
  - phoneme level
  - word level
  - utterance level
- MDD outputs:
  - binary error state `e_n` per canonical phoneme
  - diagnosis label `y_n` for the produced phoneme

`e_n = 1` indicates mispronunciation.

### C. Phoneme-level modeling

Input phoneme features combine:

- GOP features `EGOP` (LPP and LPR)
- duration features `EDur`
- energy statistics `EEng`
- SSL features `ESSL`

Concatenate and project:

- `Xp = Linear_p([EGOP; EDur; EEng; ESSL])`
- `Hp0 = Xp + Ep` where `Ep` is phoneme prompt embedding
- `Hp = PhnEnc(Hp0)` with 3 Branchformer blocks

Then three heads:

- Error detector: binary sigmoid for `e_n`
- Diagnosis predictor: softmax over pronunciation dictionary for `y_n`
- Accuracy regressor for phoneme-level score

### D. Word-level modeling

- attention pooling over phoneme representations with depthwise conv + self-attention to form word vectors
- word textual embedding added
- Word encoder (2 Branchformer blocks)
- three aspect branches for multiple word-level metrics

### E. Utterance-level modeling

- extract average-pooled SSL features at utterance level
- project/mix word-level aspect-specific representations by weighted merge
- concatenate pooled phoneme/word/SSL projections and pass through utterance encoder
- five attention pooling heads for utterance aspects (accuracy/fluency/completeness/prosody/total)

### F. Losses and training objective

APA uses weighted MSE over granularity-level/regression tasks:

- `L_APA = sum(MSE_p) + sum(MSE_w) + sum(MSE_u)`

MDD uses:

- detection loss `L_det` (binary cross-entropy/likelihood for error states)
- diagnosis loss `L_diag` (cross-entropy over diagnostic classes)
- `L_MDD = L_det + L_diag`

Total:

- `L_MuFFIN = L_APA + L_MDD`

## IV. Contrastive Phonemic Ordinal Regularizer (ConPCO)

ConPCO adds three terms:

- contrastive term `L_con`
- phonemic characteristic term `L_pc`
- ordinal term `L_o`

Total regularizer:

- `L_ConPCO = L_con + L_pc + L_o`

### 1) Contrastive term

Align phoneme encoder features with phoneme prompt embeddings in a shared space.
Uses paired sample sets with centroid construction and large-temperatureed contrastive objective.

### 2) Phonemic characteristic term

Encourages better inter-phoneme separation and preserves phonemic structure by minimizing negative centroid distances where appropriate (as implemented in paper’s objective style).

### 3) Ordinal term

Incorporates pronunciation score order.

- weights samples by distance from score constant
- encourages feature compactness that reflects score proximity

## V. Phoneme-Specific Variation (PhnVar)

Instead of standard training with hard imbalance bias, PhnVar perturbs diagnosis logits:

- for each phoneme class `k`, adjusted logit:
- `ĝ_k,n = g_k,n + δ(σ) * exp((α log(QF_k) + β log(DF_k)) / (α+β))`

where:

- `δ(σ)` is Gaussian noise
- `QF_k` is data quantity factor from inverse class frequency
- `DF_k` is pronunciation difficulty factor from mispronunciation rate
- `α = β = 1` in experiments

Effect:

- expands minority phoneme decision regions
- compensates for mispronunciation difficulty asymmetry
- improves recall/precision balance versus quantity-only perturbation

## VI. Experimental Setups

### A. Data and metrics

Dataset: `Speechocean762`

- 5,000 recordings from 250 Mandarin learners
- train: 2,500 / test: 2,500

Table I task statistics (APA):

- phoneme-level accuracy labels `[0,2]`: 47,076 train / 47,369 test
- word-level accuracy/stress/total `[0,10]`: 15,849/15,967 (acc), 15,849/15,967 (stress), 15,849/15,967 (total)
- utterance-level accuracy/completeness/fluency/prosody/total `[0,10]`: 2,500/2,500 each

Table II task statistics (MDD):

- correct canonical pronunciation labels: 45,088 / 45,959
- deletion: 450 / 396
- substitution: 914 / 593
- non-categorical: 488 / 332
- accented: 136 / 89

Metrics:

- APA: Pearson correlation coefficient (PCC) and MSE for phoneme accuracy
- MDD: recall, precision, F1 for detection; FAR/FRR/DER/PER for diagnosis

### B. Features and implementation

- GOP + energy + duration + SSL features
- SSL backbones: wav2vec2.0, wavlm, Hubert (frame-level -> phone-level aggregation via forced alignment)
- feature dimension total 3,164 (GOP 84 + energy 7 + duration 1 + SSL 3072)

Training:

- 5 independent runs per result
- 100 epochs
- Adam, initial lr 1e-3
- batch size 25
- reduce lr by 0.1 after 10 epochs no improvement
- pretrain strategy inherited from prior work
- results averaged over 5 trials, reported on best phoneme MSE setting

Model blocks:

- Phoneme encoder 3 blocks
- Word encoder 2 blocks
- Utterance encoder 1 block
- hidden size 24, single-head attention in reported setup

### C. Compared methods

Compared against:

- Single-aspect: Lin2021, Kim2022
- Multi-aspect/multi-granular: GOPT, LSTM, GFR, 3M, 3MH, HierGAT
- Multi-faceted baselines: Ryu2023, JAM

## VII. Experimental Results

### A. Qualitative analysis

#### Phoneme statistics and imbalance

Fig. 6 shows many/medium/few shot and difficulty-based groupings; occurrence count and mispronunciation rate are not aligned.

#### ConPCO visualizations

Fig. 7:

- baseline MuFFIN groups by score but weak phoneme separation
- adding `L_pc` improves phoneme separation
- adding `L_o` adds score-ordered structure

Fig. 8:

- contrastive term aligns spoken phoneme features with textual embeddings
- helps preserve phoneme-specific structure

#### PhnVar visualizations

Fig. 9:

- vanilla MuFFIN: majority phonemes dominate feature space
- `PhnVar w/o DF`: better equalization by frequency only
- `PhnVar`: combines frequency + pronunciation difficulty for better class-imbalance balancing

### B. APA performance

Table III summary:

- MuFFIN and MuFFIN variants achieve SOTA-like APA on most metrics.
- reported standout: phoneme-level accuracy PCC around 0.742 (best), phone/multi-aspect gains over 3MH.
- strongest gains across total utterance-level metrics versus strong baselines.
- word-level stress is weaker than 3MH (likely due sub-phoneme modeling advantages in 3MH).
- randomization tests show significant gains `p < 0.001`.

### C. MDD performance

Table V highlights:

- Ryu2023 and JAM are baselines with lower MDD baselines
- MuFFIN: high recall/precision and lower PER than baselines
- MuFFIN + PhnVar: further increases F1 and precision, and improves other MDD metrics significantly over MuFFIN
- trade-off remains: PhnVar improves detection recall with slight effects on diagnostic FAR/FRR versus some baselines

### D. Imbalance examination

Table VI and Fig. 11 investigate long-tail and pronunciation difficulty partitions:

- Many/med/few occurrence groups and high/med/low mispronunciation-rate groups are evaluated.
- QF-only tends to improve recall/feature balance but may reduce precision
- DF-only helps precision
- combined QF+DF gives better F1 with better balance

### E. Ablations

Table VII (APA-only granularity):

- phone-only, word-only, utterance-only are dominated by multi-granularity combinations
- Phone+Word and Phone+Word+Utt. are strongest for accuracy aspect

Table VIII (joint APA+MDD):

- adding MDD to APA improves MDD metrics
- MDD+Phone configuration often gives best recall
- multi-granularity assessment remains key for APA

## VIII. Conclusion

The paper presents MuFFIN as a unified framework for pronunciation feedback:

- performs both APA and MDD in one hierarchy-aware network
- improves phoneme-level assessment and mispronunciation handling via ConPCO
- improves training robustness to imbalance via PhnVar
- demonstrates gains on SpeechOcean762 in both APA and MDD

## Limitations and Future Work

- data is read-aloud only
- limited explainability for score outputs
- Mandarin-accent-only coverage in dataset
- no fully realistic conversational/general spoken-language scenario tested

## References

The references section in the PDF contains 61 cited works, including:

- 3M and prior MuFFIN lineage papers [24][25]
- speech model baselines and CAPT foundations [46][57] etc.
- key benchmark dataset release [26]

See the original PDF for full bibliography list.
