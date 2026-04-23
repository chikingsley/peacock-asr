---
title: "HiPPO: Exploring A Novel Hierarchical Pronunciation Assessment Approach for Spoken Languages"
authors:
  - "Bi-Cheng Yan"
  - "Hsin-Wei Wang"
  - "Fu-An Chao"
  - "Tien-Hong Lo"
  - "Yung-Chang Hsu"
  - "Berlin Chen"
affiliation: "National Taiwan Normal University, EZAI"
citation_author: "Yan et al"
year: 2025
doi: "10.1109/ICASSP49660.2025.10890778"
pages: 14
source_pdf: "paper.pdf"
extraction_method: "Manual extraction from PDF"
extracted_at: "2026-04-20"
llm_friendly: true
---


## Abstract

Automatic pronunciation assessment (APA) seeks to quantify a second language (L2) learner's pronunciation proficiency in a target language by offering timely and fine-grained diagnostic feedback. Most existing efforts on APA have predominantly concentrated on highly constrained reading-aloud tasks (where learners are prompted to read a reference text aloud); however, assessing pronunciation quality in unscripted speech (or free-speaking scenarios) remains relatively underexplored. In light of this, we first propose **HiPPO**, a hierarchical pronunciation assessment model tailored for spoken languages, which evaluates an L2 learner's oral proficiency at multiple linguistic levels based solely on the speech uttered by the learner. To improve the overall accuracy of assessment, a contrastive ordinal regularizer and a curriculum learning strategy are introduced for model training. The former aims to generate score-discriminative features by exploiting the ordinal nature of regression targets, while the latter gradually ramps up the training complexity to facilitate the assessment task that takes unscripted speech as input. Experiments conducted on the Speechocean762 benchmark dataset validates the feasibility and superiority of our method in relation to several cutting-edge baselines.

## 1 Introduction

Spurred by the global demand for foreign language proficiency in both the workforce and academia, computer-assisted pronunciation training (CAPT) has gained significant attention, which facilitates second-language (L2) learners to practice pronunciation skills with near-instant, instructive, and potentially diagnostic feedback (Norris and Davis, 2025; Moere and Downey, 2016). To meet this pressing demand, CAPT systems have become ubiquitous and appealing learning tools, transitioning the conventional pedagogical approach from teacher-led instruction to self-directed learning (Rogerson-Revell, 2021; Chen and Li, 2016; Singla et al., 2021).

Automatic pronunciation assessment (APA) aims to evaluate L2 learners' speaking proficiency and provide fine-grained feedback on specific pronunciation aspects pertaining to a target language, figuring prominently in the field of CAPT. Prior studies on APA have primarily drawn attention to highly constrained speaking tasks (such as listening and then repeating words or sentences). As exemplified in Figure 1(a), a de-facto archetype system for APA is instantiated in reading-aloud (or scripted) learning scenarios, where an L2 learner is provided with a reference text and instructed to pronounce it correctly. Methods in this line of research typically rely on an input reference text paired with the learner's speech to derive timestamps of linguistic units (i.e., phones or words) via an automatic speech recognition (ASR) system, which are then used for either pronunciation feature extraction (Gong et al., 2022; Chao et al., 2022; Do et al., 2023; Yan et al., 2024) or for neural modeling (Lin and Wang, 2021; Wang et al., 2025). Albeit achieving competitive performance in relation to inter-rater agreement (Yan and Chen, 2024; Pei et al., 2024), scripted-speech assessments fail to reflect learners' speaking abilities in real-world communication. In contrast, pronunciation assessment of spoken languages introduces new challenges to CAPT, as it attempts to quantify an L2 learner's oral skills in spontaneous speech or elicit authentic responses through short questions (Zechner and Evanini, 2019; Kheir et al., 2023). Directly grafting existing APA models to use cases of spoken language assessment, however, confronts at least two major issues. First, as shown in Figure 1(b), the utterances of an L2 learner are produced in an unscripted manner, which makes APA models struggle to extract correct pronunciation features encompassing time-alignment information (Shen et al., 2021; Deng et al., 2020; Witt and Young, 2000). What is more, owing to the free-form nature of unscripted speech, the desired APA models are required to accommodate speech input of varying lengths.

Building on these observations, this paper presents HiPPO, a novel hierarchical pronunciation assessment model for spoken languages that evaluates L2 learners' oral proficiency based on unscripted speech (or free-speaking scenarios) and provides analytical scores on various pronunciation aspects across multi-granular linguistic levels. Specifically, HiPPO strategically employs a speech foundation model along with a grapheme-to-phoneme (G2P) converter to derive the most likely phone sequence produced by an L2 learner, thereby bringing the assessment task closer to its scripted-speech counterpart, as illustrated in Figure 1(c). To overcome sequence length constraints and preserve articulatory traits across multi-granular linguistic units, HiPPO capitalizes on a tailor-made Conv-LLaMA block to stack a hierarchical neural architecture, which augments the LLaMA block (Touvron et al., 2023) with a convolutional branch and rotary position encoding (Su et al., 2024). Moreover, during training, a contrastive ordinal regularizer is put forward to modulate feature distances through the absolute differences between regression targets. By exploiting the ordinal constraints, the proposed regularizer serves as a promising approach to generate score-discriminative features, mitigating the detrimental effects of ASR errors on pronunciation assessments. We further introduce a simple yet effective curriculum learning strategy for HiPPO that progressively increases the training complexity, transforming the assessment tasks from the read-aloud scenario to the free-speaking counterpart. An extensive set of experiments conducted on Speechocean762 benchmark dataset (Zhang et al., 2021), consisting of both read-aloud and simulated free-speaking scenarios, demonstrates substantial and consistent performance gains of the proposed methods over several strong baselines.

**Figure 1 Description**: Outlines the motivations.

- (a) Existing APA models are primarily tailored for read-aloud tasks where a reference text is provided.
- (b) Directly applying APA models to free-speaking scenarios struggles to quantify oral skills based solely on speech signals.
- (c) HiPPO integrates a speech recognizer to generate transcriptions from the learner's speech, effectively reformulating free-speaking assessment as a task akin to read-aloud.

In summary, our contributions are at least four-fold: (1) to our knowledge, HiPPO is the first attempt to assess oral skills for unscripted speech with multi-faceted scores from phone to utterance levels, opening a new avenue for CAPT; (2) we propose a novel Transformer block, **Conv-LLaMA** block, as the backbone of HiPPO, elaborately designed to handle the free-from speech uttered by L2 learners; (3) to alleviate the negative effects of ASR errors, a **contrastive ordinal regularizer** is proposed to reflect the ordinality of regression targets within the feature space; and (4) a simple yet effective **curriculum learning strategy** is explored to boost the performance of pronunciation assessment in the free-speaking scenario.

## 2 Methodology

This section sets out with a problem definition for pronunciation assessments on unscripted speech (or free-speaking scenarios) and then sheds light on the proposed methods, encompassing the assessment model, training objectives, and learning strategy. Due to the space limit, the overview of related work will be given in Appendix A.

### 2.1 Problem Definition

To assess speaking skills across different linguistic granularities for unscripted speech, we first employ a speech foundation model[^1] to transcribe a speech signal $X$ produced by an L2 learner into a sequence of $M$ words $\mathbf{w} = (w_1, w_2, ..., w_M)$ and subsequently a G2P converter[^2] to generate the corresponding phonetic transcription of $N$ phones $\mathbf{p} = (p_1, p_2, ..., p_N)$, where **w** and **p** collectively serve as a proxy for the textual and phonetic realizations perceived by human raters.

Let $G = \{g^{phn}, g^{word}, g^{utt}\}$ denote the set of linguistic granularities, where $g^{phn}$, $g^{word}$, and $g^{utt}$ mark the phone-, word-, and utterance-level linguistic granularities, respectively. HiPPO is trained under a multi-task learning paradigm to estimate a set of aspect score sequences $A^g = \{a_1^g, a_2^g, ..., a_{N_g}^g\}$ for each granularity $g \in G$, where $N_g$ is the number of pronunciation aspects.

[^1]: <https://huggingface.co/openai/whisper-large-v3>
[^2]: <https://github.com/Kyubyong/g2p>

**Figure 2 Description**: Processing flow of HiPPO for qualifying the oral skills in unscripted speech. It shows the speech signal being processed by a Speech Foundation Model (Whisper) to get transcribed words, which then go through a G2P model to get a perceived phone sequence. These, along with CTC-based GOP features and SSL-based features, are fed into the HiPPO model to produce aspect scores.

### 2.2 Hierarchical Pronunciation Assessment Model (HiPPO)

**Figure 3 Description**: The overall architecture of HiPPO. It shows three major modeling stages:

1. **Phone-level Modeling**: Extracts features from the speech signal and uses a Phone Encoder (Conv-LLaMA blocks) to predict phone-level accuracy.
2. **Word-level Modeling**: Pools phone-level representations and word-level textual embeddings into a Word Encoder to predict word-level scores (Accuracy, Stress, Total).
3. **Utterance-level Modeling**: Aggregates phone and word features, combines them with utterance-level SSL features, and uses an Utterance Encoder to predict utterance-level scores (Accuracy, Fluency, Completeness, Prosody, Total).

Figure 3 depicts the model architecture of HiPPO, which encompasses three major modeling stages: phone-, word-, and utterance-level modeling. In each of these modeling stages, the corresponding encoder is constructed with the newly proposed Conv-LLaMA block. After obtaining the representations of all pronunciation aspects, a distinct regressor is used to generate the pronunciation score of each aspect.

#### Pronunciation Feature Extraction

To portray the pronunciation quality of $X$, HiPPO extracts connectionist temporal classification (CTC)-based goodness pronunciation (GOP) features for each phone in **p**, where the pronunciation quality is measured as the likelihood ratio of all valid CTC alignments of **p** to that of the deviated phonetic transcripts (Cao et al., 2024). Compared to previous studies on GOP feature extraction (Witt and Young, 2000; Hu et al., 2015; Shen et al., 2021), the CTC-based method computes GOP scores without explicit timestamps of phone segments and inherently tackles alignment errors by accounting for insertions and/or deletions in the deviated phonetic transcriptions.

Additionally, to capture supra-segmental articulation cues and mitigate the data-sparsity issue frequently occurring in L2 speech corpora (Lo et al., 2024; Bannò and Matassoni, 2022), HiPPO leverages self-supervised learning (SSL)-based features for utterance-level pronunciation modeling. The SSL-based features are extracted at the frame-level and then aggregated to the utterance-level via simple mean pooling over time (Chao et al., 2022; Kim et al., 2022).

The pronunciation feature extraction of HiPPO produces a phone-level pronunciation feature sequence $X^p \in \mathbb{R}^{d_p \times N}$ and a projected SSL-based feature vector $\mathbf{x}^{ssl} \in \mathbb{R}^{d_u \times 1}$, where $N$ is the length of the phone sequence, and $d_p$ and $d_u$ represent the hidden dimension of phone- and utterance-level modeling. The processing flow is summarized as follows:

$$ X^p = \text{Lin}_p(E^{gop}) \tag{1} $$
$$ \mathbf{x}^{ssl} = \text{Lin}_{ssl}([e^{w2v}; e^{hu}; e^{wlm}]) \tag{2} $$

where $\text{Lin}_p(\cdot)$ and $\text{Lin}_{ssl}(\cdot)$ are linear projections, and $[;]$ is a concatenation operation. $E^{gop} \in \mathbb{R}^{41 \times N}$ refers to the CTC-based GOP features extracted from a well-trained CTC-based ASR model[^3], while $e^{w2v}, e^{hu}, e^{wlm} \in \mathbb{R}^{1024 \times 1}$ are utterance-level SSL-feature vectors derived from pre-trained acoustic models, viz. wav2vec-2.0, HuBERT, and WavLM, respectively.

[^3]: <https://github.com/frank613/CTC-based-GOP.git>

#### Convolution-augmented LLaMA Block (Conv-LLaMA)

To model pronunciation feature sequences of arbitrary length and capture nuanced articulation traits across linguistic units, we introduce the Conv-LLaMA block to stack a hierarchical assessment model, which enhances the model component of LLaMA (Touvron et al., 2023) with a convolutional branch and rotary position encoding.

**Figure 4 Description**: Schematic illustration of the Conv-LLaMA block. It comprises two branches:

- One branch captures supra-segmental articulation cues via a multi-head self-attention (MHSA) module followed by a SwiGLU linear unit (SwiGLU operation from Touvron et al., 2023). The MHSA module incorporates rotary position encoding (RoPE), a relative position encoding method developed for extrapolating feature sequence lengths, which operates through **channel-wise multiplication on the key and query vectors** (Su et al., 2024).
- The other branch captures local pronunciation traits via a convolutional neural network (CNN) module, equipped with two key components: a **point-wise convolution** for capturing information across feature dimensions and a **depth-wise convolution** layer for extracting local spatial patterns.

The two branches are combined via a weighted average operation (Peng et al., 2022).

#### Hierarchical APA Modeling

**Phone-level**: $E^p$ is generated by passing phonetic transcription **p** into a phone embedding layer, and $\text{PhnEnc}(\cdot)$ is a stack of **3 Conv-LLaMA blocks**.

$$H_0^p = X^p + E^p \tag{3}$$
$$H^p = \text{PhnEnc}(H_0^p) \tag{4}$$

Subsequently, a regressor is built on top of $H^p$ to produce phone-level accuracy scores.

**Word-level**: For word-level assessments, a word representation vector is derived from its constituent phones with a dedicated attention pooling, implemented with a **1-D depth-wise convolution layer, an MHA layer, and an average operation**. The word-level input features $X^w \in \mathbb{R}^{d_w \times M}$ are obtained by feeding $X^p$ and $H^p$ through word-level attention pooling, and then packing their pooled counterparts together via a linear projection[^4]:

$$\tilde{X}^w = \text{AttPool}_{w_1}(X^p) \tag{5}$$
$$\tilde{H}^w = \text{AttPool}_{w_2}(H^p) \tag{6}$$
$$X^w = \text{Lin}_w([\tilde{X}^w; \tilde{H}^w]) \tag{7}$$

Following the integration of word-level textual embeddings $E^w$ with $X^w$, a word encoder is employed to generate a sequence of contextualized representations $H^w \in \mathbb{R}^{d_w \times M}$, where $E^w$ are obtained by mapping the transcribed word sequence **w** through **modernBERT** (Warner et al., 2024), and $\text{WordEnc}(\cdot)$ consists of **2 Conv-LLaMA blocks**:

$$H_0^w = X^w + E^w \tag{8}$$
$$H^w = \text{WordEnc}(H_0^w) \tag{9}$$

Consequently, **three distinct 1-D depth-wise convolution layers** are performed on top of $H^w$ to generate aspect representations (viz. $H^{w_1}$, $H^{w_2}$, and $H^{w_3}$). The word-level pronunciation scores (accuracy, stress, and total) are generated by passing the aspect representations into the corresponding regressors.

[^4]: For efficient parallel computation, a word-level representation is duplicated to length of constituent phones.

**Utterance-level**: For the utterance-level assessments, we first fuse $H^{w_1}$, $H^{w_2}$, and $H^{w_3}$ with a weighted average operation to produce $\bar{H}^w \in \mathbb{R}^{d_w \times M}$. After the distinct forward propagation through 1-D depth-wise convolution layers on $X^p$, $H^p$, and $\bar{H}^w$, the corresponding outputs are combined via a linear projection, and then fed into an utterance encoder — $\text{UttEnc}(\cdot)$ is a **single Conv-LLaMA block** — to generate contextualized representations $H^u$:

$$\bar{H}^w = \text{Merge}(H^{w_1}, H^{w_2}, H^{w_3}) \tag{10}$$
$$H_0^u = \text{Lin}_u([\text{DC}_1(X^p); \text{DC}_2(H^p); \text{DC}_3(\bar{H}^w)]) \tag{11}$$
$$H^u = \text{UttEnc}(H_0^u) \tag{12}$$

where $\text{DC}_1(\cdot)$, $\text{DC}_2(\cdot)$, and $\text{DC}_3(\cdot)$ are distinct 1-D depth-wise convolution layers, each with a **kernel size of 3**. Afterward, **five separate attention pooling layers** are stacked on top of $H^u$ and then integrated with the projected SSL-based feature vector $\mathbf{x}^{ssl}$ via separate residual connections. These aspect representation vectors are processed by the corresponding regressors to derive the utterance-level aspect scores (viz. accuracy, fluency, completeness, prosody, and total).

### 2.3 Training Objectives

The primary objective is a weighted sum of mean squared error (MSE) losses:

$$\mathcal{L}_{APA} = \sum_{g \in G} \lambda_g \times \frac{1}{N_g} \sum_{k=0}^{N_g-1} \mathcal{L}_{gk} \tag{13}$$

where $\lambda_g$ denotes adjustable parameter, $N_g$ is number of aspects at granularity $g$, and $\mathcal{L}_{g_k}$ represents the MSE loss computed for the $k$-th aspect score sequence.

#### Contrastive Ordinal Regularizer (CONO)

To mitigate the detrimental effects of ASR errors on assessment performance, we devise a contrastive ordinal (CONO) regularizer to extract score-discriminative features. As phone-level representations are essential for constructing a hierarchical assessment model, we first extract an utterance-level feature **z** by **averaging the outputs of the phone-level encoder $H^p$ over time**. For a training batch of $L$ utterances, the corresponding feature vectors are aggregated to form a sequence $Z = (\mathbf{z}_1, \mathbf{z}_2, ..., \mathbf{z}_L)$.

**Figure 5 Description**: Illustration of the CONO regularizer. It encourages feature vectors $Z$ to render the ordinal relationship of utterance-level accuracy scores $\mathbf{y} = (y_1, y_2, ..., y_L)$ via the synergy of a diversity term $\mathcal{L}_d$ and a tightness term $\mathcal{L}_t$.

$$\mathcal{L}_{CONO} = \lambda_d \mathcal{L}_d + \lambda_t \mathcal{L}_t \tag{14}$$

where $\lambda_d$ and $\lambda_t$ are trade-off parameters. The **diversity term** $\mathcal{L}_d$ preserves inter-score discrepancies by minimizing the negative distances between score centroid vectors $\mathbf{z}_{c_i}$ with a penalty:

$$\mathcal{L}_d = -\frac{1}{M(M-1)} \sum_{i=1}^K \sum_{i \neq j} w_{ij} \|\mathbf{z}_{c_i} - \mathbf{z}_{c_j}\|_2^2 \tag{15}$$

where $K$ is the number of score centers, and penalty $w_{ij} = |y_i - y_j|$ signifies the absolute differences between the regression targets. The score centroid vectors $\mathbf{z}_{c_i}$ and $\mathbf{z}_{c_j}$ are computed from $Z$ by averaging all feature vectors whose utterance-level accuracy scores are $y_i$ and $y_j$, respectively.

The **tightness term** $\mathcal{L}_t$ regulates intra-score compactness by pulling feature representations $\mathbf{z}_i$ towards their score centroid vectors $\mathbf{z}_{c_{y_i}}$:

$$\mathcal{L}_t = \frac{1}{L} \sum_{i=1}^L \|\mathbf{z}_i - \mathbf{z}_{c_{y_i}}\|_2^2 \tag{16}$$

The total training objective of HiPPO is a linear combination of the pronunciation assessment task $\mathcal{L}_{APA}$ and the CONO regularization $\mathcal{L}_{CONO}$:

$$\mathcal{L} = \mathcal{L}_{APA} + \lambda_{CONO} \mathcal{L}_{CONO} \tag{17}$$

where $\lambda_{CONO}$ is a tunable hyperparameter.

### 2.4 Curriculum Learning

Drawing inspiration from education systems, curriculum learning techniques improve model performance by progressively escalating training complexity from simple to hard (Bengio et al., 2009; Castells et al., 2020; Vakil and Amiri, 2023). The proposed curriculum training strategy starts from assessing pronunciation in a reading-aloud scenario $\mathcal{L}_{read}$, and gradually shifts towards assessing pronunciation in the free-speaking counterpart $\mathcal{L}_{free}$.

In $\mathcal{L}_{read}$, the pronunciation features are extracted from the learner's speech alongside the corresponding reference text, while in $\mathcal{L}_{free}$ the transcribed word sequence serve as an alternative for pronunciation feature extraction. At each training iteration $\tau$, HiPPO selects a task from $\mathcal{L}_{read}$ with a probability of $1 - \mathcal{P}(\tau)$, or from $\mathcal{L}_{free}$ with a probability of $\mathcal{P}(\tau)$, where $\mathcal{P}(\tau) = \tau/T$ is a scheduling function, with $T$ being the total number of training iterations and $\tau \in [0, T]$. The training strategy at iteration $\tau$ is defined by:

$$(1 - \mathbb{I}(\tau))\mathcal{L}_{read} + \mathbb{I}(\tau)\mathcal{L}_{free} \tag{18}$$

with the indicator function $\mathbb{I}(\tau)$ given by:

$$\mathbb{I}(\tau) = \begin{cases} 1, & \text{learning hard task (w.p. } \mathcal{P}(\tau)) \\ 0, & \text{learning easy task (w.p. } 1 - \mathcal{P}(\tau)) \end{cases} \tag{19}$$

## 3 Experimental Settings

This section describes the benchmark dataset and metrics used in this paper. Implementation details and descriptions of comparative methods are elaborated in Appendices B and C. Furthermore, HiPPO and the experimental dataset are publicly available[^5] to ensure the reproducibility of our work, accelerate CAPT research, and facilitate standardized evaluation.

[^5]: <https://github.com/bicheng1225/HIPPO/tree/main>

**Benchmark Dataset.** A series of experiments were carried out on the **Speechocean762** dataset (Zhang et al., 2021), a publicly available corpus specifically designed for CAPT research. This dataset comprises **5,000 English-speaking recordings collected from 250 Mandarin L2 learners**, with training and test sets of equal size, **each containing 2,500 utterances**. Speechocean762 was collected in a reading-aloud scenario (reading reference texts aloud) with accessible reference texts and corresponding canonical phones (phone-level reference text).

To simulate a free-speaking scenario for possible use cases of spoken language assessment, we **exclude these reference texts from the model input** and rely instead on the ASR transcribed words and their associated phones (using **Whisper-large-v3**). The WER is **19.22% for the training set** and **17.49% for the test set**. The detailed pronunciation score assignments for the free-speaking scenario are provided in Appendix D.

**Evaluation Metrics.** 1) Pearson correlation coefficient (PCC, $\uparrow$) measures the linear correlation between predicted and ground-truth scores for disparate pronunciation aspects. 2) Mean squared error (MSE, $\downarrow$) evaluates score discrepancy of the phone-level accuracy. The mean and standard deviation are reported for both metrics.

## 4 Experimental Results

**Assessments in the Free-speaking Scenarios.** At the outset, we compare our HiPPO with several current top-of-the-line APA models in the simulated free-speaking scenarios. From the results shown in Table 1, we make the following observations. 1) Our HiPPO achieves better PCC scores than all other competitive methods across different pronunciation aspects and linguistic granularities. 2) As to the ASR-free methods, both VanillaSSL and Liu2023 are limited to utterance-level assessment, lacking finer-grained aspect scores at the phone or word level. Moreover, Liu2023 outperforms VanillaSSL in assessing the utterance-level fluency, where the gains stem from the integration of frame-level phonetic information via *k*-means clustering. Note also that effectively using phonetic information to boost assessment performance has been verified in prior work (Gong et al., 2022). Subsequently, compared to MultiPA, our method extracts pronunciation feature at the phone-level and then qualifies pronunciation aspects hierarchically across linguistic granularities, resulting in superior assessment performance. 3) In comparison among the variants of HiPPO, Parallel-CTC and Parallel-LLaMA outperform Hier-LLaMA in most assessment tasks. This observation suggests that, when pronunciation features are extracted from the transcripts containing ASR errors, the parallel design offers a more flexible and robust neural architecture for assessments in free-speaking scenarios compared to the hierarchical one. Notably, HiPPO stands out in assessment performance via the synergy of the CONO regularizer and curriculum learning strategy.

**Table 1**: Performance evaluations on Speechocean762 test set in simulated free-speaking scenarios.

| Models | Phone MSE↓ | Phone PCC↑ | Word Acc↑ | Word Total↑ | Utt Acc↑ | Utt Flu↑ | Utt Pro↑ | Utt Total↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Liu2023 | - | - | - | - | - | 0.795 | - | - |
| VanillaSSL | - | - | - | - | 0.692 (±0.006) | 0.757 (±0.010) | 0.757 (±0.009) | 0.714 (±0.006) |
| MultiPA | - | - | 0.427 (±0.008) | 0.436 (±0.010) | 0.705 (±0.009) | 0.772 (±0.010) | 0.763 (±0.016) | 0.730 (±0.006) |
| Parallel-TFR | 0.240 (±0.003) | 0.330 (±0.009) | 0.416 (±0.016) | 0.417 (±0.019) | 0.717 (±0.014) | 0.797 (±0.003) | 0.791 (±0.003) | 0.741 (±0.010) |
| Parallel-LLaMA | 0.237 (±0.001) | 0.345 (±0.004) | 0.426 (±0.012) | 0.428 (±0.011) | 0.726 (±0.006) | 0.799 (±0.006) | 0.791 (±0.005) | 0.748 (±0.004) |
| Hier-LLaMA | 0.238 (±0.001) | 0.328 (±0.008) | 0.412 (±0.011) | 0.418 (±0.012) | 0.692 (±0.012) | 0.786 (±0.008) | 0.780 (±0.006) | 0.724 (±0.008) |
| **HiPPO** | **0.202** (±0.003) | **0.480** (±0.013) | **0.520** (±0.016) | **0.521** (±0.016) | **0.733** (±0.006) | **0.806** (±0.003) | **0.797** (±0.002) | **0.754** (±0.006) |
| w/o CONO | 0.213 (±0.004) | 0.448 (±0.012) | 0.513 (±0.007) | 0.516 (±0.007) | 0.720 (±0.005) | 0.797 (±0.003) | 0.791 (±0.002) | 0.743 (±0.005) |
| w/o CL | 0.241 (±0.002) | 0.331 (±0.011) | 0.404 (±0.012) | 0.404 (±0.014) | 0.698 (±0.010) | 0.790 (±0.011) | 0.785 (±0.011) | 0.728 (±0.007) |

**Ablation Studies in Free-speaking Scenarios.**

As shown in the last two columns of Table 1, we ablate HiPPO with following settings: removing the CONO regularizer (w/o CONO) and substituting the curriculum learning strategy with training on a combined dataset of reading-aloud and free-speaking scenarios (w/o CL). From these ablation studies we can observe that both the CONO regularizer and the curriculum strategy are crucial to HiPPO. Removing either one of them leads to a decline in performance across several aspects and granularities. Second, the curriculum learning strategy makes a substantial contribution to the performance. Training HiPPO with the combined dataset, in contrast, results in lower performance across all assessment tasks.

**Qualitative Analysis on the CONO Regularizer in the Free-speaking Scenarios.**

**Figure 6 Description**: Visualization of utterance-level representations $Z$, where the orange, blue, and green points indicate accuracy scores of 4.0, 6.0, and 8.0, respectively. The plots display feature points for: (a) vanilla model, (b) vanilla model with a modified diversity term $\mathcal{L}_{d'}$ where the penalty is removed, (c) vanilla model with diversity term $\mathcal{L}_d$, and (d) vanilla model with CONO regularizer $\mathcal{L}_{CONO}$.

In Figure 6, the feature points display ordinal relationships, which are sorted by their utterance-level scores, with blue points being located between red points and green points. This result can be attributed to the aggregation of representations $Z$ from the phone-level representations, which are highly correlated with the utterance-level accuracy score (Yan et al., 2024). By comparing Figures 6(b) with 6(c), it is evident that both diversity terms ($\mathcal{L}_{d'}$ and $\mathcal{L}_d$) can capture subtle differences between utterance-level scores, where feature points are clustered by their respective accuracy scores. The integration of ordinal penalty, as shown in Figure 6(c), further facilitates a clearer scattering of feature representations, with blue and green points more distinctly spread out. Finally, the impact of the tightness term $\mathcal{L}_t$ is verified in Figure 6(d), where the feature points exhibit tighter clustering in comparison with other subfigures.

**Effectiveness of CONO Regularizer across Different ASR Word Error Rate Settings.**

**Figure 7 Description**: A comparison of PCC scores for pronunciation accuracy at the phone, word, and utterance levels between HiPPO and HiPPO w/o $\mathcal{L}_{CONO}$ under varying word error rate (WER) conditions. These WERs are calculated based on the reference text and different input transcriptions which are reference text and outputs of Whisper models (viz., large-v3, medium-en, small-en).

Figure 7 examines the effectiveness of CONO regularizer $\mathcal{L}_{CONO}$ for the assessment accuracy at different granularities across various ASR word error rates (WERs), by comparing the HiPPO and its ablated version (HiPPO w/o CONO). Notably, in this set of experiments, our models were trained on the reference text and transcripts generated by whisper-large-v3 (achieving a WER of 19.6%) via proposed curricular learning strategy. First, with reference text as the input transcript, the assessment performance of both models seems comparable across granularities (phone, word, and utterance levels). Second, at the utterance-level assessment, the PCC scores of these two models appear relatively immune to WER degradation. A possible reason is that utilization of SSL-based features in utterance-level modeling, as the SSL models are often pre-trained on complex acoustic environments. Finally, the benefits of the CONO regularizer are more prominent at finer-grained linguistic levels. Specifically, the performance degrades substantially at the phone and word levels; however, the performance of HiPPO exhibits a more attenuated decline in comparison to other variants, which highlights the robustness of the proposed regularizer to ASR errors.

**Table 2**: Performance evaluations on Speechocean762 test set in the read-aloud scenario. HiPPO\* refers to the model trained without curricular strategy and CONO regularizer.

| Models | Phone MSE↓ | Phone PCC↑ | Word Acc↑ | Word Total↑ | Utt Acc↑ | Utt Flu↑ | Utt Pro↑ | Utt Total↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| AzurePA | - | - | 0.623 | - | 0.700 | 0.715 | **0.842** | 0.782 |
| GOPT | 0.085 (±0.001) | 0.612 (±0.003) | 0.533 (±0.004) | 0.549 (±0.002) | 0.714 (±0.004) | 0.753 (±0.008) | 0.760 (±0.006) | 0.742 (±0.005) |
| 3M | **0.078** (±0.001) | 0.656 (±0.005) | 0.598 (±0.005) | 0.617 (±0.002) | 0.760 (±0.008) | 0.828 (±0.006) | 0.827 (±0.008) | 0.796 (±0.002) |
| HiPAMA | 0.084 (±0.001) | 0.616 (±0.004) | 0.575 (±0.004) | 0.591 (±0.004) | 0.730 (±0.002) | 0.749 (±0.001) | 0.751 (±0.002) | 0.754 (±0.002) |
| HierTFR | 0.081 (±0.000) | 0.644 (±0.000) | 0.622 (±0.002) | 0.634 (±0.002) | 0.735 (±0.008) | 0.801 (±0.004) | 0.795 (±0.002) | 0.764 (±0.002) |
| Parallel-TFR | 0.078 (±0.001) | 0.650 (±0.009) | 0.575 (±0.018) | 0.589 (±0.013) | 0.754 (±0.011) | 0.816 (±0.006) | 0.806 (±0.007) | 0.772 (±0.010) |
| Parallel-LLaMA | 0.074 (±0.002) | 0.658 (±0.007) | 0.598 (±0.012) | 0.610 (±0.009) | 0.774 (±0.009) | 0.837 (±0.006) | 0.829 (±0.004) | 0.796 (±0.009) |
| Hier-LLaMA | 0.082 (±0.002) | 0.656 (±0.006) | 0.622 (±0.006) | 0.634 (±0.008) | 0.789 (±0.006) | 0.844 (±0.003) | 0.832 (±0.003) | 0.811 (±0.005) |
| **HiPPO\*** | 0.080 (±0.001) | **0.657** (±0.001) | **0.630** (±0.009) | **0.643** (±0.009) | **0.791** (±0.002) | **0.845** (±0.001) | 0.837 (±0.001) | **0.816** (±0.001) |

**Assessments in the Read-aloud Scenario.** In Table 2, the proposed HiPPO is evaluated in a read-aloud setting, where reference texts are employed in training and test. The main findings are presented as follows. 1) HiPPO markedly outperforms other methods in most pronunciation aspects. Notably, in contrast to prior studies, i.e., parallel models (GOPT and 3M) and hierarchical ones (HiPAMA and HierTFR), our model assesses pronunciation quality without explicit phone-level timestamps and achieves superior performance across various pronunciation aspects. 2) AzurePA stands out at the assessment of utterance-level prosody, whereas its performance on the other pronunciation aspects trails behind that of the other methods. These inferior results probably stem from that AzurePA is a commercial system that might has not been finetuned on Speechocean762. 3) As to the comparison between the variants of HiPPO (Parallel-LLaMA, Parallel-TFR, and Hier-LLaMA), Hier-LLaMA attains superior performance in most pronunciation aspects, particularly at the word and utterance levels, with a slight sacrifice in performance at the phone-level. These results are in line with the findings from previous studies (Do et al., 2023; Chao et al., 2023). By comparing HiPPO with Hier-LLaMA, we can verify that the proposed Conv-LLaMA block brings consistent improvements to pronunciation assessments.

## 5 Conclusion

In this paper, we have proposed a novel hierarchical pronunciation assessment model (dubbed HiPPO) for the spoken languages. To address arbitrarily long pronunciation feature sequences and capturing articulation traits across various linguistic granularities, we designed a Conv-LLaMA block for the proposed model. A contrastive ordinal regularizer is put forward to enhance robustness against ASR errors. Moreover, we explored a simple yet effective curriculum learning strategy for the spoken language assessment. Extensive experimental results validate the feasibility and effectiveness of the proposed methods, obtaining superior assessment performance compared to several state-of-the-art methods in both reading-aloud and stimulated free-speaking scenarios. In future work, we plan to explore more robust assessment models under various word error rate conditions for unscripted pronunciation assessments.

## 6 Limitations

Spoken language assessment gauges language competence across three sub-dimensions: pronunciation (fluency and delivery), language use (vocabulary and grammar), and topic development (content and discourse). In this paper, however, HiPPO focuses exclusively on pronunciation assessment within the broader context of spoken language evaluation. The following are several limitations of HiPPO in real-world applications:

**Transcriptions Containing ASR Errors.** Although speech foundation models have achieved near-human accuracy on public benchmark datasets, transcribing non-native English speech remains challenging. In our experiments, the word error rate (WER) for Speechocean762, transcribed using Whisper-large-v3, is 19.22% for the training set and 17.49% for the test set. Examining the performance of HiPPO through the lens of different WER conditions, we observed a significant degradation when ASR errors were severe, even with the proposed CONO regularizer.

**Lack of Accent Diversity.** The used dataset merely contains Mandarin L2 learners, hindering the generalizability of the proposed model and could be untenable when assessing the L2 learners with diverse accents.

**The Lack of Interpretability.** The model of the proposed method simply trains to mimic expert's annotations without resorting to manual assessment rubrics or other external knowledge, making it not straightforward to provide reasonable explanations for the assessment performance.

## Ethics Statement

We hereby acknowledge that all of the co-authors of this work compile with the provided ACL Code of Ethics and honor the code of conduct. Our experimental corpus, Speechocean762, is widely used and publicly available. We think there are no potential risks for this work.

---

## Appendices

### Appendix A: Related Work

**Scripted-speech Assessment**: Prior work includes GOPT (Gong et al., 2022), 3M (Chao et al., 2022), and HiPAMA (Do et al., 2023), which rely on reference texts and phone-level timestamps for assessment. HierTFR represents a hierarchical transformer approach for read-aloud scenarios.

**Unscripted-speech Assessment**: The emerging field includes MultiPA (Chen et al., 2024), a multi-task model for open response scenarios, and Liu2023, which leverages frame-level phonetic information via *k*-means clustering for utterance-level fluency assessment.

### Appendix B: Implementation Details

- **Optimizer**: Adam (LR = 0.001, Batch size = 25)
- **Model Config**: 1 head, 24 hidden units for Conv-LLaMA blocks
- **Training**: 5 independent trials; best-performing epochs selected

### Appendix C: Comparative Methods

Lists categories of models:

- **ASR-free methods**: VanillaSSL — uses only SSL features without ASR transcription
- **ASR-based methods**: MultiPA — multi-task model using ASR transcriptions
- **HiPPO variants**: Parallel-TFR (standard Transformer backbone), Parallel-LLaMA (LLaMA backbone, parallel architecture), Hier-LLaMA (LLaMA backbone, hierarchical architecture)

### Appendix D: Score Assignments for Speechocean762 in Free-speaking

**Figure 8 Description**: Illustration of the score assignment process. It shows how ASR transcriptions are aligned to reference texts to handle deletion, substitution, and insertion errors during simulated free-speaking evaluation.
