---
title: "MuFFIN: Multifaceted Pronunciation Feedback Model with Interactive Hierarchical Neural Modeling"
authors:
  - "Bi-Cheng Yan"
  - "Ming-Kang Tsai"
  - "Berlin Chen"
citation_author: "Yan et al."
year: 2025
doi: null
pages: 15
source_pdf: "paper.pdf"
extraction_method: "full manual read of local paper.pdf, paragraph-by-paragraph"
extracted_at: "2026-03-22"
llm_friendly: true
---

# MuFFIN: Multifaceted Pronunciation Feedback Model with Interactive Hierarchical Neural Modeling

**Authors**: Bi-Cheng Yan, Ming-Kang Tsai, Berlin Chen
**Year**: 2025
**Venue**: IEEE (pre-publication manuscript draft; manuscript-ID placeholder still present)
**DOI**: Not yet assigned

---

## TL;DR

MuFFIN is current SOTA on SpeechOcean762 for phone-level pronunciation accuracy (PCC **0.742**). It jointly solves automatic pronunciation assessment (APA) and mispronunciation detection and diagnosis (MDD) in one hierarchical model. Three contributions: (1) convolution-augmented Branchformer hierarchy at phone→word→utterance levels, (2) contrastive phonemic ordinal regularizer (ConPCO) that aligns speech and text phoneme embeddings while preserving score ordinality, (3) phoneme-specific variation (PhnVar) that addresses MDD class imbalance via both class frequency and mispronunciation-rate factors.

---

## Abstract

Computer-assisted pronunciation training (CAPT) manages to facilitate second-language (L2) learners to practice pronunciation skills by offering timely and instructive feedback. To examine pronunciation proficiency from multiple facets, existing methods for CAPT broadly fall into two categories: mispronunciation detection and diagnosis (MDD) as well as automatic pronunciation assessment (APA). The former aims to pinpoint phonetic pronunciation errors and provide diagnostic feedback, while the latter seeks instead to quantify pronunciation proficiency pertaining to various aspects. Despite the natural complementarity between MDD and APA, researchers and practitioners, however, often treat them as independent tasks with disparate modeling paradigms. In light of this, the paper first introduces MuFFIN, a **Mu**lti-**F**aceted pronunciation **F**eedback model with an **I**nteractive hierarchical **N**eural architecture, to jointly address the tasks of MDD and APA. To better capture the nuanced distinctions between phonemes in the feature space, a novel phoneme-contrastive ordinal regularization mechanism is then put forward to optimize the proposed model to generate more phoneme-discriminative features while factoring in the ordinality of the aspect scores. In addition, to address the intricate data imbalance problem in MDD, a simple yet effective training objective is designed which is specifically tailored to perturb the outputs of a phoneme classifier with phoneme-specific variations, so as to better render the distribution of predicted phonemes while considering their mispronunciation characteristics. A series of experiments conducted on the Speechocean762 benchmark dataset demonstrates the efficacy of the method in relation to several cutting-edge baselines, showing state-of-the-art performance on both the APA and MDD tasks.

---

## I. Introduction

Fueled by the amplified demand for foreign language acquisition, research on computer-assisted pronunciation training (CAPT) has aroused significant attention amidst the tide of globalization, figuring prominently in the field of computer-assisted language learning (CALL) [1][2]. To bridge the gap between insufficient supplies and pressing needs from language teachers and learners, CAPT systems have emerged as appealing learning tools ubiquitously, shifting the conventional pedagogy from teacher-led to self-directed learning. Beyond their critical roles in education and language learning, CAPT systems also serve as a handy reference for professionals (e.g., interviewers and examiners) in high-stakes assessments, with the goals of reducing the workload [3][4], alleviating the burdens of recruiting new human experts, and achieving consistent and objective assessment results [5][6][7].

A de-facto archetype system for CAPT is normally instantiated in a read-aloud scenario, where an L2 learner is provided with a reference text and instructed to pronounce it correctly. By taking the learner's speech paired with the reference text as input, CAPT systems are anticipated to assess the learner's oral competence from multiple facets, providing detailed and potentially diagnostic performance feedback with a near-instant turnaround. To this end, mispronunciation detection and diagnosis (MDD) and automatic pronunciation assessment (APA) are two active strands of research in developing pronunciation feedback modules for CAPT. The former seeks to pinpoint phonetic pronunciation errors and provides L2 learners with the corresponding diagnostic feedback [8][9]. The latter, in contrast, concentrates more on assessing the learner's pronunciation quality through multi-faceted pronunciation scores, reflecting his/her proficiency pertaining to specific aspects or some extent of spoken language usage [10][11]. One time-tested approach for MDD is goodness of pronunciation (GOP) and its derivatives [12][13], which calculate the ratio between the likelihoods of the canonical and most likely pronounced phonemes. Phoneme-level erroneous pronunciations are subsequently detected if the likelihood ratios of certain phoneme segments fall below predetermined thresholds. On a separate front, the models of iconic APA methods are typically trained to mimic human ratings based on surface features (viz. a set of hand-crafted features). These models either employ a classifier to predict a holistic score representing learners' oral proficiency [10] or use regressors to estimate continuous analytic scores for specific pronunciation aspects, such as phoneme-level accuracy [14], word-level lexical stress [15], and utterance-level pronunciation quality [16][17].

In spite of the complementary nature of MDD and APA, most existing efforts treat them as independent tasks, thereby developing two disparate feedback modules for use in CAPT. However, some prior studies reveal that an L2 English learner tends to have lower utterance-level assessment scores of intelligibility and fluency [18] whenever his or her utterances frequently contain phoneme-level pronunciation errors [19][20]. In the view of this, the paper first proposes a novel CAPT modeling paradigm, dubbed MuFFIN, which is a Multi-Faceted pronunciation Feedback model with an Interactive hierarchical Neural structure. MuFFIN unifies the individual feedback modules of MDD and APA into a streamlined, hierarchical neural architecture through a multi-task learning scheme. Building on a language hierarchy-aware neural architecture with the tailor-made convolution-augmented Branchformer blocks, MuFFIN can effectively capture interactions across the linguistic granularities (i.e., phoneme, word, and utterance) and preserve fine-grained articulatory cues at different linguistic units. Next, to render the subtle differences between phonemes in the feature space, a novel phoneme-contrastive ordinal regularizer (ConPCO) is introduced to facilitate the proposed model in generating more phoneme-discriminative features. This training regime leverages contrastive learning to better align the phoneme representations of a scoring model with the textual embeddings of their corresponding canonical phonemes, while also accounting for the ordinal relationships among the regression targets (i.e., phoneme-level accuracy scores). Furthermore, a simple yet effective training objective, phoneme-specific variation, is explored to ease the data imbalance problem incurred by MDD [21]. Data imbalance is a long-standing problem in MDD, where phoneme distributions are often skewed between correct and incorrect pronunciation instances.

The paper summarizes at least four contributions:

1. MuFFIN is presented: a multi-faceted pronunciation feedback model that jointly addresses the tasks of MDD and APA through an interactive hierarchical neural framework. This model signifies a paradigm shift from separate modeling of APA and MDD to a unified assessment approach, opening up a new avenue in CAPT.
2. A contrastive phonemic ordinal regularizer (ConPCO) is proposed to align the speech-derived phoneme representations with the corresponding phoneme-level textual embeddings, while organically engaging the ordinality of pronunciation accuracy scores.
3. To the best of the authors' knowledge, this is the first attempt to address data imbalance issues in MDD by incorporating phoneme-specific variations into the training process. The method highlights that the data imbalance problem in MDD stems from two intertwined and equally crucial factors, viz. the quantity and the pronunciation difficulty of the training data.
4. Extensive sets of experiments carried out on the Speechocean762 benchmark dataset [26] confirm the effectiveness of the proposed methods, which improves the performance of state-of-the-art ones on both the APA and MDD tasks.

---

## II. Related Work

Computer-assisted pronunciation training (CAPT) is a subfield of computer-assisted language learning (CALL), whose research and development can trace back to pioneering efforts in the 1960s [27] and have gained significant attention recently due to the unprecedented advancements in speech and language technologies [28][29][30]. According to the diagnostic feedback of CAPT, research endeavors typically fall into phoneme-level mispronunciation detection and diagnosis (MDD) as well as automatic pronunciation assessment (APA), both mostly developed under read-aloud learning scenarios.

### A. Mispronunciation Detection and Diagnosis

MDD manages to detect erroneous pronunciation at phoneme segments, and in turn provide L2 learners with the corresponding diagnostic feedback [31][32]. Common approaches to MDD can be grouped into three categories: pronunciation scoring-based, dictation-based, and prompt-based methods.

Pronunciation scoring-based methods typically exploit various types of confidence measurements to evaluate pronunciation quality via a well-trained ASR system (e.g., hybrid DNN-HMM ASR system). Frequently-used measurements include, but are not limited to, phoneme durations [33][34], likelihood ratios [13], phoneme posterior probabilities [35], and their combinations [36]. Given an input utterance and its corresponding canonical phoneme sequence (viz. phoneme-level text prompt), pronunciation scoring-based methods first gauge the pronunciation scores for each phoneme in the canonical phoneme sequence. Mispronounced phoneme segments are then detected when their scores fall below predetermined thresholds, signifying a deviation from the expected pronunciation. However, pronunciation scoring-based methods are untenable to provide diagnostic feedback for the detected mispronounced phoneme segments.

As a remedy, dictation-based methods strive to formulate MDD as a phoneme recognition task by employing a phoneme recognizer to dictate the most likely phoneme sequence uttered by an L2 learner. The erroneous pronunciation portions can be easily identified by comparing the dictation result with the corresponding canonical phoneme sequence. For instance, Leung et al. ventured into employing a CTC-based phoneme recognizer for L2 English learners, showing comparative performance with pronunciation scoring-based methods in the mispronunciation detection subtask, where the performance gains mainly contributed from the accurate diagnosis of mispronunciations in unvoiced phoneme segments [37]. Yan et al. exploited the hybrid CTC-Attention ASR model as the dictation model and sought to capture deviant (non-categorical) phoneme productions uttered by accented L2 learners with anti-phone modeling [38]. Both of the above-mentioned methods rely on precise alignments to identify mispronounced segments; however, in practical applications, alignment errors might arise when comparing the canonical phoneme sequence to accented or disfluent speech produced by L2 learners.

In response, prompt-based methods leverage an attention mechanism to derive the soft alignment between the canonical phoneme sequence and the learner's input speech in an end-to-end manner, offering a promising approach to reduce alignment errors. As one of the first attempts, PeppaNet aligns canonical phonemes with the learner's speech via a Transformer decoder, where any discrepancies are captured in the matching degree vectors through end-to-end neural modeling [39]. Among other things, MDDGCN introduces a graph-based prompt encoder for canonical phonemes, aiming to improve diagnosis accuracy by regularizing the relationships between canonical and actually pronounced phonemes through a pre-defined phonetic graph [40].

### B. Automatic Pronunciation Assessment

Automatic Pronunciation Assessment (APA) quantifies an L2 learner's pronunciation proficiency in a target language by providing either analytic scores (viz. continuous numerical values) for specific pronunciation aspects [41][42] or a holistic assessment (viz. discrete categorical values) to reflect overall spoken competence [10]. Early efforts in APA predominantly focused on single-aspect assessment, typically by constructing individual scoring modules to predict proficiency scores at specific linguistic levels with various sets of hand-crafted features. These hand-crafted features, extracted from the learner's input speech or its corresponding ASR-generated transcript, may include acoustic features, confidence scores of recognized linguistic units, time-alignment information, and statistical measures [43][44]. To scrutinize learners' pronunciation comprehensively, recent advances in APA have advocated multi-aspect and multi-granular pronunciation assessment, leveraging unified scoring models that evaluate pronunciation proficiency across multiple linguistic levels (viz. phoneme, word, and utterance) with diverse aspects (e.g., accuracy, fluency, and completeness).

Drawing on this research trend, Gong et al. proposed a parallel pronunciation modeling architecture dubbed GOPT, which took GOP features as input and adopted a Transformer encoder as the backbone model to jointly model multiple pronunciation aspects across various linguistic granularities [45]. Following this school of thought, 3M extended GOPT by augmenting the model's input embeddings with prosodic features and self-supervised learning (SSL)-based features, aiming to achieve multi-view, multi-granularity, and multi-aspect pronunciation modeling [46]. Despite their decent performance, the hierarchical structure of an utterance is largely set aside. To capture the language hierarchy of an utterance, Do et al. proposed a hierarchical APA model and explored a novel multi-trait attention layer to strengthen the connection between scoring aspects [47]. Chao et al. introduced sub-phoneme modeling and employed a depth-wise separable convolution layer to construct a hierarchical APA model, facilitating better modeling of local context cues at the sub-word level [48]. Apart from the above, Gradformer (GFR) leveraged a granularity-decoupled Transformer network that first separates the granularity of an utterance into lower-level (phoneme- and word-level) ones and higher-level (utterance-level) one. A Conformer encoder in turn jointly models pronunciation aspects at the lower levels, while a Transformer decoder processes a set of trainable aspect vectors and interacts with the encoder outputs for utterance-level pronunciation assessment [42]. 3MH, a previous state-of-the-art method for APA, employs 3M as the backbone model and introduces sup-phoneme modeling to capture finer articulation traits within the language hierarchy between the phoneme and word levels [48]. HierGAT devises a language-hierarchy-aware model with a series of graph attention neural networks and further strengthening the relatedness among the aspects with aspect attention mechanisms [58]. Ryu2023 introduces a unified model architecture that jointly optimizes both phoneme recognition and pronunciation assessments by independently stacking a CTC-based phoneme recognizer and a set of regressors on top of a pretrained acoustic model [59]. JAM advances 3M by integrating a phoneme classifier to predict diagnostic phonemes based on input canonical phonemes and further boosts the MDD performance by exploiting electromagnetic articulography (EMA) features to capture the articulatory movements of L2 learners [60].

---

## III. Multi-Faceted Pronunciation Feedback Model with an Interactive Hierarchical Neural Architecture

The overall architecture of the proposed MuFFIN is schematically depicted in Fig. 3(a), which contains three main components: phoneme-level modeling, word-level modeling, and utterance-level modeling. The encoder at each different linguistic level adopts a novel convolution-augmented Branchformer block [25], as shown in Fig. 3(b), which consists of two branches with one branch designed to capture supra-segmental pronunciation cues with multi-head attention (MHA) layers while the other tailored to capture fine-grained pronunciation cues with a series of convolution layers. Furthermore, as illustrated in Fig. 4, a novel phoneme-level pronunciation feedback module is devised to assess phoneme-level accuracy and perform mispronunciation detection and diagnosis.

### A. Problem Formulation

Given an input utterance U, consisting of a time sequence of audio signals X produced by an L2 learner, and a reference text prompt T with M words, which is converted into N canonical phonemes based on a pronunciation dictionary (CMU dictionary), the proposed multi-faceted pronunciation feedback model aims to estimate proficiency scores at various linguistic granularities, while pinpointing phoneme-level pronunciation errors for the canonical phoneme sequence. Formally, let G = {p, w, u} be a set of linguistic granularities, where p, w, u stands for the phoneme-, word-, and utterance-level, respectively. For each granularity g ∈ G, the model aims to predict a set of aspect score sequences A^g = {a^g_1, a^g_2, ..., a^g_{Ng}}, where N_g is the number of pronunciation aspects at granularity g. In the meantime, for the canonical phoneme sequence **q** = (q_1, q_2, ..., q_N), the proposed model seeks to detect an error state sequence **e** = (e_1, e_2, ..., e_N) and generate a phonetic diagnosis sequence **y** = (y_1, y_2, ..., y_N). Both e_n and y_n are phoneme-level pronunciation feedback for q_n, where e_n = 1 denotes a mispronounced phoneme segment and e_n = 0 for a correct one, while y_n specifies the phoneme produced by the learner.

### B. Interactive Hierarchical Neural Modeling

**Phoneme-level Modeling.** For an input utterance, various pronunciation features are extracted to portray the pronunciation quality of the L2 learner at phoneme-level, which are then concatenated and projected to obtain a sequence of condensed acoustic features X^p. The feature extraction process is formulated as:

```text
X^p = Linear_p([E^GOP; E^Dur; E^Eng; E^SSL])         (1)
```

where Linear_p(·) is a single feedforward layer, E^GOP is goodness-of-pronunciation (GOP)-based features including log phoneme posterior (LPP) and log posterior ratio (LPR) [12][14], E^Dur and E^Eng are prosodic features related to duration and energy statistics [49][50], while E^SSL are self-supervised learning (SSL)-based features [46]. We then add phoneme-level textual embeddings E^p to X^p, followed by a phoneme encoder to obtain aspect representations H^p = (h^p_1, h^p_2, ..., h^p_N):

```text
H^p_0 = X^p + E^p        (2)
H^p   = PhnEnc(H^p_0)    (3)
```

Here, E^p is generated by passing **q** into a phoneme-level prompt encoder which comprises a phoneme and position embedding layer. PhnEnc(·) is composed of a stack of 3 convolution-augmented Branchformer blocks.

Afterward, the pronunciation feedback module builds on H^p to estimate the multi-faceted pronunciation feedback, comprising three components: an error detector, a diagnosis predictor, and an accuracy score regressor. The error detector is a binary labeling model which predicts the error state ê_n, indicating whether the n-th phoneme of **q** is identified as a mispronunciation:

```text
P_det(ê_n | q, X) = Sigmoid(Linear_det(h^p_n))       (4)
```

where Linear_det(·) is a linear layer followed by layer normalization. The diagnosis predictor performs a sequential multi-class labeling process to derive the probability distribution of diagnostic feedback for the n-th canonical phoneme as:

```text
P_diag(ŷ_n | q, X) = Softmax(Linear_diag(h^p_N))     (5)
```

where Linear_diag(·) used to convert hidden dimensions into the size of pronunciation dictionary. Finally, the phoneme-level accuracy score is estimated by an accuracy score regressor.

**Word-level Modeling.** For the word-level assessments, a word-level attention pooling is introduced to produce a word representation vector from its constituent phonemes, instantiated with a 1-D depth-wise convolution layer followed by an MHA layer and an average operation. The word-level input representations X^w are computed by individually passing X^p and H^p into the word-level attention pooling and subsequently packing them together with a linear projection:

```text
X̃^w, H̃^w = AttPool_{w1}(X^p), AttPool_{w2}(H^p)    (6)
X^w        = Linear_w([X̃^w; H̃^w])                    (7)
```

Next, the word-level textual embeddings E^w are added to X^w, and a word encoder is employed to generate word-level contextualized representations H^w:

```text
H^w_0 = X^w + E^w          (8)
H^w   = WordEnc(H^w_0)      (9)
```

where E^w are obtained by mapping the text prompt T into the corresponding embedding sequence via a word and position embedding layer, and WordEnc(·) consists of 2 convolution-augmented Branchformer blocks. Finally, three distinct 1-D depth-wise convolution layers are performed on H^w to generate word-level aspect representations (H^{w1}, H^{w2}, H^{w3}), which are then transformed into the pronunciation score sequences by the corresponding word-level regressors.

**Utterance-level Modeling.** For the utterance-level assessments, a frame-level SSL-based feature Ẽ^SSL is first extracted by applying average pooling over the time dimension of frame-level SSL-based features. Next, H^{w1}, H^{w2}, and H^{w3} are merged with a weighted combination to obtain word-level representations H̃^w [51]. A sequence of utterance-level input representations H^u_0 is obtained by first applying 1-D depth-wise convolution layers to X^p, H^p, and H̃^w, followed by concatenation and linear projection. Consequently, an utterance encoder is exploited to generate contextualized representations H^u:

```text
H̃^w  = Merge(H^{w1}, H^{w2}, H^{w3})                         (10)
H^u_0 = Linear_u([DC_1(X^p); DC_2(H^p); DC_3(H̃^w)])          (11)
H^u   = UttEnc(H^u_0)                                          (12)
```

where Merge(·) is a weighted average operation [51], UttEnc(·) is a single convolution-augmented Branchformer block, and DC_1(·), DC_2(·), DC_3(·) are distinct 1-D depth-wise convolution layers, each with a kernel size of 3. Afterward, five separate attention pooling modules are applied on top of H^u to generate utterance-level aspect representation vectors. These features are then combined with Ẽ^SSL via the residual connections and converted into utterance-level aspect scores through the respective regressors.

**Training Objective.** The training objective of MuFFIN is calculated from the losses of APA and MDD:

```text
L_MuFFIN = L_APA + L_MDD     (13)
```

The APA loss is a weighted sum of the mean square error (MSE) losses gathered from different granularity levels:

```text
L_APA = Σ_{jp} L_{p|p}/N_p  +  Σ_{jw} L_{w|w}/N_w  +  Σ_{ju} L_{u|u}/N_u     (14)
```

where L_{p|p}, L_{w|w}, and L_{u|u} are phoneme-level, word-level, and utterance-level losses for disparate aspects, and N_p, N_w, N_u mark the numbers of aspects at each granularity. On a separate front, the training objective of MDD comes with the tasks of mispronunciation detection L_det and diagnosis L_diag:

```text
L_MDD  = L_det + L_diag                                         (15)
L_det  = -Σ_{n=1}^{N} log P_det(ê_n = e_n | q, X)              (16)
L_diag = -Σ_{n=1}^{N} log P_diag(ŷ_n = y_n | q, X)             (17)
```

where L_det and L_diag represent the negative log-likelihood used for training the detector and the predictor, respectively.

---

## IV. Contrastive Phonemic Ordinal Regularizer (ConPCO)

To generate more phoneme-discriminative features for the multi-faceted pronunciation assessment model, the contrastive phonemic ordinal regularizer (ConPCO) is proposed, which consists of three mathematical terms: the contrastive term L_con, the phonemic characteristic term L_pc, and the ordinal term L_o. L_con aims to simultaneously project the phoneme representations generated from a pronunciation assessment model and the embeddings of phoneme-level text prompt into a joint feature space. L_pc and L_o adjust the distances between inter- and intra-phoneme categories, where the former enhances inter-phoneme discrepancy, and the latter improves intra-phoneme compactness with ordinal relationship. The proposed ConPCO regularizer is formulated as:

```text
L_ConPCO = L_con + L_pc + L_o     (18)
```

**Contrastive Term.** Let H^p = (h^p_1, h^p_2, ..., h^p_N) stand for the phoneme representation sequence of an utterance generated by a phoneme encoder in a pronunciation scoring model, and E^p = (e^p_1, e^p_2, ..., e^p_N) denote the textual embedding of canonical phonemes generated by a phoneme-level prompt encoder. A set of paired phoneme representations M = {(z^p_i, z^t_i), i = 1, ..., M} is obtained by first applying separate linear projections to H^p and E^p, and then calculating the centroid vectors for each phoneme category. Next, as illustrated in Fig. 5, the M×M similarities are derived from M, with the contrastive term L_con aiming to maximize the similarity between paired phoneme representations while minimizing the similarity of unpaired ones [52][53]. The contrastive term L_con includes two losses, with a temperature hyper-parameter τ that controls the strength of penalties on negative samples:

```text
L_con = L_p2t + L_t2p                                                                          (19)
L_p2t = -1/M Σ_{i=1}^{M} log exp(φ(z^p_i, z^t_i)/τ) / Σ_{j=1}^{M} exp(φ(z^p_i, z^t_j)/τ)  (20)
L_t2p = -1/M Σ_{i=1}^{M} log exp(φ(z^t_i, z^p_i)/τ) / Σ_{j=1}^{M} exp(φ(z^t_i, z^p_j)/τ)  (21)
```

where φ(z^p_i, z^t_j) is dot product between L2-normalized vectors z^p_i and z^t_j (cosine similarity). During training, M is constructed from each batch, where the data instances with the highest proficiency score are empirically sampled to compute centroid vectors.

**Phonemic Characteristic Term.** The phonemic characteristic term L_pc preserves the phonemic proximity information by minimizing the negative distances between centroid vectors z^p_i:

```text
L_pc = -1/(M(M-1)) Σ_{i=1}^{M} Σ_{i≠j} ||z^p_i - z^p_j||_2     (22)
```

L_pc is equivalent to maximizing the distances between phoneme categories during the optimization process.

**Ordinal Term.** To reflect ordinal relationships of regression targets in the feature space, the ordinal term L_o is defined to minimize the distance between the feature representations h^p_i and their corresponding phoneme centroid vectors z^p_i with relative differences of proficiency score:

```text
L_o = 1/N Σ_{i=1}^{N} w_i ||h^p_i - z^p_i||_2     (23)
```

where w_i = |C - y^p_i| is a compactness weight for each h^p_i, reflecting the ordinal behaviors within the label space, with y^p_i denoting the corresponding phoneme-level accuracy score. The tunable constant C is set to be 3, representing the highest accuracy score plus a small margin.

---

## IV (cont). Phoneme-Specific Variation (PhnVar)

To balance the distribution of predicted phonemes while accounting for pronunciation difficulties, the logits of phoneme predictions generated by a phoneme predictor are perturbed with randomly sampled Gaussian noise, where the radius is determined by the phoneme-dependent variance. To this end, the proposed training scheme, phoneme-specific variation (PhnVar), consists of two factors: a data quantity factor and a pronunciation difficulty factor. The data quantity factor assigns smaller variances to majority phoneme categories and larger variance to minority ones, while the pronunciation difficulty factor modulates feature areas based on the mispronunciation rates of phonemes. Formally, revisiting Eq. (5), the probability of the n-th canonical phoneme being predicted as a diagnostic phoneme k, derived from the softmax function:

```text
p^n_k = exp(g^n_k) / Σ_{i=1}^{M} exp(g^n_i)     (24)
```

Here, g^n_k is the logit of the k-th phoneme in logit vector **g^n** = (g^n_1, g^n_2, ..., g^n_M), generated by Linear_diag(h^p_N), where M is the number of phoneme categories. Logits are augmented with phoneme-specific variance, defined as the weighted p of the data quantity factor QF_k and the pronunciation difficulty factor DF_k for phoneme k, with coefficients α and β:

```text
g̃^n_k = g^n_k + δ(σ) × exp((α×log(QF_k) + β×log(DF_k)) / (α+β))     (25)
```

where δ(σ) stands for a Gaussian distribution with a zero mean and the standard deviation σ. Both α and β are set to 1 in the experiments. The data quantity factor is defined as normalized inverse phoneme frequency operated in the logarithmic scale:

```text
QF_k = c_k / max_i c_i ;   c_k = log(Σ_{i=1}^{M} q_i / q_k)     (26)
```

where q_k is the number of instances in phoneme category k. The pronunciation difficulty factor is expressed as normalized mispronunciation rate:

```text
DF_k = d_k / max_i d_i ;   d_k = mp_k / (mp_k + cp_k)     (27)
```

where mp_k and cp_k are the number of mispronounced and correctly pronounced instances for phoneme category k, respectively.

**Mispronunciation Detection and Diagnosis via MuFFIN.** To detect mispronunciation segments, MuFFIN follows a pronunciation scoring-based paradigm, where the outputs of the phoneme-level error detector serve as indicators of mispronounced segments. Phoneme segments are identified as mispronounced if the corresponding indicators exceed a predefined threshold. Subsequently, the detected mispronunciation segments are fed into the phoneme-level predictor to generate diagnostic results. To ensure consistency between the detector and predictor, the canonical phonemes (i.e., the phonetic transcription of the text prompt) are masked during the softmax computation of the predictor.

---

## V. Experimental Setups

### A. Experimental Data and Evaluation Metrics

**Dataset.** A series of experiments were conducted on the Speechocean762 dataset, a publicly available dataset specifically designed for research on computer-assisted language learning [26]. This dataset contains **5,000 English-speaking recordings** spoken by **250 Mandarin L2 learners**. The training and test sets are of equal size, and each of them has 2,500 utterances.

For the **APA task**, pronunciation proficiency scores were evaluated at multiple linguistic granularities with various pronunciation aspects:

| Granularity | Aspect | Score Interval | # Train | # Test |
|-------------|--------|---------------|---------|--------|
| Phoneme | Accuracy | [0, 2] | 47,076 | 47,369 |
| Word | Accuracy, Stress, Total | [0, 10] | 15,849 | 15,967 |
| Utterance | Accuracy, Completeness, Fluency, Prosody, Total | [0, 10] | 2,500 | 2,500 |

For the **MDD task**, the phoneme labels follow the definitions in the CMU pronunciation dictionary, which includes a set of 39 canonical phonemes. In Speechocean762, **mispronunciation labels were manually assigned to phoneme segments with accuracy scores below 0.5** and were categorized into four types:

| Type | Description | # Train | # Test |
|------|-------------|---------|--------|
| Correctness | The uttered phoneme aligns with the canonical phoneme | 45,088 | 45,959 |
| Deletion | A canonical phoneme is omitted | 450 | 396 |
| Substitution | A canonical phoneme is mispronounced to others | 914 | 593 |
| Non-categorical Error | The uttered phoneme does not exist in the CMU dictionary | 488 | 332 |
| Accented Error | A canonical phoneme is pronounced correctly but with a strong accent | 136 | 89 |

**Evaluation Metrics.** The primary evaluation metric for APA adopts Pearson correlation coefficient (PCC), which measures the linear correlation between predicted scores and ground-truth scores. In accordance with prior studies, mean square error (MSE) is reported for phoneme-level accuracy. For MDD tasks, the evaluation metrics follow the scoring rubrics in [9]. Specifically, the mispronunciation detection subtask is evaluated using recall (RE), precision (PR), and F1-score (F1), while the mispronunciation diagnosis subtask is assessed with diagnostic error rate (DER), false rejection rate (FRR), false acceptance rate (FAR), and phoneme error rate (PER).

### B. Implementation Details

**Feature Extraction.** For the pronunciation feature extraction, the GOP features, the energy, and the duration statistics are adopted in line with previous studies [24][25]. The extraction of SSL-based features follows the processing flow suggested in [46], where features are extracted from the outputs of pretrained acoustic models, including Wav2vec2.0 [54], WavLM [55], and HuBERT [56]. The SSL-based and energy features are extracted at the frame level and then aggregated into phoneme-level representations based on timestamps of phoneme segments derived from forced-aligning the learner's speech to the reference text. The extracted phoneme-level proficiency features amount to **3,164 dimensions**, comprising:

- **84** dimensions for GOP features E^GOP
- **7** for energy statistic E^Eng
- **1** for duration value E^Dur
- **3,072** for SSL-based features E^SSL (1,024-dim last-layer from each of 3 Large SSL models)

**Training Configuration.** For the training configuration, the settings reported in [24][25] were followed, where each experiment consisted of 5 independent trials, and each trial runs for **100 epochs** with different random seeds. In each trial, the model was trained with an **Adam optimizer** with an initial learning rate of **1e-3** and a **batch size of 25**. A learning rate scheduler was used to decay the learning rate by a factor of 0.1 after the overall loss did not decrease for **10 consecutive epochs**. Furthermore, models were initialized with a pretrained model following the pretraining strategies described in [41]. The reported experimental results were averaged over the 5 trials, with evaluation based on the minimum phoneme-level MSE.

**Model Configuration.** The phoneme-level, word-level, and the utterance-level encoder (viz. PhnEnc(·), WordEnc(·), UttEnc(·)) consisted of **3, 2, and 1** convolution-augmented Branchformer blocks, respectively [25]. Within each encoder block, the self-attention branch was implemented with a **single-head attention layer**, followed by two feed-forward layers. Both the self-attention and feed-forward layers had a **hidden dimension of 24**. Meanwhile, the convolutional branch consisted of a depth-wise convolutional layer with a **1×3 kernel** and a point-wise convolutional layer with a **1×1 kernel**, both of which had **24 channels**. To aggregate word-level and utterance-level features, the attention pooling modules were composed of a depth-wise convolutional layer and a single-head self-attention layer, where the convolutional layer had 24 channels with a kernel size of 1×3 and the attention layer had a hidden dimension of 24. Furthermore, the hidden dimension of the projection layers (viz. Linear_p(·), Linear_w(·), and Linear_u(·)) was set to **24**. During the training phase, the tunable parameters of L_p, L_w, and L_u in Eq. (14) were set to 3, 1, and 1, respectively, while the temperature factor τ in Eqs. (20) and (21) was set to 1.

**Mispronunciation Detection Threshold.** A global threshold is selected via grid search, with a stride of 0.1 over a range [0.0, 1.0]. A held-out set of 500 utterances is set aside from the training set, with the remaining 2,000 utterances used for model training. This held-out set is designed to cover both correct and incorrect pronunciations of each phoneme and is then used to determine phoneme-specific thresholds by maximizing the area under the precision-recall curve. The global threshold is set to **0.4** for MuFFIN and MuFFIN+PhnVar.

### C. Compared Methods

Three categories of pronunciation assessment models:

1. **Single-aspect APA models**: Lin2021 [57] — hierarchical APA model using surface features, utterance-level only; Kim2022 [30] — layer-wise contextual representations from pretrained acoustic model for oral skills (fluency/prosody) at utterance-level.
2. **Multi-aspect and multi-granular APA models**: LSTM [45], GOPT [45], 3M [45], GFR [42], HierGAT [58], 3MH [48].
3. **Multi-faceted assessment**: Ryu2023 [59] (joint APA+MDD via CTC recognizer + regressors), JAM [60] (3M + phoneme classifier + EMA features).

---

## V. Experimental Results

### A. Qualitative Analysis

**Phoneme Statistics of Speechocean762 (Fig. 6).** The phonemes are sorted by mispronunciation rate and then categorized into three disjoint subsets: high (mispronunciation rate above 5.1%), medium (mispronunciation rate between 5.1% and 3.4%), and low (mispronunciation rate below 3.4%) regions. It is evident that the occurrence counts of phonemes and their corresponding mispronunciation rates exhibit distinct distributional patterns. For example, the high-occurrence phonemes (e.g., /AH/, /T/, and /N/) are found within the low mispronunciation region. In contrast, some low-occurrence phonemes (e.g., /ZH/, /TH/, and /NG/) are often associated with high mispronunciation rates. Building on this, to mitigate the data imbalance issue facing the MDD task, the proposed phoneme-specific variance incorporates two novel regulation terms: a quantity factor and a pronunciation difficulty factor. The former adjusts the feature distributions of phonemes, while the latter adjusts feature scatteredness according to the mispronunciation rate.

**Qualitative Visualizations for L_pc and L_o (Fig. 7).** In the second set of experiments, the impacts of the phonemic characteristic term and the ordinal term (L_pc and L_o) are graphically examined. As depicted in Fig. 7, phoneme representations H^p from the test set are extracted and visualized. From Fig. 7(a), it is observed that despite MuFFIN jointly optimizing both the phoneme recognition and the assessment tasks, the resulting phoneme representations, however, are inevitably grouped by phoneme-level accuracy scores, inadequate to explicitly capturing the subtle distinctions between phonemes in the feature space. When training MuFFIN with L_pc, as shown in Fig. 7(b), the phoneme-discriminative features are obtained, where the representations disperse according to their respective phoneme categories. However, simply separating the feature representations would omit the ordinal relationships, which might impede pronunciation assessment tasks. In response to this, the synergy of L_pc and L_o serves as a remedy, which enables the phoneme representations to reflect both categorical distinctions and ordinal relationships derived from their accuracy scores, as shown in Fig. 7(c). Specifically, integrating L_o leads to a stronger correlation between pairwise distances and phoneme-level accuracy within each phoneme category, resulting in an outward dispersion in the feature space as accuracy decreases. Grounded on these observations, incorporating L_pc and L_o during the training process of MuFFIN substantially improves the discriminability of phoneme representations and simultaneously reflects the ordinal relationships of the predicted accuracy scores in the feature space.

**Qualitative Visualizations for L_con (Fig. 8).** Subsequently, to qualitatively assess whether the contrastive term L_con aligns the speech-derived representations (colored in blue) with their corresponding textual embeddings (colored in orange) for phoneme segments, the representations H^p and E^p from MuFFIN on the test set are visualized in Fig. 8. By comparing among Fig. 8(a) and Fig. 8(b), it is observed that the proposed L_con effectively projects these two types of phoneme representations into a shared feature space, resulting in a more coherent distribution. Going one step further, a zoomed-in view is presented in Fig. 8(c), which highlights that the contrastive term not only aligns the heterogeneous phoneme representations with the corresponding textual embeddings, but also preserves the phoneme-specific characteristics across phoneme categories.

**Qualitative Visualizations of PhnVar (Fig. 9).** Finally, to qualitatively assess the effectiveness of the proposed PhnVar training scheme, the phoneme logits and decision boundaries of the diagnosis predictor are visualized. The observations from Fig. 9 are highlighted as follows. First, the logits of phonemes with higher occurrence counts tend to occupy a larger portion of the feature space, while those with lower occurrence counts are compressed into a narrower region. This is evidenced by the increasing size of feature regions for phonemes /K/, /IH/, /T/, to /AH/, which is consistent with their respective occurrence frequencies. Subsequently, in Fig. 9(b), it is observed that training MuFFIN with the variant of PhnVar (viz. PhnVar w/o DF) results in more uniformly distributed feature regions, independent of phoneme occurrence counts. However, adjusting the feature space solely factoring in data quantity fails to capture the distribution of mispronunciations. In light of this, PhnVar additionally takes the pronunciation difficulty factor into account. The phoneme logits of MuFFIN trained with PhnVar are visualized in Fig. 9(c), where the feature regions are partitioned by the phoneme mispronunciation rates, with region sizes decreasing in the order of /IH/, /AH/, /T/, and /K/.

### B. Performance of Automatic Pronunciation Assessment (Table III)

| Model | Phone MSE↓ | Phone PCC↑ | Word Acc↑ | Word Stress↑ | Word Total↑ | Utt Acc↑ | Utt Comp↑ | Utt Flu↑ | Utt Pros↑ | Utt Total↑ |
|-------|-----------|-----------|---------|------------|-----------|---------|---------|---------|---------|---------|
| Lin2021 [57] | — | — | — | — | — | — | — | — | — | 0.720 |
| Kim2022 [30] | — | — | — | — | — | — | — | 0.780 | 0.770 | — |
| LSTM [45] | 0.089 | 0.591 | 0.514 | 0.294 | 0.531 | 0.720 | 0.076 | 0.745 | 0.747 | 0.741 |
| GOPT [45] | 0.085 | 0.612 | 0.533 | 0.291 | 0.549 | 0.714 | 0.155 | 0.753 | 0.760 | 0.742 |
| 3M [45] | 0.078 | 0.656 | 0.598 | 0.289 | 0.617 | 0.760 | 0.325 | 0.828 | 0.827 | 0.796 |
| GFR [42] | 0.079 | 0.646 | 0.598 | 0.334 | 0.614 | 0.732 | 0.318 | 0.769 | 0.767 | 0.756 |
| HierGAT [58] | 0.073 | 0.683 | 0.648 | 0.327 | 0.663 | 0.798 | 0.531 | 0.840 | 0.833 | 0.821 |
| 3MH [48] | 0.071 | 0.693 | 0.682 | **0.361** | 0.694 | 0.782 | 0.374 | **0.843** | **0.836** | 0.811 |
| Ryu2023 [59] | — | — | — | — | — | 0.719 | — | 0.775 | 0.773 | 0.743 |
| JAM [60] | 0.076 | 0.664 | 0.622 | 0.241 | 0.638 | 0.773 | 0.205 | 0.831 | 0.829 | 0.805 |
| **MuFFIN** | **0.063** | **0.742** | **0.705** | 0.315 | **0.714** | **0.807** | **0.768** | 0.841 | 0.832 | **0.830** |

*(Mean PCC ± std dev over 5 seeds. MuFFIN achieves higher PCC than 3MH across all metrics except utterance-fluency, approximate randomization test p < 0.001.)*

The proposed MuFFIN outperforms other APA models by a remarkable margin in most pronunciation assessment tasks, except for the word-level stress. Specifically, MuFFIN stands out in the phoneme-level accuracy, demonstrating PCC score improvements of 4.9% and 5.9% over the prior-art models, 3MH and HierGAT, respectively. These performance gains are attributed to the proposed multi-faceted phoneme-level pronunciation feedback module, which jointly optimizes the APA and MDD tasks, thereby encouraging the phoneme encoder to learn distinct phoneme identities when evaluating the pronunciation scores. With respect to the word-level assessments, MuFFIN generally performs well across most pronunciation aspects. However, in word-level stress, the model demonstrates comparable performance against GFR and HierGAT, while trailing behind 3MH. A possible reason for the inferior performance is that 3MH leverages sub-phoneme modeling to create a pseudo (augmented)-linguistic hierarchy between phoneme and word levels, facilitating better rendering of supra-segmental information for word-level assessments.

**Assessment Performance at Utterance-level.** MuFFIN achieves the highest performance across most aspects. Compared to 3MH, MuFFIN enhances the PCC scores by 2.5% in utterance-level accuracy, 2.9% in utterance-level total, and achieves comparable performance in utterance-level fluency and prosody. MuFFIN also achieves substantial improvements in the utterance-level completeness assessment, a metric reflecting the proportion of correctly pronounced words in an utterance. This gain is attributed to the joint training of the MDD task within the APA model, which consequently enables MuFFIN to pinpoint mispronounced segments and identify corresponding phonemes in learners' speech.

**Effectiveness of PhnVar and ConPCO (Table IV):**

| PhnVar | L_con | L_pco | Phone Acc PCC | Word Acc | Word Stress | Word Total |
|--------|-------|-------|--------------|---------|------------|-----------|
| — | — | — | 0.742 | 0.705 | 0.315 | 0.714 |
| ✓ | — | — | 0.746 | 0.704 | 0.310 | 0.714 |
| ✓ | ✓ | — | **0.749** | 0.707 | 0.314 | **0.718** |
| ✓ | — | ✓ | 0.745 | 0.703 | 0.296 | 0.713 |
| ✓ | ✓ | ✓ | 0.747 | **0.708** | **0.341** | **0.718** |

From Table IV, the proposed PhnVar training scheme yields a 0.7% improvement in phoneme-level accuracy over the base model. Subsequently, the incorporation of the phoneme-level regularizers (viz. L_con and L_pco) under the PhnVar training regime benefits pronunciation assessments, as evidenced by the sustained or improved results at the phoneme- and word-level assessment tasks. Furthermore, the contrastive term primarily boosts the performance in the aspects of phoneme-level accuracy and word-level total score. In contrast, the phonemic ordinal regularizer tends to either slightly enhance performance or retain that of the vanilla MuFFIN model. In addition, training MuFFIN with ConPCO attains the best performance in the word-level assessment tasks (as shown in the last row of Table IV).

### C. Performance of Mispronunciation Detection and Diagnosis (Table V)

| Model | RE (%)↑ | PR (%)↑ | F1 (%)↑ | FAR (%)↓ | FRR (%)↓ | DER (%)↓ | PER (%)↓ |
|-------|--------|--------|--------|---------|---------|---------|---------|
| Ryu2023 [59] | **91.60** | 26.90 | 41.50 | — | — | — | 9.93 |
| JAM [60] | 34.76 | 61.10 | 45.01 | 64.32 | **0.58** | **45.23** | 2.81 |
| MuFFIN | 64.33 | 66.89 | 65.99 | 35.67 | 0.97 | 60.97 | 2.36 |
| MuFFIN + PhnVar | 68.37 | **67.60** | **67.98** | **31.63** | 1.01 | 58.82 | **2.33** |

As shown in Table V, MuFFIN outperforms other methods in the mispronunciation detection subtask, achieving outstanding performance in terms of F1-score and precision. Moreover, training MuFFIN with the phoneme-specific variation (PhnVar) leads to notable improvements in all evaluation metrics compared to the base model. This gain is further illustrated in Fig. 10, where the orange line (MuFFIN+PhnVar) exceeds the blue line (MuFFIN) in area under the precision-recall curve. Ryu2023, on the basis of a CTC-based phoneme recognizer, achieves the highest recall value but has the downside of low precision for the mispronunciation detection task. Instead of a direct free-phoneme recognition process, JAM builds upon 3M and detects mispronunciations in learners' speech by attaching a phoneme classifier to the phoneme-level encoder. The corresponding result demonstrates promising performance in terms of precision metric, though it struggles with the low recall rate. Compared to JAM, MuFFIN achieves superior performance across all metrics in the mispronunciation detection subtask.

For mispronunciation diagnosis subtask, methods achieve promising performance in terms of FAR and PER. However, a trade-off appears to exist between recall and the metrics of FRR and DER. Specifically, compared to JAM, MuFFIN achieves higher recall rate and lower PER but exhibits inferior performance in both FRR and DER. This result implies that the model detects a greater number of mispronounced segments but comes at the cost of diagnostic accuracy. This issue is left as a direction for future research.

**Systematic Examination of Data Imbalance in MDD (Table VI).** Phoneme segments are divided into two grouping criteria: occurrence count (many/medium/few) and mispronunciation rate (high/medium/low). Key findings from Table VI:

- Data quantity primarily affects average PER: PER increases significantly from many-shot to few-shot phoneme subsets. A naïve training process for a phoneme classifier based on empirical risk minimization inevitably biases the model toward majority phoneme categories.
- Pronunciation difficulty causes a steady decline in average recall as phoneme subsets shift from high to low mispronunciation rates. Infrequently mispronounced phoneme segments pose greater challenges for pronunciation error detection.
- **PhnVar** combining both QF and DF factors achieves notable performance gains in F1-score. Removing DF (w/o DF) primarily enhances recall; removing QF (w/o QF) primarily enhances precision. Both factors together achieve the balanced optimal F1.

### D. Ablation Studies

**Multi-granularity APA (Table VII — MuFFIN without MDD):**

| Training Objective | # Params | Phone Acc PCC | Word Acc PCC | Utt Acc PCC |
|-------------------|---------|--------------|-------------|------------|
| Utt. Only | 541K | — | — | 0.782 |
| Word Only | 248K | — | 0.674 | — |
| Phone Only | 126K | 0.715 | — | — |
| Phone + Word | 249K | 0.724 | 0.688 | — |
| Phone + Word + Utt. | 608K | **0.726** | **0.687** | **0.807** |

MuFFIN trained in a multi-granularity manner achieves superior results in relation to any single-granularity assessment model. For instance, MuFFIN trained with multi-granularity objectives (Phone+Word and Phone+Word+Utt.) outperforms their single-granularity counterparts (Word Only and Utt. Only), with respective gains of 14% and 13%. Furthermore, a comparison of parameter sizes reveals that utterance-level assessment models (Utt. Only and Phone+Word+Utt.) have substantially larger parameter sizes than the other assessment models due to the residual connections between the mean pooling feature Ẽ^SSL and the utterance-level regressors.

**Joint APA and MDD (Table VIII):**

| Training Objective | Phone PCC | Word Acc | Utt Acc | F1 (%) | RE (%) | PR (%) |
|-------------------|-----------|---------|--------|--------|--------|--------|
| MDD Only | — | — | — | 62.71 | 65.67 | 60.33 |
| MDD + Utt. | — | — | 0.787 | 63.34 | 63.49 | 63.45 |
| MDD + Word | — | 0.681 | — | 64.46 | 66.27 | 62.86 |
| MDD + Phone | 0.717 | — | — | **66.26** | **69.06** | 63.77 |
| MDD + Word + Phone | 0.741 | 0.696 | — | 66.04 | 67.08 | 65.36 |
| **MDD + Word + Phone + Utt.** | **0.742** | **0.705** | **0.807** | 65.99 | 64.33 | **66.89** |

Multi-faceted pronunciation models that integrate APA tasks consistently outperform the model trained solely on MDD (viz. MDD Only) across all MDD evaluation metrics, demonstrating the synergistic effect of jointly modeling MDD and APA. Among these multi-faceted pronunciation models, the model trained with phoneme-level assessment and MDD tasks (MDD+Phone) yields the optimum performance in term of the recall metric. Regarding the performance of pronunciation assessment, observations from Tables VII and VIII suggest that the integration of MDD tasks maintains or slightly improves pronunciation accuracy. However, the primary improvement for the performance of pronunciation assessment stems from the incorporation of diverse assessment tasks at various linguistic levels.

---

## VI. Conclusion

In this paper, a novel multi-faceted pronunciation feedback model dubbed MuFFIN is proposed, which is designed to qualify learners' pronunciation from multiple perspectives, including pronunciation aspects across various linguistic levels, as well as mispronunciation detection and diagnosis at phoneme-level. A novel contrastive phonemic ordinal regularizer (ConPCO) has been put forward to empower MuFFIN to generate more phoneme-discriminative features while accounting for the ordinal nature of phoneme-level accuracy scores. Furthermore, to tackle the intricate data imbalance problem of MDD, the phoneme-specific variation (PhnVar) effectively balances the distribution of predicted phonemes while incorporating considerations of pronunciation difficulty. The practical utility of the method has been verified through extensive experiments on the Speechocean762 benchmark dataset. The proposed contrastive phonemic ordinal regularizer has been thoroughly examined through a series of graphical visualizations. Moreover, this study is the first attempt to address the data imbalance problem in MDD from the perspectives of data quantity and pronunciation difficulty. The empirical results demonstrate that the model outperforms some state-of-the-art methods in both APA and MDD tasks.

**Limitations and Future Work.** The proposed method is constrained by its dependence on the "read-aloud" learning scenario and to some extent lacks explainability for the provided assessment results. Furthermore, the experimental dataset solely contains the Mandarin learners, potentially hindering the generalization abilities and applicability to learners with other accents. In future work, the authors plan to examine the proposed method on spoken language assessment, where learners speak freely or respond to a given task or question [61]. In addition, the issues of explainable pronunciation feedback are also left as a future extension.

---

## Key Implementation Facts for P010

| Item | Value |
|------|-------|
| Input features | 3,164 dims total |
| — GOP (E^GOP) | 84 dims (LPP + LPR, 84 phoneme posteriors) |
| — Energy (E^Eng) | 7 dims (RMS energy statistics) |
| — Duration (E^Dur) | 1 dim |
| — SSL (E^SSL) | 3,072 dims (3 × 1,024 last-layer) |
| SSL models | Wav2vec2.0 + WavLM + HuBERT (all Large, 1,024-dim each) |
| SSL extraction | Frame-level → phone-level mean via forced-alignment timestamps |
| embed_dim | 24 (all projections, attention, conv channels) |
| PhnEnc depth | 3 Branchformer blocks |
| WordEnc depth | 2 Branchformer blocks |
| UttEnc depth | 1 Branchformer block |
| Attention heads | 1 per block |
| Conv kernel | depth-wise 1×3, point-wise 1×1 |
| Optimizer | Adam lr=1e-3, batch_size=25 |
| LR schedule | ReduceLROnPlateau factor=0.1, patience=10 |
| Epochs | 100 per trial, 5 trials |
| Seeds | 5 independent random seeds |
| MDD threshold | 0.4 (grid search stride 0.1 on held-out 500 utterances) |
| ConPCO τ | 1 |
| PhnVar α, β | Both = 1 |
| PhnVar C | 3 (highest accuracy score + small margin) |
| MDD label derivation | accuracy_score < 0.5 → mispronounced (in SpeechOcean762) |
| Best phone PCC | **0.742** (MuFFIN, 5-seed avg, evaluation at min phone MSE) |
| Pre-training | Yes, following [41] (Do et al. hierarchical APA pretraining) |
| L_p, L_w, L_u weights | 3, 1, 1 |
