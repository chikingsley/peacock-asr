---
title: "An Effective Hierarchical Graph Attention Network Modeling Approach for Pronunciation Assessment"
authors:
  - "Bi-Cheng Yan"
  - "Berlin Chen"
citation_author: "Yan and Chen"
year: 2024
doi: "10.1109/TASLP.2024.3449111"
pages: 12
journal: "IEEE/ACM TRANSACTIONS ON AUDIO, SPEECH, AND LANGUAGE PROCESSING, VOL. 32, 2024"
source_pdf: "paper.pdf"
extraction_method: "Extracted from PDF with Manual Vision Synthesis"
extracted_at: "2026-04-18"
llm_friendly: true
---

## Abstract

Automatic pronunciation assessment (APA) manages to quantify second language (L2) learners’ pronunciation proficiency in a target language by providing fine-grained feedback with multiple aspect scores (e.g., accuracy, fluency, and completeness) at various linguistic levels (i.e., phone, word, and utterance). Most of the existing efforts commonly follow a parallel modeling framework, which takes a sequence of phone-level pronunciation feature embeddings of a learner’s utterance as input and then predicts multiple aspect scores across various linguistic levels. However, these approaches neither take the hierarchy of linguistic units into account nor consider the relatedness among the pronunciation aspects in an explicit manner. In light of this, we put forward an effective modeling approach for APA, termed HierGAT, which is grounded on a hierarchical graph attention network. Our approach facilitates hierarchical modeling of the input utterance as a heterogeneous graph that contains linguistic nodes at various levels of granularity. On top of the tactfully designed hierarchical graph message passing mechanism, intricate interdependencies within and across different linguistic levels are encapsulated and the language hierarchy of an utterance is factored in as well. Furthermore, we also design a novel aspect attention module to encode relatedness among aspects. To our knowledge, we are the first to introduce multiple types of linguistic nodes into graph-based neural networks for APA and perform a comprehensive qualitative analysis to investigate their merits. A series of experiments conducted on the speechocean762 benchmark dataset suggests the feasibility and effectiveness of our approach in relation to several competitive baselines.

## I. Introduction

With the rising trend of globalization, an ever-growing number of people are willing or being asked to learn foreign languages. In response to this surging demand, computer-assisted pronunciation training (CAPT) systems have garnered significant research attention, as they offer L2 (second-language) learners a range of stress-free and self-directed scenarios to practicing pronunciation skills [1]. CAPT systems have a broad spectrum of applications, providing timely and informative feedback for L2 learners, but also serving as a handy reference for professionals (e.g., interviewers and examiners) on standardized tests to relieve their workload [4].

As a crucial ingredient of CAPT, automatic pronunciation assessment (APA) aims to quantify oral proficiency and provide fine-grained feedback to learners by predicting multiple aspect scores at various linguistic levels [5], [6]. An APA system is typically instantiated in a read-aloud scenario, where an L2 learner is presented with a text prompt and instructed to pronounce it correctly. Early studies for APA mostly focused on single-aspect assessment, typically developed by extracting sets of hand-crafted features to construct scoring modules accordingly, such as phone-level accuracy [7], [8], [9], word-level lexical stress [10], [11], or various aspects of utterance-level proficiency scores [12], [13], [14].

More recently, with the synergistic breakthroughs in neural model architectures and optimization algorithms [15], [16], research endeavors have advocated for the notion of multi-aspect and multi-granular pronunciation assessment, which creates a unified scoring model to jointly evaluate pronunciation proficiency at various linguistic levels (i.e., phone, word, and utterance) with diverse aspects (e.g., accuracy, fluency, and completeness). The running example in **Fig. 1** illustrates the evaluation flow of an APA system in the reading-aloud training scenario, which offers an L2 learner in-depth pronunciation feedback. Prior studies along this line of research usually follow a parallel modeling paradigm [17], [18], [19], wherein Transformer-based neural networks serve as the archetype to take as input a sequence of phone-level pronunciation feature embeddings of a learner’s utterance while simultaneously predicting multiple aspect scores across different linguistic levels without accounting for their subtle dependency.

### The Limitations of Parallel Modeling

Albeit effective, such parallel modeling approaches suffer from at least two weaknesses:

1. **Hierarchical Neglect:** First, these approaches fall short in taking advantage of the hierarchical structure of an utterance, which assumes that all phones within a word are of equal importance and insufficiently capture the word-level structure cues that are prominent in the composition of an utterance-level representation when solely based on phone-level pronunciation features.
2. **Aspect Relatedness:** Second, the relatedness among pronunciation aspects is mostly sidelined. As an illustration, we visualize the correlation matrix in **Fig. 2**, which shows the Pearson Correlation Coefficient (PCC) between any pair of expert annotated aspect scores on the training set. We can observe that except for the aspects of utterance-completeness and word-stress, the remaining aspects present strong correlations not only within the same linguistic level but also across different linguistic levels.

**Key Contribution:** Building on these observations, we present a novel APA method, dubbed HierGAT, which leverages hierarchical graph attention architecture to jointly model the intrinsic structure of an utterance and meanwhile considers the transitions among disparate aspects at the same and across different linguistic levels. HierGAT is the first to construct a heterogeneous graph structured hierarchically with utterance, word, and phone nodes, capable of capturing relations including utterance-word, word-word, phone-word, and phone-phone.

## II. Related Work

Research and development on CAPT date back to pioneering efforts conducted in the 60's of the last century [7], [25], which has attracted surging attention in recent years, showing good promise by leveraging many advanced deep learning technologies [26], [27], [28]. According to the types of diagnostic feedback being provided, research endeavors of CAPT fall into two broad categories: mispronunciation detection and diagnosis (MDD), and automatic pronunciation assessment (APA).

### A. Mispronunciation Detection and Diagnosis (MDD)

The goal of MDD focuses on pinpointing phone-level erroneous pronunciation segments and provide L2 learners with the corresponding diagnostic feedback [28], [29], [30]. Early works relied on pronunciation scoring based approaches, which make use of a well-trained acoustic model to derive various types of confidence measures as indicators of mispronunciation. Commonly used indicators include, but is not limited to, phone durations [32], [33], likelihood ratios [29], [34], phone posterior probabilities [35], and their combinations [39]. Goodness of pronunciation (GOP) and its descendants are the most iconic instantiations [7].

In order to better obtain informative diagnostic feedback, dictation-based methods alternatively frame MDD as a phone recognition task by employing a free-phone recognition process to dictate the most likely phone sequence uttered by an L2 learner. Consequently, the erroneous pronunciation portions can be easily identified by comparing the dictation result with the corresponding canonical phone sequence. To this end, for example, Leung et al. made attempts to employ a phone recognizer trained with the connectionist temporal classification (CTC) loss [36]. As a workaround, Yan et al. [37] exploited the hybrid CTC-Attention ASR model as the dictation model and sought to capture deviant (non-categorical) phone productions by augmenting the canonical phone dictionary. To integrate historical mispronunciation patterns of L2 learners, Zhang et al. utilized a phonetic recurrent neural network Transducer (RNN-T) to transcribe learners’ speech, which synergized RNN-T modeling with weakly supervised data augmentation and diversified beam search.

### B. Automatic Pronunciation Assessment (APA)

Automatic pronunciation assessment concentrates more on assessing and providing a suite of comprehensive pronunciation scores on a few specific aspects or traits of spoken language usage to reflect a learner’s pronunciation quality [39], [40], [41]. Prior arts on APA focused exclusively on the single-aspect assessment, typically through constructing scoring modules individually to predict a holistic pronunciation proficiency score on a targeted linguistic level or some specific aspect with different sets of hand-crafted features. These hand-crafted features can be extracted directly from a learner’s input speech signal or the associated transcription generated by automatic speech recognition (ASR), which may consist of acoustic features, confidence of recognized linguistic units (phones, syllables, or words) [43], time-alignment information [44], and other statistic measures such as fundamental frequency [45], speech rate, and filled pause [46]. For example, Ferrer et al. quantified word-level stress according to the time-alignment information at the syllable nucleus, where Gaussian mixture models were employed to represent the distributions of prosody- and spectrum-related features.

Due to the unprecedented breakthroughs brought about by deep learning, the notion of multi-aspect and multi-granular pronunciation assessment has made inroad into APA with good promise. Several neural scoring models have been proposed to jointly evaluate pronunciation proficiency at various linguistic levels with diverse aspects. For example, Lin et al. streamlined three linguistic-level scoring modules and introduced a single-aspect multi-granular hierarchical APA architecture, utilizing an attention mechanism to extract and aggregate representations from low to high linguistic levels for multi-granularity proficiency estimation [39]. Gong et al. proposed a GOP feature-based Transformer (GOPT) to jointly model multi-aspect pronunciation assessment at multiple granularities with a multi-task learning mechanism [19]. Since then, several subsequent extensions to the GOPT framework were developed. For example, Chao et al. integrated prosodic and self-supervised learning (SSL) based features into GOPT to achieve multi-view, multi-granularity, and multi-aspect (3M) pronunciation modeling [17]. Do et al. investigated the issue of data imbalance incurred by APA and proposed a score-balanced loss function that aims to nudge the prediction bias of a neural model towards the majority scores. HierGAT departs from these "parallel" Transformer architectures by representing an input utterance as a hierarchical graph, updating node representations via message passing.

---

## III. Methodology

### A. Problem Formulation

In this paper, we explore the task of multi-aspect and multi-granular automatic pronunciation assessment (APA), as illustrated in **Fig. 1**. Given an input utterance $U$ which consists of a sequence of audio signals $X$ uttered by an L2 learner and a text prompt $T$ that the learner is expected to pronounce correctly, the objective of APA is to estimate proficiency scores for multiple aspects across various linguistic granularities. Formally, we denote a set of linguistic granularities $G = \{p, w, u\}$, where $p, w, u$ stands for the phone, word, and utterance levels, respectively. For a linguistic level $g \in G$, our APA model targets to quantify pronunciation skill of an L2 learner with respect to $N_g$ multiple aspects, represented by $A^g = \{a_1^g, a_2^g, ..., a_{N_g}^g\}$, where $N_g$ is the number of aspects, and each $a_j^g$ is framed as a regression task that estimates a sequence of aspect score $y_j^g \in [0, 2]$ for the phone level and $[0, 10]$ for word and utterance levels. The overall model architecture of HierGAT is depicted in **Fig. 3**, which mainly consists of three parts: 1) node representation initialization; 2) hierarchical graph layer; and 3) aspect assessments on nodes.

### B. Graph Construction

For an input text prompt $T$ with $M$ words and $N$ phones, we first represent it as a hierarchical graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$, where $\mathcal{V}$ stands for the node set and $\mathcal{E}$ are edges between nodes. In order to utilize the linguistic structures of the text prompt, the undirected hierarchical graph $\mathcal{G}$ contains phone nodes, word nodes, and an utterance node, defined by $\mathcal{V} = \mathcal{V}_p \cup \mathcal{V}_w \cup \mathcal{V}_u$, where each phone node $v_{p_n} \in \mathcal{V}_p$ corresponds to a phone $p_n$ in the canonical phone sequence of $T$, $v_{w_m} \in \mathcal{V}_w$ represents a word $w_m$ in $T$, and $v_u \in \mathcal{V}_u$ is a special supernode that signifies the whole utterance. The edge connection of $\mathcal{G}$ is defined as $\mathcal{E} = \mathcal{E}_p \cup \mathcal{E}_w \cup \mathcal{E}_{pw} \cup \mathcal{E}_{wu}$, where $\mathcal{E}_p$ denotes the connections between phone nodes within a particular word, $\mathcal{E}_w$ stands for connections between word nodes within the text prompt, $\mathcal{E}_{pw}$ is the cross-linguistic connections between a word node and its constituent phone nodes, and $\mathcal{E}_{wu}$ describes cross-linguistic connections between an utterance node and its constituent word nodes. A schematic depiction of the hierarchical graph is illustrated in **Fig. 4**.

**Edge Connection:** This hierarchical graph $\mathcal{G}$ is an unweighted graph; namely, the connected node pairs have weight 1, and disconnected node pairs have weight 0 in the adjacency matrix $\mathcal{A}$. For the phone-level connections, an edge $e_{p_{i,j}}$ connects phone nodes $v_{p_i}$ and $v_{p_j}$ if they are within the same word, facilitating the aggregation of intra-word information. All word nodes are fully-connected by word-level edges $\mathcal{E}_w$ which seeks to capture inter-word information. For the cross-linguistic relations, the phone-to-word edge $e_{pw_{i,k}}$ connects the phone node $v_{p_i}$ to its corresponding word node $v_{w_k}$, enabling message passing from the phone nodes to word nodes. All word nodes are linked to an utterance supernode $v_u$ with word-to-utterance connections, thereby gathering information from the word nodes to an utterance node. In the resulting hierarchical graph, each phone node can only interact with neighboring phone nodes within the same word, while interacting indirectly with the phone nodes of other words via word-level node connections.

### C. Node Representation Initialization

**Pronunciation Feature Extraction:** For an input utterance $U$, we start by converting the text prompt into a canonical phone sequence through looking up a pronunciation dictionary. Next, various pronunciation features are extracted to assess the L2 learner’s pronunciation quality at the phone level, which are then concatenated and projected to obtain a sequence of dense pronunciation features $\mathbf{X}_p = (\mathbf{x}_{p1}, \mathbf{x}_{p2}, ..., \mathbf{x}_{pN})$:
$$ \mathbf{X}_p = \mathbf{W}_x \cdot \tilde{\mathbf{X}}_p + \mathbf{b}_x \quad (1) $$
$$ \tilde{\mathbf{X}}_p = [E^{GOP} || E^{Eng} || E^{Dur} || E^{Fbank}] \quad (2) $$
where $E^{GOP}$ is goodness of pronunciation-based (GOP) feature [7], [26], $E^{Eng}$ and $E^{Dur}$ are prosodic features of duration and energy statistics, and spectral features $E^{Fbank}$ (viz. log Mel-filterbank features). $\mathbf{W}_x$ and $\mathbf{b}_x$ are learnable parameters, and $||$ denotes concatenation operation. Notably, the extracted pronunciation features include both frame- and phone-level features. To align with the phone-level features, the frame-level features are averaged over time frames based on aligned phone boundaries. In addition, the word-level pronunciation features are denoted by $\mathbf{X}_w = (\mathbf{x}_{w1}, \mathbf{x}_{w2}, ..., \mathbf{x}_{wM})$, where $\mathbf{x}_{wm}$ stands for the features of the $m$-th word, which is the sum of its constituent (connected) phone-level pronunciation features.

**Node Representation Initialization:** We explore to use a convolution-augmented Branchformer (ConvBFR) [56] to initialize node features at both the phone and word levels, with the aim of capturing contextualized pronunciation patterns at their respective granularities. Subsequently, the utterance-level node is initialized by summing the features of its connected words. More specifically, the proposed ConvBFR comprises two parallel branches to dynamically model various ranged contexts at different linguistic granularities, with one branch following the original Transformer network architecture employing self-attention to capture long-range dependencies [53] and the other branch utilizing a convolution module introduced in [55] to capture local dependencies. Specifically, for the phone-level nodes, we first map the canonical phone sequence into phone embeddings $E_p$ via a phone embedding layer, which are then point-wisely added to $\mathbf{X}_p$ to provide a rendition of the positional information and phonetic characteristics. Next, a phone encoder is followed to initialize the phone-level node representations $\tilde{\mathbf{H}}_p$:
$$ \tilde{\mathbf{H}}_p^0 = \mathbf{X}_p + E_p \quad (3) $$
$$ \tilde{\mathbf{H}}_p = PhnEnc(\tilde{\mathbf{H}}_p^0) \quad (4) $$
where $PhnEnc(\cdot)$ consists of 3 stacked ConvBFR blocks. Afterward, for the word-level nodes, $\mathbf{X}_w$ are enriched with the textual information $E_w$, and then a word encoder is employed to generate the initial node representations $\tilde{\mathbf{H}}_w$:
$$ \tilde{\mathbf{H}}_w^0 = \mathbf{X}_w + E_w \quad (5) $$
$$ \tilde{\mathbf{H}}_w = WordEnc(\tilde{\mathbf{H}}_w^0) \quad (6) $$
where $E_w$ is generated by passing the text prompt $T$ into a word and position layer, and $WordEnc(\cdot)$ encompasses a stack of 3 ConvBFR blocks. For the utterance node representation $\tilde{\mathbf{h}}_u$, it is initialized by summing the representations of its connected word nodes $v_w \in \mathcal{V}_w$.

### D. Hierarchical Graph Layer

After constructing the hierarchical graph $\mathcal{G}$ with the adjacency matrix $\mathcal{A}$ and node representations at three linguistic levels ($\tilde{\mathbf{H}}_p, \tilde{\mathbf{H}}_w, \tilde{\mathbf{h}}_u$), we use the graph attention network (GAT) [54] to update the node representations.

**Graph Attention Network:** Given a constructed graph $\mathcal{G}$ with the corresponding hidden representations of input nodes $\mathbf{H}$, a GAT layer updates a node $v_i$ with the representation $\mathbf{h}_i$ as follows:
$$ e_{ij} = LeakyReLU(\mathbf{W}_a[\mathbf{W}_q\mathbf{h}_i || \mathbf{W}_k\mathbf{h}_j]) \quad (7) $$
$$ \alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{l \in \mathcal{N}_i} \exp(e_{il})} \quad (8) $$
$$ \mathbf{u}_i = \sigma \left( \sum_{j \in \mathcal{N}_i} \alpha_{ij} \mathbf{W}_v \mathbf{h}_j \right) \quad (9) $$
where $\sigma$ is an activation function instantiated with rectified linear units (ReLU), $\mathcal{N}_i$ is the set of neighboring nodes of $v_i$, $\alpha_{ij}$ stands for the attention weight between $\mathbf{h}_i$ and $\mathbf{h}_j$, and $\mathbf{W}_a, \mathbf{W}_q, \mathbf{W}_k$, and $\mathbf{W}_v$ are trainable weight matrices. The multi-head attention can be expressed by
$$ \mathbf{u}_i = ||_{t=1}^T \sigma \left( \sum_{j \in \mathcal{N}_i} \alpha_{ij}^t \mathbf{W}_v^t \mathbf{h}_j \right) \quad (10) $$
where $T$ is the number of independent attention mechanisms, $\alpha_{ij}^t$ is normalized attention weights computed by the $t$-th attention mechanism, and $\mathbf{W}_v^t$ is the corresponding transformation matrix. Next, a residual connection is in turn employed to prevent gradient vanishing. The updated node representation $\mathbf{h}_i'$ can be denoted by
$$ \mathbf{h}_i' = \mathbf{h}_i + \mathbf{W}_o \mathbf{u}_i \quad (11) $$
where $\mathbf{W}_o$ is a linear projection adjusting the dimension of $\mathbf{u}_i$ to align with $\mathbf{h}_i$. Finally, stacking on each graph attention layer, we introduce a position-wise feed-forward (FFN) layer consisting of two linear transformations, in the same vein as Transformer [15].

**Hierarchical Message Passing:** The proposed hierarchical graph layer begins by updating representations of phone nodes using their locally-neighboring phones within a word via the intra-word message passing. Then, the intermediate representations of a word node $\tilde{\mathbf{H}}_w'$ are derived by gathering information from its constituent phone nodes:
$$ \mathbf{H}_{p \leftarrow p} = GAT(\tilde{\mathbf{H}}_p, \tilde{\mathbf{H}}_p) \quad (12) $$
$$ \tilde{\mathbf{H}}_{w}' = GAT(\tilde{\mathbf{H}}_w, \mathbf{H}_{p \leftarrow p}) \quad (13) $$
where $\mathbf{H}_{p \leftarrow p}$ is updated representations of phone nodes. $GAT(\tilde{\mathbf{H}}_p, \tilde{\mathbf{H}}_p)$ denotes that $\tilde{\mathbf{H}}_p$ is linear projected to form query, key, and value matrices, respectively, while $GAT(\tilde{\mathbf{H}}_w, \mathbf{H}_{p \leftarrow p})$ means that $\tilde{\mathbf{H}}_w$ is used as query matrix, and $\mathbf{H}_{p \leftarrow p}$ serves as the key and value matrices, respectively. To propagate information from word nodes to the utterance node, we first perform inter-word message passing to update the representations of word nodes for capturing the interactions among words. The representation of the utterance node is then refined by aggregating information from its connected word nodes:
$$ \mathbf{H}_{w \leftarrow w} = GAT(\tilde{\mathbf{H}}_w', \tilde{\mathbf{H}}_w') \quad (14) $$
$$ \mathbf{h}_{u \leftarrow w} = GAT(\tilde{\mathbf{h}}_u, \mathbf{H}_{w \leftarrow w}) \quad (15) $$
where $\mathbf{H}_{w \leftarrow w}$ and $\mathbf{h}_{u \leftarrow w}$ are updated representations of word and utterance nodes, respectively. In $GAT(\tilde{\mathbf{h}}_u, \mathbf{H}_{w \leftarrow w})$, $\tilde{\mathbf{h}}_u$ acts as a query vector, and $\mathbf{H}_{w \leftarrow w}$ is projected to construct the key and value matrices. In this way, HierGAT updates and learns hierarchy-aware node representations through the hierarchical graph layer at three linguistic levels.

### E. Aspect Assessments on Nodes

The proposed HierGAT model is a unified architecture that can be optimized in an end-to-end manner using the mean square error (MSE) loss for each aspect at different linguistic levels. Once the aspect representations are obtained, a fully-connected layer acting as the regressor is in turn employed to calculate the corresponding aspect score sequence.

**Aspect Assessments via Phone-level Nodes:** For the phone-level node aspect assessment, we first concatenate phone node representations with their corresponding word node representations, which are then activated by the ReLU function to derive the aspect representations $\mathbf{H}_p$ for phone nodes:
$$ \mathbf{H}_p = \sigma(\mathbf{W}_p [\mathbf{H}_{p \leftarrow p} || \mathbf{H}_{w \leftarrow w}^{p}]) \quad (16) $$
$\mathbf{H}_{w \leftarrow w}^{p}$ is a sequence of augmented word-level node representations, repeated for each phone node based on the phone-to-word connections. Next, the regression head is built on top of $\mathbf{H}_p$ to access phone accuracy scores.

**Aspect Assessments via Word-level Nodes:** For the word-level node aspect assessments, the word node representations are first concatenating with the average representations of their constituent phone nodes:
$$ \mathbf{H}_w = \sigma(\mathbf{W}_w [\mathbf{H}_{w \leftarrow w} || \bar{\mathbf{H}}_{p \leftarrow p}^{w}]) \quad (17) $$
where $\bar{\mathbf{H}}_{p \leftarrow p}^{w} = (\bar{\mathbf{h}}_{w1}, \bar{\mathbf{h}}_{w2}, ..., \bar{\mathbf{h}}_{wM})$, with $\bar{\mathbf{h}}_{wm}$ being the average vector of constituent phone-level representations derived from $\mathbf{H}_{p \leftarrow p}$ for the $m$-th word. Afterward, an aspect attention mechanism is introduced to capture the relatedness among the aspects [20], [41]. Specifically, for the $j$-th word-level aspect, the intermediate aspect representations $\hat{\mathbf{H}}^{w_j}$ are linearly projected from $\mathbf{H}_w$, and a multi-head cross-attention (MHCA) with a masking strategy is followed to derive word-level aspect representations $\mathbf{H}^{w_j}$ from a collection of all intermediate representations $\mathbf{H}^{w \setminus j} = \{\hat{\mathbf{H}}^{w_1}, \hat{\mathbf{H}}^{w_2}, \dots, \hat{\mathbf{H}}^{w_{N_w}}\}$. The following equations illustrate the operations of aspect attention:
$$ \hat{\mathbf{H}}^{w_j} = \mathbf{W}_{w_j} \cdot \mathbf{H}_w + \mathbf{b}_{w_j} \quad (18) $$
$$ \mathbf{H}^{w_j} = MHCA(\hat{\mathbf{H}}^{w_j}, \mathbf{C}^w) \quad (19) $$
where $\mathbf{W}_{w_j}$ and $\mathbf{b}_{w_j}$ are aspect specific projection weights. In the operation of MHCA, $\hat{\mathbf{H}}^{w_j}$ is linearly projected as query matrix, while $\mathbf{C}^w$ serves as key and value matrices. The masking strategy ensures that the output representation at a specific position is only influenced by the other aspects of the word. Lastly, the aspect representations $\mathbf{H}^{w_j}$ are taken as the input to the corresponding regressor for evaluating the $j$-th word-level pronunciation aspect.

**Utterance-level Node Aspect Estimations:** For the utterance-level node aspect assessments, the node representations $\mathbf{H}_{p \leftarrow p}$ and $\mathbf{H}_{w \leftarrow w}$ are individually fed into an attention pooling mechanism to obtain holistic vector representations $\bar{\mathbf{h}}_{p \leftarrow p}$ and $\bar{\mathbf{h}}_{w \leftarrow w}$ at the phone and word levels, respectively. The utterance node representation $\mathbf{h}_u$ is then generated by packing these vectors together via concatenation and projection:
$$ \bar{\mathbf{h}}_{p \leftarrow p} = AttPool_p (\mathbf{H}_{p \leftarrow p}) \quad (20) $$
$$ \bar{\mathbf{h}}_{w \leftarrow w} = AttPool_w (\mathbf{H}_{w \leftarrow w}) \quad (21) $$
$$ \mathbf{h}_u = \sigma(\mathbf{W}_u [\bar{\mathbf{h}}_{p \leftarrow p} || \bar{\mathbf{h}}_{w \leftarrow w} || \mathbf{h}_{u \leftarrow w}]) \quad (22) $$
where $\sigma$ is the ReLU function, and $\mathbf{W}_u$ and $\mathbf{b}_u$ are trainable parameters. After that, an aspect attention mechanism is performed on $\mathbf{h}_u$ to derive various aspect representations $\mathbf{h}^{u_j}$, which are then passed through regression heads to derive the utterance-level proficiency scores.

**Model Optimization:** The total loss is computed as a weighted sum of the MSE losses from different levels, where the loss at each linguistic level is calculated as an average of multiple aspects:
$$ \mathcal{L}_{APA} = \frac{\lambda_p}{N_p} \sum \mathcal{L}_{p_{in}} + \frac{\lambda_w}{N_w} \sum \mathcal{L}_{w_{in}} + \frac{\lambda_u}{N_u} \sum \mathcal{L}_{u_{in}} \quad (23) $$
where $\lambda_p, \lambda_w$, and $\lambda_u$ are phone-level, word-level, and utterance-level losses at disparate aspects, respectively; $\lambda_p, \lambda_w$, and $\lambda_u$ are adjustable parameters controlling the influence of different granularities; and $N_p, N_w$, and $N_u$ refer to the numbers of aspects at phone, word, and utterance levels.

## IV. EXPERIMENTAL SETUPS

### A. Experimental Data

We conducted APA experiments on the speechocean762 dataset, a publicly available open-source dataset specifically designed for multi-aspect and multi-granular pronunciation assessment [21]. This dataset contains 5000 English-speaking recordings spoken by 250 Mandarin L2 learners, each of which has 2500 utterances. Training and test sets are of equal size, each of which has 2500 utterances. This corpus contains comprehensive annotation information, and the pronunciation proficiency scores were evaluated at multiple linguistic granularities alongside disparate aspects. **Table I** summarizes the detailed statistics of the used speech corpus. Each score was independently assigned by five experts using the same rubrics, and the final score was determined by selecting the median value from the five scores.

### B. Pronunciation Feature Extraction

**GOP Feature:** To extract the GOP feature, we first aligned audio signals $X$ with the text prompt $T$ by using an ASR model to obtain the timestamps for each phone in the canonical phone sequence. Next, frame-level phonetic posterior probabilities were produced by the ASR model and then averaged over time based on the phone-level timestamps. The resulting phone-level posterior probabilities are converted into a GOP feature vector as a combination of log phone posterior (LPP) and log posterior ratio (LPR). Owing to the used ASR model containing 42 phones, the GOP feature of a canonical phone $p$ was thus represented by an 84-dimensional vector:
$$ [LPP(p_1), \dots, LPP(p_{42}), LPR(p_1|p), \dots, LPR(p_{42}|p)] \quad (24) $$
$$ LPP(p_i) = \log p(p_i|\mathbf{o}; t_s, t_e) = \frac{1}{t_e - t_s + 1} \sum_{t=t_s}^{t_e} \log p(p_i|o_t) \quad (25) $$
$$ LPR(p_i|p) = \log p(p_i|\mathbf{o}; t_s, t_e) - \log p(p|\mathbf{o}; t_s, t_e) \quad (26) $$
where LPR is the log posterior ratio between phones $p_i$ and $p$; $t_s$ and $t_e$ are the start and end timestamps of phone $p$, and $o_t$ is the input acoustic observation of the time frame $t$.

**Energy Feature:** The energy feature is a 7-dimensional vector comprised of statistics (viz. [mean, std, median, max, min, RMSE]) employed to compute energy value for each time frame, with 25-millisecond windows and a stride of 10 milliseconds.

**Duration Feature:** The duration feature is a 1-dimensional vector indicating the length of each phone segment in seconds.

**Log Mel-filterbank Feature:** The log Mel-filterbank feature is an 80-dimensional vector computed over 25-millisecond windows with 10-millisecond strides, which are then averaged over each phone segment to from the corresponding phone-level feature.

### C. Implementation Details

**Model Configurations:** In accordance with [19], we normalized utterance-level and word-level scores to the same scale as the phone-level score [0, 2] for training APA models. Both the feature encoders at the phone and word levels consist of three blocks, each with a single-head attention mechanism and 24 hidden units. The proposed hierarchical graph layer consists of 3 stack graph attention layers, each with a single attention head and a hidden size of 24.

**Training Configurations:** In the training phase, we use a batch size of 25 and apply Adam optimizer with a learning rate 1e-3. To ensure the reliability of our experimental results, we repeated 5 independent trials, each consisting of 100 epochs using different random seeds with a learning rate scheduler that warms up at the beginning and cuts in half every five epochs after the 20-th epoch. The experimental results are reported by averaging 100 experiments with the minimum phone-level MSE values, where the mean and standard deviation values for different evaluation metrics, as described below, are reported.

**Evaluation Metrics:** The primary evaluation metric is PCC, which measures the linear correlation between predicted scores and ground-truth scores. In addition, mean squared error (MSE) is used to assess phone-level accuracy.

### D. Compared Methods

We first report the inter-annotator agreement for the five annotators (**Human-agreement**), and compare the proposed model with the following top-of-the-line methods:

- **Lin2021 [14]:** Uses a single-aspect multi-granular pronunciation scorer with a hierarchical architecture.
- **Kim2022 [28]:** Employs a single-aspect pronunciation assessment model designed to separately measure oral skills on the utterance level.
- **LSTM [19]:** This model frames multi-aspect and multi-granular pronunciation assessment as sequential labeling tasks, deriving a sequence of phone-level features and utilizing a 3-layer LSTM.
- **GOPT [19]:** Extends the sequential modeling strategy by replacing the backbone model of LSTM with a 3-stacked Transformer block.
- **Ryu2023 [40]:** This method leverages a unified model architecture that adopts a self-supervised model as the backbone model, which is optimized with phone recognition and utterance-level pronunciation assessment tasks jointly.
- **Gradformer (GFR) [42]:** Models multi-aspect and multi-granular pronunciation assessment tasks with a granularity-decoupled Transformer network.
- **HiPAMA [41]:** Built on top of a hierarchical architecture for multi-aspect and multi-granular pronunciation assessment, using a simple average pooling mechanism.
- **3M [17]:** Enhances the input features of GOPT with three types of SSL-based features to capture supra-segmental pronunciation cues.
- **HierCB [56]:** A cutting-edge APA model with a hierarchical neural structure, stacking multiple ConvBFR blocks at three linguistic granularities.

## V. EXPERIMENTAL RESULTS

### A. Qualitative Analysis

**Distributions of Aspect Scores:** Before launching into a series of experiments on the APA tasks, we perform quantitative analysis on the score distributions of aspects across different linguistic granularities on both the training and test sets. As shown in **Fig. 5**, the speechocean762 is a well-curated dataset, where though the majority of aspect scores skew towards high proficiency scores, the distributional trends are consistent between the training and test sets. Furthermore, both the distributions of utterance-completeness and word-stress demonstrate a notable high-score-biased phenomenon.

**Qualitative Visualization of Attention Weights in the Aspect Attention Mechanisms:** In the second set of experiments, we examine the relatedness among disparate aspects at both word and utterance levels on the training set by analyzing the attention weights of the aspect attention mechanisms when assessing a specific aspect score. **Fig. 6(a)** presents attention weights among the word-level aspects, which reveals the attention weights for the assessments on the accuracy and total aspects are influenced by various other aspects. In contrast, the aspect of stress is a specific evaluation task concerned with identifying emphasis on particular syllables within a word, resulting in attention weights being focused on itself [52]. We then move to analyzing the relatedness among the utterance-level pronunciation aspects. As shown in **Fig. 6(b)**, the attention weights for the prosody and the total aspects are more uniformly distributed, whereas the fluency aspect is primarily complemented by the prosody aspect. This could be attributed to the fact that the total and prosody scores measure holistic oral skills, including speaking style, rhythm, and intonation.

### B. Main Results

**Table II** presents the APA results on the speechocean762 dataset, organized into two groups, where the first group includes the results of models built upon the GOP-based features, while the second group for other models utilizing the SSL-based features. Furthermore, for fair comparisons, we report on the performance of GOPT and HierGAT variants, where the input features of these models are enhanced by concatenating GOP features with three types of SSL-based features.

With respect to the models built on the GOP-based features (the first group of **Table II**), we can make the following observations. First, on the whole, our model (HierGAT) consistently outperforms human-annotator agreement on all assessment tasks, expect for the aspect of utterance-completeness. Second, Lin2020, a single-aspect assessment method, fails to harness the dependency between aspects through the multi-task learning scheme, resulting in inferior performance compared to other multi-aspect and multi-granular pronunciation assessment models. Third, compared to the baseline methods with the parallel modeling techniques, HierGAT excels on most assessment tasks, particularly for assessments of higher linguistic granularities (utterance and word levels), achieving average improvements of 9.94%, and 8.28% over LSTM, and GOPT, respectively. This performance gain underscores the significance of capturing the hierarchical structure of an utterance when modeling cross-linguistic relationships with the proposed hierarchical graph layer.

### C. Ablation Studies

To better understand the contributions of different modules to the performance of HierGAT, we conduct here a series of ablation studies for in-depth analysis.

**Comparison of Model Components:** The first part of **Table III** presents an ablation study with the following settings: 1) replacing the concatenation operator with a weighted average mechanism for merging two branches in both phone and word encoders [53], 2) removing the aspect attention mechanism, and 3) replacing the hierarchical graph layer with a simple attention pooling. First, we can observe that the weighted average mechanism is slightly worse than the concatenation operator, where we performance drops at phone and utterance levels and a modest improvement at the word level. Next, we notice the performance significantly declines at the utterance level and slightly drops at the word-level when the aspect attention mechanisms are removed from the proposed hierarchical architecture. The proposed aspect attention mechanism can effectively leverage the relatedness among aspects, as evident by the proportional decrease in performance corresponding to the number of aspects at different linguistic granularities. Finally, the employ of the hierarchical graph layer is indispensable for HierGAT, as the removal of such a layer leads to performance degrades for all linguistic granularities.

## VI. CONCLUSION

In this paper, we have proposed HierGAT, a hierarchical graph-based architecture for automatic pronunciation assessment. Notably, we are the first to explore constructing a heterogeneous graph network to streamline the three linguistic units for the pronunciation assessment. Evaluation on the speechocean762 benchmark datasets proves the effectiveness of HierGAT and demonstrates capturing the language hierarchy and interactions between pronunciation aspects are beneficial to the assessments.

---

## References

1. A. Van Moere and R. Downey, “Technology and artificial intelligence in language assessment,” in Handbook of Second Language Assessment. Boston, MA, USA: De Gruyter Mouton, 2016, pp. 341–357.
2. M. Eskenazi, “An overview of spoken language technology for education,” Speech Commun., vol. 51, no. 10, pp. 832–844, 2009.
3. K. Evanini and X. Wang, “Automated speech scoring for nonnative middle school students with multiple task types,” in Proc. Annu. Conf. Int. Speech Commun. Assoc., 2013, pp. 2435–2439.
4. K. Evanini, M. C. Hauck, and K. Hakuta, “Approaches to automated scoring of speaking for K–12 English language proficiency assessments,” ETS Res. Rep. Ser., vol. 2017, pp. 1–11, 2017.
5. K. Li, X. Wu, and H. Meng, “Intonation classification for L2 English speech using multi-distribution deep neural networks,” Comput. Speech Lang., vol. 43, pp. 18–33, 2017.
6. S. Banno, B. Balusu, M. J. F. Gales, K. M. Knill, and K. Kyriakopoulos, “View-specific assessment of L2 spoken English,” in Proc. Annu. Conf. Int. Speech Commun. Assoc., 2022, pp. 4471–4475.
7. S. M. Witt and S. J. Young, “Phone-level pronunciation scoring and assessment for interactive language learning,” Speech Commun., vol. 30, no. 2/3, pp. 95–108, 2000.
8. K. Li, X. Qian, and H. Meng, “Mispronunciation detection and diagnosis in L2 English speech using multi-distribution deep neural networks,” IEEE/ACM Trans. Audio Speech Lang. Process., vol. 25, no. 1, pp. 193–207, Jan. 2017.
9. S. Mao, F. Soong, Y. Xia, and J. Tien, “A universal ordinal regression for assessing phone-level pronunciation,” in Proc. IEEE Int. Conf. Acoust. Speech, Signal Process., 2022, pp. 6807–6811.
10. L. Ferrer, H. Bratt, C. Richey, H. Franco, V. Abrash, and K. Precoda, “Classification of lexical stress using spectral and prosodic features for computer-assisted language learning systems,” Speech Commun., vol. 69, pp. 31–45, 2015.
11. D. Korzekwa et al., “Detection of lexical stress errors in non-native (L2) English with data augmentation and attention,” in Proc. Annu. Conf. Int. Speech Commun. Assoc., 2021, pp. 3915–3919.
12. E. Coutinho et al., “Assessing the prosody of non-native speakers of English: Measures and feature sets,” in Proc. Lang. Resour. Eval. Conf., 2016, pp. 1328–1332.
13. C. Cucchiarini et al., “Quantitative assessment of second language learners’ fluency by means of automatic speech recognition technology,” J. Acoust. Soc. Amer., vol. 107, no. 2, pp. 989–999, 2000.
14. B. Lin and L. Wang, “Deep feature transfer learning for automatic pronunciation assessment,” in Proc. Annu. Conf. Int. Speech Commun. Assoc., 2021, pp. 4438–4442.
15. A. Vaswani et al., “Attention is all you need,” in Proc. Adv. Neural Inf. Process. Syst., 2017, pp. 5998–6008.
16. J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: Pretraining of deep bidirectional transformers for language understanding,” in Proc. Conf. North Amer. Chapter Assoc. Comput. Linguistics, 2019, pp. 4171–4186.
17. F.-A. Chao, T.-H. Lo, T.-I. Wu, Y.-T. Sung, and B. Chen, “3M: An effective multi-view, multigranularity, and multi-aspect modeling approach to English pronunciation assessment,” in Proc. IEEE Asia-Pac. Signal Inf. Process. Assoc. Annu. Summit Conf., 2022, pp. 575–582.
18. H. Do, Y. Kim, and G. G. Lee, “Score-balanced loss for multi-aspect pronunciation assessment,” in Proc. Annu. Conf. Int. Speech Commun. Assoc., 2023, pp. 4998–5002.
19. Y. Gong, Z. Chen, I.-H. Chu, P. Chang, and J. Glass, “Transformer-based multi-aspect multigranularity non-native English speaker pronunciation assessment,” in Proc. IEEE Int. Conf. Acoust., Speech, Signal Process., 2022, pp. 7262–7266.
20. R. Ridley, L. He, X.-Y. Dai, S. Huang, and J. Chen, “Automated crossprompt scoring of essay traits,” in Proc. AAAI Conf. Artif. Intell., 2021, vol. 35, pp. 13745–13753.
21. J. Zhang et al., “Speechocean762: An open-source non-native English speech corpus for pronunciation assessment,” in Proc. Annu. Conf. Int. Speech Commun. Assoc., 2021, pp. 3710–3714.
22. A. Baevski, Y. Zhou, A. Mohamed, and M. Auli, “Wav2vec 2.0: A framework for self-supervised learning of speech representations,” in Proc. Adv. Neural Inf. Process. Syst., 2020, pp. 124449–12460.
23. W.-N. Hsu, B. Bolte, Y.-H. H. Tsai, K. Lakhotia, R. Salakhutdinov, and A. Mohamed, “HuBERT: Self-supervised speech representation learning by masked prediction of hidden units,” IEEE Trans. Audio Speech Lang. Process., vol. 29, pp. 3451–3460, 2021.
24. S. Chen et al., “WavLM: Large-scale self-supervised pre-training for full stack speech processing,” IEEE J. Sel. Topics Signal Process., vol. 16, no. 6, pp. 1505–1518, Oct. 2022.
25. E. B. Page, “Statistical and linguistic strategies in the computer grading of essays,” in Proc. Conf. Comput. Linguistics, 1967, pp. 1–13.
26. W. Hu, Y. Qian, F. K. Soong, and Y. Wang, “Improved mispronunciation detection with deep neural network trained acoustic models and transfer learning based logistic regression classifiers,” Speech Commun., vol. 67, pp. 154–166, 2015.
27. Y. Qian et al., “Neural approaches to automated speech scoring of monologue and dialogue responses,” in Proc. IEEE Int. Conf. Acoust., Speech, Signal Process., 2019, pp. 8112–8116.
28. E. Kim, J.-J. Jeon, H. Seo, and H. Kim, “Automatic pronunciation assessment using self-supervised speech representation learning,” in Proc. Annu. Conf. Int. Speech Commun. Assoc., 2022, pp. 1411–1415.
29. J. Shi, n. Huo, and Q. Jin, “Context-aware goodness of pronunciation for computer-assisted pronunciation training,” in Proc. Annu. Conf. Int. Speech Commun. Assoc., 2020, pp. 3057–3061.
30. B.-C. Yan, H.-W. Wang, Y.-C. Wang, and B. Chen, “Effective graph-based modeling of articulation traits for mispronunciation detection and diagnosis,” in Proc. IEEE Int. Conf. Acoust., Speech, Signal Process., 2023, pp. 1–5.
31. C. Richter and J. Guðnason, “Relative dynamic time warping comparison for pronunciation errors,” in Proc. IEEE Int. Conf. Acoust., Speech, Signal Process., 2023, pp. 1–5.
32. Q.-T. Truong, T. Kato, and S. Yamamoto, “Automatic assessment of L2 English word prosody using weighted distances of F0 and intensity contours,” in Proc. Annu. Conf. Int. Speech Commun. Assoc., 2018, pp. 2186–2190.
33. C. Graham and F. Nolan, “Articulation rate as a metric in spoken language assessment,” in Proc. Annu. Conf. Int. Speech Commun. Assoc., 2019, pp. 3564–3568.
34. S. Sudhakara, M. K. Ramanathi, C. Yarra, and P. K. Ghosh, “An improved goodness of pronunciation (GOP) measure for pronunciation evaluation with DNN-HMM system considering hmm transition probabilities,” in Proc. Annu. Conf. Int. Speech Commun. Assoc., 2019, pp. 954–958.
35. S. Mao, Z. Wu, R. Li, X. Li, H. Meng, and L. Cai, “Applying multitask learning to acoustic-phonemic model for mispronunciation detection and diagnosis in L2 English speech,” in Proc. IEEE Int. Conf. Acoust., Speech, Signal Process., 2018, pp. 6254–6258.
36. W.-K. Leung, X. Liu, and H. Meng, “CNN-RNN-CTC based end-to-end mispronunciation detection and diagnosis,” in Proc. IEEE Int. Conf. Acoust., Speech, Signal Process., 2018, pp. 8132–8136.
37. B.-C. Yan, M.-C. Wu, H.-T. Hung, and B. Chen, “An end-to-end mispronunciation detection system for L2 English speech leveraging novel anti-phone modeling,” in Proc. Annu. Conf. Int. Speech Commun. Assoc., 2020, pp. 3032–3036.
38. D. Y. Zhang, S. Saha, and S. Campbell, “Phonetic RNN-transducer for mispronunciation diagnosis,” in Proc. IEEE Int. Conf. Acoust., Speech, Signal Process., 2023, pp. 1–5.
39. B. Lin, L. Wang, X. Feng, and J. Zhang, “Automatic scoring at multigranularity for L2 pronunciation,” in Proc. Annu. Conf. Int. Speech Commun. Assoc., 2020, pp. 3022–3026.
40. H. Ryu, S. Kim, and M. Chung, “A joint model for pronunciation assessment and mispronunciation detection and diagnosis with multi-task learning,” in Proc. Annu. Conf. Int. Speech Commun. Assoc., 2023, pp. 959–963.
41. H. Do, Y. Kim, and G. G. Lee, “Hierarchical pronunciation assessment with multi-aspect attention,” in Proc. IEEE Int. Conf. Acoust., Speech, Signal Process., 2023, pp. 1–5.
42. H.-C. Pei, H. Fang, X. Luo, and X.-S. Xu, “Gradformer: A framework for multi-aspect multi-granularity pronunciation assessment,” IEEE/ACM Trans. Audio, Speech, Lang. Process., vol. 32, pp. 554–563, 2024.
43. P. Muller, F. De Wet, C. Van Der Walt, and T. Niesler, “Automatically assessing the oral proficiency of proficient L2 speakers,” in Proc. Workshop Speech Lang. Technol. Educ., 2009, pp. 29–32.
44. H. Franco et al., “EduSpeak: A speech recognition and pronunciation scoring toolkit for computer-aided language learning applications,” Lang. Testing, vol. 27, no. 3, pp. 401–418, 2010.
45. K. Laskowski, J. Edlund, and M. Heldner, “An instantaneous vector representation of delta pitch for speaker-change prediction in conversation dialogue system,” in Proc. IEEE Int. Conf. Acoust., Speech, Signal Process., 2008, pp. 5041–5044.
46. C. Cucchiarini, H. Strik, and L. Boves, “Quantitative assessment of second language learners’ fluency by means of automatic speech recognition technology,” J. Acoust. Soc. Amer., vol. 107, no. 2, pp. 989–999, 2000.
47. K. Li, S. Mao, X. Li, Z. Wu, and H. Meng, “Automatic lexical stress and pitch accent detection for L2 English speech using multi-distribution deep neural networks,” Speech Commun., vol. 96, pp. 28–36, 2018.
48. L. Chen, J. Tao, S. Ghaffarzadegan, and Y. Qian, “End-to-end neural network based automated speech scoring,” in Proc. IEEE Int. Conf. Acoust., Speech, Signal Process., 2018, pp. 6234–6238.
49. W. Liu et al., “An ASR-free fluency scoring approach with self-supervised learning,” in Proc. IEEE Int. Conf. Acoust., Speech, Signal Process., 2023, pp. 1–5.
50. K. Fu, S. Gao, S. Shi, X. Tian, W. Li, and Z. Ma, “Phonetic and prosodyaware self-supervised learning approach for non-native fluency scoring,” in Proc. Annu. Conf. Int. Speech Commun. Assoc., 2023, pp. 949–953.
51. S. Cheng, Z. Liu, L. Li, Z. Tang, D. Wang, and T. F. Zheng, “ASR-free pronunciation assessment,” in Proc. Annu. Conf. Int. Speech Commun. Assoc., 2020, pp. 3047–3051.
52. D. Korzekwa, J. Lorenzo-Trueba, T. Drugman, and B. Kostek, “Computerassisted pronunciation training—Speech synthesis is almost all you need,” Speech Commun., vol. 142, pp. 22–33, 2022.
53. Y. Peng, S. Dalmia, I. Lane, and S. Watanabe, “Branchformer: Parallel MLP-attention architectures to capture local and global context for speech recognition and understanding,” in Proc. Int. Conf. Learn. Representations, 2022, pp. 17627–17643.
54. P. Velickovic, G. Cucurull, A. Casanova, A. Romero, P. Lio, and Y. Bengio, “Graph attention networks,” in Proc. Int. Conf. Learn. Representations, 2018.
55. A. Gulati et al., “Conformer: Convolution-augmented transformer for speech recognition,” in Proc. Annu. Conf. Int. Speech Commun. Assoc., 2020, pp. 5036–5040.
56. B.-C. Yan, Y.-C. Wang, J.-T. Li, H.-W. Wang, W.-C. Chao, and B. Chen, “ConPCO: Preserving phoneme characteristics for automatic pronunciation assessment leveraging contrastive ordinal regularization,” 2024, arXiv:2406.02859.
