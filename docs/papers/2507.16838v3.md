# Segmentation-Free Goodness of Pronunciation

**Xinwei Cao, Zijian Fan, Torbjørn Svendsen**, *Senior Member, IEEE*, **Giampiero Salvi**, *Senior Member, IEEE*

---

## Abstract

Mispronunciation detection and diagnosis (MDD) is a significant part in modern computer-aided language learning (CALL) systems. Most systems implementing phoneme-level MDD through goodness of pronunciation (GOP), however, rely on pre-segmentation of speech into phonetic units. This limits the accuracy of these methods and the possibility to use modern CTC-based acoustic models for their evaluation. In this study, we first propose self-alignment GOP (GOP-SA) that enables the use of CTC-trained ASR models for MDD. Next, we define a more general segmentation-free method that takes all possible segmentations of the canonical transcription into account (GOP-SF). We give a theoretical account of our definition of GOP-SF, an implementation that solves potential numerical issues as well as a proper normalization which allows the use of acoustic models with different peakiness over time. We provide extensive experimental results on the CMU Kids and speechocean762 datasets comparing the different definitions of our methods, estimating the dependency of GOP-SF on the peakiness of the acoustic models and on the amount of context around the target phoneme. Finally, we compare our methods with recent studies over the speechocean762 data showing that the feature vectors derived from the proposed method achieve state-of-the-art results on phoneme-level pronunciation assessment.

**Index Terms** — Computer-aided pronunciation training, mispronunciation detection and diagnosis, speech assessment, goodness of pronunciation, CTC, child speech, L2.

---

## I. Introduction

Computer-aided language learning (CALL) and computer-assisted pronunciation training (CAPT) are becoming more important and helpful among language learners and teachers both because they are ubiquitously available and because they maintain a high degree of self-controlled manner of study. One of the desirable features for these systems is the ability to provide instant feedback and intervention when the learner makes any mispronunciation. However, the automatic mispronunciation detection and/or diagnosis (MDD) modules currently available are not sufficiently reliable.

MDD can be performed at different linguistic levels: phoneme, word, and utterance. One of the main challenges with MDD is the scarcity of data that is specifically annotated for the task, including information about mispronunciations. This problem is especially severe for phoneme-level assessment which is the focus of this paper.

Witt et al. [1] proposed a widely used method for MDD at the phoneme level that is based on a measure called goodness of pronunciation (GOP). The advantage of this method is that it relies on acoustic models exclusively trained for automatic speech recognition (ASR). The ASR models are used to score new utterances, and only a small amount of MDD annotations are required to optimize a detection threshold that separates acceptable pronunciations from mispronunciations. ASR models can also be used to transcribe speech at the word, character, or phoneme level and detect mispronunciations by comparing the resulting output sequence with the canonical pronunciation or orthographic transcription, respectively [2], [3], [4], [5], [6].

Another approach is to train end-to-end models for MDD. However, training such models from scratch would require large amounts of MDD annotated data. A solution is to use foundation models that are either trained on large amounts of unlabeled speech data, or fine-tuned for ASR. These models can be further fine-tuned with small amounts of MDD annotated data for the MDD task. For example, Xu et al. [7] follow this strategy starting from a Wav2Vec2.0 model [8] whereas, the authors of [9], [10] start from a HuBERT model [11]. Liu et al. [12] attempt to perform MDD at utterance-level based on hidden representations directly derived from the foundation models without further fine-tuning.

In this paper, our goal is to combine the advantages of the GOP method, the foundation models, and end-to-end ASR training based on CTC loss [13]. This raises a number of challenges due to the fact that GOP requires segmentation of speech into phonetic units. This is typically obtained by forced alignment of the canonical pronunciation with the spoken utterance. If the pronunciation is correct, the obtained phonetic boundaries may vary due to coarticulation effects. In case of mispronunciations, the segmentation may be even less reliable. Finally, CTC trained models tend to give activations that are not necessarily aligned with acoustic speech segmentation. In [14] we introduced a framework for combining CTC trained models and GOP. In particular, we introduced 1) a self-aligned GOP method that uses the same activations from CTC trained models for alignment and GOP evaluation, and 2) a segmentation-free GOP method that can assess a specific speech segment without committing to a particular segmentation.

In this paper, we enhance these methods by making the following novel contributions:

- We define proper segment length normalization for our segmentation-free GOP definition without committing to a particular segmentation.
- We provide a theoretical derivation of the segmentation-free GOP method that exposes the assumptions required for its definition.
- We introduce a novel implementation of our method that eliminates numerical problems.
- We provide extensive experimental results that compare the different methods we propose to the state-of-the-art on binary and ternary MDD tasks.

---

## II. Background and Related Work

In this section we give background information and review the related work that are necessary to understand the proposed method. We will focus on phoneme-level pronunciation assessment that is the goal of this paper. We first need to distinguish between continuous measurement for pronunciation assessment and the specific task of MDD. In the first case, the goal is to introduce a metric that can indicate how a specific pronunciation of a phoneme is from the acceptable variability in the language. An example of this kind of metric is GOP. The MDD task, on the other hand, is to provide a binary or sequences ternary decision on whether the pronunciation is correct or not. This task can be performed at different linguistic levels on a single GOP-like score by defining thresholds for the different output classes.

### A. The definition of GOP

GOP was initially proposed as a measure of how closely the pronunciation of a specific phoneme matches its expected canonical pronunciation [1]. This is a segmental measure that it relies on acoustic models exclusively trained for automatic speech recognition (ASR). The ASR models are used to score new utterances, and only a small amount of MDD annotations are required to optimize a detection threshold that separates acceptable pronunciations from mispronunciations.

Witt's original definition of GOP [1] corresponds to the log posterior of the canonical phoneme i, given the sequence of observations O_i^T = {o_t, ..., o_t}, in the segment under test, normalized by the sequence length:

$$\text{GOP}(l_i) = \frac{\log p(l_i | O_{t_1}^{t_2})}{t_2 - t_1}$$ (1)

Mispronunciations are detected on the basis of the GOP value and an empirically determined threshold.

The estimation of the posteriors was originally implemented using Hidden Markov Models and Gaussian Mixture Models (HMM-GMM) and by using Bayes rule:

$$p(l_i | O_{t_2}^t) = \frac{p(O_{t_2}^t | l_i) P(l_i)}{\sum_{q \in Q} p(O_{t_2}^t | q) P(q)}$$ (2)

where P(q) represents the prior probability of each phoneme in the phoneme inventory Q and the likelihood p(O_{t_2}^t | l_i) can be directly evaluated using the acoustic model, for example with the forward algorithm. For efficiency reasons, Witt approximates the summation in the denominator by obtaining the best path over the full utterance through a phone-loop model. From Eq. (2), GOP is then the ratio of the likelihood of the segment estimated from the canonical phone model (numerator) and the likelihood of the best path (allowing any phone sequence) through the segment (denominator).

The phoneme boundaries t_1 and t_2 can be obtained by human annotations. However, this is not practical: During model training, annotating large amounts of data would be too costly. More importantly, it would make the methods not suitable for providing immediate and automated feedback to the students. The segmentation step is, therefore, commonly automated by forced alignment using the canonical pronunciation L_C = {l_1, ..., l_{|Lc|}} and a phonetic model trained for ASR. This model does not need to be the same as that used for the GOP evaluation, and it is typically a context-independent HMM-GMM model.

Although Witt's GOP has been successful, many later works argue that the phone-loop approximation is not reliable in estimating the denominator of Eq. 2. For example, Lattice-based GOP [15] includes contributions from N-best hypotheses to avoid underestimating the value of the denominator. In [16], the authors show performance improvements that can be obtained when the phone-loop is evaluated multiple times over the target segment for each phoneme rather than once along the whole utterance. This implementation follows more closely the original definition of the denominator in Eq. 2.

Phonological rules from the learner's first language (L1) may also be included in GOP methods, where the acoustic models are trained with target language (L2) only, to achieve better accuracy [17], [18].

### B. The alignment issues for GOP

There are several issues with traditional definitions of GOP related to speech segmentation. The first is the ability to perform a perfect alignment of phonetic segments to the recorded speech. Phonetic segmentation is an ill-posed problem because it is based on the assumption that speech is produced as a sequence of discrete events. However, coarticulation effects in speech production question the existence of reliable phonetic boundaries. This explains the disagreement on segmentation even between trained phoneticians. An example is shown in Figure 1, where it is difficult to tell the exact beginning and end of the sonorant [w] as it happens in the transition of two vowel sounds [u] and [a].

Even assuming the existence of well-defined phonetic boundaries, alignment errors may occur in case of mispronunciations because the models used for segmentation are usually trained on correctly pronounced speech. Furthermore, as studied in [24], [25], [26] forced alignment may be unreliable due to speaker variability: age, accent, dialect, health condition, or name a few. The uncertainty over phonetic boundaries has a strong impact on traditional GOP definitions which consider the boundaries as deterministic.

Another important limitation of traditional GOP emerges from the characteristics of the models used in modern ASR training. Even assuming perfect segmentation of speech, there is no guarantee that the activations of the models used for GOP evaluation are aligned with this segmentation. A typical case is with end-to-end transformer-based models where the alignment between input speech and output symbols is somewhat arbitrary. This aspect is rarely emphasized in the literature, where the method used for alignment is often not specified.

In [27] the authors propose to use the general posterior probability (GPP) to mitigate alignment issues by allowing evaluation over any segments that overlap with the target segment. Zhang et al. [28] propose a pure end-to-end method that uses a sequence-to-sequence model without having to segment the speech. However, this method requires a large number of human annotations for MDD and an additional step that compares the canonical phoneme sequence with the human-annotated sequence before training the model.

In this work, we propose two methods to relax the dependency of GOP on the accuracy of segmentation. The first has the goal of reducing the effect of misalignment between the acoustic model activations and the speech segments, the second is completely segmentation-free. The latter also allows for detecting insertion and deletion errors as well as substitutions.

### C. End-to-end ASR models: CTC and peaky behavior

In this section, we introduce some details of the Connectionist Temporal Classification (CTC) loss used in modern end-to-end ASR models. This premise is important to understand our methods in Section III.

End-to-end ASR models were initially introduced to map speech to output symbols, typically phonemes or graphemes, directly. In general, the sequence of output symbols L = {l_1, ..., l_{|L|}} has a different rate that the input speech feature vectors O_T = {o_1, ..., o_T}. To overcome this problem in training, the CTC loss [13] was introduced. This makes use of an additional blank (∅) output symbol. Given a speech sequence O_T of length T, we define V ∅ as the set of all output symbols including ∅. Then any vector u = {u_1, ..., u_T} ∈ U = V^T is an alignment path between the input sequence and the output symbols. The probability of a path u under the assumption of conditional independence is ∏_{t=1}^T p(u_t | o_t). As a consequence ∑_{u∈U: B(u)=L} ∏_{t=1}^T p(u_t | o_t) = 1 representing all possible paths given the model and the speech. During training, for a given target output sequence L and the speech sequence O_T, the model learns to minimize the following loss:

$$\mathcal{L}_{CTC}(L) = -\log \sum_{u∈U:B(u)=L} p(π | O_T^t)$$ (4)

where π spans over the paths that can be mapped to L using the many-to-one function B : U → L by removing all repeated symbols and blank symbols, e.g., B(aaa∅bbbccc) = abc.

Since training the CTC does not require frame-level alignment, the timing information tends to be ignored by the model. A well-known phenomenon of CTC-trained model is the "peaky" behavior of model output, as illustrated in the right column of Figure 2. There are two dimensions of peakiness: "peaky over time" (POT) and "peaky over state" (POS). POT behavior corresponds to the fact that the blank symbol is found for most of the time steps, whilst the non-blank symbols that correspond to the target label sequence only become activated for a few time steps. Forced alignment (Viterbi search) performed with CTC-trained models can result in distorted alignment between input speech and output symbols. On the other hand, POT behavior in observed because the model activations at each frame are always close to 1.0 for one output and 0 for all the others.

For these reasons, CTC is not feasible for applications where the time alignment is essential, such as speech segmentation, or conventional GOP evaluation for MDD. An indication of this is the lack of literature in these areas with CTC-based models. However, several works tried to mitigate the peaky behavior of CTC-based models. ASR e.g. [29], [30], [31].

The two methods proposed in this paper, have a goal to make it possible to use CTC-trained models with phoneme output symbols as a basis for computing GOP for MDD tasks.

---

## III. Methods

The goals of our methods are (i) to make it possible to use CTC-based models for a reliable evaluation of GOP, (ii) to reduce the sensitivity of GOP to precise speech segmentation and to consider the uncertainty of phonetic alignment, (iii) to introduce context awareness into the definition of GOP and (iv) to extend the method of GOP to allow for detecting insertion and deletion errors as well as substitutions.

To achieve these goals we propose two methods that will be detailed in Section III-A and III-B.

### A. Self-alignment GOP (GOP-SA)

Firstly, we consider the problem of mismatch between the segmentation of speech into phonetic units and the activations of the models used for GOP estimation as mentioned in Section II-B. We propose to use the same GOP definition as for DNN models (Eq. 12) to perform the alignment of the target segment (l_i, t_2) based on the same model used for GOP evaluation instead of an external forced aligner. We refer to this method as self-alignment GOP (GOP-SA). Figure 2 shows this alignment with red dashed vertical lines for the CTC-less trained models. We want to stress that the goal of alignment in this method is not to find the segment corresponding to the target phoneme, but rather, to find the activations of the model corresponding to the target phoneme.

We hypothesize that using CTC-based models for GOP would reduce the impact of the alignment errors due to mispronunciations which we mentioned in Sec II-B. We will show that this method leads to improvements in MDD accuracy not only when using the standard definition of GOP but also when using alternative methods to compute the loss, besides standard CTC.

### B. Segmentation-free GOP (GOP-SF)

The second method is segmentation-free (SF) and evaluates the GOP for a particular speech segment without the need for explicit segmenting the utterance end. The motivation for this method is to overcome the following limitations of standard GOP: 1) the evaluation of pronunciation of each phoneme is exclusively based on the corresponding speech segment (see Figure 1); 2) the evaluation of GOP is sensitive to the specific alignment (i, t_2) and does not take into account the uncertainty in alignment; 3) the common implementation of GOP uses Viterbi decoding and therefore only considers one path through the model possibly leaving out part of the probability mass, finally, 4) standard GOP does not allow for insertion and deletion errors.

We assume that we have recorded an utterance with a canonical transcription L_C with l_1, ..., l_{|L_C|} and canonical transcription L_C = {l_1, ..., l_{|Lc|}}. We are interested in evaluating the pronunciation of a phone l_i. We can therefore split the canonical transcription into three parts, the left context (L_L), the target phone l_i, and the right context (L_R):

$$L_C = \{l_1, \ldots, l_{i-1}, \quad l_i, \quad l_{i+1}, \ldots, l_{|Lc|}\}$$

As in previous sections, we define t_1 and t_2 as the start and end frame indices for the target phone l_i. Instead of committing to a specific segmentation in standard GOP, in the proposed segmentation-free GOP (GOP-SF) we compute the log posterior for the target phone l_i given the full observation sequence O_T and all the phonemes in the left and right context:

$$\text{GOP-SF}(l_i) = \log(p(l_i | O_T^t, L_L, L_R))$$ (5)

This is the same definition that we introduced in [14], although not explicitly written as in Eq. 5, there.

In this work, we also introduce a version of this definition that is normalized by the estimated length of the model activations for the target speech segment:

$$\text{GOP-SF-Norm}(l_i) = \frac{\log(p(l_i | O_T^t, L_L, L_R))}{\mathbb{E}[t_2 - t_1 | O_T, L_L, L_R]}$$ (6)

The reason for this new definition is to reduce the variance of the GOP estimates with the different lengths of the activations. This is similar to the standard GOP definition of Eq. 1, where the posterior is normalized by |t_2 - t_1| with the following difference: 1) our normalization factor is not related to the length of the target segment, but, rather, to the length of the activations of the model corresponding to the target segment. This accommodates both peaky and non-peaky behavior; 2) we do not commit to a specific speech segmentation and incorporate the uncertainty in the alignment into our estimation.

In the following subsections, we give both theoretical and practical accounts of how to estimate the numerator and denominator in Eqs. 5 and 6.

### C. Segmentation-free target phoneme posterior estimation

In this section we show how to compute the alignment-free estimation of the log function in Eqs. 5 and 6, that is the alignment-free estimation of the posterior for the target phoneme p(l_i | O_T^t, L_L, L_R).

We first rewrite the expression using the chain rule of probabilities:

$$p(l_i | L_L, L_R) = \frac{p(L_L, l_i, L_R | O_T^t)}{p(L_L, L_R | O_T^t)}$$ (7)

We now consider a specific alignment (t_1, t_2) for the target phone l_i and we define the set of all possible alignments as:

$$\mathcal{A}(l_i) = \{(t_1, t_2) : i - 1 < t_1 \leq t_2 < T - (|L_C| - i)\}$$ (8)

Because the left context i_L and the right context L_R contain respectively i - 1 and |L_C| - i symbols, the lower bound for t_1 and the upper bound for t_2 ensure that there is at least one frame for each of the left and right context.

We can now expand numerator and denominator of Eq. 7 by considering the specific alignment (t_1, t_2) and by summing over all possible alignments A(l_i):

$$\frac{p(l_i, l_i, L_R | O_T^t)}{p(L_L, L_R | O_T^t)} = \frac{\sum_{(t_1,t_2) \in \mathcal{A}(l_i)} p(L_L, l_i, L_R, t_1, t_2 | O_T^t)}{p(L_L, L_R | O_T^t)} = \frac{\sum_{(t_1,t_2) \in \mathcal{A}(l_i)} p(L_L, l_i, L_R, t_1, t_2 | O_T^t)}{\sum_{(t_1,t_2) \in \mathcal{A}(l_i)} p(L_L, L_R, t_1, t_2 | O_T^t)}$$ (9)

Special attention must be paid to the term p(t_1, t_2 | O_T^t) within both sums at the numerator and denominator of Eq. 9. This is the probability of a certain alignment for the i-th segment, given the observation sequence, but independent of the actual transcription. We can interpret this as a prior with respect to the transcription over all possible segmentations.

This distribution could be estimated from the training data. However, in our definition of GOP-SF, we make the simplifying assumption that this distribution is uniform. Under this assumption p(t_1, t_2 | O_T^t) is a constant and can be taken out of the sums and finally cancels out between the numerator and the denominator of Eq. 9 which becomes:

$$\frac{p(L_L, l_i, L_R | O_T^t)}{p(L_L, L_R | O_T^t)} = \frac{\sum_{(t_1,t_2) \in \mathcal{A}(l_i)} p(L_L, l_i, L_R, t_1, t_2 | O_T^t)}{\sum_{(t_1,t_2) \in \mathcal{A}(l_i)} p(L_L, L_R, t_1, t_2 | O_T^t)}$$ (10)

Finally, the terms within the sums in Eq. 10 can be computed with the CTC's forward variables α and β as defined in [13]:

$$\sum_{(t_1,t_2) \in \mathcal{A}(l_i)} α_{t_1-1}(i - 1) \prod_{j=t_1}^{t_2} y_t(l_j) β_{t_2+1}(i + 1)$$
$$\frac{\sum_{(t_1,t_2) \in \mathcal{A}(l_i)} α_{t_1-1}(i - 1) β_{t_2+1}(i + 1)}{\quad}$$ (11)

In fact, the numerator in Eq. 11 is equivalent to the original definition of the CTC loss for the canonical pronunciation L_C and the acoustic features O_T^t, except for a – log(·) term. Similarly, the denominator can be computed with the CTC's forward variables α and β as defined in [13] if we define a modified canonical pronunciation L_SDI as:

$$L_{SDI} = \{l_1, \ldots, l_{i-1}, \quad l_{i+1}, \ldots, l_{|Lc|}\}$$

This is the set of all possible transcriptions in which the left and right contexts are equal to the canonical transcription, but we allow any sequence of phonemes in place of the target phoneme l_i. This corresponds to allowing any number of "substitution", "deletion" and "insertion" errors in pronunciation.

In summary, Eq. 5 and Eq. 11 can be computed as the difference between the CTC loss for the pair (L_SDI, O_T^t) and the CTC loss for the pair (L_C, O_T^t) (see Figure 3).

Note that, because we are considering contributions from all possible segmentations, our method is able to deal with uncertainty in alignment. Also, the definition of L_SDI with a graph as in Figure 3 gives us flexibility on the kind of pronunciation errors we consider. If the full graph is used, all substitution (S), deletion (D) and insertion (I) errors are considered. We refer to this version of the method as GOP-SF-SD (substitution deletion). If we remove the blue path, we only consider substitutions as in the traditional GOP (GOP-SF-S) and the modified canonical pronunciation is

$$L_S = \{l_1, \ldots, l_{i-1}, \quad l_{i+1}, \ldots, l_{|Lc|}\}$$

In our implementation, we also consider alternative methods to compute the loss, besides standard CTC.

### D. Segmentation-free activation length estimation

We now turn to the denominator of Eq. 6, that is on the estimation of the model activation length for the target phoneme ℰ[t_2 - t_1 | O_T^t, L_L, L_R].

We call N_C the set of the central nodes (yellow and purple) in the graph in Figure 3 that correspond to the "i-th" term in L_SDI.

Then, the expected value of the duration of the activations corresponding to l_i is the sum over the whole observation sequence of the normalized forward variables α corresponding to the nodes in N_C:

$$\mathbb{E}[t_2 - t_1 | O_T^t, L_L, L_R] = \sum_{t=1}^T \sum_{s \in N_C} \bar{α}_t(s) := \text{Occ}(t)$$ (13)

This expression can be efficiently computed with the forward algorithm that is also used to estimate the posterior of the target phoneme. As detailed in the previous section, Note that the definitions of L_SDI and L_SP allow deletion of the target segment as a possible mispronunciation error. In this case, ℰ[t_2 - t_1 | O_T^t, L_L, L_R] will tend to diverge. In the actual implementation, therefore, we define a floor value of 1.

We expect this normalization factor to be most relevant for models that are not peaky in time. Peaky models tend to have activations that span over a single frame, thus making GOP-SF roughly equivalent to GOP-SF-Norm.

### E. Computational complexity

We can compute the complexity of calculating GOP-SF (Eq. 12) with the help of dynamic programming. First we note that we only need to evaluate the CTC loss twice, once for the first term L(L_C, O_T^t) and once for the second term L(L_SDI, O_T^t) (Figure 3, top) contains |G| = 2|L_C| + 1 nodes (the transcript input labels and one additional blank), whereas L(L_SDI, O_T^t) (Figure 3, bottom) contains |G| = 2|L_C| + |V| nodes, where V is the set of output symbols from the neural network. The complexity of running dynamic programming on these graphs is therefore O(T × |G|), that is, it is linear both in the length of the utterances, in the number of symbols in the canonical transcription augmented by the number of output symbols, which is usually a constant.

For GOP-SF-Norm, the only overhead is the summation in Eq. 13 because the normalized forward variables α are already computed.

### F. Segmentation-free GOP features

Similarly to the approaches in [21], [32], [33], the GOP-SF of the i-th phoneme in a canonical sequence can be expanded into a feature vector using LPP (log posterior probability) and LPR (log posterior ratio vector):

$$\text{FGOP-SF}(l_i) = \{\text{LPP}, \text{LPR}(l_i)\}$$ (14)

where LPP = log p(L_C | O_T^t) = −ℒ_{CTC}(L_C) follows the definition of CTC's log-posterior in Eq. 4 and LPR is a vector of length |L_C| − 1:

$$\text{LPR}(i) = \left\{\log \left(\frac{p(L_C | O_T^t)}{p(L_i | O_T^t)}\right) \text{ with } L \in L_{SDI}(l_i)\right\}$$ (15)

where L_{SDI}(l_i) has already been defined in Section III-C. We do not include insertion errors here because that would result in infinite length for the feature vectors. Similar to GOP-SF-Norm, we append the expected count in Eq. 13 as an extra dimension of the feature vector which forms:

$$\text{FGOP-SF-Norm}(l_i) = \{\text{LPP}, \text{LPR}(l_i), \text{Occ}(t)\}$$ (16)

The feature vectors can be fed into multi-dimensional MDD classification models that take the feature vectors as input.

### G. Measures of Peakiness

In Section II-B, we have introduced the peaky behavior of CTC-trained models both with respect to time (POT) and symbols (POS). Our two methods were introduced in part to cope with this behavior. Due to the definitions given in Section III, we expect the performance of GOP-SA (self-alignment) to increase with both POT and POS. In contrast to standard methods, we also expect GOP-SF (segmentation-free) to be less affected by POT and POS.

For a given model M:

$$\text{BC}(\mathcal{D}, M) = \frac{1}{N} \sum_{n=1}^{N} \frac{1}{T_n} \sum_{t \in L_C} p(o_t^n | M)$$ (17)

where the dataset D contains N utterances, each represented by a sequence of speech features O^n of length T_n and the corresponding transcriptions L^n. We call this the blank coverage (BC). High BC corresponds to very short activations of the non-blank output symbols and, therefore, to high peakiness in time (POT).

For POS, we borrow the idea from [30] and define the average conditional entropy (ConEn) as:

$$\text{ConEn}(\mathcal{D}, M) = \frac{1}{N} \sum_{n=1}^{N} H(p(π | L^n, O^n))$$ (18)

where

$$H(p(π | L^n, O^n)) = - \sum_{π ∈ Π(B(π)=L^n)} p(π | L^n, O^n) \log p(π | L^n, O^n)$$ (19)

Low ConEn corresponds to high peakiness in symbols (POS).

---

## IV. Experimental Settings

### A. Data

We perform our experiments on two datasets including child speech. CMU Kids and speechocean762.

CMU Kids [34] comprises 9.1 hours of speech from children in the range 6–11 years old, and a total of 5180 sentences spoken by 63 females and 24 males. In the CMU Kids, each utterance is equipped with a phonetic transcription and utterances including mispronunciations are marked. We determine the distribution of pronunciation errors by comparing the phonetic transcription with the canonical pronunciation from the CMU pronunciation dictionary provided by Kaldi [35].

There are 150899 labeled phonemes in total, among which 90.2% are correct, 3.2% are substitution errors, 2.3% are deletion errors, and 4.5% are insertion errors. We refer to these labels obtained this way as "real errors".

Following [36], we also create an alternative MDD task based on the CMU Kids data, where we systematically change each phoneme in the canonical pronunciation to any other phoneme. We call this task "simulated errors" because we pretend that the recorded speech was incorrectly pronounced.

Finally, the speechocean762 dataset [37] includes 5000 utterances read by 2670 speakers, half of which are children in the range from 5 years- to 15 years-old. The dataset is collected specifically for pronunciation assessment tasks with real annotations at different linguistic levels. We focus on the phoneme-level where each phoneme is labeled as one of the three categories: 0 (mispronounced), 1 (strongly accented) and 2 (correctly spoken). speechocean762 provides canonical phone-level transcriptions using the same phone inventory as the CMU pronunciation dictionary that we used for CMU Kids. We are also interested in comparing the results for children and adults. In order to do this we split the speechocean762 test set according to the reported age of the speaker. The children (age 6–11) account for 38.8% of the utterances whereas adults (age 12–43) account for 61.6%. Note however, that the results on speechocean762 children are not comparable to those in CMU because the two data sets define different tasks (ternary vs binary assessment).

**Note on simulated errors:** We performed only for those utterances that are marked as pronounced correctly. For the correct utterances, on the other hand, we simply label all the canonical phonemes as correct.

### B. Baseline segmentation model

The external segmentation used for all the experiments in this study is based on the same baseline segmentation model. The model is a context-independent GMM-HMM model that is trained using the Kaldi recipes "gmn-align" for obtaining the segmentation for both CMU Kids and speechocean762 for further experiments.

### C. Pre-training and fine-tuning of the acoustic models

The acoustic models "DNN" (CMU Kids baseline) and "TDNN" (speechocean762 baseline) are trained from scratch using the Kaldi recipes. All the other models are based on the wav2vec2.0-large-xlsr-53 [39] which is available on Huggingface and has been frozen, was line-tuned on the Librispeech "train-100-clean" set according to the corresponding loss functions:

- **CE: cross entropy**
- **CTC: Connectionist temporal classification**
- **EnCTC: CTC entropy regularization [30]**
- **EsCTC: CTC with equal space constraint [30]**

Fine-tuning was performed for at most 10 epochs or by early stopping based on the validation loss, with learning rate 1e-4 and batch size 32. The same model was used for the CMU Kids and speechocean762 experiments.

### D. Evaluation

The evaluation on CMU Kids is based on the canonical transcription for each utterance. Each phone in the canonical transcription is marked as mispronounced if it deviates from the phonetic transcription obtained from the data at testing because we want a robust method to evaluate the performance of our GOP definition that is not dependent on a specific threshold chosen to perform MDD task. This also allows us to compare between different GOP methods with different ranges. We compute the received operative characteristic curve (AUC-ROC) according to [41].

For the MDD-oriented dataset speechocean762 the task is to predict three different classes. In this case we train and evaluate several hyper multi-task MDD models using the standard training and test splits. We follow the recommendations from the dataset's paper [37] as MDD performance metric.

---

## V. Results

### A. Method comparison

Table I shows the results comparing the different methods on the CMU Kids data both for simulated and real errors. AUC values are reported together with 95% confidence intervals computed according to [41]. The methods based on CTC outperform both models trained with standard CTC outputs but it should be noted that the external standard GOP average is still expected, because GOP-X-Avg uses the same baseline segmentation that was used to train the CE model.

The CTC and EnCTC models are comparable. The left part of the table reports blank coverage (BC) as a measure of POT and conditional entropy (ConEn) as a measure of POS, as well as phone error rates (PER). Beam-search decoding without language models is used for recognition.

To give a reference on the quality of the different acoustic models in this study, we also report on phoneme recognition results measured with phoneme error rates (PER). Beam-search decoding without language models is used for recognition.

| Method | AUC (95% confidence interval) | Real errors |
|--------|-----|---|
| GOP-DNN-Avg [36] | 0.824 (± 1.6E-3) | 0.796 (± 1.8E-3) |
| GOP-CE-Avg | 0.967 (± 7.2E-4) | 0.851 (± 3.0E-3) |
| GOP-CTC-SA | 0.988 (± 4.3E-4) | 0.908 (± 2.1E-3) |
| GOP-CTC-SF-S | 0.989 (± 4.1E-4) | 0.891 (± 2.4E-3) |
| GOP-CTC-SF-SD | 0.986 (± 4.7E-4) | 0.914 (± 2.0E-3) |
| GOP-CTC-SF-SDI | 0.938 (± 9.0E-4) | 0.859 (± 2.9E-3) |

**Table I: Mispronunciation Detection on CMU Kids**

### B. Performance versus peakiness

Table II shows the results of testing the effect of peakiness of the acoustic models on the real errors of the CMU Kids. All the models were trained with dropout/peakiness. The left part of the table reports blank coverage (BC) as a measure of POT and conditional entropy (ConEn) as a measure of POS, as well as phone error rates (PER). Beam-search decoding without language models is used for recognition.

Standard CTC results in the highest BC and lowest ConEn, which means that CTC is the peakiest model both with respect to time and symbols. The left part of the table reports blank coverage (BC) as a measure of POT and conditional entropy (ConEn) as a measure of POS, as well as phone error rates (PER). Beam-search decoding without language models is used for recognition.

For the EnCTC models, POS decreases as the weight β of the entropy term in the loss function increases. In our tests we varied β in the range [0.0, 0.15, 0.30]). This is not surprising because the entropy term is similar to ConEn. More interesting is the fact that POT also decreases with β as shown by BC. The EnCTC performed with better phone error rates compared to standard CTC but with a much lower POS (higher ConEn). This is in line with the experiments in [30], because the model trained with CE has the lowest BC because it is trained frame-wise according to baseline segmentation. Surprisingly, the ConEn of the CE-trained model is still lower than EnCTC implying that the distribution of possible alignment paths are more concentrated for CE-trained model.

The model trained with the EsCTC has by far the worst phone recognition performance. However, the model's peakiness [sic] with the worst phone recognition performance. However, the model's peakiness [sic] both BOTH (POT and POS) may be compared with CE in MDD performance when combined with GOP-X-SA.

| Models | Peakiness and Phone Recognition | GOP Methods (AUC, 95% confidence intervals) | | |
|--------|-----|---|---|---|
| | BC (%) | ConEn | PER (%) | GOP-X-Avg | GOP-X-SA | GOP-X-SF |
| CE | 42.22 | 2.2 0886 | 13.25 | 0.860 (±2.91E-3) [14] | 0.885 (±2.50E-3) | 0.870 (±2.75E-3) |
| EnCTC-0.20 | 79.10 | 2.7 8672 | 11.63 | 0.842 (±3.19E-3) | 0.860 (±2.91E-3) | 0.913 (±2.01E-3) |
| EnCTC-0.15 | 79.55 | 3.5 9145 | 11.64 | 0.841 (±3.20E-3) | 0.861 (±2.81E-3) | 0.914 (±1.99E-3) |
| EnCTC-0.04 | 80.11 | 3.4 3024 | 11.47 | 0.580 (±6.00E-3) | 0.884 (±2.52E-3) | 0.896 (±2.31E-3) |
| EsCTC | 88.52 | 14.2370 | 21.06 | 0.580 (±6.00E-3) | 0.884 (±2.52E-3) | 0.909 (±1.23E-3) |
| CTC | 88.62 | 2.8287 | 11.46 | 0.824 (±3.46E-3) | 0.909 (±3.08E-3) | 0.914 (±1.99E-3) |

**Table II: Phone Recognition and Pronunciation Assessment Results vs Peakiness**

whether to round the output before calculating the PCC; using polynomial order two for polynomial regression; applying the Radial Basis Function (RBF) kernel for the SVR models etc. Due to relatively high variance of the GOP metric, same as in [40], we run the training of GOP 5 times with random initialization, selecting the best model for each run and then averaging the results. Following the same idea in [36] to reduce the variance of the GOPT model, we limit the tests to polynomial regression, support vector regression (SVR) and GOPT as in [40]. The PCC is computed between the model's output value and the true label. For the sake of fair comparison, we preserve all the details for evaluations as the baseline papers, e.g.

### C. GOP-SF and context length

By definition, the segmentation-free GOP method that we have proposed (GOP-SF) is computed considering contributions from the entire utterance, even if those contributions are weighted by how likely it is that each part of the utterance belongs to the target phone. A reasonable question to ask is how the length of left context (L_L) and right context (L_R) affect the pronunciation assessment results.

Figure 4 displays the AUC results on CMU Kids where we have varied the number of phones in L_L and L_R from 0 to 7 increments of 2. We compare our results with the original utterances (full context). In this case, the length of the context depends on the specific utterance, but it is always greater than 14. The left plots simulated errors, while the right plots real errors.

For simulated errors, the AUC values for different context lengths remain at a high level, and the impact of context length can be neglected. The results show that the GOP-CF-SF is robust to different context length in the ideal case.

Also for real errors, the dependency on context length is usually under the variability described by the 95% confidence intervals. The only exception is the observation when we reduce the context to zero, which corresponds to the self-aligned GOP definition (GOP-SA). This confirms the superiority of the segmentation-free method that it can take advantage of the context to assess the pronunciation of the target phone. Using context lengths 8 and 14 is visually indistinguishable from using the full context length. This suggests that the information needed to assess the pronunciation of the target phoneme is relatively local in the utterance.

### D. Comparison to state-of-the-art

In order to compare the performance of our best method (GOP-X-SF-SD) with respect to the state-of-the-art, we report results on the speechocean762 dataset in Table III. The table compares different evaluation of GOP with a simple polynomial regression MDD model. It also includes results with GOP feature vectors in combination with SVR or GOPT models.

Similarly to the previous experiments on CMU Kids, we find that the methods trained on CTC models are less performant than those trained with the CE loss. Results for the method "GOP-CE-SF-SD" are missing due to numerical problems. These were solved in the new implementation ("GOP-CE-SF-SD-numerical"). The results are further improved using the definition that normalizes GOP.

| Model | PCC (Pearson Correlation Coefficient) | | |
|--------|-----|---|---|
| TGOP-TDNN [14] | 0.361 ± 0.008 | | |
| FGOP-CTC-SF-numerical | 0.580 ± 0.006 | | |
| FGOP-CTC-SF-Norm | 0.581 ± 0.006 | | |
| TGOP-TDNN [37] | 0.605 ± 0.002 | | |
| FGOP-CTC-SF | N/A | | |
| FGOP-CTC-SF-numerical | 0.646 ± 0.002 | | |
| FGOP-CTC-SF-Norm | 0.648 ± 0.002 | | |

**Table III: speechocean762, reported Pearson Correlation Coefficient**

by the estimated length of the target segment ("GOP-CE-SF-SD-Norm").

In all cases, when combined with a CTC-trained model, our segmentation-free methods (GOP-CTC-SF, and FGOP-CTC-SF) outperform the baseline methods that rely on segmentation. GOP-X-SA is always better than traditional GOP regardless of the characteristics of the acoustic model used for the evaluation. GOP-SF obtains the overall best results for both the CMU Kids and for the speechocean762 material. On the speechocean762 data we obtain state-of-the-art results keeping the MDD model constant. We also show how the peakiness of the acoustic model affects the MDD results for the standard GOP definition and for the two proposed methods. Finally we show that the performance of GOP-SF is robust with respect context length ("GOP-SF-SF") that considers the full utterance to assess the pronunciation of the target phoneme, is not affected by the length of the context.

We believe these proposed methods are potentially very appealing for phoneme-level pronunciation assessment, both because of their high performance but also for their simple implementation and very low computational cost.

---

## VI. Conclusions

In this work, we propose improvements to a framework for pronunciation assessment that we recently introduced in [14]. We propose two methods with the goal of allowing the use of goodness of pronunciation (GOP) features extracted from modern high-performance automatic speech recognition (ASR) models. The main idea of our methods is to release the dependencies of the GOP definition on accurate speech segmentation. This raises a number of challenges due to the fact that GOP requires segmentation of speech into phonetic units.

In this paper, our goal is to combine the advantages of the GOP method, the foundation models, and end-to-end ASR training based on CTC loss [13]. This raises a number of challenges due to the fact that GOP requires segmentation of speech into phonetic units. The first method (GOP-SA) with the goal of combining CTC trained models and GOP. In particular, we introduced 1) a self-aligned GOP method that uses the same activations from CTC trained models for alignment and GOP evaluation, and 2) a segmentation-free GOP method that can assess a specific speech segment without committing to a particular segmentation.

In this paper, we propose improvements to a framework for pronunciation assessment that we recently introduced in [14]. We propose two methods with the goal of allowing the use of goodness of pronunciation (GOP) features extracted from modern high-performance automatic speech recognition models.

---

## References

[1] S. M. Witt and S. J. Young, "Phone-level pronunciation assessment for interactive language learning," *Spe. Comm.*, vol. 30, no. 2-3, pp. 95–108, 2000.

[2] W. K. Leung, X. Liu, and H. Meng, "CNN-RNN-CTC based end-to-end pronunciation detection and diagnosis," in *IEEE ICASSP*, 2019, pp. 8132–8136.

[3] E. Hu, L. Liu, D. Ke, X. Wang, and B. Lin, "A full text-dependent end to end mispronunciation detection and diagnosis with easy data augmentation techniques," *ArXiv*, vol. abs/2104.09328, 2021.

[4] Y. Feng, G. Fu, Q. Chen, and K. Chen, "SED-MDD: Towards sentence-dependent end to end mispronunciation detection and diagnosis," in *IEEE ICASSP*, 2020, pp. 3492–3496.

[5] L. Zhang et al., "End-to-end automatic pronunciation error detection based on improved hybrid ctc/attention mechanism," *Sensors*, vol. 20, no. 7, 2020.

[6] Y. Getman et al., "wav2vec2-based speech rating system for children with speech sound disorder," in *Interspeech*, 2022, pp. 26–30.

[7] X. Xu, Y. Kang, S. Cao, B. Lin, and L. Ma, "Explore wav2vec 2.0 for mispronunciation detection," in *Interspeech*, 2021, pp. 4428–4432.

[8] A. Baevski, H. Zhou, A. Mohamed, and A. Auli, "wav2vec 2.0: A framework for self-supervised learning of speech representations," in *NeurIPS*, 2020.

[9] E. Kim, J.-J. Jeon, H. Seo, and H. Kim, "Automatic Pronunciation Assessment using Self-Supervised Speech Representation Learning," in *Interspeech*, 2022, pp. 1411–1415.

[10] Y. Shen, Q. Liu, Z. Fan, J. Liu, and A. Wumaier, "Self-supervised pre-trained speech representation learning based end-to-end mispronunciation detection and diagnosis of mandarin," *IEEE Access*, vol. 10, pp. 106451–106462, 2022.

[11] W.-N. Hsu, B. Bolte, Y.-H. Tsai, K. Lakhotia, R. Salakhutdinov, and A. Mohamed, "HuBERT: Self-supervised speech representation learning by masked prediction of hidden units," in *IEICE Trans. Audio, Speech, Language Process.*, pp. 3451–3460, 2021.

[12] H. Liu, M. Shi, and Y. Wang, "Zero-Shot Automatic Pronunciation Assessment in Interspeech," 2023, pp. 1009–1013.

[13] A. Graves, S. Fernández, F. Gomez, and J. Schmidhuber, "Connectionist temporal classification: Labelling unsegmented sequence data with recurrent neural networks," in *ICML*, 2006, pp. 369–376.

[14] X. Cao, Z. Fan, T. Svendsen, and G. Salvi, "A framework for phoneme-level pronunciation assessment using CTC," in *Interspeech 2024*, 2024, pp. 306–306.

[15] Y. Song, W. Liang, and R. Liu, "Lattice-based GOP in automatic pronunciation evaluation," in *ICAE*, vol. 3, 2010, pp. 598–602.

[16] D. Luo, Y. Qiao, N. Minematsu, Y. Yamauchi, and K. Hirose, "Analysis and utilization of MLLR speaker adaptation technique for learner's pronunciation evaluation," in *Interspeech*, 2009, pp. 45–48.

[17] A. M. Harrison, W.-K. Lo, X.-j. Qian, and H. Meng, "Implementation of an extended recognition network for mispronunciation detection and diagnosis in computer-aided pronunciation training," in *SLaTE*, 2009, pp. 45–48.

[18] S. Dudy, S. Bedrick, M. Asgari, and A. Kain, "Automatic analysis of pronunciations for children with speech sound disorders," *Computer Speech & Language*, vol. 50, pp. 26–43, 2018.

[19] S. Kanters, C. Cucchiarini, and H. Strik, "The goodness of pronunciation algorithm: a detailed performance analysis," in *SLATE*, 2009, pp. 49–52.

[20] W. Hu, Y. Qian, and F. K. Soong, "A new DNN-based high quality pronunciation evaluation for computer-aided language learning (CALL)," in *Interspeech*, 2013, pp. 1886–1890.

[21] W. Hu, Y. Qian, F. Soong, and Y. Wang, "Improved mispronunciation detection with deep neural network trained acoustic models and transfer based logistic regression classifiers," *Spe. Comm.*, vol. 67, 2015.

[22] S. Sudhakara, M. Ramanathi, C. Yarra, and P. Ghosh, "An Improved Goodness of Pronunciation (GoP) Measure for Pronunciation Evaluation with HMM-HMM System Considering HMM Transition Probabilities," in *Interspeech*, 2019, pp. 954–958.

[23] J. Shi, N. Huo, and Q. Jin, "Context-Aware Goodness of Pronunciation for Computer-Assisted Pronunciation Training," in *Interspeech*, 2020, pp. 3057–3061.

[24] V. C M et al., "The impact of forced-alignment errors on automatic pronunciation evaluation," *Interspeech*, 2021, pp. 1922–1926.

[25] T. Mahr, V. Barisha, K. Kawabata, J. Liss, and K. Hustad, "Performance of forced-alignment algorithms on children's speech," *JSHIR*, vol. 64, pp. 1–10, 2021.

[26] W. Hu, Y. Qian, and F. K. Soong, "An improved DNN-based approach to mispronunciation detection and diagnosis of L2 learners' speech," in *SLaTE*, 2015, pp. 71–76.

[27] L. Wai Kit and F. Soong, "Generalized posterior probability for minimum error verification of recognized sentences," in *IEEE ICASSP*, 2005, pp. 85–88.

[28] Z. Zhang, W. Wang, and J. Yang, "Text-conditioned transformer for automatic pronunciation error detection," in *Spe. Comm.*, vol. 130, no. C, pp. 55–63, 2021.

[29] A. Zeyer, R. Schlüter, and H. Ney, "Why does CTC result in peaky behavior?" *arXiv* preprint arXiv:1409.4049, 2021.

[30] H. Liu, S. Jin, and C. Zhang, "Connectionist temporal classification with maximum entropy regularization," in *NeurIPS*, 2018.

[31] R. Huang et al., "Less peaky and more accurate CTC by label priors," in *IEEE ICASSP*, 2024.

[32] J. Dornemann, C. Cucchiarini, and H. Strik, "Using non-native patterns to improve pronunciation verification," in *Interspeech*, 2010, pp. 590–593.

[33] S. Wei, G. Hu, Y. Hu, and R.-H. Wang, "A new method for mispronunciation detection using support vector machines based on pronunciation space models," *Spe. Comm.*, vol. 51, no. 10, pp. 896–905, 2009.

[34] M. Eskenazi, J. Mostow, and D. Graff, *The CMU Kids Corpus Linguistic Data Consortium*, 1996.

[35] D. Povey et al., "The Kaldi speech recognition toolkit," in *ASRU*, 2011.

[36] X. Cao, Z. Fan, T. Svendsen, and G. Salvi, "An Analysis of Goodness of Pronunciation on Child Speech," in *Interspeech*, 2023, pp. 4613–4617.

[37] J. Zhang et al., "speechocean762: An open-source non-native English speech corpus for pronunciation assessment," in *Interspeech*, 2021, pp. 3710–3714.

[38] V. Panayotov, G. Chen, D. Povey, and S. Khudanpur, "Librispeech: An ASR corpus based on public domain audio books," in *IEEE ICASSP*, 2015, pp. 5206–5210.

[39] A. Conneau, A. Baevski, R. Collobert, A. Mohamed, and A. Auli, "Unsupervised Cross-Lingual Representation Learning for Speech Recognition," in *Interspeech*, 2021, pp. 2426–2430.

[40] Y. Gong, Z. Chen, I.-H. Chu, P. Chang, and J. Glass, "Transformer-based multi-aspect multi-granularity non-native english speaker pronunciation assessment," *IEEE ICASSP*, pp. 7262–7266, 2022.

[41] J. Hanley and B. McNeil, "The meaning and use of the area under the receiver operating characteristic (ROC) curve.," *Radiology*, vol. 148, pp. 29–36, 1982.
