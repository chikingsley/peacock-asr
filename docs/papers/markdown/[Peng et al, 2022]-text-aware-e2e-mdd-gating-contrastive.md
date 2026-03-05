# Text-Aware End-to-end Mispronunciation Detection and Diagnosis

###### Abstract

Mispronunciation detection and diagnosis (MDD) technology is a key component of computer-assisted pronunciation training system (CAPT). In the field of assessing the pronunciation quality of constrained speech, the given transcriptions can play the role of a teacher. Conventional methods have fully utilized the prior texts for the model construction or improving the system performance, e.g. forced-alignment and extended recognition networks. Recently, some end-to-end based methods attempt to incorporate the prior texts into model training and preliminarily show the effectiveness. However, previous studies mostly consider applying raw attention mechanism to fuse audio representations with text representations, without taking possible text-pronunciation mismatch into account. In this paper, we present a gating strategy that assigns more importance to the relevant audio features while suppressing irrelevant text information. Moreover, given the transcriptions, we design an extra contrastive loss to reduce the gap between the learning objective of phoneme recognition and MDD. We conducted experiments using two publicly available datasets (TIMIT and L2-Arctic) and our best model improved the F1 score from 57.51% to 61.75% compared to the baselines. Besides, we provide a detailed analysis to shed light on the effectiveness of gating mechanism and contrastive learning on MDD https://github.com/vocaliodmiku/wav2vec2mdd-Text.

Index Terms: mispronunciation detection and diagnosis (MDD), computer-aided pronunciation training (CAPT), text aware, gate mechanism

## 1 Introduction

Computer-Assisted Pronunciation Training (CAPT) system can meet people’s needs for language learning in fragmented time with flexible devices. Mispronunciation detection and diagnosis system (MDD) is an indispensable component of the CAPT system. Similar to the role of teachers in oral practice lessons, MDD can provide instant feedback about pronunciation problem for users to improve their speaking skills. Considering the rapidly increasing number of language learners, a high-performance MDD is needed to assure the precise diagnosis of pronunciation errors at the phonetic and prosodic levels. Here, we focus on phonetic mispronunciation in second-language learning.

Here, we consider assessing the pronunciation quality of constrained speech, that is, the text uttered by speakers is known to the system. Popular pronunciation error detection framework can be roughly divided into two categories, both of which have fully made use of the transcriptions. The first category is based on confidence measures which are mainly obtained from automatic speech recognition (ASR). Whether the pronunciation is correct or not is decided by calculating the confidence score of the frame/phoneme level with the help of forced alignment [1, 2, 3], which is a technique to align acoustic frames with given texts. The second category is based on extended search lattice and one of the most popular approaches is extended recognition network (ERN) [4]. ERN analyzes the text first and then incorporates a finite number of phonetic error patterns into the decoding network based on handcrafted or data-driven rules.

Recent studies have proposed various network architectures to improve the MDD performance [5, 6, 7, 8, 9, 10]. Similar to conventional methods, a line of studies attempt to leverage the prior linguistic information to provide guidance extracted from the given transcriptions [7, 8, 9, 10]. [8] feeds phoneme sequences into a sentence encoder and then combines with audio features via attention. Frame-level cross-entropy loss is calculated with the help of manually labeled phoneme boundary. [9] designs multiple data augment techniques based on the given transcriptions to alleviate the data sparsity problem. Despite the effective network design, most previous studies directly incorporate textual features into speech representation via a naive attention mechanism. We contend that textual features contribute very differently when they are assigned to attend different acoustic features. For correct pronunciation, transcription can guide the model step towards text-audio joint representation for better inference. However, it is difficult to align prior phonemes with acoustic features when mispronunciation occurs and hence limits the potential performance improvement.

End-to-end MDD shows its success in modeling simplicity and performance improvement. The main idea is to train a phoneme recognition model on L1/L2/L1-L2 hybrid datasets, and then perform MDD by comparing the reference and the inference. However, most previous models are optimized with a sole phoneme recognition objective directly or implicitly constrained by extra error states towards the correct diagnosis. Such a single recognition loss tries to predict each phoneme equally. To some extent, we hope the system can report more mispronunciations with little/no sacrifice of performance on canonicals. Some works utilize an extra error-state-related loss function to carry out MDD implicitly [7]. Due to the gap between the learning objective of phoneme recognition and MDD, previous methods fail to focus on mispronunciations explicitly and thus being less effective in detection and diagnosis.

In this study, we propose a *Text-Aware end-to-end* model for MDD, which incorporates the prior text modality to learn a good joint representation of acoustic units. We leverage an effective Text-Audio gate control module to effectively fuse prior transcriptions. It can enforce the model to align textual information to the most related acoustic regions while ignoring irrelevant parts automatically. To further unleash the power of prior texts, we refine the loss to bridge the learning objective gap between phoneme recognition and MDD by explicitly discriminating the probability of reference and annotation sequences. We experiment on the L2-Arctic dataset. Results confirm our main hypothesis that modeling text with gate control and explicitly distinguishing the reference and annotation benefit performance.

## 2 Text-Aware E2E MDD

We first briefly introduce the model structure we used for exploring information control. Then we explain the notion of contrastive learning object which will be performed for E2E MDD. Due to space limitations, we scatter the proposed architecture into Figure 1 and Figure 2. The pre-trained model we used comes from fairseq toolkit [11].

### 2.1 Audio encoder

In our previous work [12] and work [13] from another group, pretrained acoustic model have achieved great success on MDD. Here we inherit the pretrained model wav2vec2.0 [14] as audio encoder. It consists of a CNN-based encoder network, a transformer-based context network and a vector quantization module. We omit quantization module because it is out of the scope of this paper. The encoder network encodes the raw audio sample point into latent speech representation . Combining multiple layer normalization and GELU activation layers, the convolutional module compresses about 25 ms of 16 kHZ audio every 20 ms. Then context representations are obtained by a context network which scans over the entire latent speech representations.

### 2.2 Text encoder

Although we have included such a powerful acoustic model into our MDD system, there is still possibly more room for improvement by combining with reference texts. A common approach is to convert the reference text into phonemes and transform each phoneme into an N-dimensional linguistic feature vector via a parameterized lookup table. We construct the text encoder with a network of two Transformer layers. Given a canonical phoneme sequence with length , the reference text representation can be derived by the text encoder.

### 2.3 Textual modulation gate

In accordance with the postulates given in the introduction, we design a *Textual Modulation Gate* based on attention fusion. Compared with the annotation transcription, some phonemes in reference text are replaced with the corresponding prompts which reflect the actual acoustic parameters. The “polluted” reference text is thus not paired with associated audio features. On the textual side, we run an information monitor to filter out texts whose prior knowledge is strong enough to deteriorate the performance. For and , we have:

| (1) | |||
| (2) | |||
| (3) | |||
| (4) | |||
| (5) |

where is element-wise product. We compute attention weight between frame and which is used for re-weighting the textual representation. Then we choose the implementation of linear projection, summation, and sigmoid activation sequentially to generate the textual gate before feeding them into the Transformer layer for CTC prediction. We refer to the formula above as *TextGate*. Furthermore, we further explore variants of gate modulating (Figure 2) and conduct experiments to evaluate them.

### 2.4 Contrastive learning

Based on our experimental results, we found that better phoneme recognition model implementations can not always report better results in the context of MDD. In Figure 4, all the results in terms of phone error rate and F1 score are leaked in advance. As the phone error rate decreases, the F1 score performance trend is hard to conclude. The failure is due to the mismatch of learning objectives between MDD and phoneme recognition. Phoneme recognition aims to infer phonemes from the annotation correctly as much as possible, irrespective of whether we should pay more attention to mispronunciations. In this situation, performance improvement in recognition can be achieved by detecting more canonicals in proportion. With the given prior texts, we propose an objective base on contrastive learning to bridge the gap. Contrastive learning, a kind of technique that maximizes the intra-class similarity and minimizes the inter-class similarity has been used extensively over the years in various applications [15, 16]. In the context of MDD, we can anchor the transcription in order to generate the dissimilarity/similarity.

While we cannot directly construct negative pairs and positive pairs as usual to define the similarity, we introduce a supervised contrastive loss derived from CTC [17]. Addressing the variable length (T) input frames, , U length associated reference characters, , and U length associated reference characters, , conditionally independent probability of label sequence:

where denotes the softmax output of label at time t, is a map function which can generate all possible intermediate label representations from unmodified label sequence. A modified label sequence is made by inserting the blank symbols between each label including the beginning and the end (i.e., ). Suppose there is only one substitution mispronunciation occurred at position t, for each possible , we can obtain a paired from and therefore, for paired -, , . Then we can define the dissimilarity for modified annotation and sequence.

| (6) |

We incorporate margin into the dissimilarity and sum up all possible negative pairs. Then our contrastive loss can be expressed as:

In order to train the network, we use besides the contrastive loss one additional loss functions:

| (7) |

where is a loss for phoneme sequence recognition222CTC computes the probability of all possible intermediate sequence via dynamic programming. Similarly, we implemented the loss as followed approximately. is set to 16 empirically :
(8)
. In practice, contrastive learning can be realized easily on an attention-based encoder-decoder model for the convenience of accessing phoneme-level likelihoods. Based on our experimental results, we find that the encoder-decoder structure cannot achieve better results and conclude that the data used in this paper is too sparse for the decoder to generalize well. An alternative approach to reducing the gap is to directly optimize the F1 score metric with the reinforcement learning technique [18].

## 3 Experiments

### 3.1 Speech corpora and model architecture

Datasets. We use the publicly available datasets TIMIT [19] and L2-arctic [20] to conduct our experiments. TIMIT is a native (L1) English corpus containing 6,300 utterances from 630 speakers. We use its original training subset. The L2-arctic corpus https://psi.engr.tamu.edu/l2-arctic-corpus/ :L2-ARCTIC-V2.0 is a non-native English speech corpus that is intended for research in voice conversion, accent conversion, and mispronunciation detection. It contains utterances with mispronunciations of 24 (12 males and 12 females) non-native speakers whose L1 languages include Hindi, Korean, Spanish, Arabic, Vietnamese and Chinese. Following prior works [8, 9, 12], six speakers (NJS, TLV, TNI, TXHC, YKWK, ZHAA) were selected as the test set while the rest were merged to build the training set. We further generated a subset from the training set as the development set by randomly selecting 20% sentences for each speaker. There was no overlap between training and developing set. For the phone set, we mapped the TIMIT 61-phone to 39-phone according to the mapping table from [21] and combined it into L2-arctic phone set.

Implementation Details. For audio encoder, we used two publicly available pre-trained wav2vec2.0 models as our backbones: wav2vec2.0-BASE https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt and wav2vec2.0-XLSR https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr_53_56k.pt.
All models were trained 142 epochs using the Adam optimizer with an initial learning rate of on one RTX3060 GPU. The dimension of attention and gating mechanism was set to 768. The audio encoder was frozen in the first 10,000 steps.

### 3.2 Performance evaluation

We followed the evaluation metrics of previous studies [22]. For the end-to-end model, the detection of pronunciation errors can be achieved by comparing the prediction sequence and the reference text sequence after alignment. For canonical phones, true accept (TA) means the recognized phone sequence is consistent with the reference text while rejection (FR) means inconsistence. For mispronounced phones, true rejection (TR) indicates the mispronunciation has been detected while false accept (FA) fails to do it. Further, true rejection can be divided into correct diagnosis and diagnosis error. Other metrics like recall (TR/(FA + TR)), precision (TR/(FR + TR)) and the F-1 score (2*((precision*recall)/(precision+recall))) can be calculated based on the accumulated statistics.

### 3.3 Experimental results

Evaluating Baselines. Under fair architecture and training settings, our Baseline surpasses other baselines. Compared with Baseline from [9], audio encoder wav2vec2.0 in our Baseline provides more powerful representations. For a simple fusion operation, concatenation is better than addition. Note that all our systems except BaselineAdd use add implementation. Tuning to concatenation will bring further improvement.

Evaluating Textual Modulation Gate. As expected, the proposed approaches Double Gate and TextGate outperform the Baseline/BaselineAdd method by +0.5%/1.8% and +1.4%/2.7%, respectively. The textual modulation gate successfully plays the role of validating information that comes from texts. Figure 4 shows attention weights output by the TextGate and Baseline model. Since the Textual Modulation Gate can take responsibility for turning on/off textual information flow, attention patterns look neat and natural, while audio-text correlation maps for the model without gate would be chaotic.

Within the TextGate series, the difference in performance between sigmoid and softmax is small. However, tanh gives a lower performance ever compared to the baseline and we also notice that “AudioGate”, performing control on the audio branch, reports poor performance. These results suggest that even incorporating the extra reference text, the fusion framework needs to be carefully designed. We further tried to utilize a more powerful acoustic model (XLSR) for further improvement and attribute the minor difference to the learning object gap. Relevant result analyses are shown below.

Evaluating Contrastive Learning Object. We integrated the proposed TextGate into XLSR model where a contrastive loss was used. As shown in Table 1 (bottom), TextGateXLSRContrast obtains a performance gain of 1.5% F1 score compared model TextGateContrast and achieves the best performance with an F1 score of 61.75%, suggesting that our methods work practically well with a contrastive loss. Furthermore, TextGateXLSRContrast reports the lowest False Accept number, which corresponds to our discussion — Given the prior text, the model learns more discriminative features about the reference and annotation phonemes which are hard to distinguish, naturally making the prior transcripts more informative.

## 4 Conclusion and future work

In this work, we have presented *Text-Aware end-to-end* model that explicitly incorporates prior transcription in model training and effectively learns a better refined text-audio representation for MDD. With the text-aware module, our best model improves the baseline by +2.8% in absolute F1 score on L2-arctic dataset. Moreover, we notice the existence of differences between current methods and MDD in optimization objectives. Contrastive-based loss is proposed to bridge the gap and outperforms the baseline methods by +4.24%. Analyses show that these simple modifications help the mispronunciation-sensitive representation learning among given reading texts and acoustic inputs. In future work, we will investigate extracting more information from transcriptions, such as transferring phonetic knowledge to constrain the text-audio attention matrix and optimize the learning object toward MDD.

## 5 Acknowledgements

This study was supported by Advanced Innovation Center for Language Resource and Intelligence (KYR17005), National Social Science Foundation of China (18BYY124), Wutong Innovation Platform of Beijing Language and Culture University (19PT04), the Science Foundation and Special Program for Key Basic Research fund of Beijing Language and Culture University (the Fundamental Research Funds for the Central Universities) (21YJ040004). Jinsong Zhang is the corresponding author.

## References

-
[1]
S. M. Witt and S. J. Young, “Phone-level pronunciation scoring and assessment
for interactive language learning,”
*Speech communication*, vol. 30, no. 2-3, pp. 95–108, 2000. -
[2]
W. Hu, Y. Qian, and F. K. Soong, “A new dnn-based high quality pronunciation
evaluation for computer-aided language learning (call).” in
*Interspeech*, 2013, pp. 1886–1890. -
[3]
J. Zheng, C. Huang, M. Chu, F. K. Soong, and W.-p. Ye, “Generalized segment
posterior probability for automatic mandarin pronunciation evaluation,” in
*2007 IEEE International Conference on Acoustics, Speech and Signal Processing-ICASSP’07*, vol. 4. IEEE, 2007, pp. IV–201. -
[4]
A. M. Harrison, W.-K. Lo, X.-j. Qian, and H. Meng, “Implementation of an
extended recognition network for mispronunciation detection and diagnosis in
computer-assisted pronunciation training,” in
*International Workshop on Speech and Language Technology in Education*, 2009. -
[5]
W.-K. Leung, X. Liu, and H. Meng, “Cnn-rnn-ctc based end-to-end
mispronunciation detection and diagnosis,” in
*ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*. IEEE, 2019, pp. 8132–8136. -
[6]
B.-C. Yan, M.-C. Wu, H.-T. Hung, and B. Chen, “An End-to-End Mispronunciation
Detection System for L2 English Speech Leveraging Novel Anti-Phone
Modeling,” in
*Proc. Interspeech 2020*, 2020, pp. 3032–3036. [Online]. Available: http://dx.doi.org/10.21437/Interspeech.2020-1616 -
[7]
N. Zheng, L. Deng, W. Huang, Y. T. Yeung, B. Xu, Y. Guo, Y. Wang, X. Jiang, and
Q. Liu, “Cca-mdd: A coupled cross-attention based framework for streaming
mispronunciation detection and diagnosis,”
*arXiv preprint arXiv:2111.08191*, 2021. -
[8]
Y. Feng, G. Fu, Q. Chen, and K. Chen, “Sed-mdd: Towards sentence dependent
end-to-end mispronunciation detection and diagnosis,” in
*ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*. IEEE, 2020, pp. 3492–3496. -
[9]
K. Fu, J. Lin, D. Ke, Y. Xie, J. Zhang, and B. Lin, “A full text-dependent end
to end mispronunciation detection and diagnosis with easy data augmentation
techniques,”
*arXiv preprint arXiv:2104.08428*, 2021. -
[10]
S.-W. F. Jiang, B.-C. Yan, T.-H. Lo, F.-A. Chao, and B. Chen, “Towards robust
mispronunciation detection and diagnosis for l2 english learners with
accent-modulating methods,”
*arXiv preprint arXiv:2108.11627*, 2021. -
[11]
M. Ott, S. Edunov, A. Baevski, A. Fan, S. Gross, N. Ng, D. Grangier, and
M. Auli, “fairseq: A fast, extensible toolkit for sequence modeling,” in
*NAACL-HLT (Demonstrations)*, 2019. -
[12]
L. Peng, K. Fu, B. Lin, D. Ke, and J. Zhan, “A Study on Fine-Tuning
wav2vec2.0 Model for the Task of Mispronunciation Detection and Diagnosis,”
in
*Proc. Interspeech 2021*, 2021, pp. 4448–4452. -
[13]
X. Xu, Y. Kang, S. Cao, B. Lin, and L. Ma, “Explore wav2vec 2.0 for
mispronunciation detection,”
*Proc. Interspeech 2021*, pp. 4428–4432, 2021. -
[14]
A. Baevski, Y. Zhou, A. Mohamed, and M. Auli, “wav2vec 2.0: A framework for
self-supervised learning of speech representations,”
*Advances in Neural Information Processing Systems*, vol. 33, 2020. -
[15]
A. Jaiswal, A. R. Babu, M. Z. Zadeh, D. Banerjee, and F. Makedon, “A survey on
contrastive self-supervised learning,”
*Technologies*, vol. 9, no. 1, p. 2, 2020. -
[16]
P. H. Le-Khac, G. Healy, and A. F. Smeaton, “Contrastive representation
learning: A framework and review,”
*IEEE Access*, vol. 8, pp. 193 907–193 934, 2020. -
[17]
A. Graves, S. Fernández, F. Gomez, and J. Schmidhuber, “Connectionist
temporal classification: labelling unsegmented sequence data with recurrent
neural networks,” in
*Proceedings of the 23rd international conference on Machine learning*, 2006, pp. 369–376. -
[18]
L. P. Kaelbling, M. L. Littman, and A. W. Moore, “Reinforcement learning: A
survey,”
*Journal of artificial intelligence research*, vol. 4, pp. 237–285, 1996. -
[19]
J. S. Garofolo, L. F. Lamel, W. M. Fisher, J. G. Fiscus, and D. S. Pallett,
“Darpa timit acoustic-phonetic continous speech corpus cd-rom. nist speech
disc 1-1.1,”
*NASA STI/Recon technical report n*, vol. 93, p. 27403, 1993. -
[20]
G. Zhao, S. Sonsaat, A. O. Silpachai, I. Lucic, E. Chukharev-Hudilainen,
J. Levis, and R. Gutierrez-Osuna, “L2-arctic: A non-native english speech
corpus,”
*Perception Sensing Instrumentation Lab*, 2018. -
[21]
K.-F. Lee and H.-W. Hon, “Speaker-independent phone recognition using hidden
markov models,”
*IEEE Transactions on Acoustics, Speech, and Signal Processing*, vol. 37, no. 11, pp. 1641–1648, 1989. -
[22]
K. Li, X. Qian, and H. Meng, “Mispronunciation detection and diagnosis in l2
english speech using multidistribution deep neural networks,”
*IEEE/ACM Transactions on Audio, Speech, and Language Processing*, vol. 25, no. 1, pp. 193–207, 2016.
