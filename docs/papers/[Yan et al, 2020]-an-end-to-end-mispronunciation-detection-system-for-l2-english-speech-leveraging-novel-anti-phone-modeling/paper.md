     An End-to-End Mispronunciation Detection System for L2 English Speech
                    Leveraging Novel Anti-Phone Modeling
                      Bi-Cheng Yan1,2, Meng-Che Wu2, Hsiao-Tsung Hung2, Berlin Chen1
                                     1
                                         National Taiwan Normal University, Taiwan
                                                   2
                                                     ASUS AICS, Taiwan
                           {bicheng_yan, meng_wu, alexht_hung}@asus.com, berlin@ntnu.edu.tw

                                                                     functionality of mispronunciation detection, but lack the ability
                          Abstract                                   of providing appropriate mispronunciation diagnosis. The
Mispronunciation detection and diagnosis (MDD) is a core             second category of methods aims to assess the details of
component of computer-assisted pronunciation training                mispronunciations, providing diagnosis feedback about specific
(CAPT). Most of the existing MDD approaches focus on                 errors such as phone substitutions, deletions and insertions [4],
dealing with categorical errors (viz. one canonical phone is         [5], [6]. A well-known method of this category is the extend
                                                                     recognition network (ERN) method, which extends the
substituted by another one, aside from those mispronunciations
caused by deletions or insertions). However, accurate detection      decoding network of ASR with phonological rules and thus can
and diagnosis of non-categorial or distortion errors (viz.           readily provide diagnosis feedback based on comparison
approximating L2 phones with L1 (first-language) phones, or          between an ASR output and the corresponding text prompt.
                                                                     Nevertheless, on one hand, it is difficult to enumerate and
erroneous pronunciations in between) still seems out of reach.
                                                                     include sufficient phonological rules into the decoding network
In view of this, we propose to conduct MDD with a novel end-
to-end automatic speech recognition (E2E-based ASR)                  for all L1-L2 language pairs. On the other hand, inclusion of
                                                                     too many phonological rules would degrade ASR accuracy,
approach. In particular, we expand the original L2 phone set
                                                                     thereby leading to poor MDD performance. More recently, the
with their corresponding anti-phone set, making the E2E-based
MDD approach have a better capability to take in both                end-to-end (E2E) based ASR paradigm instantiated with
categorical and non-categorial mispronunciations, aiming to          connectionist temporal classification (CTC) [7] has also been
provide better mispronunciation detection and diagnosis              introduced to MDD to with promising results, in comparison to
feedback. Furthermore, a novel transfer-learning paradigm is         the GOP-based method that builds on the hybrid deep neural
devised to obtain the initial model estimate of the E2E-based        network-hidden Markov model (DNN-HMM) based acoustic
                                                                     model [2]. Among others, there also has been some follow-up
MDD system without resource to any phonological rules.
Extensive sets of experimental results on the L2-ARCTIC              work of using disparate E2E-based methods to address the
                                                                     MDD problem [8], [9].
dataset show that our best system can outperform the existing
E2E baseline system and pronunciation scoring based method               However most of the aforementioned methods have
(GOP) in terms of the F1-score, by 11.05% and 27.71%,                focused exclusively on detecting categorical pronunciation
respectively.                                                        errors (e.g., phoneme substitutions, insertions or deletions),
                                                                     whereas paying less attention to detecting mispronunciations
Index Terms: computer-assisted pronunciation training
                                                                     that belong to non-categorial or distortion errors [10], [11]. As
(CAPT), mispronunciation detection and diagnosis (MDD),
end-to-end ASR, anti-phone model                                     an illustration, Figure 1 shows the MDD results of a
                                                                     mispronounced utterance of an L2 English speaker, where the
                                                                     yellow blocks correspond to mispronunciation segments. The
                    1. Introduction                                  canonical phone-level pronunciation for word “The” is [dh iy]
Computer-assisted pronunciation training (CAPT) systems              but an L2 speaker uttered [d ah] instead, where [dh]→[d] and
provide opportunities of self-directed language learning for         [iy]→[ah] are categorical errors (viz., substitutions). In addition,
second-language (L2) learners. It can supplement the teachers’       the canonical pronunciation of the consonant in word “He”
instructions, offer individualized feedback and also mitigate the    should be [hh], but it is instead pronounced as [hh*], which in
problem of teacher shortage. The mispronunciation detection          fact is a non-categorical pronunciation error.
and diagnosis (MDD) module play an integral role in CAPT                 In view of this, we propose to approach MDD with a
systems, since this module facilitates to pinpoint                   specially-tailored E2E-based ASR model structure, where the
mispronunciation segments and provide phone-level diagnosis          involved E2E-based model is embodied with a hybrid CTC-
feedback.                                                            Attention model [12]. By combining the strengths of both CTC
     The MDD methods developed so far can be roughly                 and the attention-based model, it is anticipated that the resulting
grouped into two categories. The first is pronunciation scoring      composite model can utilize CTC to assist the attention-based
based methods, which compute phone-level pronunciation               model to compensate for the misalignment problem and
scores based on confidence measures derived from ASR, e.g.,          improve the speed of the decoding process, though the
phone durations, phone posterior probability scores and              attention-based model provides flexible soft-alignment
segment duration scores [1], [2], [3]. Goodness of                   between the output label sequence and the input acoustic feature
pronunciation (GOP), based on the log-likelihood ratio test, and     vector sequence without any Markov assumptions as CTC does.
its variants are the most representative methods of this category.   Furthermore, we expand the original L2 phone set with their
However, these methods typically can only provide the                corresponding anti-phone set, making the proposed E2E-based
                                                                     MDD approach have the ability to take in both categorical and
         Figure 1: Analysis of the results generated by
         different baseline E2E-based MDD methods.
                                                                             Figure 2: A schematic depiction of the hybrid CTC-
non-categorial mispronunciations, in order to provide better                 Attention model architecture for MDD.
mispronunciation detection and diagnosis feedback [13].
Furthermore, a novel transfer learning paradigm is devised to           where 𝑆 is the length of the hidden vector sequence and usually
obtain the initial model estimate of the E2E-based ASR system           𝑆 < 𝑇 due to the subsampling operation. The decoder network
without resource to any phonological rules. The rest of the             is a disparate unidirectional long short-term memory network
paper is organized as follows. We first elucidate the model             (ULSTM), which predicts the incoming phone label 𝑦"
architectures of the proposed E2E-based MDD methods in                  conditioning on the previous output 𝑦"(! , the current decoder
Section 2, followed by the experimental setup and results in            state 𝐡-
                                                                               " and the context vector 𝐜" :
Section 3. Finally, we conclude the paper and suggest avenues                                                                    (3)
                                                                             𝑦 = Softmax BLinB ELSTMF𝑦 , 𝐜 , 𝐡- GHI,
                                                                              "                                   "(!   "   "
for future work in Section 4.
                                                                                  𝐡-                           -
                                                                                   " = ULSTMF𝑐𝑎𝑡(𝑦"(! , 𝐜" ), 𝐡"(! G,               (4)
            2. End-to-End MDD Model
                                                                        where LinB(⋅) is a linear transformation. The input of the
In this section, we first describe the hybrid CTC-Attention             ULSTM in (4) consists of the previous decoder’s state 𝐡-"(! and
model that we capitalize on for E2E-based MDD. After that, we           the concatenation of 𝐜" and 𝑦"(! . The context vector 𝐜" can be
explain the notion of anti-phone modeling that will be realized         calculated using an attention mechanism which communicates
for E2E-based MDD. Then, we shed light on the trick for                 information between the encoder’s holistic representation H+
estimating the initial anti-phone probabilities and the full            and the current decoder’s states 𝐡-" . The attention mechanism
training procedure for the E2E-based model.                             is summarized as follows:
                                                                                                      ,
2.1. CTC/Attention-based Modeling Architecture
                                                                                              𝐜" = O 𝐚",/ 𝐡+/ ,                      (5)
We adopt a hybrid CTC-Attention model (or CTC-ATT for                                                /*!
short) architecture, originally designed for E2E-based ASR, to                           𝐚",/ = AlignF𝐡-    +
                                                                                                       " , 𝐡/ G,
tackle the MDD problem [12]. In this architecture, the attention-
based model takes the primary role in determining output                                              exp (ScoreF𝐡-     +
                                                                                                                   " , 𝐡/ G)         (6)
                                                                                               =                               .
symbols, through the use of effective attention mechanisms to                                        ,
                                                                                                    ∑/*! exp (ScoreF𝐡" , 𝐡+/ G)
                                                                                                                       -
perform flexible alignment between an input acoustic vector
sequence and the associated output symbol sequence. On the              The soft-alignment (association) between a hidden acoustic
other hand, CTC, normally sharing the encoder with the                  vector state and a decoder state is quantified with a normalized
attention-based model, makes use the Markov assumptions to              score function Score(·,·); here we adopt the location-based
alleviate irregular alignment between input acoustic vector             scoring function [14].
sequence and the output symbol sequence. CTC plays an                     On a separate front, CTC first generates a frame-wise symbol
auxiliary role here to assist the attention-based model for more        sequence 𝐳 = 𝑧! . . 𝑧" . . 𝑧% . The probability of an output symbol
accurate MDD performance. The CTC-Attention model will                  sequence Y compute by:
predict an L-length phone sequence Y = y! . . y" . . y# (e.g., y"
belongs to the standard IPA symbol set) given a T-length input             𝑃0$0 (Y|O) = O . 𝑃(𝑧/ |𝑧/(! , Y)𝑃(𝑧/ |O)𝑃(Y),             (7)
                                                                                          𝒛     /
acoustic feature vector sequence O = 𝐨! . . 𝐨$ . . 𝐨% . In the
context of MDD, the output phone sequence Y can be viewed               where 𝑃(𝑧/ |𝑧/(! , Y) represents the state transition probability,
as the diagnosis result, in relation a text prompt that corresponds     which satisfies the monotonic alignment constraint posed by
to O. For the attention-based model, the probability distribution       CTC. In the context of MDD, the inclusion of 𝑃(𝑧/ |𝑧/(! , Y) can
𝑃&$$ (Y|O) is computed by multiplying the sequence of                   bring benefit to the MDD task, since the model will learn
conditional probabilities of label 𝑦" given the past history y!:"(! :   transitions between mispronunciations and correct
                          #                                             pronunciations from the training corpus. 𝑃(z/ |O) is the frame-
                                                                        level label probability and computed by
          𝑃&$$ (Y|O) = . 𝑃&$$ (𝑦) |𝑦!:"(! , O).               (1)
                         "*!                                                       𝑃(z/ |O) = Softmax ELinBF𝐡,+ GH.                  (8)
Subsequently, 𝑃&$$ (𝑦" |O, 𝑦!:"(! ) is obtained with the joint             In the training phase, the loss of CTC and the loss of the
encoder and decoder networks. The encoder network can be a              attention-based model are combined with an interpolation
bidirectional long short-term memory (BLSTM) which extracts             weight 𝜆 ∈ [0, 1], so as to encourage monotonic alignments.
a high-level hidden acoustic vector sequence H+ = (𝐡!+ , … , 𝐡,+ )
from the input acoustic feature vector sequence O:                                   ℒ234 = 𝜆ℒ0$0 + (1 − 𝜆)ℒ&$$ .                    (9)
                   H+ = BLSTM(O),                        (2)            The hybrid CTC-Attention model architecture is also adopted
                                                                        in the test phase. The additional incorporation of the CTC
objective is expected to provide fast and accurate inference          Table 1: Statistics of the experimental speech corpora.
during the training and test phases, thanks to its monotonic-                  Corpus             subsets Spks. Utters. Hrs.
alignment property. Figure 2 shows a schematic depiction of the                TIMIT+              Train    989 27801 87.90
hybrid CTC-Attention model architecture for MDD.                     L1
                                                                               LS-sub              Dev.     108 2871        8.83
2.2. Creation of the Anti-phone Set                                                                Train     18 17384 48.18
                                                                                           CP      Dev.      2     1962     4.91
In order to model non-categorical errors, we introduce the                                         Test      4     3928 11.44
notion of anti-phones to the CTC-Attention model architecture,       L2 L2-ARCTIC
                                                                                                   Train     18    2697     7.58
which is designed to accommodate non-categorical                                           MP      Dev.      2      300     0.75
mispronunciations of each L2 phone. To create an anti-phone                                        Test      4      596     1.75
set, each phone symbol in the L2 canonical phone set 𝒰567 is
appended with a token # at its beginning to designate its anti-          Table 2: The confusion matrix of mispronunciation
phone to be added into the anti-phone set 𝒰6789 . As such, the                      detection and diagnosis task
resulting augmented phone symbol set 𝒰 for the E2E-based                  Total                    Ground Truth
MDD model will be the union of the canonical phone set and             Conditions             CP                  MP
the anti-phone set:                                                    Model    CP True Positive (TP) False Positive (FP)
                  𝒰 = 𝒰567 ∪ 𝒰6789 .                    (10)         Prediction MP False Negative (FN) True Negative (TN)
Taking advantage of this augmented phone symbol set, it is
                                                                     Section 2.3, while the encoder network of the accent-contained
anticipated that the associated E2E-based MDD model can
separate mispronunciations into categorical errors and non-          E2E model is finetuned with this augmented dataset. Finally, at
                                                                     the third stage, the whole accent-contained E2E model is
categorical errors. In this way, for a mispronunciation that is in
between a L2 (target) canonical phone and some L1 (mother-           finetuned with the rest L2 English training utterances that
                                                                     contain mispronunciations (L2:MP; cf. Section 3.1). Note also
tongue) phone pronunciations, or is a distortion of the canonical
                                                                     that for a mispronounced phone segment of a given training
phone pronunciation, it would be possible to detect and classify
                                                                     utterance, its phone label in the transcript of the utterance is
this mispronunciation with the associated anti-phone label of
                                                                     replaced with its corresponding anti-phone label. We argue that
the canonical phone.
                                                                     the aforementioned training procedure could enable the
2.3. Data Augmentation with Label Shuffling                          resulting accent-contained E2E model not only to identify and
                                                                     diagnose categorical mispronunciation errors accurately, but
In this subsection, we describe a novel data-augmentation            also to detect non-categorical mispronunciations to some extent.
process for E2E-based anti-phone modeling, which creates
additional speech training data with a label-shuffling scheme.                           3. Experiments
Specifically, for every utterance in the original speech training
dataset, the label of a phone 𝜑 at each position of its reference    3.1. Speech Corpora and Model Architecture
transcript is either kept unchanged or randomly substituted with
an arbitrary anti-phone label (excluding the anti-phone label        We used the L2-ARCTIC 1 corpus for our experiments [15],
that corresponds to 𝜑) with a predefined probability. As such,       which is a publicly-available non-native English speech corpus
we can duplicate the original speech training data, having the       intended for research in mispronunciation detection, accent
new copy be equipped with the label-shuffled transcripts that        conversion, and others. This corpus contains correctly
contain anti-phone labels. Note here that “the original speech       pronounced utterances (denoted by CP) and mispronounced
training dataset” mentioned above refers to the part of non-         utterances (denoted by MP) of 24 non-native speakers, whose
native English utterances in the training dataset of the L2-         L1 languages are Hindi, Korean, Mandarin, Spanish, Arabic
ARCTIC corpus [15] (L2:CP; cf. Section 3.1) that were                and Vietnamese. We divided each of these two parts of
correctly pronounced without any pronunciation errors.               utterances into training, development and test subsets,
                                                                     respectively. As mentioned in Section 2.4, a suitable amount of
2.4. Training of the E2E-based MDD Model                             native (L1) English speech data compiled from the TIMIT
                                                                     corpus and a small portion of the Librispeech corpus [16] were
The training process of the proposed E2E-based model for             used to bootstrap the training of the E2E-based model. Table 1
MDD can be broken down into three stages. At the first stage,        summarizes some basic statistics of these speech datasets.
an accent-free E2E model is trained on a publicly-available
                                                                          The encoder network of the E2E-based model is composed
English speech dataset that contain utterances of native
                                                                     of a 4-layer bidirectional long short-term memory (BLSTM)
speakers (which was compiled from the TIMIT corpus and a
small portion of the Librispeech corpus [16]; cf. Section 3.1).      with 320 hidden units in each layer, while the input to the
                                                                     encoder network are 80-dimensional Mel-filter-bank feature
The second stage is to train an accent-contained E2E model. To
this end, we first adopt the notion of transfer learning to          vectors. In addition, the decoder network consists of a single-
initialize the encoder network of the accent-contained E2E           layer LSTM with 300 hidden units. the English canonical phone
model with the corresponding parameters of the accent-free           set was defined based on the CMU pronunciation dictionary.
E2E model trained at the first stage [17]. Then, the decoder         3.2. Performance Evaluation
network of the accent-contained E2E model is trained on the
augmented dataset (containing only of non-native English             For the mispronunciation detection subtask, we follow the
training utterances without mispronunciations) described in          hierarchical evaluation structure adopted in [18], while the


1
    https://psi.engr.tamu.edu/l2-arctic-corpus/
corresponding confusion matrix for four test conditions is
                                                                        Table 3: Mispronunciation detection results of our
illustrated in Table 2. Based on the statistics accumulated from
                                                                         proposed methods and the GOP-based method.
the four test conditions, we can calculate the values of different
metrics like recall (RE; TN/(FP + TN)), precision (PR; TN/                                         CTC-ATT           CTC-ATT
                                                                                       GOP
(FN + TN)) and the F-1 measure (F-1; the harmonic mean of                                           (Anti)            (Unk)
the precision and recall), so as to evaluate the performance of
                                                                         PR(%)        19.42          46.57             38.99
mispronunciation detection.
                                                                         RE(%)        52.19          70.28             53.12
     For the mispronunciation diagnosis subtask, we first                F1(%)        28.31          56.02             44.97
address in those mispronounced phone labels in the text
prompts of test utterances that have been correctly detected,
                                                                        Table 4: Mispronunciation detection results of our
referred to as true negative (TN; cf. Table 2), to calculate the
                                                                        proposed methods with different model structures.
diagnostic accuracy rate (DAR). Furthermore, we also analyze
                                                                                       CTC         Attention       CTC-ATT
the performance statistics like the number (ratio) of categorical
                                                                                      (Anti)        (Anti)           (Anti)
errors and non-categorical errors of the true mispronunciations
(FP + TN) that we can provide correct diagnoses, respectively.           PR(%)        41.17          43.89           46.57
                                                                         RE(%)        76.48          64.54           70.28
3.3. Experimental results                                                F1(%)        53.52          52.25           56.02

3.3.1. Evaluations on Mispronunciation Detection                       Table 5: Mispronunciation diagnosis accuracy results
At the outset, we assess the performance level of our proposed        (DAR%) of our proposed methods with different model
E2E-based method on mispronunciation detection, in relation                                structures.
to the cerebrated GOP-based method building on the DNN-                                 CTC         Attention    CTC-ATT
HMM model. Specifically, the DNN component of GOP is a 5-                              (Anti)        (Anti)         (Anti)
layer time-delay neural network (TDNN) and 1,280 neurons in              DAR(%)        32.46          37.02         40.66
each layer, whose parameters were trained on the training sets
of L1 and L2-CP (cf. Table 1). The corresponding results are             Table 6: Numbers of categorical errors and non-
shown in Table 3, where our methods were either implemented            categorical errors of the true mispronunciations that
with phone-specific anti-phone modeling, viz. CTC-ATT(Anti),                our models can provide correct diagnoses.
or with a simplified version, viz. CTC-ATT(Unk), which used                              Non-categorical        Categorical
a single symbol Unk instead to accommodate all non-                                            errors              errors
categorical mispronunciations. Looking at Table 3, we can                 Ground               100%                100%
make at least three observations. First, our CTC-ATT(Anti)                 Truth               (771)              (3,310)
method outperforms the GOP-based method by a significant                CTC-ATT                8.4%               19.63%
margin, demonstrating the promise of using E2E-based model                 (Unk)                (65)               (650)
structure for the mispronunciation detection subtask. Second,           CTC-ATT                9.4%               33.02%
CTC-ATT(Anti) yields considerably better performance than                  (Anti)               (73)              (1,093)
CTC-ATT(Unk), which reveals that finer-grained anti-phone
modeling is desirable. Third, the aforementioned methods are         CAPT systems, the mispronunciation diagnosis subtask is even
still far from perfect for the mispronunciation detection subtask    more challenging that the mispronunciation detection subtask.
on the L2-ARCTIC corpus.                                                 As a final note, we report on a statistical analysis of the
     We then set out to analyze the impacts of leveraging            numbers (ratios) of categorical errors and non-categorical errors
different model architectures on the mispronunciation detection      we could provide correct diagnoses with our two methods, viz.
subtask. Here apart from the hybrid CTC-Attention model              CTC-ATT(Anti) and CTC-ATT(Unk). Inspection of Table 6,
(CTC-ATT), either CTC or the attention-based model (denoted          reveals that through the use of phone-specific anti-phone
by Attention for short) were investigated for this purpose. Here     modeling, viz. CTC-ATT(Anti), both the categorical and non-
all the three methods were implemented with phone-specific           categorical mispronunciations can be better diagnosed than that
anti-phone modeling as well (cf. Section 2). As can be seen          using coarse-grained anti-phone modeling, viz. CTC-
from Table 4, mispronunciation detection using CTC-ATT               ATT(Unk).
delivers a superior F1-score than that with CTC or Attention in
isolation. If we compare among CTC and Attention, it is evident              4. Conclusion and Future Work
that CTC stands out in performance when using recall as the
evaluation metric, whereas the situation is reversed when using      In this paper, we have presented an effective end-to-end neural
precision as the metric. This also confirms our anticipation that    modeling framework for mispronunciation detection and
CTC-ATT is able to harness the synergistic power of CTC and          diagnosis (MDD), capitalizing on a hybrid CTC-Attention
ATT for use in mispronunciation detection.                           model structure and a novel anti-phone modeling technique. A
3.3.2. Evaluations on Mispronunciation Diagnosis                     series of empirical experiments carried on the L2-ARCTIC
                                                                     non-native English corpus have demonstrated its practical
In the third set of experiments, we turn to evaluating the           utility. As to future work, we are intended to investigate more
mispronunciation diagnosis performance of our methods with           sophisticated      modeling     techniques    to    characterize
different model architectures, viz. CTC-ATT, CTC and                 mispronunciations that contain non-categorial or distortion
Attention. As shown in Table 5, though the DAR results of            errors [19], as well as to apply and extend our methods to other
these three models still falls short of expectation, using CTC-      L2 CAPT tasks, such as MDD for Mandarin Chinese.
ATT stands out in comparison to using either CTC or Attention
in isolation. This also indicates that for the development of
                       5. References
[1]  S. M. Witt and S. J. Young, “Phone-level pronunciation scoring
     and assessment for interactive language learning,” Speech
     Communication, vol. 30, pp. 95–108, 2000.
[2] W. Hu, et al, “Improved mispronunciation detection with deep
     neural network trained acoustic models and transfer learning
     based logistic regression classifiers,” Speech Communication, vol.
     67, pp.154–166, 2015.
[3] S. Sudhakara, et al, “An improved goodness of pronunciation
     (GoP) measure for pronunciation evaluation with DNN-HMM
     system considering HMM transition probabilities,” Proceedings
     of the INTERSPEECH, pp. 954–958, 2019.
[4] W. Li, et al, “Improving non-native mispronunciation detection
     and enriching diagnostic feedback with DNN-based speech
     attribute modeling,” Proceedings of the ICASSP, pp. 6135–6139,
     2016.
[5] W. Lo, S. Zhang, and H. Meng, “Automatic derivation of
     phonological rules for mispronunciation detection in a computer-
     assisted pronunciation training system,” Proceedings of the
     INTERSPEECH, pp.765–768, 2010.
[6] X. Qian, F. K. Soong, and H. Meng, “Discriminative acoustic
     model for improving mispronunciation detection and diagnosis in
     computer-aided pronunciation training (CAPT),” in International
     Speech Communication Association, 2010.
[7] W. Leung, X. Liu and H. Meng, “CNN-RNN-CTC Based End-to-
     end Mispronunciation Detection and Diagnosis,” Proceedings of
     the ICASSP, pp. 8132–8136, 2019.
[8] Y. Feng, et al, “SED-MDD: towards sentence dependent end-to-
     end mispronunciation detection and diagnosis,” Proceedings of
     the ICASSP, pp. 3492-3496, 2020
[9] L. Chen, et al, “End-to-End neural network based automated
     speech scoring,” Proceedings of the ICASSP, pp. 6234–6238,
     2018.
[10] S. Mao, et al, “Unsupervised Discovery of an Extended Phoneme
     Set in L2 English Speech for Mispronunciation Detection and
     Diagnosis,” Proceedings of the ICASSP, pp. 6244–6248, 2018.
[11] X. Li, et al, “Unsupervised discovery of non-native phonetic
     patterns in L2 English speech for mispronunciation detection and
     diagnosis” Proceedings of the INTERSPEECH, pp. 2554–2558,
     2018.
[12] S. Watanabe, et al, “Hybrid CTC/attention architecture for end-
     to-end speech recognition,” IEEE Journal of Selected Topics in
     Signal Processing, vol. 11, no. 8, pp. 1240–1253, 2017.
[13] O. Ronen, L. Neumeyer, and H. Franco, “Automatic detection of
     mispronunciation for language instruction,” Proceedings of
     Speech Communication and Technology, pp. 649–652, 1997.
[14] J. K. Chorowski, et al, “Attention-based models for speech
     recognition,” Proceedings of NIPS, pp. 577–585, 2015.
[15] G. Zhao, et al. “L2-ARCTIC: A Non-native English Speech
     Corpus,” Proceedings of the INTERSPEECH, pp. 2783–2787,
     2018.
[16] V. Panayotov, et al, “Librispeech: an asr corpus based on public
     domain audio books,” Proceedings of the ICASSP, pp. 5206–5210,
     2015.
[17] J. Cho, et al, “Multilingual sequence-to-sequence speech
     recognition: architecture, transfer learning, and language
     modeling,” Proceedings of the SLT, pp. 521–527, 2018.
[18] K. Li, X. Qian and H. Meng, “Mispronunciation Detection and
     Diagnosis in L2 English Speech Using Multidistribution Deep
     Neural Networks,” IEEE/ACM Transactions on Audio, Speech,
     and Language Processing, vol. 25, 193–207, 2016.
[19] S. Mao, et al, “Applying multitask learning to acoustic-phonemic
     model for mispronunciation detection and diagnosis in L2 English
     speech,” Proceedings of the ICASSP, pp. 6254–6258, 2018.
