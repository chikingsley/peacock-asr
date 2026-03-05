Interspeech 2025
17-21 August 2025, Rotterdam, The Netherlands



 GoP2Vec: A few shot learning for pronunciation assessment with goodness of
 pronunciation (GoP) based representations from an i-vector framework and
                               augmentation
                                        Meenakshi Sirigiaju1 , Chiranjeevi Yarra1
        1
            Speech Processing Lab, Language Technologies Research Center (LTRC), IIITH, India
         meenakshi.sirigiraju@research.iiit.ac.in, chiranjeevi.yarra@research.iiit.ac.in


                          Abstract                                          lution, and a score-restraint attention pooling mechanism to
                                                                            capture both local and global contextual cues. Yang et al. [8]
Automatic pronunciation assessment is a critical component in               combined Multi-Head Self-Attention with Convolutional Neu-
computer assisted language learning. Typically, modeling pro-               ral Networks to extract local and global features, incorporating
nunciation assessment tasks need labels, which are difficult to             multi-layer feature fusion and multi-task loss weight optimiza-
obtain as it requires expert annotators. Thus, it is essential to           tion to enhance performance.
build an accurate model with less annotated data. In this work,
                                                                                 In addition to end-to-end approaches, fine-tuning pre-
an approach is proposed that considers a few speech samples
                                                                            trained models has also been widely adopted for pronuncia-
using the i-vector framework. Each sample, first, is length-
                                                                            tion assessment. Liang et al. [9] developed an encoder-decoder
ened by T factor by concatenating the augmented samples of
                                                                            framework with a mask-predict strategy to avoid misalignment
the same speech. The augmentation is obtained using time-scale
                                                                            issues, pre-training the model on ASR datasets and fine-tuning
modification (TSM), pitch-scale modification (PSM) and both.
                                                                            it with expert annotations. Ryu et al. [10] fine-tuned a pre-
Next, phoneme-level goodness-of-pronunciation scores of con-
                                                                            trained Wav2Vec 2.0 model on the TIMIT dataset for phone
catenated speech are converted to a vector (GoP2Vec) with the
                                                                            recognition, optimized jointly for pronunciation assessment and
i-vector framework. Experiments on two datasets revealed that
                                                                            mispronunciation detection and diagnosis tasks. Similarly, Kim
the proposed GoP2Vec outperforms the state-of-the-art (SOTA)
                                                                            et al. [11] fine-tuned Self-Supervised Learning models such as
unsupervised methods and is on par with the SOTA supervised
                                                                            HuBERT and Wav2Vec 2.0 using Connectionist Temporal Clas-
methods when it is used to train a simple neural model with a
                                                                            sification loss, extracting layer-wise contextual representations
few samples.
                                                                            for pronunciation scoring.
Index Terms: Pronunciation assessment, i-vector, few-shot
learning, CALL, data augmentation                                                While neural network (NN)-based pronunciation models
                                                                            have become more popular than traditional methods, as the NN
                                                                            models learn the pronunciation patterns directly from data, they
                     1. Introduction                                        are data-hungry. In the pronunciation assessment modelling,
Pronunciation assessment is a critical component of language                it is challenging to obtain a large amount of speech data col-
learning, particularly for second language (L2) learners. Ac-               lected from L2 learners, followed by annotations from experts.
curate pronunciation is essential for effective communication,              In contrast to NN-based models, the traditional pronunciation
as it directly impacts intelligibility and fluency. Traditionally,          assessment models do not require expert annotated ratings in
pronunciation assessment relies on human raters to evaluate                 the modelling [1, 2, 3, 4, 5]. As these methods do not utilise
speech samples, which can be subjective, time-consuming, and                annotated ratings, the performance from these models is below
resource-intensive. For language learners, timely and objective             that of the recent E2E and finetuning-based approaches.
feedback on pronunciation is crucial for identifying areas of                    To address the performance gap with a small amount of an-
improvement and accelerating the learning process. However,                 notated speech data, this work proposes an approach to convert
the scalability of human-based assessment is limited, creating a            the GoP score sequence of an utterance to a vector representa-
pressing need for automated systems that can provide consistent             tion using an i-vector framework and spoken data augmentation
and reliable evaluations.                                                   strategies. For augmenting a speech sample, a set of methods
     In the literature, pronunciation assessment was explored               is chosen such that it would not affect the pronunciation quality
through both supervised and unsupervised approaches. The                    of the augmenting speech sample. For augmenting speech, the
most common feature used is the Goodness of Pronunciation                   speech sample is modified with its time scale, pitch scale and
(GoP) [1] score, which relies on Automatic speech recognition               both. The augmented samples are randomly selected specific
(ASR) and forced alignment process in ASR. Many works were                  to each modification approach and concatenated to the original
proposed [2, 3, 4, 5] utilising GoP. Recently, end-to-end (E2E)             sample to increase overall duration. From the resultant con-
models have gained significant popularity due to their ability to           catenated speech, a GoP score sequence is computed using the
learn directly from raw audio inputs without requiring exten-               traditional approach and obtain a vector (GoP2Vec) representa-
sive feature engineering. For instance, Gong et al. [6] proposed            tion using i-vector modelling. For this, the background Gaus-
a Transformer-based model (GOPT) that uses multi-task learn-                sian mixture model (GMM) [12] and total variability matrix
ing to assess pronunciation at multiple granularities—phoneme,              are trained with the concatenated samples from the respective
word, and utterance levels—leveraging GOP features. Sim-                    augmentation technique. The GoP2Vec is used to train an NN-
ilarly, Chao et al. [7] introduced a hierarchical model for                 based classifier with a set of few samples to predict pronuncia-
multi-aspect and multi-granular pronunciation assessment, uti-              tion quality. Experiments on SpeechOcean762 and voisTUTOR
lizing sub-phoneme embeddings, depth-wise separable convo-                  corpora revealed that the proposed approach is on par with the




                                                                     5063                           10.21437/Interspeech.2025-2359
                           Figure 1: Block diagram of the proposed GoP2Vec based pronunciation assessment


state-of-the-art supervised E2E approaches and a significant im-              consisting GMM and extracting i-vectors using the Total Vari-
provement from the SOTA traditional unsupervised approaches.                  ability matrix. Since speech signals vary in length, the i-vector’s
                                                                              ability to provide a compact and fixed-dimensional representa-
                         2. Dataset                                           tion makes it highly effective for various speech applications. It
                                                                              is widely used in tasks such as speaker recognition [17, 18], lan-
For the experiments conducted in this work, the L2 datasets                   guage recognition [19], accent recognition [20], acoustic event
voisTUTOR [13] and SpeechOcean762 [14] are utilized.                          detection [21], emotion recognition [22], and speaker diariza-
                                                                              tion [23]. Using i-vectors for these applications has proven to
2.1. voisTUTOR                                                                be highly effective [24]. A key observation noted by Kenny et
                                                                              al. [25] is that i-vectors extracted from short utterances tend to
The dataset consists of 1,676 unique stimuli, featuring speech                be less reliable, whereas their reliability improves with longer
recordings from 16 Indian L2 learners of English. The learners,               utterances. Motivated by this insight, we propose augmentation
aged between 19 and 25, represent six native language back-                   strategies to extend speech sample length in this work.
grounds: Malayalam (4), Kannada (5), Telugu (3), Tamil (2),
Hindi (1), and Gujarati (1), with an equal distribution of 8 male
and 8 female speakers. The stimuli include words with a min-                                       4. Methodology
imum of 1 and a maximum of 26 in length. The pronunciation                    The block diagram in Figure 1 shows the steps involved in ex-
quality of each utterance was evaluated by an expert on a scale               tracting the proposed GoP2Vec and its use in pronunciation
of 0 to 4, reflecting the overall quality of the speech. The dataset          score prediction. In the first step, each speech sample length is
spans a total duration of 14 hours.                                           increased by T times by concatenating the augmented samples
                                                                              obtained from the respective speech. The augmented speech
2.2. SpeechOcean762                                                           sample is obtained by changing its length and pitch using time-
                                                                              scale modification (TSM) and pitch-scale modification (PSM),
It consists of 5,000 English utterances from 250 L2 learners,
                                                                              respectively. Further, the speech sample also increased by con-
with train and test splits of 2,500 utterances each. The learners
                                                                              sidering the combination of augmented samples from both mod-
are evenly split between children and adults, all native Mandarin
                                                                              ifications. In the second step, goodness of pronunciation scores
speakers with a 1:1 gender ratio. Each utterance was manually
                                                                              are obtained at the phoneme level using force-aligned phoneme
rated by five experts at the phoneme, word, and sentence levels.
                                                                              segment boundaries. In the third step, phoneme level GoP
The phoneme-level score reflects the pronunciation accuracy of
                                                                              scores for the entire increased utterance are converted to a vec-
each phone, while the word-level scores include accuracy and
                                                                              tor (GoP2Vec) using i-vector-based modelling [15]. For each
stress. At the sentence level, scores capture accuracy, complete-
                                                                              augmentation method, a background GMM and total variability
ness, fluency, and prosody. The dataset spans 6 hours, with
                                                                              matrix are learned separately and considered for the respective
each learner reading 20 sentences ranging from 1 to 20 words
                                                                              augmented speech sample. In the fourth step, the GoP2Vec is
in length. In this work, we consider the median of the five expert
                                                                              passed to the NN-based score predictor for obtaining a pronun-
scores provided for each utterance, utilizing the sentence-level
                                                                              ciation quality score.
total scores that indicate overall pronunciation quality on a scale
of 0 to 10.
                                                                              4.1. Data Augmentation

                      3. Background                                           Generally, the vector obtained from i-vector computation is ef-
                                                                              fective when the length of the speech sample is greater than 5s,
An i-vector (identity vector) [15] is a compact, fixed-                       i.e., a sequence of more than 500 frames [15]. However, the
dimensional representation of variable-length sequences. It                   phoneme level score sequences are much shorter than this re-
builds on the concept of factor analysis to capture underlying                quirement. To achieve a longer length sequence, augmentation
variability in a concise form. Introduced as an improvement                   strategies that do not affect the pronunciation quality are con-
over Joint Factor Analysis (JFA) [16], the key advantage of the               sidered for increasing the speech sample length with the con-
i-vector approach is that it consolidates all variabilities into a            catenation process. The learners speaking with a normal rate
single total variability space. The process generally involves                range may not affect the pronunciation quality due to minimal
two main steps: training a Universal Background Model (UBM)                   changes in the phoneme length. The TSM technique changes




                                                                       5064
the duration of the sample without affecting the message con-                  Table 1: Comparison of the proposed approach using three data
tent and speaker identity. In this work, the augmented samples                 augmentation strategies with the baselines.
are generated with TSM, considering the scale ranging from 0.8
                                                                                   Dataset                    Proposed approach              Baselines
to 1.2. The phoneme level scores are independent of the pitch in                                 # Train samples TSM PSM          TSM+PSM   SV     USV
the speech sample. The PSM technique changes the pitch from                       voisTUTOR            150         0.66   0.69      0.67      -    0.61
                                                                                SpeechOcean762         241         0.68   0.71      0.69    0.74   0.62
one value to another without changing the duration and message
content of the speech sample. The augmented speech samples
are generated considering a pitch scale ranging from -20Hz to
20Hz. Further, the same time and pitch scale ranges are applied                4.4. NN-based score Computation
when combined augmentation from both techniques is consid-                     A simple multi-layer perception (MLP) is considered for pre-
ered. The augmented samples of these ranges were randomly                      dicting a score in a supervised manner. The considered MLP
selected and used in the concatenation process in all three types              layer has the following architecture: The MLP architecture con-
of length-increasing strategies.                                               sists of an input layer, followed by two hidden layers with ReLU
                                                                               activation and an output layer. The output layer has r units,
4.2. GoP Score Computation                                                     where ’r’ represents the number of rating classes. During train-
                                                                               ing, the logits are passed through a softmax function to compute
Typically, the GoP computation methods provide phoneme level                   class probabilities, and the model is optimized using CrossEn-
scores indicating its pronunciation quality in reference to the re-            tropyLoss. The architectural choices, including the number of
spective native speaker’s phoneme pronunciation. These meth-                   hidden layers and hidden units, are determined based on perfor-
ods were developed using the ASR framework, considering                        mance on validation set. It is to be noted that the architecture
phoneme segment boundaries, which were obtained by force-                      is simple and needs a few samples with labels to predict a score
aligning second language learner’s speech with native speaker                  for the pronunciation quality.
phoneme sequence. For this, the ASR is trained on the native
speaker’s spoken data and its respective pronunciation lexicon.                Table 2: Cross-corpus analysis along with varying word lengths
Further, in the force-aligning, the native phoneme sequence is
automatically identified by the ASR when the native pronunci-                        Train                                  Test (Correlation)
                                                                                                    Word length      voisTUTOR SpeechOcean762
ation lexicon is used. We compute GoP scores for second lan-
                                                                                  voisTUTOR             1                0.64              0.65
guage learners’ English speech following the work by Sweekar                                           2-7               0.73              0.74
et al. [5]. The choice is due to its rating-independent state-                                         >7                0.76              0.77
of-the-art nature in GoP computation and its effectiveness in                                          All               0.69              0.70
correlating with ground-truth ratings. Following their work, for                SpeechOcean762          1                0.63              0.66
the computation, we have considered Kaldi ASR tool kit [26]                                            2-7               0.72              0.75
                                                                                                       >7                0.75              0.78
trained on English speech data from Libri-speech [27] for force-                                       All               0.68              0.71
alignment and GoP computation.

4.3. Gop2Vec Computation                                                                         5. Experimental setup
                                                                               We conduct the experiments on two datasets: voisTUTOR and
Gop2Vec pl embeds pronunciation quality for lth concatenated
                                                                               SpeechOcean762. From voisTUTOR data, a random 150 out of
speech sample (Al ) is computed as follows:
                                                                               12,535 samples are used for training, while the remaining sam-
     1) The GoP scores sequence (Gl ) for Al is obtained for                   ples serve as the test set. From the SpeechOcean762 data, 241
each phoneme using the GoP computation indicated as Gl =                       out of 2,500 training samples are utilized for training while for
{gi ; 1 ≤ i ≤ nl }, where nl is the number of phonemes in Al .                 the testing all 2500 samples in the test set are used. The train-
     2) The pl is computed as, (I + VT Σ−1 NV)−1 VT Σ−1 F,                     ing samples were chosen to ensure balanced representation of
where, I is the identity matrix; V is the total variability matrix;            all rating levels. We use Kaldi toolkit for implementing i-vector
Σ is the background GMM covariance matrix; N is the diag-                      framework. We consider two baselines: 1. A supervised ap-
onal matrix with diagonal entries as Nk and F is vector with                   proach proposed in [6], which integrates GoP features with a
elements as Fk . The matrices V and Σ are obtained from the                    Transformer-based architecture. 2. An unsupervised approach
training data sample, whereasPNk and Fk are specific                           proposed in [5], which relies solely on GoP features. We evalu-
                                 nl                  Pnlto the Al              ate model performance using the Pearson correlation coefficient
and is computed as, Nk =         i=1 γk,i ,  Fk =       i=1 γk,i gi ,
where Nk , Fk represents the zeroth and first-order statistics of              [28], which measures the correlation between predicted and ac-
kth mixture for a given GoP score sequence Gl , and γk,i is the                tual ratings.
posterior probability of the kth mixture of background GMM                                                6. Results
given gi .
                                                                               In this section, we present the results as follows: 1) Comparison
     3) Training: The GMM is trained on the GoP score se-
                                                                               of the baseline and proposed approach; 2) Effect of sentence
quences of all the utterances present in the train set and ob-
                                                                               length under cross and matched conditions and 3) Analysis on
tained its parameters: µk ; σk ; λk ∀k ∈ 1 ≤ k ≤ K, where K
                                                                               GMM components & GoP2Vec dimensions.
is total number of mixture components. We obtain diagonal ma-
trix Σ whose diagonal elements are σk . Using the super-vector
                                                                               6.1. Comparison with baselines:
µ = [µ1 , . . . , µk ]T from the background GMM, we obtain the
total variability matrix V by solving the following equation it-               Table 1 presents the correlations achieved by the proposed
eratively; µl = µ + V ql + ϵl , where, the covariance matrix of                approach, which utilizes three data augmentation strate-
ϵl is approximated by Σ.                                                       gies—TSM, PSM, and TSM+PSM—against two baselines:




                                                                        5065
Table 3: Correlation with varying no of Gaussian compo-                      strategy, and highlights the impact of sentence length on pro-
nents and GoP2Vec dimensions for both voisTUTOR & Spee-                      nunciation assessment. For the voisTUTOR dataset, the correla-
chOcean762 (in brackets) test datasets                                       tion improves as the sentence length increases. The correlation
                                                                             for single-word utterances is 0.63, but for sentences with 2–7
 GMM (k)                         Correlation                                 length, the correlation increases to 0.73, showing a 15.9% rel-
                            GoP2Vec dimension (d)
                                                                             ative improvement. For sentences with more than 7 length, the
                 2        5      8        20      50        100
               0.59     0.6     0.61     0.62   0.63        0.64             correlation further increases to 0.76, with a 20.6% relative im-
     2                                                                       provement compared to single-word utterances. Similarly, for
              (0.66)   (0.65) (0.65) (0.67) (0.67)         (0.68)
                0.6     0.62    0.63     0.64   0.65        0.66             the SpeechOcean762 dataset, the correlation for single-word ut-
     4                                                                       terances starts at 0.66, increases to 0.72 for sentences with 2–7
              (0.66)   (0.66) (0.67) (0.67) (0.68)         (0.68)
               0.61     0.63    0.65     0.66   0.67        0.67             length (a relative improvement of 9.1%), and further increases
     8
              (0.67)   (0.67) (0.67) (0.68) (0.68)         (0.68)            to 0.75 for words with more than 7 length (a relative improve-
               0.63     0.69    0.68     0.67   0.66        0.65             ment of 13.6%).
     16
              (0.67)   (0.67) (0.71) (0.68) (0.68)         (0.69)
                                                                                  These results suggest that longer sentences provide richer
               0.62     0.68    0.67     0.66   0.65        0.64
     32
              (0.67)   (0.68) (0.68) (0.68) (0.69)         (0.69)
                                                                             pronunciation information, leading to more reliable and ac-
               0.61     0.67    0.66     0.64   0.63        0.62             curate pronunciation assessments. The cross-dataset analysis
     64                                                                      also reveals similar trends. For instance, when the model is
              (0.68)   (0.68) (0.68) (0.69) (0.70)         (0.70)
               0.60     0.65    0.64     0.62   0.61        0.60             trained on voisTUTOR and tested on SpeechOcean762, the
    128                                                                      correlation increases from 0.64 for single-word utterances to
              (0.68)   (0.68) (0.69) (0.69) (0.70)         (0.70)
               0.59     0.63    0.62     0.61   0.60        0.59             0.74 for sentences with 2-7 length (15.6% relative improve-
    256
              (0.69)   (0.69) (0.69) (0.70) (0.70)         (0.71)            ment), and reaches 0.77 for sentences with more than length
                                                                             7 (20.3% relative improvement). Conversely, when trained on
                                                                             SpeechOcean762 and tested on voisTUTOR, the model shows
the Supervised Baseline (SV) and the Unsupervised Baseline                   similar improvements, with correlations rising as word length
(USV). The results are summarized for two datasets, voisTU-                  increases.
TOR and SpeechOcean762.                                                           Therefore, the results indicate that the proposed approach
     voisTUTOR: The proposed approach achieves the highest                   is not only effective within a single dataset but also exhibits
correlation of 0.69 using the PSM strategy. This result outper-              strong cross-dataset generalization, performing well across dif-
forms the TSM (0.66) and TSM+PSM (0.67) variations, as well                  ferent word lengths and datasets.
as the USV baseline (0.61). The supervised baseline (SV) is un-
available for the voisTUTOR dataset because it lacks detailed                6.3. GMM components & GoP2Vec dimensions analysis:
utterance and word level annotations.
     SpeechOcean762: A similar trend is observed. The pro-                   Table 3 presents the correlation values for pronunciation assess-
posed approach with PSM obtains the best correlation of 0.71,                ment performance across both the test datasets. The rows rep-
followed by TSM (0.68) and TSM+PSM (0.69). While the pro-                    resent different values of k (ranging from 2 to 256), while the
posed method surpasses the USV baseline (0.62), it falls slightly            columns correspond to different values of d (ranging from 2 to
short of the supervised SV baseline, which achieves a correla-               100). Each cell contains two correlation values: the first for
tion of 0.74.                                                                the voisTUTOR dataset and the value in parentheses for Spee-
                                                                             chOcean762. The results indicate a general trend of increasing
     Additionally the supervised baseline was trained on approx-
                                                                             correlation as both k and d grow. However, the best perfor-
imately 2500 samples, whereas the proposed approach achieves
                                                                             mance is observed at k = 16 with d = 5 for voisTUTOR and
competitive performance using only the minimal sample sizes
                                                                             d = 8 for SpeechOcean762. While higher values of k and d
indicated in the table. PSM outperforms TSM as it involves
                                                                             provide similar performance, they are computationally less ef-
modifying the pitch of speech while maintaining the original
                                                                             ficient. Interestingly, the optimal performance occurs when the
duration and message content, preserving the overall structure
                                                                             GoP2Vec dimension matches the number of classes, suggesting
of pronunciation. Since pitch variations are speaker-dependent
                                                                             that i-vector training may be learning class-specific information
and do not inherently lead to incorrect pronunciation, PSM al-
                                                                             in each dimension. These findings emphasize the importance of
lows for more natural speech variations, which might explain
                                                                             balancing model complexity and computational efficiency for
its superior performance. On the other hand, TSM alters the
                                                                             optimal pronunciation assessment.
duration of the speech while keeping the pitch constant. When
the duration of speech is modified, it can disrupt the natural
flow and rhythm, potentially leading to degraded speech quality                                   7. Conclusion
and affecting the perceived accuracy of the pronunciation. This
disruption can be particularly detrimental when evaluating the               In this work, we addressed the challenge of automatic pronun-
correlation between predicted and actual ratings, which is why               ciation assessment focusing on reducing the reliance on large
TSM may result in lower correlation scores compared to PSM.                  amounts of annotated data. We proposed a novel approach that
                                                                             leverages a few data samples by augmenting speech through
                                                                             TSM, PSM and combination of both. We develop a method
6.2. Effect of sentence length on performance:
                                                                             to obtain a vector (GoP2Vec) representation leveraging the i-
Table 2 presents the correlation results for sentences with differ-          vector framework. Our experiments on two datasets demon-
ent lengths across both the voisTUTOR and SpeechOcean762                     strated that the proposed GoP2Vec approach outperforms un-
datasets, including both within-dataset (matching) and cross-                supervised methods and performs on par with supervised ap-
dataset (training on one dataset and testing on the other) eval-             proaches when trained with a limited number of samples. Fu-
uations. The analysis primarily focuses on the performance of                ture works include to deduce strategies for better augmentation
the proposed approach, utilizing the PSM-based augmentation                  strategies in GoP2Vec computation for better performance.




                                                                      5066
                         8. References                                              [16] P. Kenny, “Joint factor analysis of speaker and session variability:
                                                                                         Theory and algorithms,” CRIM, Montreal,(Report) CRIM-06/08-
 [1] S. M. Witt, “Use of speech recognition in computer-assisted lan-                    13, vol. 14, 2005.
     guage learning,” Ph.D. dissertation, University of Cambridge, De-
     partment of Engineering, 2000.                                                 [17] S. Biswas, J. Rohdin, and K. Shinoda, “i-vector selection for
                                                                                         effective plda modeling in speaker recognition,” ODYSSEY, The
 [2] J. J. H. C. van Doremalen, C. Cucchiarini, and H. Strik, “Using                     Speaker and Language Recognition Workshop, pp. 100–105,
     non-native error patterns to improve pronunciation verification,”                   2014.
     in Proceedings of Interspeech – 11th Annual Conference of the
     International Speech Communication Association (ISCA), 2010.                   [18] P.-M. Bousquet, D. Matrouf, and J.-F. Bonastre, “Intersession
                                                                                         compensation and scoring methods in the i-vectors space for
 [3] Y. Song, W. Liang, and R. Liu, “Lattice-based gop in automatic                      speaker recognition,” in Proceedings of Interspeech – 12th Annual
     pronunciation evaluation,” in The 2nd International Conference                      Conference of the International Speech Communication Associa-
     on Computer and Automation Engineering (ICCAE), 2010.                               tion (ISCA), 2011.
 [4] D. Luo, Y. Qiao, N. Minematsu, Y. Yamauchi, and K. Hirose,                     [19] D. Martinez, O. Plchot, L. Burget, O. Glembek, and P. Matějka,
     “Analysis and utilization of mllr speaker adaptation technique for                  “Language recognition in ivectors space,” in Proceedings of In-
     learners’ pronunciation evaluation,” in Proceedings of Interspeech                  terspeech – 12th Annual Conference of the International Speech
     – 10th Annual Conference of the International Speech Communi-                       Communication Association (ISCA), 2011.
     cation Association (ISCA), 2009.
                                                                                    [20] H. Behravan, V. Hautamäki, and T. Kinnunen, “Factors affecting
 [5] S. Sudhakara, M. K. Ramanathi, C. Yarra, and P. K. Ghosh, “An                       i-vector based foreign accent recognition: A case study in spoken
     improved goodness of pronunciation (gop) measure for pronunci-                      finnish,” Speech Communication, vol. 66, pp. 118–129, 2015.
     ation evaluation with dnn-hmm system considering hmm transi-
                                                                                    [21] Z. Huang, Y.-C. Cheng, K. Li, V. Hautamäki, and C.-H. Lee, “A
     tion probabilities.” in Proceedings of Interspeech – 20th Annual
                                                                                         blind segmentation approach to acoustic event detection based on
     Conference of the International Speech Communication Associa-
                                                                                         i-vector.” in Proceedings of Interspeech – 14th Annual Conference
     tion (ISCA), 2019.
                                                                                         of the International Speech Communication Association (ISCA),
 [6] Y. Gong, Z. Chen, I.-H. Chu, P. Chang, and J. Glass,                                2013.
     “Transformer-based multi-aspect multi-granularity non-native en-               [22] R. Xia and Y. Liu, “Using i-vector space model for emotion recog-
     glish speaker pronunciation assessment,” in 2022 IEEE Interna-                      nition.” in Proceedings of Interspeech – 13th Annual Conference
     tional Conference on Acoustics, Speech and Signal Processing                        of the International Speech Communication Association (ISCA),
     (ICASSP), pp. 7262–7266.                                                            2012.
 [7] F.-A. Chao, T.-H. Lo, T.-I. Wu, Y.-T. Sung, and B. Chen, “A                    [23] J. Silovsky and J. Prazak, “Speaker diarization of broadcast
     hierarchical context-aware modeling approach for multi-aspect                       streams using two-stage clustering based on i-vectors and co-
     and multi-granular pronunciation assessment,” arXiv preprint                        sine distance scoring,” in 2012 IEEE International Conference
     arXiv:2305.18146, 2023.                                                             on Acoustics, Speech and Signal Processing (ICASSP), pp. 4193–
 [8] J. Yang, A. Wumaier, Z. Kadeer, L. Wang, S. Guo, and J. Li,                         4196.
     “Attention-cnn combined with multi-layer feature fusion for en-                [24] P. Verma and P. K. Das, “i-vectors in speech processing appli-
     glish l2 multi-granularity pronunciation assessment,” in 2023                       cations: a survey,” International Journal of Speech Technology,
     IEEE 4th International Conference on Pattern Recognition and                        vol. 18, pp. 529–546, 2015.
     Machine Learning (PRML), pp. 449–457.
                                                                                    [25] P. Kenny, T. Stafylakis, P. Ouellet, M. J. Alam, and P. Dumouchel,
 [9] Y. Liang, K. Song, S. Mao, H. Jiang, L. Qiu, Y. Yang, D. Li,                        “Plda for speaker verification with utterances of arbitrary du-
     L. Xu, and L. Qiu, “End-to-end word-level pronunciation assess-                     ration,” in 2013 IEEE International Conference on Acoustics,
     ment with mask pre-training,” arXiv preprint arXiv:2306.02682,                      Speech and Signal Processing (ICASSP), pp. 7649–7653.
     2023.
                                                                                    [26] D. Povey, A. Ghoshal, G. Boulianne, L. Burget, O. Glembek,
[10] H. Ryu, S. Kim, and M. Chung, “A joint model for pronunci-                          N. Goel, M. Hannemann, P. Motlicek, Y. Qian, P. Schwarz,
     ation assessment and mispronunciation detection and diagnosis                       J. Silovsky, G. Stemmer, and K. Vesely, “The kaldi speech recog-
     with multi-task learning,” in Proceedings of Interspeech – 24rd                     nition toolkit,” in IEEE 2011 Workshop on Automatic Speech
     Annual Conference of the International Speech Communication                         Recognition and Understanding.
     Association (ISCA), 2023.
                                                                                    [27] V. Panayotov, G. Chen, D. Povey, and S. Khudanpur, “Lib-
[11] E. Kim, J.-J. Jeon, H. Seo, and H. Kim, “Automatic pronunciation                    rispeech: an asr corpus based on public domain audio books,”
     assessment using self-supervised speech representation learning,”                   in 2015 IEEE international conference on acoustics, speech and
     arXiv preprint arXiv:2204.03863, 2022.                                              signal processing (ICASSP), pp. 5206–5210.
[12] A. P. Dempster, N. M. Laird, and D. B. Rubin, “Maximum likeli-                 [28] I. Cohen, Y. Huang, J. Chen, J. Benesty, J. Benesty, J. Chen,
     hood from incomplete data via the em algorithm,” Journal of the                     Y. Huang, and I. Cohen, “Pearson correlation coefficient,” Noise
     royal statistical society: series B (methodological), vol. 39, no. 1,               reduction in speech processing, pp. 1–4, 2009.
     pp. 1–22, 1977.
[13] C. Yarra, A. Srinivasan, C. Srinivasa, R. Aggarwal, and P. K.
     Ghosh, “voistutor corpus: A speech corpus of indian l2 en-
     glish learners for pronunciation assessment,” in 22nd Conference
     of the Oriental COCOSDA International Committee for the Co-
     ordination and Standardisation of Speech Databases and Assess-
     ment Techniques (O-COCOSDA). IEEE, 2019, pp. 1–6.
[14] J. Zhang, Z. Zhang, Y. Wang, Z. Yan, Q. Song, Y. Huang, K. Li,
     D. Povey, and Y. Wang, “speechocean762: An open-source non-
     native english speech corpus for pronunciation assessment,” arXiv
     preprint arXiv:2104.01378, 2021.
[15] N. Dehak, P. J. Kenny, R. Dehak, P. Dumouchel, and P. Ouellet,
     “Front-end factor analysis for speaker verification,” IEEE Trans-
     actions on Audio, Speech, and Language Processing, vol. 19,
     no. 4, pp. 788–798, 2010.




                                                                             5067
