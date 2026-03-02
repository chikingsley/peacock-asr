# A Full Text-Dependent End to End Mispronunciation Detection and Diagnosis with Easy Data Augmentation Techniques

###### Abstract

Recently, end-to-end mispronunciation detection and diagnosis (MD&D) systems has become a popular alternative to greatly simplify the model-building process of conventional hybrid DNN-HMM systems by representing complicated modules with a single deep network architecture. In this paper, in order to utilize the prior text in the end-to-end structure, we present a novel text-dependent model which is difference with sed-mdd, the model achieves a fully end-to-end system by aligning the audio with the phoneme sequences of the prior text inside the model through the attention mechanism. Moreover, the prior text as input will be a problem of imbalance between positive and negative samples in the phoneme sequence. To alleviate this problem, we propose three simple data augmentation methods, which effectively improve the ability of model to capture mispronounced phonemes. We conduct experiments on L2-ARCTIC, and our best performance improved from 49.29 to 56.08 in F-measure metric compared to the CNN-RNN-CTC model.

Index Terms: mispronunciation detection and diagnosis, computer-aided pronunciation training, end-to-end model, attention mechanism

## 1 Introduction

With the development of speech technology and the promotion of online learning, computer-aided pronunciation training (CAPT) has been more fully applied to language teaching [1, 2]. As an important part of CAPT technology, mispronunciation detection and diagnosis (MD&D) can be used to detect the mispronunciation in a L2 learner‘s speech, and further diagnose and give learners effective feedback.

A great deal of research on mispronunciation detection has been carried out [3-9], these methods can be grouped into two categories. The first are pronunciation scoring based on confidence measures originally proposed for automatic speech recognition (ASR). A namely goodness of pronunciation (GOP) scores [3] is computed by log-posterior probability based on force alignment, the method and its prominent variants [4, 5] currently mostly popular and achieve a promising performance on mispronunciation detection. But, this kind of method is not only unable to deal with the insertion errors in pronunciation, but also fail to give more detailed diagnosis to learners.

The second category aims to assess the type of the mispronunciation and provide informative feedback on specific errors. A well-know method of this category is extend recognition network (ERN) [6], which incorporate expected pronunciation error patterns into the lexicon to constrain the recognition paths to the canonical pronunciation and the likely phonetic mispronunciations. These ERNs [6, 7, 8] compiled by handcrafted or data-driven rules [9] have the advantage that the errors and the error types are detected together, and thus can be used for the system to provide diagnostic feedback. But it is difficult to build ERNs that incorporate as many as possible mispronunciation paths so that the recall performance improvement is limited [10].

Recently, the end-to-end structure shows a good talent for ASR tasks [11, 12], and gets promising performance in MDD [10, 13, 14]. A CNN-RNN-CTC free phone recognize model was proposed [10] firstly, which showed an outperformed result comparied with previously approaches [15]. The hybrid CTC-Attention architecture [14] with expanding the original L2 phone set was used to detect categorial and non-categorial errors. Those CTC-based methods does not require forced alignment, and intergrate the whole training pipeline. However, in the case of the reading text already known, the above research does not use the existing prior text information. To utilize the prior linguistic information, previous study [13] proposed a text-dependent end to end model that uses attention mechanism to combine the high-level hidden features of acoustics and the sentence-to-character encoder feature.

The model proposed in this paper is similar to sed-mdd [13] to utilize the prior text information, but there are still some differences. Feng et.al convert the prior text into a sequence of characters for sentence-embedding, The task of MD&D aims to detect errors in phoneme-level, so it makes sense to convert text into phoneme sequences and fed them into the sentence encoder. In addition, we also noticed that in the sed-mdd, manually labeled phoneme boundary was used to calculate the frame-level cross-entropy loss function. In this paper, connectionist temporal classification (CTC) [16] loss function was used without any labeled time information.

In this work, we propose an end-to-end mispronunciation detection framework based on the prior text attention mechanism, and explore the effects of sentence-to-phoneme. Furthermore, there are many more correctly pronounced phonemes than incorrectly pronounced in the training set, there will be a problem of imbalance between positive and negative samples in the phoneme sequence we send to the attention model, which leads to the tendency to output the input phoneme sequences. For this we propose three easy data augment techniques to explore the influence of modifying the input phoneme sequences. Finally, in our experiment, all datasets, metrics and baseline systems are open source https://github.com/cageyoko/CTC-Attention-Mispronunciation.

The rest of this paper is organized as follows. The following section describes our system in detail and introduces several methods of data augmentation. In section 3, experimental results will demonstrate. Section 4 concludes with a discussion of potential future work.

## 2 Our Methods

### 2.1 System overview

The input of the model is the phoneme sequence converted from prior text and the fbank acoustic feature, and the output of the model is the phoneme sequence corresponding to the audio. Specifically, it is assumed that the audio input is , where represents the feature vector of speech at frame and represents the total number of frames of a speech. The -length phoneme sequence input is , is a phoneme at position in the prior phoneme sequence. Our model contains three modules: the sentence encoder, the audio encoder and decoder with the attention mechanism. The two encoders extract high-level feature representations from the input speech feature and sentence-to-phoneme sequences :

| (1) |

| (2) |

where the and are the query and key. In our work, wo adopt CNN-RNN struct as the audio encoder that has been experimented in the previous research [10] and showed a good results, Bi-LSTM as the sentence encoder that has strong sequential modeling ability. Then the attention mechanism takes the , and as input and forms a fixed-length context vector:

| (3) |

The workflow of the entire model is shown in Figure 1. we detail the encoders and the attention mechanism in the following subsections.

#### 2.1.1 Sentence Encoder

The input of the encoder is phoneme sequences of the prior text, after feeding the phoneme sequence into the model, each phoneme is embedded into a vector and fed into a Bi-LSTM with hidden size=384 and dropout=0.2 to obtain the output sequence , denoted as value, where is 2*hidden size. This value is fed into linear layer with 2*hidden size as input and 2*hidden size as output to obtain the output sequence , denoted as key, where is equal to 2*hidden size. the key and value will be fed into the decoder module in the next step.

| (4) |

| (5) |

| (6) |

where represents function of Bi-LSTM cell and represents fully connected layer.

#### 2.1.2 Audio Encoder

The input of the encoder is a 243-dimensional feature. It is worth noting that the stacking of the left and right frames and the current frame is used here, and the original audio feature is an 80-dim fbank and 1-dim energy. After the audio features are fed into the model, they first go through the CNN-RNN module to get the output sequence , denoted as query. denotes the length of the original -frame speech after the CNN downsampling operation, where .

| (7) |

where represents function of CNN-RNN layers. The CNN-RNN module consists of two CNN layers and four Bi-LSTM layers. Batch normalization is employed following each layer to help the model converge better.

#### 2.1.3 Decoder with attention

Recall that the attention mechanism forms a context vector c that contains the information of key . Notably, the attention mechanism can use both past and future frames of a time sequence. Thus, a normalized attention weight is learned using and :

| (8) |

where . is the attention weight, which makes a monotonic alignment between the audio and the prior text input. Finally, we compute the context vector as the weighted average of :

| (9) |

Considering that the text information used may be too strong in the model, here we use the previous value as acoustic residual. Given and , the final framewise probability is:

| (10) |

where the [·;·] denotes the concatenation of two vectors. Beamsearch is used here to generate recognized phoneme sequence with .

### 2.2 Data Augmentation

One difficult of the model our proposed is the imbalance between positive and negative samples in training data. Here, we define a positive sample as a mispronunciation phone, and negative samples as one that is the canonical. Without data augmentation, the model would tend to output the prior phonemes and ignore the learn‘s mispronunciation, i.e., higher accuracy and lower recall, because the prior phoneme sequences were fed into the model, the most of which is negative samples, i.e., the number of negative samples is high and the number of positive samples is less, which leads to the model preferring to output canonical phonemes. A key goal of mispronunciation detection is to detect the wrong phonemes, thus to make our model can learn more features of positive samples, we increase the number of positive samples by the following three easy random replacement data augmentation operations:

1. Phoneme set based(PS): Randomly choose phonemes from phoneme sequence of the prior text, replace each of these phonemes with the phoneme set at random, e.g. . It is worth noting that ”blank” symbol may be used instead of the phoneme, which is equivalent to an INSERT type error. It may also happen that the ”blank” is replaced by a phoneme in the reading text, which is equivalent to a DELETE type error.

2. The vowels and consonants set based(VC): The vowels are more likely to be mispronounced with vowels and consonants with consonants based on L2-ARCTIC statistics [17] such as the frequent substitution: . So randomly choose phones from phoneme sequences of the prior text, if the phoneme belong to vowel/consonant, replace it with a phoneme in vowel/consonant set at random.

3. The confusing pairs based(CP): Firstly, we count the confused pairs of learner pronunciation on the phonemes in the L2-Arctic part of the training set, then, we randomly choose phones from phoneme sequences of the reading text, if the phoneme belong to confused pairs, replace it with with its confusing phoneme at random.

## 3 EXPERIMENTS

### 3.1 Speech Corpus

We conducte TIMIT [18] and L2-ARCTIC [17] datasets to evaluate the performance of the our model, both corpora are publicly available. The L2-ARCTIC is a non-native English corpus built for CAPT and other tasks, it contains 24 non-native speakers (12 males and 12 females), whose L1 languages include Hindi, Korean, Mandarin, Spanish, Arabic and Vietnamese.

To unify the phone set of these two datasets, we map the 61-phone set in TIMIT and the 48-phone set in L2-Arctic to the same 39-phone set. All TIMIT data are used as training data. for L2-ARCTIC that has been annotated by experts, we select 6 speakers (”NJS”, ”TLV”, ”TNI”, ”TXHC”, ”YKWK”, ”ZHAA”) to build the test set, and the 6 speakers (”MBMPS”, ”THV”, ” SVBI”, ”NCC”, ”YDCK”, ”YBAA”) are used as development data to save our best model, remainding 12 speakers as training data. The detail of data split is shown in Table 1.

In our experiments, the input phoneme sequences were provided by these two corpora. Certainly, we also can produce it by the Montreal forced-aligner [19] as [17]. Our all models are trained using same parameters, such as learning rate, batch size, etc.

| TIMIT | L2-ARCTIC | |||
|---|---|---|---|---|
| Train | Train | Dev | Test | |
| Speakers | 630 | 12 | 6 | 6 |
| Utterances | 6300 | 1800 | 897 | 900 |
| Hours | 4.5 | 1.84 | 0.94 | 0.88 |

### 3.2 Performance for phone recognition

In this section, phone recognition evaluation metric employed is the phone error rate(PER), which is computed by aligning the annotated phone sequence and recognized phone sequence by the model using edit distance algorithm. The experimental results of phone recognition are shown in Table 3, which shows that the PER of the free phone recognition using the CNN-RNN-CTC is 27.75%, the result means that the original model on these data does not perform well. After introducing a priori text information, the setence-to-phoneme and the setence-to-character reported on sed-mdd have performances of 16.06% and 20.66% respectively in our proposed model structure, we can find that taking phoneme recognition as the goal and using text phoneme sequence as input is helpful to improve the performance of phoneme recognition. Although we modified the prior phoneme sequence in data augmentation, the performance did not decrease, and randomly selected 10% of the input phoneme sequence and randomly modified it according to the confusion-pairs(CP10%) to achieve the best PER of 15.48%.

| Annotation | |||||||
| aa | ah | ae | eh | ih | iy | ||
| CNN-RNN-CTC | aa | 231 | 46 | 15 | 7 | 0 | 0 |
| ah | 59 | 2035 | 37 | 51 | 72 | 16 | |
| ae | 20 | 32 | 630 | 62 | 5 | 0 | |
| eh | 1 | 45 | 75 | 544 | 43 | 3 | |
| ih | 0 | 105 | 29 | 49 | 1256 | 170 | |
| iy | 1 | 19 | 0 | 6 | 177 | 1097 | |
| Attention-VC(10%) | aa | 310 | 12 | 14 | 1 | 0 | 0 |
| ah | 33 | 2421 | 10 | 42 | 29 | 7 | |
| ae | 19 | 7 | 757 | 19 | 0 | 1 | |
| eh | 1 | 15 | 32 | 694 | 14 | 1 | |
| ih | 0 | 13 | 8 | 19 | 1567 | 115 | |
| iy | 0 | 10 | 0 | 3 | 77 | 1214 | |
| most frequency misrecognized vowels | |||||||
| d | dh | t | sh | s | z | ||
| CNN-RNN-CTC | d | 1067 | 193 | 77 | 0 | 1 | 5 |
| dh | 74 | 125 | 7 | 0 | 2 | 12 | |
| t | 173 | 14 | 1280 | 3 | 8 | 5 | |
| sh | 0 | 0 | 2 | 313 | 3 | 1 | |
| s | 8 | 4 | 12 | 10 | 1457 | 185 | |
| z | 3 | 3 | 4 | 0 | 97 | 228 | |
| Attention-VC(10%) | d | 1278 | 191 | 68 | 0 | 0 | 1 |
| dh | 108 | 171 | 6 | 0 | 2 | 17 | |
| t | 57 | 3 | 1433 | 1 | 5 | 2 | |
| sh | 0 | 0 | 0 | 322 | 7 | 0 | |
| s | 2 | 0 | 8 | 7 | 1466 | 138 | |
| z | 3 | 1 | 0 | 0 | 120 | 311 | |
| most frequency misrecognized consonants |

| Models | canonicals | mispronunciations | Recall | Precision | F-measure(%) | PER(%) | ||||
| True Accept | False Rejection | False Accept | True Rejection | |||||||
|
|
|||||||||
| CNN-RNN-CTC[10] | 78.53%(20194) | 21.47%(5520) | 25.22%(1082) | 64.57%(2072) | 35.43%(1137) | 74.78% | 36.76% | 49.29 | 27.75 | |
| +Character-Attention | 87.30%(22449) | 12.70%(3265) | 37.68%(1617) | 69.52%(1859) | 30.48%(815) | 62.32% | 45.02% | 52.28 | 20.66 | |
| +Phoneme-Attention | 93.06%(23929) | 6.94%(1785) | 49.59%(2128) | 73.78%(1595) | 26.26%(568) | 50.41% | 54.79% | 52.51 | 16.06 | |
| Performance of different model architecture | ||||||||||
| +PS(10%) | 92.72%(23841) | 7.28%(1873) | 45.07%(1934) | 75.31%(1775) | 24.69%(582) | 54.93% | 55.72% | 55.32 | 15.52 | |
| +PS(15%) | 92.23%(23715) | 7.77%(1999) | 43.18%(1853) | 74.65%(1820) | 25.35%(618) | 56.82% | 54.95% | 55.87 | 16.13 | |
| +PS(20%) | 92.60%(23810) | 7.4%(1904) | 44.63%(1915) | 73.40%(1744) | 26.60%(632) | 55.37% | 55.51% | 55.44 | 15.96 | |
| +VC(10%) | 92.65%(23825) | 7.35%(1889) | 43.88%(1883) | 74.96%(1805) | 25.04%(603) | 56.12% | 56.04% | 56.08 | 15.58 | |
| +VC(15%) | 92.27%(23726) | 7.73%(1988) | 43.32%(1859) | 74.63%(1815) | 25.37%(617) | 56.68% | 55.02% | 55.84 | 15.71 | |
| +VC(20%) | 91.77%(23650) | 8.03%(2064) | 43.59%(1870) | 73.90%(1789) | 30.42%(812) | 56.42% | 53.98% | 55.17 | 16.33 | |
| +CP(10%) | 92.83%(23870) | 7.17%(1844) | 45.14%(1937) | 75.45%(1776) | 24.55%(578) | 54.86% | 56.07% | 55.46 | 15.48 | |
| +CP(15%) | 92.62%(23817) | 7.38%(1897) | 46.10%(1978) | 73.07%(1690) | 26.93%(623) | 53.90% | 54.94% | 54.42 | 15.91 | |
| +CP(20%) | 91.95%(23644) | 8.05%(2070) | 42.32%(1816) | 74.10%(1834) | 25.90%(641) | 57.68% | 54.46% | 56.02 | 16.20 | |
| Performance of different data augmentation with phoneme-attention model |

Note: The CNN-RNN-CTC is our baseline model and audio encoder. “Character-Attention” denotes that we use character sequences as sentence input on attention model our proposed. “Phoneme-Attention” denotes that we use phoneme sequences as sentence input. To modify the sentence input sequence, three data augmentation methods was used to here, ”PS” means phoneme set based, ”VC” means the vowels and consonants set base, ”CP” means the confusing pairs based, we tried to modify separately 10%,15%,20% of the phonemes in the input phoneme sequence

The confusion matrices of most frequently misrecognized vowels and consonants by CNN-RNN-CTC and our best MD&D model(Attention-VC10%) are shown in Table 2. Our model significantly obtains better results than CNN-RNN-CTC for all confusable consonants and vowels. In addition, we find that this tendency to confusable after the introducing the prior text still remains(i.e. //-//, //-//, etc.), and there is no obvious damage to the acoustic information.

### 3.3 Results of MD&D

Following previous work [10, 15], the hierarchical evaluation structure are used to measure the MD&D system performance. The true accept(TA) and true rejection(TR) indicates predict correctly, while the false accept(FA) and false rejection(FR) indicates predict incorrectly. The true rejection(TR) is divided into correct diagnosis and diagnosis error. Three main evaluation indicators precision, recall and f-measure are calculated by the following formula:

| (11) |

| (12) |

| (13) |

For the model architecture, it can be found that our proposed phoneme-attention model improves the F-measure from 49.29% to 52.51% when comparing to CNN-RNN-CTC baseline, which is slightly better than character-attention model as well. We also notice a significant decrease in recall due to the large number of phonemes classified as input phonemes after the inclusion of prior text constraints, which is reflected in the TA reaching 93.06%. In the actual application of the MD&D, in order for learners to perceive the system as reliable, the TA needs to maintain a very high performance, which means learner’s many correct pronunciation in the canonicals detected by the machine are correct.

For the data augmentation, it can be seen that we achieved the best result in the experiment 56.08% after using VC(10%) method, which is an absolute improvement of 6.79% compared with the baseline. In addition, all data augmentation experiments maintain high TA and f-measure performance.

Through the analysis of the correct diagnosis and the incorrect diagnosis, we find that the main reason for poor recall performance are the decrease in the number of diagnosed error. What’s more, we show a comparison of the results in detecting the type of mispronunciation as in Table 4, the error diagnosis of the three types is greatly reduced after adding a priori text information, especially the two types of substitution and deletion.

| Mispronunciations | |||||||||||
| Error types | Substitution | Insertion | Deletion | ||||||||
| Error nums |
|
|
|
||||||||
| Correct Diag. | CNN-RNN-CTC |
|
|
|
|||||||
|
|
|
|
||||||||
| Diag. Error | CNN-RNN-CTC |
|
|
|
|||||||
|
|
|
|

## 4 Conclusions

In this paper, we present a fully text-dependent end to end model for mispronunciation detection and diagnosis (MD&D). The model does not need to use any forced alignment information and only requires audio and prior phoneme sequences to be available for training, which makes the model easier to train and a wider range of applications. Furthermore, three easy data augmentation techniques are used to make up for the shortcomings of our proposed model. By doing experiments on public corpora, we demonstrate that our proposed method shows more effective performance than the baseline system on the TA and F-measure. In the future, considering that we don’t have a lot of training data, the pre-trained model and unsupervised learning will be investigated to improve the performance with our model.

## References

-
[1]
A. Neri, O. Mich, M. Gerosa, and D. Giuliani, “The effectiveness of computer
assisted pronunciation training for foreign language learning by children,”
*Computer Assisted Language Learning*, vol. 21, no. 5, pp. 393–408, 2008. -
[2]
Y. Xie, X. Feng, B. Li, J. Zhang, and Y. Jin, “A Mandarin L2 Learning APP
with Mispronunciation Detection and Feedback,” in
*Proc. Interspeech 2020*, 2020, pp. 1015–1016. -
[3]
S. M. Witt and S. Young, “Phone-level pronunciation scoring and assessment for
interactive language learning,”
*Speech Commun.*, vol. 30, pp. 95–108, 2000. -
[4]
W. Hu, Y. Qian, F. K. Soong, and Y. Wang, “Improved mispronunciation detection
with deep neural network trained acoustic models and transfer learning based
logistic regression classifiers,”
*Speech Communication*, vol. 67, pp. 154–166, 2015. -
[5]
J. Shi, N. Huo, and Q. Jin, “Context-aware goodness of pronunciation for
computer-assisted pronunciation training,” in
*INTERSPEECH*, 2020. -
[6]
A. M. Harrison, W. Lo, X. Qian, and H. Meng, “Implementation of an extended
recognition network for mispronunciation detection and diagnosis in
computer-assisted pronunciation training,” in
*SLaTE*, 2009. - [7] L. Wai Kit, S. Zhang, and H. Meng, “Automatic derivation of phonological rules for mispronunciation detection in a computer-assisted pronunciation training system.” 01 2010, pp. 765–768.
-
[8]
X. Qian, F. K. Soong, and H. Meng, “Discriminative acoustic model for
improving mispronunciation detection and diagnosis in computer-aided
pronunciation training (capt),” in
*Eleventh Annual Conference of the International Speech Communication Association*, 2010. -
[9]
H. Meng, Y. Y. Lo, L. Wang, and W. Y. Lau, “Deriving salient learners’
mispronunciations from cross-language phonological comparisons,” in
*2007 IEEE Workshop on Automatic Speech Recognition & Understanding (ASRU)*. IEEE, 2007, pp. 437–442. -
[10]
W.-K. Leung, X. Liu, and H. Meng, “Cnn-rnn-ctc based end-to-end
mispronunciation detection and diagnosis,” in
*ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*. IEEE, 2019, pp. 8132–8136. -
[11]
A. Graves and N. Jaitly, “Towards end-to-end speech recognition with recurrent
neural networks,” in
*International conference on machine learning*. PMLR, 2014, pp. 1764–1772. -
[12]
D. Amodei, S. Ananthanarayanan, R. Anubhai, J. Bai, E. Battenberg, C. Case,
J. Casper, B. Catanzaro, Q. Cheng, G. Chen
*et al.*, “Deep speech 2: End-to-end speech recognition in english and mandarin,” in*International conference on machine learning*. PMLR, 2016, pp. 173–182. -
[13]
Y. Feng, G. Fu, Q. Chen, and K. Chen, “Sed-mdd: Towards sentence dependent
end-to-end mispronunciation detection and diagnosis,” in
*ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*. IEEE, 2020, pp. 3492–3496. -
[14]
B.-C. Yan, M.-C. Wu, H.-T. Hung, and B. Chen, “An end-to-end mispronunciation
detection system for l2 english speech leveraging novel anti-phone
modeling,”
*arXiv preprint arXiv:2005.11950*, 2020. -
[15]
K. Li, X. Qian, and H. Meng, “Mispronunciation detection and diagnosis in l2
english speech using multidistribution deep neural networks,”
*IEEE/ACM Transactions on Audio, Speech, and Language Processing*, vol. 25, no. 1, pp. 193–207, 2016. -
[16]
A. Graves, S. Fernández, F. Gomez, and J. Schmidhuber, “Connectionist
temporal classification: labelling unsegmented sequence data with recurrent
neural networks,” in
*Proceedings of the 23rd international conference on Machine learning*, 2006, pp. 369–376. -
[17]
G. Zhao, S. Sonsaat, A. O. Silpachai, I. Lucic, E. Chukharev-Hudilainen,
J. Levis, and R. Gutierrez-Osuna, “L2-arctic: A non-native english speech
corpus,”
*Perception Sensing Instrumentation Lab*, 2018. -
[18]
J. S. Garofolo, L. F. Lamel, W. M. Fisher, J. G. Fiscus, and D. S. Pallett,
“Darpa timit acoustic-phonetic continous speech corpus cd-rom. nist speech
disc 1-1.1,”
*NASA STI/Recon technical report n*, vol. 93, p. 27403, 1993. -
[19]
M. McAuliffe, M. Socolof, S. Mihuc, M. Wagner, and M. Sonderegger, “Montreal
forced aligner: Trainable text-speech alignment using kaldi.” in
*Interspeech*, vol. 2017, 2017, pp. 498–502.
