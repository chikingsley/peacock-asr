# Automatic Pronunciation Assessment using Self-Supervised Speech Representation Learning

###### Abstract

Self-supervised learning (SSL) approaches such as wav2vec 2.0 and HuBERT models have shown promising results in various downstream tasks in the speech community. In particular, speech representations learned by SSL models have been shown to be effective for encoding various speech-related characteristics. In this context, we propose a novel automatic pronunciation assessment method based on SSL models. First, the proposed method fine-tunes the pre-trained SSL models with connectionist temporal classification to adapt the English pronunciation of English-as-a-second-language (ESL) learners in a data environment. Then, the layer-wise contextual representations are extracted from all across the transformer layers of the SSL models. Finally, the automatic pronunciation score is estimated using bidirectional long short-term memory with the layer-wise contextual representations and the corresponding text. We show that the proposed SSL model-based methods outperform the baselines, in terms of the Pearson correlation coefficient, on datasets of Korean ESL learner children and Speechocean762. Furthermore, we analyze how different representations of transformer layers in the SSL model affect the performance of the pronunciation assessment task.

Index Terms: automatic pronunciation assessment, pronunciation scoring, self-supervised speech representation learning, wav2vec 2.0, HuBERT

## 1 Introduction

The need for English-as-a-second-language (ESL) learners to improve their English pronunciation is increasing owing to globalization. The computer-assisted pronunciation training (CAPT) system, which can conduct assessments and provide detailed feedback on pronunciation proficiency, is thus attracting attention as an ESL learning service and platform [1, 2]. There are two technical approaches to the CAPT system: mispronunciation detection and diagnosis (MDD) [3, 4, 5, 6, 7, 8, 9, 10, 11, 12] and automatic pronunciation assessment [6, 13, 14, 15, 16, 17, 10]. MDD is a task of detecting pronunciation errors by calculating multiple measures using estimated and canonical phones from an automatic speech recognizer. In this study, we focus on the automatic pronunciation-scoring problem, which estimates a pronunciation score based on pronunciation-relevant characteristics from spoken English data to achieve a high correlation with the scores annotated by experts.

Most previous studies that measured speech pronunciation used automatic speech recognition (ASR) systems and acoustic features to estimate the pronunciation score. With the ASR system, SpeechRater [3] is an effective method for predicting pronunciation using several handcrafted speech features. Another common feature used for pronunciation assessment is the Goodness of Pronunciation (GOP) measure [14, 18] and its variations, while using the ASR system and forced alignment. In addition, many studies have utilized acoustic features, including prosody, intensity, rhythm, and cepstrum, for pronunciation assessments.

Several recent studies on automatic pronunciation assessment were based on deep neural networks [6, 15, 16, 17, 19]. Bidirectional long short-term memory (BLSTM)-based time-sequence features and time-aggregated features were optimized using a multilayer perceptron (MLP) for automatic scoring [15]. BLSTM, with attention structure, simultaneously estimates the pronunciation score with both acoustic and linguistic cues [16]. In [17], phone distance metric features extracted from a Siamese network based on BLSTM were used for end-to-end training using an attention mechanism. In [19], deep features from the acoustic model and the self-attention mechanism for a scoring module were utilized directly. According to end-to-end (E2E) ASR advances, E2E-based research has been introduced to pronunciation assessment with competitive results. An E2E model consisting of two encoders for text and audio followed the attention mechanism to combine the two features [6].

Recently, self-supervised learning (SSL) [20, 21, 22, 23, 24] has shown promising results in the downstream tasks of speech processing applications, such as ASR, phoneme recognition, emotion recognition, speaker diarization, and speaker verification [22, 24, 25]. These studies applied contextual representations using pre-trained models or a fine-tuned model for related tasks. In particular, speech representations learned across the transformer layers of SSL models have been shown to be highly effective in learning high-level representations by encoding various speech-related properties and linguistic information contents [26, 27, 28]. However, for the CAPT system, researchers have investigated the capability of the wav2vec 2.0 model only in MDD tasks [10, 11, 12].

In this paper, we proposed an SSL-model-based automatic pronunciation assessment method. We hypothesized that SSL would be effective for learning pronunciation-relevant latent representations. To test this hypothesis, we propose a novel pronunciation assessment method that incorporates pre-trained SSL models to fine-tune and adapt them to the pronunciation of non-native English language learners. Subsequently, the weighted average of all context representations was extracted across the transformer layers of the SSL models. Finally, BLSTM was constructed on top of the models incorporating script information to estimate the utterance-level pronunciation score. We observed a significant improvement from fined-tuned SSL models over other baselines in terms of the Pearson correlation coefficient (PCC) with two datasets of Korean children: ESL learners (KESL) and Chinese speakers (Speechocean762). To the best of our knowledge, this is the first study on pronunciation scoring using the SSL method.

## 2 Method

An overview of the proposed method is shown in Figure 1. The method includes three stages: (i) fine-tuning the pre-trained contextual transformer to adapt to a non-native spoken data environment with connectionist temporal classification (CTC) loss [29], (ii) extracting layer-wise contextual representations from the contextual transformer, and (iii) estimating the utterance-level pronunciation score using the neural network constructed on top of the contextual transformer. Our target SSL-based architecture is the wav2vec 2.0 and HuBERT models.

### 2.1 Pre-trained Models

Wav2vec 2.0 [22], which has been well verified [25] is an effective structure for learning representations from speech data using SSL models. The wav2vec 2.0 model consists of a convolutional encoder, context encoder, and quantization module. The convolutional encoder contains time-stacking convolutional layer maps that transform the raw waveform into latent speech representations. The context encoder, consisting of multiple transformer blocks, takes a latent speech representation and outputs the context representation. The latent speech representations are quantized into an embedding with a fixed number of latent representations stored in a codebook for the prediction task. The contrastive score between context representation and vector-quantized embedding is maximized during network training [30].

HuBERT [24] has been investigated for outperforming wav2vec 2.0 for multiple downstream recognition and generation tasks. It uses the same structure as wav2vec 2.0, which consists of a convolutional encoder followed by a transformer context encoder. In the training phase, HuBERT builds pseudo-labels through iterative refinement of clustering with mel-frequency cepstrum coefficients (MFCC) and hidden units of transformers, while training wav2vec 2.0 through a quantization module using Gumbel-softmax [22]. With the discovered hidden units, the model was trained using cross-entropy loss over the masked regions only.

### 2.2 Fine-tuning

Fine-tuning in SSL has the advantage of adapting to the characteristics of a small dataset. Following the wav2vec 2.0 and HuBERT fine-tuning procedure [22, 24], pre-trained models are fine-tuned with the given non-native spoken training data, which are optimized with the CTC loss [29]. We assumed that it would be more beneficial for the model to adapt to the non-native spoken dataset environment with the ASR fine-tuning process for pronunciation scoring. The final hidden state of the transformers is convoluted using a 1D convolutional layer and then fed into a softmax layer, as shown in Figure 1 (a). All model weights, except for the convolutional layer, were fine-tuned.

### 2.3 Pronunciation Representation Modeling

Recent studies [27, 28] have shown that various properties including acoustic and linguistic information tend to be encoded in different layers of transformers. We hypothesized that the transformers in the SSL models would learn pronunciation-related features when the model is optimized using a pronunciation dataset. To capture all encoded information related to pronunciation effectively, we extract layer-wise context representations by averaging the hidden states of the context representations of all transformer layers, as illustrated in Figure 1 (b). In addition, we investigated the effectiveness of utilizing the convolution layer output and context representation outputs of different transformer layers in the SSL models on the pronunciation assessment task.

### 2.4 Scoring Module for Pronunciation Assessment

To achieve the final goal of predicting pronunciation scores, this study was approached as a regression problem by optimizing the mean square error loss between the predicted and human-annotated scores. As shown in Figure 1 (c), we adopted the BLSTM model to reflect the dynamics of the acoustic and linguistic representations for the utterance-level score, as in [15][16]. First, we generated two unidirectional encoded vectors in the forward and backward directions from BLSTM with the audio context representations of the transformer layers of the SSL models. Then, we obtained the utterance-level full audio context vector by concatenating the two unidirectional encoded vectors, followed by the linear layer. To encode the dynamics of linguistic information for the spoken utterance, characters of the script of the spoken utterance were converted into embedded vectors by the embedding layer, and then the embedded vectors were transformed to two BLSTM-based unidirectional encoded vectors followed by a linear layer to obtain the bidirectional full linguistic context vector. The final score was obtained by applying global average pooling (GAP) of audio context vectors and linguistic context vectors over time dimension and script characters, respectively, and then a linear layer.

## 3 EXPERIMENT

### 3.1 Dataset

For the assessment of non-native English spoken pronunciation, two different human-labeled datasets were utilized to demonstrate the effectiveness of our proposed method. The first corpus was an in-house dataset recorded by Korean ESL learner (KESL) children. It consisted of 17800 utterances by 300 Korean speakers with ages ranging from 10 to 12 years. Five native experts annotated the sentence level using five pronunciation continuous measures ranging from 1 to 5, including the holistic impression of pronunciation, segmental accuracy, stress, pauses, and intonation. Utterance-level scores were obtained by averaging the scores of five experts for each label. A holistic score distribution of the dataset is depicted in Figure 2. The second dataset was a public dataset called Speechocean762 [31]. It contained 5000 English sentences recorded by 250 English non-native English speakers, in which the gender and age of the speakers were proportionately balanced. The dataset provides multidimensional scores, such as accuracy, completeness, fluency, and prosody, in terms of word-level, phoneme level, and sentence level. Sentence-level labels were used to evaluate pronunciation. The dataset was divided into a training set and test set at a ratio of 5:5. The sampling rate of all the speech data was 16,000 Hz.

### 3.2 Baselines

For traditional features for pronunciation-scoring tasks, we used time-aggregated features, time-sequence handcrafted acoustic features, GOP, and their combination features, which were used in previous studies [15, 16, 19]. For *GOP*, the frame-level posterior was generated using the DNN-based acoustic model as the likelihood ratio between the forced alignment likelihood and the maximum likelihood obtained from the ASR engine [32]. The frame-level posterior matrix was generated by forwarding propagation on the native acoustic model, and the matrix was used for forced alignment and computing to obtain the GOP-based features, whose definitions can be found in [32]. The ASR system used in this study was an in-house DNN-HMM hybrid ASR system trained on 4000 h of native and non-native spoken English data. In addition, we utilized the time-aggregated features (*AggFeat*) used in previous works [15, 16] based on SpeechRater [3] related to several aspects of the speech construct, including fluency, rhythm, intonation, stress, pronunciation, grammar, and vocabulary use. For the time-sequence features indicated as *SeqFeat*, the mean and standard deviation of 23 low-level descriptors using the eGeMAPS set [33], including the MFCC, loudness, pitch, jitter, and shimmer over the segment implemented in OpenSmile [34] were extracted. Feature-wise zero-mean and unit-variance normalizations were used for all features. For the scoring module of the baselines, we averaged the segment features along the time dimension using GAP and then concatenated the baseline features, followed by two linear layers, 256 and 1 unit, respectively, to predict the pronunciation score.

### 3.3 Experimental Setup

Experiments on SSL models used the pre-trained wav2vec2-large-960h [22], wav2vec2-base-960h [22], wav2vec2-large-robust [22], HuBERT-base-ls960h [24], and HuBERT-large-ls960h [24] from Fairseq [35]. For the fine-tuning stage of all models, we finetuned 150k steps with eight batch sizes and used an Adam optimizer, where the learning rate was warmed up from 1e-4 with a 1k warm-up step using the training implementation on *HuggingFace* [36]. The model with the lowest word error rate score was used for the development set. To train the pre-trained model with CTC, the vocabulary was built from all distinct letters of the training and test data, including *CTC blank*, *apostrophe*, and *space* tokens. A 10-fold cross-validation method based on the speaker was used. For all scoring modules, we experimented with a single-layer BLSTM with 128 hidden and 64 embedding dimensions. The model was trained using the Adam optimizer with a learning rate of 1e-4 and early stopping with a patience of 3 on the validation loss. To evaluate the models, we compared the PCC of the predicted and human-annotated scores.

### 3.4 Results

| Method | KESL | Speechocean762 | ||||
| Holistic | Fluency | Prosodic | ||||
| GOP | 0.63 | 0.65 | 0.64 | |||
| Agg + Seq | 0.55 | 0.51 | 0.59 | |||
| Agg + Seq + GOP | 0.64 | 0.67 | 0.66 | |||
| pre-trained | ||||||
| wav2vec2 Base | 0.65 | 0.72 | 0.72 | |||
| wav2vec2 Large | 0.71 | 0.72 | 0.72 | |||
| wav2vec2 Robust | 0.76 | 0.73 | 0.73 | |||
| HuBERT Base | 0.69 | 0.72 | 0.71 | |||
| HuBERT Large | 0.75 | 0.75 | 0.74 | |||
| Finetuned | ||||||
| wav2vec2 Base | 0.68 | 0.73 | 0.72 | |||
| wav2vec2 Large | 0.78 | 0.73 | 0.72 | |||
| wav2vec2 Robust | 0.79 | 0.75 | 0.74 | |||
| HuBERT Base | 0.75 | 0.74 | 0.73 | |||
| HuBERT Large | 0.82 | 0.78 | 0.77 | |||

To evaluate the performance of the SSL-based methods, we explored improvements to the SSL-based pronunciation assessment models with two corpora: the KESL and Speechocean762 datasets. Table 1 compares the performance of SSL-based pre-trained models, fine-tuned models of wav2vec 2.0 (wav2vec2), and HuBERT with traditional baselines. All fine-tuned models were fine-tuned on two non-native datasets, KESL and Speechocean72. All SSL models used layer-wise context representations. First, both HuBERT and wav2vec2 exhibit performance improvements over the existing baseline model by using only the layer-wise context representations of the pre-trained model, as shown in Table 1. In particular, the wav2vec2 robust pre-trained model [23] showed high performance, and it can be inferred that wav2vec2 robust is a pre-trained model with speech in real scenarios, making the representation more robust for non-native speaker conditions. Second, we observed that the fine-tuned models consistently outperformed the pre-trained models in both the wav2vec2 and HuBERT models. This shows that the fine-tuned model for ASR gains is beneficial for estimating pronunciation scores compared to the pre-trained model, which did not adapt to the non-native spoken data environment. Finally, we can see that the fine-tuned HuBERT Large model achieved the best results for PCCs of holistic, fluency, and prosodic measures on the KESL and Speechocean762 datasets. We can see that the holistic, fluency, and prosodic aspects of the HuBERT Large model outperform those of the wav2vec2 robust model for the PCCs by , , and , respectively.

### 3.5 Ablation Studies

In this section, we focus on our experiment using the HuBERT model as an example to study the factors affecting the performance of the SSL model in automatic pronunciation assessment.

#### 3.5.1 Effect of contextual representations of the HuBERT model

First, we explore the effectiveness of using representations from different transformer layers of the pre-trained and fine-tuned HuBERT large models for the pronunciation assessment task. As shown in Figure 3, we can observe that there is a performance difference among the different layers, and the tendencies of the PCC according to different layers are similar among the prosodic and SpeechOcean762 datasets. Figure 3 shows that the performance of both the fine-tuned and pre-trained models in the upper layer parts (from the 11th to 22nd layers) performed significantly better than those in the lower layer parts (from 1st to 9th layer). This indicates that the HuBERT Large model can better learn pronunciation-related information in the upper parts. Table 2 shows that transformer-based contextual representations perform better than convolutional encoder representations. We can observe that the hidden layer from the 20th layer obtains the best PCC among the transformer layers. Finally, the layer-wise context vector, which is the average of the hidden states of all layers, achieved the best PCC score. This indicates that it is beneficial for the pronunciation assessment task to use the information of all layers, including the various properties of speech.

| Feature | KESL | Speechocean762 | |
| Holistic | Fluency | Prosodic | |
| Local | 0.56 | 0.60 | 0.62 |
| Layer 20 | 0.81 | 0.76 | 0.76 |
| All Layers (Proposed) | 0.82 | 0.78 | 0.77 |

#### 3.5.2 Comparison of different regression methods

The last ablation involved a pronunciation-scoring module. We tested commonly used regression methods for pronunciation task, including linear regression (LR) and the MLP on top of the transformer of HuBERT to aggregate an utterance-level signal vector. The PCCs of holistic, fluency, and prosodic of the BLSTM scoring module outperformed all the baseline performances on two corpora, KESL and Speechocean762, by 0.02, 0.05, and 0.03, respectively. The results demonstrate that the BLSTM-based scoring module performs better than simple regression methods with average pooling method.

| Scoring Module | KESL | Speechocean762 | |
| Holistic | Fluency | Prosodic | |
| LR | 0.78 | 0.72 | 0.72 |
| MLP | 0.80 | 0.73 | 0.74 |
| BLSTM (Proposed) | 0.82 | 0.78 | 0.77 |

## 4 CONCLUSION

In this paper, we presented an automatic pronunciation assessment method utilizing effective contextual representations of SSL models such as wav2vec 2.0 and HuBERT. We showed that pre-trained SSL models are beneficial for estimating pronunciation-scoring tasks. In addition, fine-tuning SSL models with CTC is beneficial for improving pronunciation-scoring performance. Finally, we demonstrated the effectiveness of the layer-wise representation of transformer layers from the perspective of a pronunciation assessment task. The experiments were conducted with two datasets, KESL and Speechocean762, in terms of PCC. In future research, we plan to study the effectiveness of fine-tuning SSL models in terms of pronunciation score.

## References

-
[1]
P. M. Rogerson-Revell, “Computer-assisted pronunciation training (capt):
Current issues and future directions,”
*RELC Journal*, vol. 52, no. 1, pp. 189–205, 2021. - [2] K. Kyriakopoulos, “Deep learning for automatic assessment and feedback of spoken english,” Ph.D. dissertation, Queens’ College, 2021.
-
[3]
K. Zechner, D. Higgins, X. Xi, and D. M. Williamson, “Automatic scoring of
non-native spontaneous speech in tests of spoken english,”
*Speech Communication*, vol. 51, 2009. -
[4]
W.-K. Leung, X. Liu, and H. Meng, “Cnn-rnn-ctc based end-to-end
mispronunciation detection and diagnosis,” in
*IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2019. -
[5]
B.-C. Yan, M.-C. Wu, H.-T. Hung, and B. Chen, “An end-to-end mispronunciation
detection system for l2 english speech leveraging novel anti-phone
modeling,”
*INTERSPEECH*, 2020. -
[6]
B. Lin and L. Wang, “Attention-based multi-encoder automatic pronunciation
assessment,” in
*IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2021. -
[7]
T.-H. Lo, S.-Y. Weng, H.-J. Chang, and B. Chen, “An effective end-to-end
modeling approach for mispronunciation detection,”
*INTERSPEECH*, 2020. -
[8]
Y. Feng, G. Fu, Q. Chen, and K. Chen, “Sed-mdd: Towards sentence dependent
end-to-end mispronunciation detection and diagnosis,” in
*ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2020. -
[9]
T.-H. Lo, Y.-T. Sung, and B. Chen, “Improving end-to-end modeling for
mispronunciation detection with effective augmentation mechanisms,” in
*IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2022. -
[10]
L. Peng, K. Fu, B. Lin, D. Ke, and J. Zhan, “A Study on Fine-Tuning
wav2vec2.0 Model for the Task of Mispronunciation Detection and Diagnosis,”
in
*INTERSPEECH*, 2021. -
[11]
M. Wu, K. Li, W.-K. Leung, and H. Meng, “Transformer Based End-to-End
Mispronunciation Detection and Diagnosis,” in
*INTERSPEECH*, 2021. -
[12]
X. Xu, Y. Kang, S. Cao, B. Lin, and L. Ma, “Explore wav2vec 2.0 for
Mispronunciation Detection,” in
*INTERSPEECH*, 2021. -
[13]
H. Li, S. Huang, S. Wang, and B. Xu, “Context-dependent duration modeling with
backoff strategy and look-up tables for pronunciation assessment and
mispronunciation detection,” in
*INTERSPEECH*, 2011. -
[14]
S. M. Witt and S. J. Young, “Phone-level pronunciation scoring and assessment
for interactive language learning,”
*Speech communication*, vol. 30, 2000. -
[15]
Z. Yu, V. Ramanarayanan, D. Suendermann-Oeft, X. Wang, K. Zechner, L. Chen,
J. Tao, A. Ivanou, and Y. Qian, “Using bidirectional lstm recurrent neural
networks to learn high-level abstractions of sequential features for
automated scoring of non-native spontaneous speech,” in
*ASRU*, 2015. -
[16]
L. Chen, J. Tao, S. Ghaffarzadegan, and Y. Qian, “End-to-end neural network
based automated speech scoring,” in
*IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2018. -
[17]
K. Kyriakopoulos, K. Knill, and M. Gales, “A deep learning approach to
assessing non-native pronunciation of english using phone distances,” in
*INTERSPEECH*, 2013. -
[18]
S. Sudhakara, M. K. Ramanathi, C. Yarra, and P. K. Ghosh, “An improved
goodness of pronunciation (gop) measure for pronunciation evaluation with
dnn-hmm system considering hmm transition probabilities.” in
*INTERSPEECH*, 2019. -
[19]
B. Lin and L. Wang, “Deep Feature Transfer Learning for Automatic
Pronunciation Assessment,” in
*INTERSPEECH*, 2021. -
[20]
A. T. Liu, S.-w. Yang, P.-H. Chi, P.-c. Hsu, and H.-y. Lee, “Mockingjay:
Unsupervised speech representation learning with deep bidirectional
transformer encoders,” in
*IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2020. -
[21]
A. T. Liu, S.-W. Li, and H.-y. Lee, “Tera: Self-supervised learning of
transformer encoder representation for speech,”
*IEEE/ACM Transactions on Audio, Speech, and Language Processing*, vol. 29, pp. 2351–2366, 2021. -
[22]
A. Baevski, H. Zhou, A. Mohamed, and M. Auli, “wav2vec 2.0: A framework for
self-supervised learning of speech representations,” in
*NeurIPS*, 2020. -
[23]
W.-N. Hsu, A. Sriram, A. Baevski, T. Likhomanenko, Q. Xu, V. Pratap, J. Kahn,
A. Lee, R. Collobert, G. Synnaeve, and M. Auli, “Robust wav2vec 2.0:
Analyzing Domain Shift in Self-Supervised Pre-Training,” in
*INTERSPEECH*, 2021. -
[24]
W.-N. Hsu, B. Bolte, Y.-H. H. Tsai, K. Lakhotia, R. Salakhutdinov, and
A. Mohamed, “Hubert: Self-supervised speech representation learning by
masked prediction of hidden units,”
*IEEE/ACM Transactions on Audio, Speech, and Language Processing*, vol. 29, pp. 3451–3460, 2021. -
[25]
S.-w. Yang, P.-H. Chi, Y.-S. Chuang, C.-I. J. Lai, K. Lakhotia, Y. Y. Lin,
A. T. Liu, J. Shi, X. Chang, G.-T. Lin
*et al.*, “Superb: Speech processing universal performance benchmark,” in*INTERSPEECH*, 2021. -
[26]
D. Ma, N. Ryant, and M. Liberman, “Probing acoustic representations for
phonetic properties,” in
*IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2021. -
[27]
A. Pasad, J.-C. Chou, and K. Livescu, “Layer-wise analysis of a
self-supervised speech representation model,”
*ASRU*, 2021. -
[28]
J. Shah, Y. K. Singla, C. Chen, and R. R. Shah, “What all do audio transformer
models hear? probing acoustic representations for language delivery and its
structure,”
*arXiv preprint arXiv:2101.00387*, 2021. -
[29]
A. Graves, S. Fernández, F. Gomez, and J. Schmidhuber, “Connectionist
temporal classification: labelling unsegmented sequence data with recurrent
neural networks,” in
*ICML*, 2006. -
[30]
A. v. d. Oord, Y. Li, and O. Vinyals, “Representation learning with
contrastive predictive coding,” in
*INTERSPEECH*, 2019. -
[31]
J. Zhang, Z. Zhang, Y. Wang, Z. Yan, Q. Song, Y. Huang, K. Li, D. Povey, and
Y. Wang, “speechocean762: An open-source non-native english speech corpus
for pronunciation assessment,” in
*INTERSPEECH*, 2021. -
[32]
W. Hu, Y. Qian, F. K. Soong, and Y. Wang, “Improved mispronunciation detection
with deep neural network trained acoustic models and transfer learning based
logistic regression classifiers,”
*Speech Communication*, vol. 67, 2015. -
[33]
F. Eyben, K. R. Scherer, B. W. Schuller, J. Sundberg, E. André, C. Busso,
L. Y. Devillers, J. Epps, P. Laukka, S. S. Narayanan
*et al.*, “The geneva minimalistic acoustic parameter set (gemaps) for voice research and affective computing,”*IEEE transactions on affective computing*, vol. 7, 2015. -
[34]
F. Eyben, M. Wöllmer, and B. Schuller, “Opensmile: the munich versatile
and fast open-source audio feature extractor,” in
*Proceedings of the 18th ACM international conference on Multimedia*, 2010, pp. 1459–1462. -
[35]
M. Ott, S. Edunov, A. Baevski, A. Fan, S. Gross, N. Ng, D. Grangier, and
M. Auli, “fairseq: A fast, extensible toolkit for sequence modeling,” in
*Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics (Demonstrations)*. ACL, 2019, pp. 48–53. -
[36]
T. Wolf, L. Debut, V. Sanh, J. Chaumond, C. Delangue, A. Moi, P. Cistac,
T. Rault, R. Louf, M. Funtowicz, J. Davison, S. Shleifer, P. von Platen,
C. Ma, Y. Jernite, J. Plu, C. Xu, T. L. Scao, S. Gugger, M. Drame, Q. Lhoest,
and A. M. Rush, “Transformers: State-of-the-art natural language
processing,” in
*Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*. ACL, 2020.
