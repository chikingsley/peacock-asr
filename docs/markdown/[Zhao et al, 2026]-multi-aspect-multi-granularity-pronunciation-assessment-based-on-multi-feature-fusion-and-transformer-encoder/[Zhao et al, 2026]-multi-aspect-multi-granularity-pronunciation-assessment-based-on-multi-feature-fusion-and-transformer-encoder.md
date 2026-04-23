---
title: "A Multi-aspect Multi-granularity Pronunciation Assessment Method Based on Multi-feature Fusion and Transformer Encoder"
authors:
  - "Yawei Zhao"
  - "Aishan Wumaier"
  - "Xueliang Guo"
  - "Yaxuan Lv"
citation_author: "Zhao et al"
year: 2026
doi: "10.1007/978-981-95-6960-1_37"
pages: 13
source_pdf: "../papers/capt-systems/[Zhao et al, 2026]-multi-aspect-multi-granularity-pronunciation-assessment-based-on-multi-feature-fusion-and-transformer-encoder/978-981-95-6960-1_37.pdf"
source_html: null
extraction_method: "Extracted from PDF"
extracted_at: "2026-04-04T00:00:00-07:00"
llm_friendly: true
---

## Abstract

The Computer-Assisted Pronunciation Training (CAPT) system provides non-native (L2) speakers with an efficient learning path. As a core component of CAPT, Automatic Pronunciation Assessment (APA) quantifies learners' pronunciation abilities and offers precise feedback. However, current APA methods fail to model hierarchical dependencies across different granularities and lack robust speech representations. To address these issues, we propose the Cwacformer, a hierarchical multi-granularity pronunciation assessment model with a convolution-augmented Transformer encoder. Key innovations include: (1) word timing features for capturing rhythm and prosodic patterns, (2) a dual-branch architecture combining global Transformer and local CNN pathways with learnable fusion, (3) multi-scale convolution for enhanced word-level modeling, (4) adversarial training with Generative Adversarial Networks (GANs) to improve robustness, and (5) cross-attention mechanisms for explicit hierarchical relationship modeling. Extensive experiments on the Speechocean762 dataset show significant improvements across all granularities, with phoneme-level mean squared error (MSE) reduced to 0.073 and word-level stress recognition PCC reached 0.483, a 51% relative improvement over the baseline. Cwacformer achieves superior performance in most evaluation metrics, particularly excelling in stress detection and fluency assessment, while maintaining robustness across different speakers and acoustic conditions.

## 1 Introduction

Improving pronunciation is a key challenge for non-native (L2) learners, and Computer-Assisted Pronunciation Training (CAPT) systems aim to provide effective solutions for this task. Both Mispronunciation Detection and Diagnosis (MDD) and Automatic Pronunciation Assessment (APA) are core components of CAPT systems. While MDD focuses on error identification, APA provides a comprehensive evaluation of pronunciation quality, covering aspects such as fluency, naturalness, intonation, stress, and rhythm. These systems are essential because they provide learners with personalized, real-time feedback, which is difficult to achieve in traditional classroom settings.

Current APA methods face three critical limitations: (1) insufficient modeling of hierarchical relationships between granularities, which leads to inconsistent cross-level predictions; (2) the lack of suprasegmental timing information, resulting in poor assessment of rhythm and stress; and (3) limited robustness to acoustic variations, causing performance degradation across different speakers and conditions. These limitations arise from fundamental architectural choices that treat granularities independently and ignore the temporal-prosodic structure of speech. Existing approaches either employ parallel architectures that overlook inter-granularity dependencies or use simplistic attention mechanisms that fail to capture complex contextual relationships between phonemes, words, and utterances.

To address these challenges, we propose Cwacformer, a multi-granularity pronunciation assessment model based on multi-feature fusion and Transformer encoders. Our key contributions are as follows: (1) novel word timing features that capture rhythm-aware suprasegmental information, (2) adversarial training mechanisms using GANs to enhance robustness, (3) cross-attention modules that explicitly model hierarchical relationships between granularities, and (4) comprehensive experimental validation demonstrating significant improvements across various evaluation aspects.

## 2 Related Work

### 2.1 Multi-granularity and Multi-aspect Assessment Evolution

Early research on Automatic Pronunciation Assessment (APA) focused on single-granularity evaluations, offering a limited assessment scope. To provide a more comprehensive skill evaluation, multi-granularity methods were introduced to assess different linguistic levels (phoneme, word, and utterance). The GOPT model was one of the pioneers of multi-dimensional prediction but used parallel architectures that overlooked hierarchical relationships. The Hipama model introduced a hierarchical structure with multi-aspect attention but did not adequately model inter-granularity dependencies.

Traditional approaches focused on single aspects, whereas real-world applications require the simultaneous assessment of accuracy, fluency, and intonation. Recent multi-aspect models have utilized multi-task learning frameworks: Ryu et al. integrated APA and MDD with shared representations, and Chao et al. developed HMamba, a model using hierarchical state space models with deXent loss.

### 2.2 Feature Enhancement Approaches

Deep learning approaches have advanced APA through self-supervised learning, ASR transfer learning, and end-to-end architectures. Chao et al. combined GOP features with self-supervised learning and suprasegmental information for multi-level assessment, demonstrating excellent performance but lacking explicit hierarchical modeling and robust representations.

Our work addresses these limitations by incorporating word timing features for suprasegmental modeling, adversarial training for robustness enhancement, and cross-attention mechanisms for explicit hierarchical relationship modeling.

## 3 Method

### 3.1 Overview

We introduce Cwacformer, a hierarchical model designed for multi-granularity pronunciation assessment. At the phoneme level, a lightweight Transformer encoder is combined with an enhanced convolutional branch, which are learnably fused to capture both global context and local phonetic details. The model’s robustness is further improved through an adversarial perturbation module during training. At the word level, multi-scale convolution and a tri-branch Bi-LSTM network are employed to capture local temporal cues and bidirectional context. At the utterance level, we replace word-feature averaging with concatenation and self-attention, and integrate word- and utterance-level information using cross-attention. This architecture effectively integrates multi-granularity modeling, local-global feature fusion, and robustness optimization, resulting in significant improvements in automatic pronunciation assessment (APA). The model structure is shown in Fig. 1.

### 3.2 Model Architecture

**Model Inputs:** For the model input, we follow the approach outlined in the baseline model to ensure a fair comparison. The automatic speech recognition (ASR) acoustic module takes both the audio and its transcription as input and predicts frame-level phonetic posterior probabilities. These probabilities are then transformed into 84-dimensional GOP features.

Traditional GOP-based features provide information at the phoneme segment level but fail to capture suprasegmental information adequately. To address this limitation, we follow the input setup, adding two additional features: duration $x_{Dur}$ and energy $x_{Eng}$. While features like $x_{Dur}$ and $x_{Eng}$ offer a preliminary way to incorporate suprasegmental information, they often represent low-level acoustic properties rather than higher-level linguistic patterns. From a linguistic perspective, pronunciation quality emerges from the interaction between segmental accuracy (phoneme-level) and suprasegmental patterns (rhythm and stress). Therefore, to better bridge this gap, we propose novel word timing features $x_{Wdr}$, which encode duration patterns that are linguistically meaningful for stress and rhythm assessment.

The timing feature vector for the i-th word is constructed as $f_{timing}^{(i)} = [\mu_i, \sigma_i, r_i, \delta_i] \in R^4$. To maintain compatibility with phoneme-level processing, each word’s timing features are propagated to all constituent phonemes: $x_{Wdr}^{(j)} = f_{timing}^{(i)}$ for all $j \in [s_i, e_i]$.

To maintain consistency with the baseline model, we apply one-hot encoding to the standardized phonemes and map them to a specified dimension to obtain the phoneme representation $x_{Phn}$. The input feature structure of the model is as follows:
$x = [x_{Gop}; x_{Dur}; x_{Eng}; x_{Wdr}]$
$x_{input} = Linear(x) + x_{Phn}$

**Adversarial Training Block:** To enhance model robustness against input feature perturbations, we introduce an adversarial perturbation module based on Generative Adversarial Networks (GANs). This module optimizes feature stability and generalization through Generator-Discriminator competition.

**Convolution-Augmented Transformer Encoder:** Accurately capturing both local details and global context is crucial for phoneme-level APA modeling. We employ a dual-branch architecture: a Transformer encoder captures long-range dependencies through self-attention for global semantic representations, while an enhanced gated convolutional block extracts local information and pronunciation anomalies.

**Word-Level Convolution Aggregator:** To address limitation, we propose a multi-scale convolution structure to better capture local and neighboring contexts for word-level scoring. Stress detection requires analyzing variations in phoneme duration, energy patterns, and pitch changes across different context windows. Small kernels (size 1) capture point-wise phoneme characteristics, medium kernels (size 3) model local phoneme transitions, and large kernels (size 5) capture broader prosodic patterns that are essential for stress identification.

Although multi-scale convolution captures local features from different receptive fields, temporal dependencies and long-range contextual relationships are equally crucial for APA. Therefore, we introduce three parallel Bi-LSTM branches to process the multi-scale convolution outputs.

**Utterance-Level Modeling:** The baseline model averages word-level features (accuracy, stress, total score) before utterance evaluation, potentially obscuring feature differences and limiting multi-aspect modeling capabilities. We replace averaging with feature concatenation to preserve distinctiveness and enhance utterance-level expressiveness. To improve global context capture, we employ multi-head attention followed by cross-attention inspired by image classification.

### 3.3 Loss

We employ mean squared error (MSE) as the primary loss function for pronunciation assessment tasks, following standard practice in the field. The total loss combines multi-granularity evaluation losses with adversarial training.

## 4 Experiments

### 4.1 Experiments Setup

We evaluate our method on the Speechocean762 dataset, which contains 5,000 speech samples from 250 Mandarin-speaking L2 English learners (2,500 training/2,500 testing). Each sample is annotated by five professional reviewers across three granularities (utterance, word, phoneme) with scores normalized to [0, 2].

### 4.2 Result and Discussions

We compare our model with the following four existing evaluation models: (1) The GOPT model, which uses a Transformer encoder in a parallel architecture; (2) The hierarchical architecture model HiPAMA, which uses multi-aspect attention; (3) A parallel architecture model named Gradformer, which employs a Transformer encoder-decoder architecture for granularity decoupling; (4) The Bfhaformer model, which employs an LSTM-augmented BranchFormer encoder.

As demonstrated in the experimental results, our model achieves the highest PCC scores in both phoneme and word-level evaluation tasks.

**Phoneme-level Analysis:** Our model achieves an MSE of 0.073 and PCC of 0.687, significantly outperforming Bfhaformer and previous models.

**Word-level Analysis:** Our model substantially outperforms others, particularly in stress evaluation, clearly surpassing Bfhaformer. Multi-scale convolution extracts local temporal features for recognizing energy and duration variations, while Bi-LSTM captures contextual information from preceding and succeeding words, enabling better understanding of stress locations.

**Utterance-level Analysis:** Our model outperforms previous models in all metrics except completeness. Cross-attention fuses word-level Bi-LSTM contextual information with global utterance-level self-attention features, providing comprehensive representations for utterance evaluation.

### 4.3 Ablation Studies

To explore the specific factors that influence Cwacformer performance, we performed five ablation experiments to verify the impact of each component on overall model performance. Each experiment involved removing components from the baseline model and incrementally adding our model components from the previous experiment.

## 5 Conclusion

This paper introduces Cwacformer, a multi-granularity pronunciation assessment model designed to enhance accuracy and robustness for L2 learners through hierarchical structure integration and GANs-based adversarial training. The model leverages Bi-LSTM, multi-scale convolution, and cross-attention mechanisms to improve evaluation across stress, fluency, and accuracy dimensions. Experimental results on the Speechocean762 dataset demonstrate Cwacformer’s superior performance, particularly in word-level stress evaluation. Comparative experiments show consistent outperformance of existing baselines, and ablation studies confirm the effectiveness of each component. Future work will focus on addressing limitations in utterance-level completeness evaluation, potentially incorporating adaptive learning and transfer learning techniques.

---

## References

1. Arias, J.P., Yoma, N.B., Vivanco, H.: Automatic intonation assessment for computer aided language learning. Speech Commun. 52(3), 254–267 (2010)
2. Chao, F.A., Chen, B.: Towards efficient and multifaceted computer-assisted pronunciation training leveraging hierarchical selective state space model and decoupled cross-entropy loss. arXiv preprint arXiv:2502.07575 (2025)
3. Chao, F.A., Lo, T.H., Wu, T.I., Sung, Y.T., Chen, B.: A hierarchical context-aware modeling approach for multi-aspect and multi-granular pronunciation assessment. arXiv preprint arXiv:2305.18146 (2023)
4. Chen, C.F.R., Fan, Q., Panda, R.: Crossvit: Cross-attention multi-scale vision transformer for image classification. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 357–366 (2021)
5. Cheng, S., Liu, Z., Li, L., Tang, Z., Wang, D., Zheng, T.F.: Asr-free pronunciation assessment. arXiv preprint arXiv:2005.11902 (2020)
6. Cucchiarini, C., Strik, H., Boves, L.: Quantitative assessment of second language learners’ fluency by means of automatic speech recognition technology. J. Acoust. Soc. Am. 107(2), 989–999 (2000)
7. Dahl, G.E., Yu, D., Deng, L., Acero, A.: Context-dependent pre-trained deep neural networks for large-vocabulary speech recognition. IEEE Trans. Audio Speech Lang. Process. 20(1), 30–42 (2011)
8. Do, H., Kim, Y., Lee, G.G.: Hierarchical pronunciation assessment with multi-aspect attention. In: ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). pp. 1–5. IEEE (2023)
9. Du, W., Wumaier, A., Shi, Y., Yi, N., Liu, D.: A multi-aspect multi-granularity pronunciation assessment method based on branchformer encoder and hierarchical aggregation. In: Ide, I., et al. (eds) International Conference on Multimedia Modeling. pp. 16–29. Springer (2025).
10. Gong, Y., Chen, Z., Chu, I.H., Chang, P., Glass, J.: Transformer-based multi-aspect multi-granularity non-native english speaker pronunciation assessment. In: ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). pp. 7262–7266. IEEE (2022)
11. Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., Bengio, Y.: Generative adversarial networks. Commun. ACM 63(11), 139–144 (2020)
12. Hu, W., Qian, Y., Soong, F.K., Wang, Y.: Improved mispronunciation detection with deep neural network trained acoustic models and transfer learning based logistic regression classifiers. Speech Commun. 67, 154–166 (2015)
13. Lin, B., Wang, L.: Attention-based multi-encoder automatic pronunciation assessment. In: ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). pp. 7743–7747. IEEE (2021)
14. Lin, B., Wang, L.: Deep feature transfer learning for automatic pronunciation assessment. In: Interspeech. vol. 2021, pp. 4438–4442 (2021)
15. Lin, B., Wang, L.: A noise robust method for word-level pronunciation assessment. In: Interspeech. pp. 781–785 (2021)
16. Pei, H.C., Fang, H., Luo, X., Xu, X.S.: Gradformer: a framework for multi-aspect multi-granularity pronunciation assessment. IEEE/ACM Trans. Audio Speech Lang. Process. 32, 554–563 (2023)
17. Ryu, H., Kim, S., Chung, M.: A joint model for pronunciation assessment and mispronunciation detection and diagnosis with multi-task learning. In: Proc. Annu. Conf. Int. Speech Commun. Assoc. pp. 959–963 (2023)
18. Shi, J., Huo, N., Jin, Q.: Context-Aware Goodness of Pronunciation for Computer-Assisted Pronunciation Training. arXiv preprint arXiv:2008.08647 (2020)
19. Tepperman, J., Narayanan, S.: Automatic syllable stress detection using prosodic features for pronunciation evaluation of language learners. In: Proceedings. (ICASSP’05). IEEE International Conference on Acoustics, Speech, and Signal Processing, 2005. vol. 1, pp. I–937. IEEE (2005)
20. Vaswani, A., et al.: Attention is all you need. Adv. Neural Inf. Process. systems 30 (2017)
21. Wang, Y.B., Lee, L.S.: Improved approaches of modeling and detecting error patterns with empirical analysis for computer-aided pronunciation training. In: 2012 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). pp. 5049–5052. IEEE (2012)
22. Witt, S.M., Young, S.J.: Phone-level pronunciation scoring and assessment for interactive language learning. Speech Commun. 30(2–3), 95–108 (2000)
23. Yan, B.C., et al.: An effective pronunciation assessment approach leveraging hierarchical transformers and pre-training strategies. In: Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). pp. 1737–1747 (2024)
24. Zhang, J., et al.: speechocean762: An Open-Source Non-native English Speech Corpus for Pronunciation Assessment. arXiv preprint arXiv:2104.01378 (2021)
