---
title: "ConPCO: Preserving Phoneme Characteristics For Automatic Pronunciation Assessment Leveraging Contrastive Ordinal Regularization"
authors:
  - "Bi-Cheng Yan"
  - "Yi-Cheng Wang"
  - "Jiun-Ting Li"
  - "Meng-Shin Lin"
  - "Hsin-Wei Wang"
  - "Wei-Cheng Chao"
  - "Berlin Chen"
citation_author: "Yan et al."
year: 2025
doi: "10.1109/ICASSP49660.2025.10890778"
pages: 5
source_pdf: "paper.pdf"
extraction_method: "Extracted from PDF OCR"
extracted_at: "2026-04-17"
llm_friendly: true
---

## Abstract

Automatic pronunciation assessment (APA) manages to evaluate the pronunciation proficiency of a second language (L2) learner in a target language. Existing efforts typically draw on regression models for proficiency score prediction, wherein the models are trained to estimate target values without explicitly accounting for phoneme-awareness in the feature space. In this paper, we propose a contrastive phonemic ordinal regularizer (ConPCO) tailored for regression-based APA models to generate more phoneme-discriminative features while factoring in the ordinal relationships among the regression targets. The proposed ConPCO first aligns the phoneme representations of an APA model and textual embeddings of phonetic transcriptions via contrastive learning. Afterward, the phoneme characteristics are retained by regulating the distances between inter- and intra-phoneme categories in the feature space while allowing for the ordinal relationships among the output targets. We further design and develop a hierarchical APA model to evaluate the effectiveness of our regularizer. A series of experiments conducted on the speechocean762 benchmark dataset suggests the feasibility and effectiveness of our approach in relation to several competitive baselines.

## Keywords

computer-assisted language learning, automatic pronunciation assessment, contrastive learning

## I. Introduction

Fueled by the surging demand for foreign language learning, developments of computer-assisted pronunciation training (CAPT) systems have aroused ever-increasing attention amidst the tide of globalization. CAPT systems are designed to offer tailored and informative feedback for L2 (second-language) learners to practice pronunciation skills in stress-free and self-directed learning scenarios [1][2][3]. As an indispensable component of CAPT systems, automatic pronunciation assessment (APA) aims to determine the extent of second language (L2) learners’ oral proficiency and then provide detailed feedback on specific pronunciation aspects pertaining to a target language [4][5].

A de-facto standard for APA systems is instantiated in a reading-aloud learning scenario, where an L2 learner is presented with a text prompt and instructed to pronounce it accordingly [6][7]. Through the synergistic processing of input speech and the reference text prompt, an APA system is anticipated to assess the learner’s speaking skills and provide immediate feedback, including the overall proficiency (holistic scores) or specific aspects of pronunciation (analytic scores). To offer in-depth feedback on learners’ pronunciation quality, recent research endeavors have drawn attention to multi-aspect and multi-granular pronunciation assessments, which devise a unified scoring model to jointly evaluate pronunciation proficiency at various linguistic levels (i.e., phoneme, word, and utterance) with diverse aspects (e.g., accuracy, fluency, and completeness) via advanced parallel [8][9] or hierarchical neural architectures [10]. Due to the continuity of output targets, which can be infinite and boundless [11], existing methods typically adopt a regression loss function, such as mean-squared error (MSE), as the training objective to mimic expert’s evaluations. Although some promising results have been achieved, the distinct features of language units (e.g., phonetic information [12][13] and word semantics [14][15]) are nearly sidelined in the optimization process.

In this work, we identify three limitations in existing regression-based APA models: (1) the phoneme representations of input speech and the textual embeddings of phoneme-level text prompts are located in separate feature spaces; (2) different phoneme representations belonging to the same proficiency level are inadvertently forced to be close to one another, harming pronunciation clarity [16]; and (3) the ordinal relationships among the regression targets are almost overlooked. To address these limitations, we present a novel training regime, termed contrastive phonemic ordinal regularizer (ConPCO).

## II. Methodology

### A. Contrastive Phonemic Ordinal Regularizer (ConPCO)

The proposed ConPCO regularizer consists of three mathematical terms: the contrastive term $\mathcal{L}_{con}$, the phonemic characteristic term $\mathcal{L}_{pc}$, and the ordinal term $\mathcal{L}_{o}$.

#### 1) Contrastive Term

The contrastive term $\mathcal{L}_{con}$ aims to simultaneously project phoneme representations from an APA model and embeddings of phoneme-level text prompt into a joint feature space. It maximizes the similarity between paired phoneme representations while minimizing the similarity of unpaired ones [18][19].

#### 2) Phonemic Characteristic Term

The phonemic characteristic term $\mathcal{L}_{pc}$ preserves the phonemic proximity information by minimizing the negative distances between centroid vectors, equivalent to maximizing the distances between phoneme categories.

#### 3) Ordinal Term

To reflect ordinal relationships of regression targets in the feature space, the ordinal term $\mathcal{L}_{o}$ is defined to minimize the distance between the feature representations and their corresponding phoneme centroid vectors with relative differences of the proficiency score.

### B. Hierarchical APA Model (HierCB)

The HierCB model comprises three main components: phoneme-level, word-level, and utterance-level modeling, each adopting the novel convolution-augmented Branchformer block.

## III. Experiments

### A. Experimental Settings

Dataset: We conducted APA experiments on the speechocean762 dataset, which contains 5,000 English-speaking recordings spoken by 250 Mandarin L2 learners. Table I summarizes the statistics.

| Granularities | Aspects | Score Interval | # of Counts (Train) | # of Counts (Test) |
| :--- | :--- | :--- | :--- | :--- |
| Phoneme | Accuracy | [0, 2] | 47,076 | 47,369 |
| Word | Accuracy, Stress, Total | [0, 10] | 15,849 | 15,967 |
| Utterance | Accuracy, Completeness, Fluency, Prosody, Total | [0, 10] | 2,500 | 2,500 |

*Table I: Statistics of the speechocean762 dataset.*

### B. Experimental Results

Table II and Table III present the performance evaluation results.

| Input Feats. | Models | Phone Score (MSE) | Phone Score (PCC) | Word Score (PCC) Acc | Word Score (PCC) Stress | Word Score (PCC) Total |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| GOP | LSTM [8] | 0.089 | 0.591 | 0.514 | 0.294 | 0.531 |
| GOP | GOPT [8] | 0.085 | 0.612 | 0.533 | 0.291 | 0.549 |
| GOP | GFR [6] | 0.079 | 0.646 | 0.598 | 0.334 | 0.614 |
| GOP | HiPAMA[10] | 0.084 | 0.616 | 0.575 | 0.320 | 0.591 |
| SSL | GOPT-SSL | 0.081 | 0.640 | 0.584 | 0.352 | 0.603 |
| SSL | 3M [9] | 0.078 | 0.656 | 0.598 | 0.289 | 0.617 |
| SSL | HierBFR | 0.082 | 0.639 | 0.591 | 0.300 | 0.609 |
| SSL | HierCB | 0.076 | 0.680 | 0.630 | 0.355 | 0.645 |
| SSL | +PCO [16] | 0.078 | 0.688 | 0.648 | 0.347 | 0.622 |
| SSL | +ConPCO | 0.071 | 0.701 | 0.669 | 0.437 | 0.682 |

*Table II: Experimental results of various methods for pronunciation assessments at phoneme and word levels.*

| Models | Acc. | Comp. | Fluency | Prosody | Total |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 3M [9] | 0.760 | 0.325 | 0.828 | 0.827 | 0.796 |
| GOPT-SSL | 0.748 | 0.290 | 0.817 | 0.807 | 0.778 |
| HierCB | 0.772 | 0.677 | 0.827 | 0.822 | 0.796 |
| +ConPCO | 0.780 | 0.749 | 0.830 | 0.823 | 0.803 |

*Table III: Experimental results of various methods on utterance-level pronunciation assessments (PCC).*

## IV. Conclusion

In this paper, we have proposed a novel training regime, ConPCO, seeking to learn phoneme-aware representations while preserving the ordinal relationships among the regression targets in the learned feature space. In addition, we also developed a hierarchical APA model to verify the efficacy of the proposed regularizer. The practical utility of our method has been verified through extensive experiments on speechocean762 benchmark dataset.

---

## References

1. P. M Rogerson-Revell, “Computer-assisted pronunciation training (CAPT): Current issues and future directions,” RELC Journal, vol. 52, pp. 189–205, 2021.
2. A. V. Moere and R. Downey, “Technology and artificial intelligence in language assessment,” Handbook of Second Language Assessment, pp. 341–358, 2016.
3. M. Eskenazi, “An overview of spoken language technology for education,” Speech Communication, vol. 51, pp. 832–844, 2009.
4. S. Bannò, B. Balusu, M. Gales, K. Knill, and K. Kyriakopoulos, “View-specific assessment of L2 spoken English,” in Proceedings of Interspeech (INTERSPEECH), pp. 4471–4475, 2022.
5. N. F. Chen, and H. Li, “Computer-assisted pronunciation training: From pronunciation scoring towards spoken language learning,” in Proceedings of the Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC), pp. 1–7, 2016.
6. H.-C. Pei, H. Fang, X. Luo and X.-S. Xu, “Gradformer: A framework for multi-aspect multi-granularity pronunciation assessment," in IEEE/ACM Trans. on Audio, Speech, and Language Processing, vol. 32, pp. 554–563, 2024.
7. B.-C. Yan, J.-T. Li, Y.-C. Wang, H. W. Wang, T.-H. Lo, Y.-C. Hsu, W.-C. Chao, and B. Chen, “An effective pronunciation assessment approach leveraging hierarchical transformers and pre-training strategies,” in Proceedings of the Association for Computational Linguistics (ACL), pp. 1737–1747, 2024.
8. Y. Gong, Z. Chen, I.-H. Chu, P. Chang, and J. Glass, “Transformer based multi-aspect multigranularity non-native English speaker pronunciation assessment,” in Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 7262–7266, 2022.
9. F. A. Chao, T. H. Lo, T. I. Wu, Y. T. Sung, B. Chen, “3M: An effective multi-view, multigranularity, and multi-aspect modeling approach to English pronunciation assessment,” in Proceedings of the Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC), pp. 575–582, 2022.
10. H. Do, Y. Kim, and G. G. Lee, “Hierarchical pronunciation assessment with multi- aspect attention,” in Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 1–5, 2023.
11. Y. Yang, K. Zha, Y. Chen, H. Wang, and D. Katabi, “Delving into deep imbalanced regression,” in Proceedings of the International Conference on Machine Learning (PMLR), pp. 11842–11851.
12. P. C. English, J. Kelleher, and J. Carson-Berndsen, “Domain informed probing of wav2vec 2.0 embeddings for phonetic features,” in Proceedings of the SIGMORPHON Workshop on Computational Research in Phonetics, Phonology, and Morphology, pp. 83–91, 2022.
13. V. Zouhar, K. Chang, C. Cui, N. B. Carlson, N. R. Robinson, M. Sachan, and D. R. Mortensen, “PWESuite: Phonetic word embeddings and tasks they facilitate,” in Proceedings of the Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING), pp. 13344–13355, 2024.
14. Z. Fu, W. Zhou, J. Xu, H. Zhou, and Lei Li. “Contextual representation learning beyond masked language modeling,” in Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL), pp. 2701–2714, 2022.
15. A. Borah, M. P. Barman, and A. Awekar, “Are word embedding methods stable and should we care about it?” in Proceedings of the ACM Conference on Hypertext and social media, pp. 45-55, 2021.
16. B.-C. Yan, H.-W. Wang, Y.-C. Wang, J.-T. Li, C.-H. Lin, and B. Chen, “Preserving phonemic distinctions for ordinal regression: A novel loss function for automatic pronunciation assessment,” in Proceedings of the IEEE Automatic Speech Recognition and Understanding Workshop (ASRU), pp. 1–7, 2023.
17. Y. Peng, S. Dalmia, I. Lane, and S. Watanabe, “Branchformer: Parallel MLP-attention architectures to capture local and global context for speech recognition and understanding, in Proceedings of the International Conference on Machine Learning (PMLR), vol. 162, pp. 17627–17643, 2022.
18. A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and Ilya Sutskever, “Learning transferable visual models from natural language supervision,” in Proceedings of the International Conference on Machine Learning (PMLR), vol. 139, pp. 8748–8763, 2021.
19. B. Elizalde, S. Deshmukh, M. A. Ismail, and H. Wang, “Clap: Learning audio concepts from natural language supervision,” in Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 1–5, 2023.
20. A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” in Proceedings of the Conference on Neural Information Processing Systems (NeurIPS), pp. 5998–6008, 2017.
21. A. Gulati, J. Qin, C.-C. Chiu, N. Parmar, Y. Zhang, J. Yu, W. Han, S. Wang, Z. Zhang, Y. Wu, and R. Pang, “Conformer: Convolution augmented transformer for speech recognition,” in Proceedings of Interspeech (INTERSPEECH), pp 5036–5040, 2020.
22. J. Sakuma, T. Komatsu, and R. Scheibler, “MLP-based architecture with variable length input for automatic speech recognition,” arXiv preprint arXiv:2202.08456, 2022.
23. Y. N. Dauphin, A. Fan, M. Auli, D. Grangier, “Language modeling with gated convolutional networks,” in Proceedings of the International Conference on Machine Learning (PMLR), vol. 70, pp. 933–941, 2017.
24. S. M. Witt and S. J. Young, “Phone-level pronunciation scoring and assessment for interactive language learning,” Speech Communication, vol. 30, pp. 95–108, 2000.
25. C. Zhu, T. Kunihara, D. Saito, N. Minematsu, N. Nakanishi, “Automatic prediction of intelligibility of words and phonemes produced orally by Japanese learners of English,” in IEEE Spoken Language Technology Workshop (SLT), pp. 1029–1036, 2022.
26. S. Yang, Y. Ayano, S. Daisuke, M. Nobuaki, and K. Saito, “Optimized prediction of fluency of L2 English based on interpretable network using quantity of phonation and quality of pronunciation,” in IEEE Spoken Language Technology Workshop (SLT), pp. 698–704, 2021.
27. J. Zhang, Z. Zhang, Y. Wang, Z. Yan, Q. Song, Y. Huang, K. Li, D. Povey, and Y. Wang, “Speechocean762: An open-source non-native English speech corpus for pronunciation assessment,” In Proceedings of Interspeech (INTERSPEECH), pp. 3710–3714, 2021.
28. A. Baevski, H. Zhou, A. Mohamed, and M. Auli, “Wav2vec 2.0: A framework for self-supervised learning of speech representations,” in Proceedings of the International Conference on Neural Information Processing Systems (NIPS), pp. 12449–12460, 2020.
29. S. Chen et al., “Wavlm: Large-scale self-supervised pre-training for full stack speech processing,” IEEE Journal of Selected Topics in Signal Processing, pp. 1505–1518, 2022.
30. W.-N. Hsu et al., “HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units,” IEEE/ACM Transactions on Audio, Speech and Language Processing, pp. 3451–3460, 2021.
31. Y. Wang, M.J.F. Gales, K. M Knill, K. Kyriakopoulos, A. Malinin, R. C van Dalen, M. Rashid, “Towards automatic assessment of spontaneous spoken English,” Speech Communication, vol. 104, pp. 47–56, 2018.
32. J. Park and S. Choi, “Addressing cold start problem for end-to-end automatic speech scoring,” in Proceedings of Interspeech (INTERSPEECH), pp. 994–998, 2023.
