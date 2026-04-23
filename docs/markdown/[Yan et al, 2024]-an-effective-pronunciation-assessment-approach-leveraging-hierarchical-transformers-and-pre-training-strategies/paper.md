---
title: "An Effective Pronunciation Assessment Approach Leveraging Hierarchical Transformers and Pre-training Strategies"
authors:
  - "Bi-Cheng Yan"
  - "Jiun-Ting Li"
  - "Yi-Cheng Wang"
  - "Hsin-Wei Wang"
  - "Tien-Hong Lo"
  - "Yung-Chang Hsu"
  - "Wei-Cheng Chao"
  - "Berlin Chen"
citation_author: "Yan et al."
year: 2024
doi: "10.18653/v1/2024.acl-long.95"
pages: "1737-1747"
source_pdf: "paper.pdf"
extraction_method: "Extracted from PDF OCR"
extracted_at: "2026-04-17"
llm_friendly: true
---

## Abstract

Automatic pronunciation assessment (APA) manages to quantify second language (L2) learner's pronunciation proficiency in a target language by providing fine-grained feedback with multiple pronunciation aspect scores at various linguistic levels. Most existing efforts on APA typically parallelize the modeling process, namely predicting multiple aspect scores across various linguistic levels simultaneously. This inevitably makes both the hierarchy of linguistic units and the relatedness among the pronunciation aspects sidelined. Recognizing such a limitation, we in this paper first introduce HierTFR, a hierarchal APA method that jointly models the intrinsic structures of an utterance while considering the relatedness among the pronunciation aspects. We also propose a correlation-aware regularizer to strengthen the connection between the estimated scores and the human annotations. Furthermore, novel pre-training strategies tailored for different linguistic levels are put forward so as to facilitate better model initialization. An extensive set of empirical experiments conducted on the speechocean762 benchmark dataset suggest the feasibility and effectiveness of our approach in relation to several competitive baselines.

## Keywords

computer-assisted language learning, automatic pronunciation assessment, deep regression models, pre-training mechanism.

## 1 Introduction

With the rising trend of globalization, more and more people are willing or being demanded to learn foreign languages. This surging need calls for developing computer-assisted pronunciation training (CAPT) systems, as they can offer tailored and informative feedback for L2 (second-language) learners to practice pronunciation skills in a stress-free and self-directed learning manner. As a crucial ingredient of CAPT, automatic pronunciation assessment (APA) aims to evaluate the extent of L2 learners’ oral proficiency and then provide fine-grained feedback on specific pronunciation aspects in response to a target language. A de-facto standard for APA systems is typically instantiated with a “reading-aloud” scenario, where an L2 learner is presented with a text prompt and instructed to pronounce it correctly. To offer in-depth feedback on learners’ pronunciation quality, recent efforts have drawn attention to the notion of multi-aspect and multi-granular pronunciation assessments, which normally devises a unified scoring model to jointly evaluate pronunciation proficiency at various linguistic levels (i.e., phone-, word-, and utterance-levels) with diverse aspects (e.g., accuracy, fluency, and completeness), as the running example depicted in Figure 1.

## 2 Methodology

### 2.1 Problem Formulation

Given an input utterance U, consisting of a time sequence of audio signals X uttered by an L2 learner, and a reference text prompt T with M words and N phones, an APA model is trained to estimate the proficiency scores pertaining to multiple pronunciation aspects at various linguistic granularities.

## 4 Experimental Results

### 4.1 Main Results

Table 1 reports the results on the speechocean762 dataset.

| Models | Phone Score (MSE) | Phone Score (PCC) | Word Score (PCC) Acc | Word Score (PCC) Stress | Word Score (PCC) Total | Utterance Score (PCC) Acc | Utterance Score (PCC) Comp | Utterance Score (PCC) Flu | Utterance Score (PCC) Pros | Utterance Score (PCC) Total |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Lin2021 | - | - | - | - | - | - | - | - | - | 0.720 |
| Kim2022 | - | - | - | - | - | - | - | 0.780 | 0.770 | - |
| Ruy2023 | - | - | - | - | - | 0.719 | - | 0.775 | 0.773 | 0.743 |
| LSTM | 0.089 | 0.591 | 0.514 | 0.294 | 0.531 | 0.720 | 0.076 | 0.745 | 0.747 | 0.741 |
| GOPT | 0.085 | 0.612 | 0.533 | 0.291 | 0.549 | 0.714 | 0.155 | 0.753 | 0.760 | 0.742 |
| HiPAMA | 0.084 | 0.616 | 0.575 | 0.320 | 0.591 | 0.730 | 0.276 | 0.749 | 0.751 | 0.754 |
| HierTFR | 0.081 | 0.644 | 0.622 | 0.325 | 0.634 | 0.735 | 0.513 | 0.801 | 0.795 | 0.764 |

*Table 1: The performance evaluations of our model and all compared methods on speechocean762 test set.*

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
