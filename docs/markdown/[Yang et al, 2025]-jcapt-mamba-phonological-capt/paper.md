---
title: "JCAPT: A Joint Modeling Approach for CAPT"
authors:
  - "Tzu-Hsuan Yang"
  - "Yue-Yang He"
  - "Berlin Chen"
citation_author: "Yang et al"
year: 2025
doi: null
arxiv: "2506.19315v2"
pages: 5
source_pdf: "2506.19315v2.pdf"
extraction_method: "Manual rewrite from the local PDF and LaTeX source (template.tex), cross-referenced with template.bbl for bibliography entries."
extracted_at: "2026-03-27T17:30:00-07:00"
llm_friendly: true
---

JCAPT: A Joint Modeling Approach for CAPT

## Metadata

- Authors: Tzu-Hsuan Yang, Yue-Yang He, Berlin Chen
- Affiliation: National Taiwan Normal University, Taipei, Taiwan
- Citation author: Yang et al
- Year: 2025
- arXiv: `2506.19315v2`
- Venue: Interspeech 2025 (camera-ready)
- Keywords: computer-assisted pronunciation training, speech attributes, Mamba, L2 speech assessment, multi-aspect scoring

## TL;DR

This paper presents `JCAPT`, a joint CAPT framework that simultaneously handles automatic pronunciation assessment (`APA`) and mispronunciation detection and diagnosis (`MDD`) using a parallel architecture built on bidirectional Mamba (a selective state space model). Two key additions differentiate it from prior joint systems: (1) phonological attribute features that encode articulatory properties of each canonical phoneme, and (2) learnable "think tokens" appended to the input sequence to give the encoder extra computational depth before making predictions. On `speechocean762`, `JCAPT` improves over the `JAM` baseline across nearly all APA metrics and raises MDD `F1` from `45.01%` to `51.05%`, with a particularly large gain in the completeness aspect (`0.205` to `0.551` PCC).

## Abstract

Effective pronunciation feedback is critical in second language (L2) learning, for which computer-assisted pronunciation training (CAPT) systems often encompass two key tasks: automatic pronunciation assessment (APA) and mispronunciation detection and diagnosis (MDD). Recent work has shown that joint modeling of these two tasks can yield mutual benefits. Our unified framework leverages Mamba, a selective state space model (SSM), while integrating phonological features and think token strategies to jointly enhance interpretability and fine-grained temporal reasoning in APA and MDD. To our knowledge, this is the first study to combine phonological attribution, SSM-based modeling, and prompting in CAPT. A series of experiments conducted on the speechocean762 benchmark demonstrate that our model consistently outperforms prior methods, particularly on the MDD task.

## Research Question

Can phonological attribute features and a "think token" reasoning mechanism, combined with a bidirectional Mamba encoder in a joint APA+MDD framework, improve both multi-granular pronunciation scoring and mispronunciation detection over existing joint baselines?

## Method

### Overview

`JCAPT` jointly models APA and MDD through a parallel architecture with five key components:

1. A comprehensive feature extraction module integrating multiple speech representations.
2. A bi-directional Mamba encoder for contextual modeling.
3. A contemplative reasoning mechanism via think tokens.
4. An attention-based pooling layer.
5. Multi-level scoring heads for APA and MDD.

### Feature Extraction

Given a speech utterance from an L2 learner and the canonical phone sequence `p = {p_1, p_2, ..., p_N}` of the corresponding text prompt, the system extracts phone-level features by combining goodness of pronunciation (GOP) with self-supervised representations.

**Goodness of Pronunciation (GOP):** GOP measures the likelihood of each phone being correctly pronounced. The system follows the standard pipeline: forced alignment via a DNN-HMM acoustic model, canonical phoneme decoding, and posterior probability estimation. This produces phone-aligned GOP features that directly reflect pronunciation accuracy.

**Self-Supervised Representations:** Three SSL models are used: `wav2vec 2.0`, `HuBERT`, and `WavLM`, all pre-trained on large-scale unlabeled speech data. Frame-level hidden features are extracted from each model and aligned to canonical phone boundaries using forced alignment. The aligned features are concatenated with GOP scores to form a comprehensive phoneme-level feature vector. The resulting vectors are projected through a dense layer to obtain phone-level embeddings `x_{1:N}`, where `N` is the number of canonical phones.

**Canonical Phone Embedding:** For each phoneme `p_i`, a one-hot vector `Phn_onehot` is concatenated with a phonological attribute vector `Phn_attr` that encodes articulatory properties. The resulting symbolic representation is projected to the same dimension as `x_{1:N}` and fused as an auxiliary input.

### Bi-directional Mamba Encoder

A bi-directional Mamba encoder inspired by the Dual-Mamba architecture is adopted for its linear scaling and efficient temporal representation. The acoustic features and canonical phoneme embeddings are fused:

```text
x_hat_i = x_i + c_i,  for i = 1, ..., N
```

where `x_i` is the projected acoustic feature and `c_i` is the symbolic canonical embedding.

The resulting sequence is passed through a stack of bidirectional Mamba blocks. Learnable think tokens are appended to the input sequence (see below). The encoder output is a sequence of contextualized phoneme-level representations:

```text
H = BiMamba(X_hat, Emb_think) = {h_1, h_2, ..., h_N}
```

### Contemplative Reasoning via Think Tokens

Inspired by contemplative prompting, learnable think tokens are postpended (not interleaved) to the end of the input sequence. This allows the model to perform additional internal computation before making phoneme-level predictions. The tokens are jointly optimized during training and are particularly effective for enhancing MDD diagnostic capacity and multi-aspect APA consistency.

### Attention-based Feature Pooling

For utterance-level representations, aspect-specific attention-based pooling is used. Given encoder output `H in R^{N x d}`, a separate attention module is defined for each assessment aspect `a` (accuracy, fluency, prosody, etc.).

Attention weights for aspect `a`:

```text
alpha_i^(a) = exp(w_a^T tanh(W_a h_i)) / sum_j exp(w_a^T tanh(W_a h_j))
```

where `W_a in R^{d_a x d}` and `w_a in R^{d_a}` are learnable parameters.

The utterance-level representation for aspect `a`:

```text
h_u^(a) = sum_i alpha_i^(a) * h_i
```

This allows each aspect to attend to different parts of the input sequence.

### Multi-level Scoring Heads

Hierarchical prediction heads support both tasks:

- **Phoneme-level:** A regression head estimates APA scores; a classification head predicts MDD labels.
- **Word-level:** Scores are derived by aggregating phoneme-level outputs based on forced alignment boundaries.
- **Utterance-level:** Each pooled aspect-specific representation `h_u^(a)` passes through an individual regression head for that aspect.

### Optimization

The model is trained under multi-task learning (MTL):

**APA loss** (sum of MSE at each granularity):

```text
L_APA = L_phn + L_word + L_utt
```

**MDD loss** (cross-entropy for phoneme classification):

```text
L_MDD = -sum_i sum_p y_{i,p} log(y_hat_{i,p})
```

where `N` is the number of training instances, `P` is the number of phoneme classes, `y_{i,p}` is the one-hot ground truth, and `y_hat_{i,p}` is the predicted probability.

**Total loss:**

```text
L = (1 - alpha) * L_APA + alpha * L_MDD
```

where `alpha = 0.3` following the JAM baseline, found to yield stable performance.

## Data

- **Dataset:** `speechocean762`
- **Size:** `5,000` English utterances from `250` Mandarin-speaking L2 learners, evenly split into training and test sets.
- **APA labels:** Human-rated pronunciation scores at utterance, word, and phoneme levels, assessed by five expert raters using standardized rubrics.
- **MDD labels:** Canonical and realized phone-level transcriptions aligned at the phoneme level. Uses a `39`-phone set based on the CMU pronunciation dictionary, extended with `<del>` and `<unk>` tokens for deleted and non-categorizable phones. The dataset does not include insertion errors.
- **Evaluation:** Five independent runs with different random seeds; average metrics reported. APA evaluated via PCC and MSE. MDD evaluated via F1-score (primary), recall, precision, PER, and correct diagnosis rate.

## Results

### Main Results (Table 1)

Comparison on `speechocean762`:

| Model | Phn MSE ↓ | Phn PCC ↑ | Word Acc ↑ | Word Stress ↑ | Word Total ↑ | Utt Acc ↑ | Utt Comp ↑ | Utt Flu ↑ | Utt Pros ↑ | Utt Total ↑ | MDD RE% ↑ | MDD PR% ↑ | MDD F1% ↑ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Joint-CAPT-L1 | - | - | - | - | - | 0.719 | - | 0.775 | 0.773 | 0.743 | **91.40** | 26.70 | 41.40 |
| JAM | 0.076 | 0.664 | 0.622 | 0.241 | 0.638 | 0.773 | 0.205 | 0.831 | **0.829** | 0.805 | 34.76 | 64.10 | 45.01 |
| JCAPT | **0.066** | **0.720** | **0.699** | **0.270** | **0.711** | **0.783** | **0.551** | **0.834** | 0.824 | **0.806** | 40.23 | **69.89** | **51.05** |

Key observations:

- `JCAPT` achieves the lowest phoneme-level MSE (`0.066`) and highest PCC (`0.720`).
- Word-level improvements are substantial: accuracy `0.622` to `0.699`, stress `0.241` to `0.270`, total `0.638` to `0.711`.
- The most striking utterance-level gain is completeness: `0.205` (JAM) to `0.551` (JCAPT).
- MDD F1 improves from `45.01%` to `51.05%`, driven by gains in both recall and precision.
- `Joint-CAPT-L1` has much higher recall (`91.40%`) but very low precision (`26.70%`), yielding a lower F1.

### Ablation Studies — APA (Table 2)

| Model | Phn MSE ↓ | Phn PCC ↑ | Word Acc ↑ | Word Stress ↑ | Word Total ↑ | Utt Acc ↑ | Utt Comp ↑ | Utt Flu ↑ | Utt Pros ↑ | Utt Total ↑ |
|---|---|---|---|---|---|---|---|---|---|---|
| JCAPT | **0.066** | **0.720** | **0.699** | 0.270 | **0.711** | 0.783 | 0.551 | 0.834 | 0.824 | 0.806 |
| w/o phonological | **0.066** | 0.716 | 0.689 | 0.239 | 0.701 | 0.775 | **0.644** | **0.840** | **0.826** | **0.808** |
| w/o think tokens | **0.066** | **0.720** | **0.699** | **0.309** | 0.710 | **0.784** | 0.556 | 0.833 | 0.818 | **0.808** |
| w/o both | 0.068 | 0.708 | 0.687 | 0.273 | 0.699 | 0.779 | 0.547 | 0.834 | 0.822 | **0.808** |

### Ablation Studies — MDD (Table 3)

| Model | Recall% ↑ | Precision% ↑ | F1% ↑ | PER% ↓ | Correct Diag% ↑ |
|---|---|---|---|---|---|
| JCAPT | 40.23 | 69.89 | 51.05 | **2.66** | **54.42** |
| w/o phonological | **42.00** | 68.98 | **52.21** | 2.70 | 52.67 |
| w/o think tokens | 39.76 | 69.95 | 50.61 | 2.67 | 54.35 |
| w/o both | 41.27 | **70.07** | 51.92 | 2.67 | 52.80 |

Ablation findings:

- Removing phonological features hurts phoneme- and word-level APA (especially PCC and stress), confirming their value for modeling fine-grained articulatory patterns. Utterance-level metrics remain stable.
- Removing think tokens mainly affects MDD recall and F1, suggesting they help capture disfluency-related cues. Minor decreases in precision indicate a potential trade-off.
- Removing both degrades performance across the board, suggesting complementary roles: phonological features provide linguistic grounding while think tokens enhance reasoning sensitivity.
- Interestingly, the `w/o phonological` variant scores higher on completeness (`0.644` vs `0.551`) and some utterance-level metrics, suggesting some tension between phonological features and global prosodic modeling.

## Limitations / Notes

- **Limited accent diversity:** All learners are Mandarin L1 speakers. Generalizability to other L1 backgrounds is unverified.
- **Read-aloud only:** The framework is evaluated in a read-aloud scenario. Spontaneous or open-ended speech is not addressed.
- **Ablation paradox on completeness:** The `w/o phonological` ablation actually outperforms the full model on completeness (`0.644` vs `0.551`) and some utterance-level metrics, which is not discussed in the paper.
- **MDD recall is moderate:** At `40.23%`, `JCAPT` detects fewer than half of actual mispronunciations. `Joint-CAPT-L1` reaches `91.40%` recall but at much lower precision.
- **No insertion error handling:** The dataset lacks insertion errors, so this error type is not evaluated.
- **Think token implementation details sparse:** The number of think tokens used and sensitivity to this hyperparameter are not reported.
- **Venue:** The LaTeX source includes Interspeech 2025 camera-ready formatting (`\interspeechcameraready`), suggesting acceptance at Interspeech 2025.

## Conclusion and Future Work

`JCAPT` is a unified CAPT framework that jointly addresses APA and MDD through a parallel architecture built on the Mamba state space model. By integrating phonological features and adopting a think token strategy for fine-grained temporal reasoning, it enhances both diagnostic interpretability and predictive performance. Results on `speechocean762` show improvements over baselines, especially in mispronunciation detection and completeness. Future work will explore generalizability across diverse learner populations, languages, and spontaneous speech, and will target item-specific enhancements for scoring aspects with lower correlation (stress and completeness).

## Acknowledgement

This work was supported by the Language Training and Testing Center (LTTC), Taiwan.

## References

1. P. Munday, "Duolingo. Gamified learning through translation," *Journal of Spanish Language Teaching*, vol. 4, no. 2, pp. 194--198, 2017.
2. A. Kholis, "ELSA Speak app: Automatic speech recognition (ASR) for supplementing English pronunciation skills," *Pedagogy: Journal of English Language Teaching*, vol. 9, no. 1, pp. 01--14, 2021.
3. Y. Gong, Z. Chen, I.-H. Chu, P. Chang, and J. Glass, "Transformer-based multi-aspect multi-granularity non-native English speaker pronunciation assessment," in *ICASSP 2022*, IEEE, 2022, pp. 7262--7266.
4. B.-C. Yan, H.-W. Wang, Y.-C. Wang, J.-T. Li, C.-H. Lin, and B. Chen, "Preserving phonemic distinctions for ordinal regression: A novel loss function for automatic pronunciation assessment," in *ASRU 2023*, 2023, pp. 1--7.
5. B.-C. Yan, J.-T. Li, Y.-C. Wang, H.-W. Wang, T.-H. Lo, Y.-C. Hsu, W.-C. Chao, and B. Chen, "An effective pronunciation assessment approach leveraging hierarchical transformers and pre-training strategies," in *ACL 2024*, 2024, pp. 1737--1747.
6. H.-W. Wang, B.-C. Yan, H.-S. Chiu, Y.-C. Hsu, and B. Chen, "Exploring non-autoregressive end-to-end neural modeling for English mispronunciation detection and diagnosis," in *ICASSP 2022*, IEEE, 2022, pp. 6817--6821.
7. W. Ye, S. Mao, F. Soong et al., "An approach to mispronunciation detection and diagnosis with acoustic, phonetic and linguistic (APL) embeddings," in *ICASSP 2022*, IEEE, 2022, pp. 6827--6831.
8. B.-C. Yan, H.-W. Wang, and B. Chen, "PeppaNet: Effective mispronunciation detection and diagnosis leveraging phonetic, phonological, and acoustic cues," in *SLT 2022*, 2023, pp. 1045--1051.
9. H. Ryu, S. Kim, and M. Chung, "A joint model for pronunciation assessment and mispronunciation detection and diagnosis with multi-task learning," *INTERSPEECH*, 2023.
10. Y. Y. He, B. C. Yan, T. H. Lo, M. S. Lin, Y. C. Hsu, and B. Chen, "JAM: A unified neural architecture for joint multi-granularity pronunciation assessment and phone-level mispronunciation detection and diagnosis towards a comprehensive CAPT system," in *APSIPA ASC 2024*, IEEE, Dec. 2024, pp. 1--6.
11. F.-A. Chao and B. Chen, "Towards efficient and multifaceted computer-assisted pronunciation training leveraging hierarchical selective state space model and decoupled cross-entropy loss," *arXiv preprint arXiv:2502.07575*, 2025.
12. M. Shahin and B. Ahmed, "Phonological-level mispronunciation detection and diagnosis," *Interspeech 2024*, Sep. 2024.
13. B.-C. Yan, H.-W. Wang, Y.-C. Wang, and B. Chen, "Effective graph-based modeling of articulation traits for mispronunciation detection and diagnosis," in *ICASSP 2023*, IEEE, 2023, pp. 1--5.
14. J. Wei, X. Wang, D. Schuurmans, M. Bosma, F. Xia, E. Chi, Q. V. Le, D. Zhou et al., "Chain-of-thought prompting elicits reasoning in large language models," *NeurIPS*, vol. 35, pp. 24824--24837, 2022.
15. T. J. Yang, A. Rosenberg, and B. Ramabhadran, "Contemplative mechanism for speech recognition: Speech encoders can think," *Proceedings of Interspeech*, pp. 3455--3459, 2024.
16. A. Gu and T. Dao, "Mamba: Linear-time sequence modeling with selective state spaces," *arXiv preprint arXiv:2312.00752*, 2023.
17. M. Yoshimura, T. Hayashi, and Y. Maeda, "MambaPEFT: Exploring parameter-efficient fine-tuning for Mamba," *arXiv preprint arXiv:2411.03855*, 2024.
18. B.-C. Yan, H.-W. Wang, Y.-C. Wang, J.-T. Li, C.-H. Lin, and B. Chen, "Preserving phonemic distinctions for ordinal regression: A novel loss function for automatic pronunciation assessment," in *ASRU 2023*, IEEE, 2023, pp. 1--7.
19. S. M. Witt and S. J. Young, "Phone-level pronunciation scoring and assessment for interactive language learning," *Speech Communication*, vol. 30, no. 2-3, pp. 95--108, 2000.
20. W. Hu, Y. Qian, F. K. Soong, and Y. Wang, "Improved mispronunciation detection with deep neural network trained acoustic models and transfer learning based logistic regression classifiers," *Speech Communication*, vol. 67, pp. 154--166, 2015.
21. J. Shi, N. Huo, and Q. Jin, "Context-aware goodness of pronunciation for computer-assisted pronunciation training," *arXiv preprint arXiv:2008.08647*, 2020.
22. A. Baevski, Y. Zhou, A. Mohamed, and M. Auli, "Wav2vec 2.0: A framework for self-supervised learning of speech representations," *NeurIPS*, vol. 33, pp. 12449--12460, 2020.
23. W.-N. Hsu, B. Bolte, Y.-H. H. Tsai, K. Lakhotia, R. Salakhutdinov, and A. Mohamed, "HuBERT: Self-supervised speech representation learning by masked prediction of hidden units," *IEEE/ACM TASLP*, vol. 29, pp. 3451--3460, 2021.
24. S. Chen, C. Wang, Z. Chen, Y. Wu, S. Liu, Z. Chen, J. Li, N. Kanda, T. Yoshioka, J. Wu, X. Xiao, L. Zhou, C. Li, S. Ren, Y. Zhang, F. Yu, Q. Fu, and F. Wei, "WavLM: Large-scale self-supervised pre-training for full stack speech processing," *IEEE JSTSP*, vol. 16, no. 6, pp. 1505--1518, 2022.
25. X. Jiang, C. Han, and N. Mesgarani, "Dual-path Mamba: Short and long-term bidirectional selective structured state space models for speech separation," in *ICASSP 2025*, IEEE, 2025, pp. 1--5.
26. J. Zhang, Z. Zhang, Y. Wang et al., "speechocean762: An open-source non-native English speech corpus for pronunciation assessment," *arXiv preprint arXiv:2104.01378*, 2021.
27. R. Weide, "The Carnegie Mellon pronouncing dictionary [cmudict. 0.6]," Pittsburgh, PA: Carnegie Mellon University, 2005.
28. K. Li, X. Qian, and H. Meng, "Mispronunciation detection and diagnosis in L2 English speech using multidistribution deep neural networks," *IEEE/ACM TASLP*, vol. 25, no. 1, pp. 193--207, 2017.
