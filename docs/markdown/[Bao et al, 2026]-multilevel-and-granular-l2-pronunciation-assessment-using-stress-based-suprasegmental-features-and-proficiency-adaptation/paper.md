---
title: "Multilevel and Granular L2 Pronunciation Assessment Using Stress-Based Suprasegmental Features and Proficiency Adaptation"
authors:
  - "Wenqian Bao"
  - "Jingsong Zhang"
citation_author: "Bao et al"
year: 2026
doi: "10.1007/978-981-95-5382-2_2"
pages: 11
journal: "NCMMMSC 2025, CCIS 2662, pp. 13-23, 2026"
source_pdf: "978-981-95-5382-2_2.pdf"
extraction_method: "Extracted from PDF with Manual Vision Synthesis"
extracted_at: "2026-04-18"
llm_friendly: true
---

## Abstract

The joint modeling of mispronunciation detection and diagnosis (MDD) and automatic pronunciation assessment (APA) in computer-assisted pronunciation training (CAPT) systems has been proven effective. While existing approaches (e.g., HMamba) employ self-supervised learning (SSL) to extract rich speech representations incorporating prosodic features like phoneme duration and silent segments, they still face two key challenges: On one hand, SSL representations lack targeted modeling of crucial suprasegmental features such as stress; on the other hand, directly joint-training segmental features (local phoneme accuracy) and suprasegmental features (global prosodic patterns) leads to performance conflicts due to their differing granularities of focus. To address these issues, this paper proposes a stress-enhanced framework. First, we explicitly model word-level and sentence-level stress features based on vowel formants (F1/F2) and spectral balance. Subsequently, we employ a proficiency-aware attention matching mechanism that adaptively adjusts fusion weights between segmental and suprasegmental features according to learners’ second-language proficiency.

## 1 Introduction

Mispronunciation Detection and Diagnosis (MDD) and Automatic Pronunciation Assessment (APA) have traditionally been treated as separate tasks in Computer-Assisted Pronunciation Training (CAPT) systems. While MDD has achieved significant progress with state-of-the-art models reaching 63% F1-score on the L2-ARCTIC dataset [14], these approaches primarily focus on segmental-level features, neglecting crucial suprasegmental feedback that is equally important for second language (L2) learners.

The field of APA witnessed a major advancement with the release of the Speechocean762 dataset [18] and the introduction of GOPT [9], which established a new baseline. Subsequent studies [4, 17] have mainly focused on architectural modifications of GOPT with limited improvements. Recent works [6, 12, 15] attempted to incorporate different ASR models for multi-level assessment but achieved marginal gains.

The potential of joint MDD-APA modeling was first demonstrated by [13] on proprietary data, while HMamba [3] recently extended this approach to Speechocean762 using Mamba architecture for computational efficiency, showing improvements across all assessment levels. However, English being a stress-timed language (e.g., “REcord” vs. “reCORD” where stress determines lexical meaning), the evaluation of word stress remains understudied. Current stress-level Pearson Correlation Coefficient (PCC) scores show room for improvement, and sentence-level completeness scores stagnate around 0.2, significantly lagging behind other metrics.

Second language (L2) proficiency significantly impacts pronunciation assessment outcomes, with advanced learners demonstrating better segmental accuracy and stress patterns compared to beginners, as shown in longitudinal studies of L2 English learners [11]. Recent work further confirms that incorporating L2 proficiency levels as a conditional variable improves automatic assessment accuracy by 12.7% in state-of-the-art models [10].

### Proposed Framework

Our goal is to develop a CAPT framework capable of effectively modeling both segmental and suprasegmental information, with particular focus on optimizing word stress assessment through proficiency-aware attention mechanisms. Current approaches suffer from limited feature diversity, primarily relying on goodness of pronunciation (GOP) features that only capture phoneme-level information. Although recent work (e.g., HMamba) has incorporated pause duration and phoneme length, these features lack specificity for stress detection. To address this, we propose targeted stress features including:

* **Formant trajectories ($F_1/F_2$)** and spectral balance.
* **Continuous pitch contours**, extracted using PyWORLD's DIO algorithm.
* **ABM Module:** A novel hierarchical attention matching mechanism (inspired by medical image segmentation work [2]) that operates at both phoneme and word levels, adaptively combining L2 proficiency indicators with suprasegmental features.

## 2 Method

### 2.1 Feature Representation

Our framework processes speech signals through two parallel paths: segmental (phonological) and suprasegmental (prosodic).

**Features Extraction for Segmental and Phonological Modeling.** The Goodness of Pronunciation (GOP) metric [16] is the core segmental feature. It measures the log-likelihood ratio between the canonical phoneme and the most likely recognized phoneme:
$$ G(p) = \frac{1}{\tau(p)} \ln \left| \frac{\mathcal{L}(p|o^p)}{\sum_{q \in Q} \mathcal{L}(q|o^q)} \right| \quad (1) $$
where $\tau(p)$ is the duration of phoneme $p$ in frames, $\mathcal{L}(p|o^p)$ is the likelihood of $p$ given acoustic observations $o^p$, and $Q$ is the candidate phoneme set.
The GOP extraction utilizes a hybrid DNN-HMM acoustic model trained on 80-dimensional logmel filterbank features with speaker-level CMVN. Phoneme representations are further enhanced with:

* **Relative Positional Embeddings:** BIES tags (Beginning, Inside, End, Single) representing phoneme position within a word.
* **Absolute Positional Embeddings:** Learned embeddings indexed by the phoneme order in the utterance.

**Acoustic Correlates of Stress.** We explicitly model suprasegmental features to capture English stress patterns:

* **Stress Feature Vector (6D):** Combines vowel formants ($F_1, F_2$) sampled at midpoints using Parselmouth's Burg method (2D) and spectral balance features (4D). Spectral balance captures band energies $E_{a:b} = 10 \log_{10}(\mathbb{E}[|S(f)|^2])$ across four ranges: 0–500 Hz, 500–1000 Hz, 1000–2000 Hz, and 2000–4000 Hz.
* **Continuous Pitch Contour:** Extracted using PyWORLD’s DIO algorithm (20ms frame/hop), clamped to 100–600 Hz, and converted to 256 Mel-scaled bins.
* **Duration and Energy:** Phoneme durations are obtained via forced alignment, while energy features follow RMS statistics [7].

### 2.2 Proficiency Embeddings

Following the stratification in [18], we categorize speakers into three proficiency levels (low, medium, high) based on their mean scores across the dataset. These discrete levels are mapped to learned embeddings to condition the assessment process:
$$ \mathbf{p} = \text{nn.Embedding}(3, d) \in \mathbb{R}^{L \times d} \quad (2) $$
where $L$ is sequence length and $d$ is the embedding dimension.

### 2.3 Feature Fusion with ABM Module

Inspired by the Attention-Based Matching (ABM) mechanism [2], we adapt it to adaptively fuse proficiency indicators with suprasegmental features. The process involves three steps:

**1. Linear Transformation:** Given proficiency embedding $\mathbf{s} \in \mathbb{R}^{L \times 128}$ and query features $\mathbf{Q}$:
$$ \begin{cases} Q_{phone} = \mathbf{H} \in \mathbb{R}^{L \times d} & \text{(Phone-level ABM)} \\ Q_{word} = \text{MLP}(\text{concat}[\mathbf{d}, \mathbf{e}, \mathbf{st}, \mathbf{p}]) & \text{(Word-level ABM)} \end{cases} \quad (3) $$
where $\mathbf{d}$ (duration), $\mathbf{e}$ (energy), $\mathbf{st}$ (stress), and $\mathbf{p}$ (pitch) are suprasegmental features.

**2. Projection:**
$$ \mathbf{F}_s = \mathbf{s} \mathbf{W}_s, \quad \mathbf{F}_q = \mathbf{Q} \mathbf{W}_q \quad (4) $$
where $\mathbf{W}_s, \mathbf{W}_q \in \mathbb{R}^{d \times d}$.

**3. Attention Computation:** We compute the similarity between proficiency and acoustic features using cosine-based attention:
$$ A(\mathbf{F}_s, \mathbf{F}_q) = \sigma \left( \frac{\mathbf{F}_s \mathbf{F}_q^\top}{||\mathbf{F}_s|| ||\mathbf{F}_q||} \right) \in \mathbb{R}^{L \times L} \quad (5) $$

## 3 Experiments

### 3.1 Datasets

The Speechocean762 dataset [18] is used for evaluation. It consists of 5,000 English utterances from 250 Mandarin speakers, split equally into training and test sets.

* **Granularities:** Annotations exist at the utterance level (accuracy, fluency, completeness, prosody, total), word level (accuracy, stress, total), and phoneme level (accuracy).
* **Scores:** Utterance and word levels use a 0–10 scale, while phoneme accuracy is 0–2. In this study, we re-scale all scores to 0–2 for consistency.
* **MDD Task:** The dataset includes 46 phones, with `[unk]` for unknown phones and `[DEL]` for deletion errors.

### Table 1: Statistical Summary of the Speechocean762 Dataset

| Metric | Train | Test |
| :--- | :--- | :--- |
| Speakers | 125 | 125 |
| Duration (h) | 2.7 | 2.8 |
| Utterances | 2,500 | 2,500 |
| Words | 15,849 | 15,967 |
| Phones | 47,076 | 47,369 |
| L1 | Chinese | Chinese |

### 3.2 Experimental Settings
Our framework is built upon **Hmamba** [3]. We employ the Adam optimizer with a tri-phase learning rate scheduler. The initial learning rate is set to 2e-3, except for the utterance-level APA module where it is 9e-5.

To address the class imbalance in MDD (where correct pronunciations significantly outnumber errors), we adapt the **Decoupled Cross-Entropy Loss (deXent)**:
$$ \mathcal{L}_{MDD} = \mathcal{L}_{Xent}^{hit} + \left( \frac{\mu^h}{\mu^m} \right)^\alpha \mathcal{L}_{Xent}^{mis} \quad (6) $$
where $\mathcal{L}_{Xent}^{hit}$ and $\mathcal{L}_{Xent}^{mis}$ are cross-entropy losses for correct ($\mathcal{H}$) and mispronounced ($\mathcal{M}$) segments, $\mu^h$ and $\mu^m$ are their respective frequencies in the training data, and $\alpha$ controls re-weighting intensity (set to 0.5).

Evaluation metrics include PCC and MSE for APA, and precision, recall, F1-score, and PER (Phone Error Rate) for MDD. Results are averaged over 5 independent trials of 20 epochs.

## 4 Experimental Results and Discussion

### 4.1 ABM Placement Matters: Feature Fusion Strategies Analyzed
As shown in **Table 2**, our approach significantly outperforms the GOPT [9] and HMamba [3] baselines, particularly in suprasegmental metrics at the word and utterance levels.

### Table 2: Overall Performance of APA

| Model | Utt-Prosodic↑ | Utt-Fluency↑ | Utt-Complete↑ | Word-Stress↑ | Phone-Total↑ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| GOPT | 0.760 | 0.753 | 0.155 | 0.291 | 0.612 |
| HMamba | 0.835 | 0.842 | 0.172 | 0.313 | 0.736 |
| **Our Method (prosodyEmb)** | **0.862** | **0.862** | **0.262** | **0.362** | **0.720** |

**Key Findings:**
*   **Word-level Stress:** The explicitly designed stress features (formants + spectral balance) provide a significant advantage. This stems from targeted modeling of $F_1/F_2$ vowel structures and energy distribution in specific bands (500–1000 Hz), which are often overlooked by generic SSL features.
*   **Utterance Completeness:** Continuous pitch contours demonstrate strong explanatory power here. From a linguistic-cognitive perspective, pitch reset (the rise at the start of a turn) and pitch convergence signal semantic completion.
*   **Synergy:** Configuration 5 (prosodyEmb + proficiency features) achieves the best results, showing that the ABM mechanism effectively adjusts interaction intensity between acoustic features and learner proficiency.

### 4.2 Effectiveness of Different Prosody-Related Features
Ablation studies in **Table 3** highlight the contribution of each prosodic feature to suprasegmental scoring and segmental MDD.

### Table 3: Impact of Supersegmental Features on Score and MDD

| Model | Stress (PCC) | Fluency (PCC) | Complete (PCC) | MDD F1↑ | MDD PER↓ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| prosodyEmb | 0.362 | 0.862 | 0.262 | 63.05% | 3.16 |
| $-$dur | 0.362 | 0.857 | 0.197 | 60.08% | 3.72 |
| $-$dur-energy | 0.372 | 0.857 | **0.314** | **63.92%** | **3.09** |


Removing duration and energy actually improves MDD performance (F1=63.92%), suggesting that while these features help prosodic scoring, they can introduce noise or conflicts during local phoneme recognition.

## 5 Conclusion

This study enhances CAPT performance by explicitly incorporating stress-related features (formants, spectral balance, pitch) and accounting for learner proficiency through an adaptive ABM mechanism. Experimental results demonstrate consistent improvements in word-level stress and utterance-level completeness. Future work will extend this to cross-linguistic formant normalization and tone-stress interaction modeling.

---

## References

1. Bentum, M., ten Bosch, L., Lentz, T.: The processing of stress in end-to-end automatic speech recognition models. In: Proc. INTERSPEECH (2024).
2. Bo, Y., Zhu, Y., Li, L., Zhang, H.: Famnet: frequency-aware matching network for cross-domain few-shot medical image segmentation. arXiv:2412.09319 (2024).
3. Chao, F.A., Chen, B.: Towards efficient and multifaceted computer-assisted pronunciation training leveraging hierarchical selective state space model and decoupled cross-entropy loss (2025).
4. Chen, L., Yang, Y., Wang, D., Li, X.: 3m: a multi-view multi-task multi-level framework for automatic pronunciation assessment. In: Proc. ACM MM (2022).
5. Chen, X., Liu, Y., Zhang, Z.: Vocabulary-phonology interdependence in l2 speech evaluation. In: Proc. INTERSPEECH (2022).
6. Do, H., Kim, Y., Lee, G.G.: Hierarchical pronunciation assessment with multi-aspect attention. In: Proc. ICASSP (2023).
7. Dong, B., Zhao, Q., Zhang, J., Yan, Y.: Automatic assessment of pronunciation quality. In: Proc. ISCSLP (2004).
8. Flege, J.E.: Second-language speech learning: theory, findings, and problems. In: Speech Perception and Linguistic Experience (1995).
9. Gong, Y., Chen, Z., Chu, I.H., Chang, P., Glass, J.: Transformer-based multi-aspect multi-granularity non-native english speaker pronunciation assessment. In: Proc. ICASSP (2022).
10. Kim, S.: Proficiency-aware neural models for pronunciation scoring. In: Proc. ACL (2023).
11. Lee, H., Wang, Y.: L2 proficiency effects in pronunciation assessment. ACM Trans. Speech Lang. Process. 14(3) (2021).
12. Li, Y., Ling, Z., Liu, Q., Fan, Y., Wang, D.: A multi-aspect multi-granularity pronunciation assessment method based on branchformer encoder and hierarchical aggregation. In: Proc. MMM (2025).
13. Lin, B., Wang, L., Feng, X., Zhang, J.: Automatic scoring at multi-granularity for l2 pronunciation. In: Proc. INTERSPEECH (2020).
14. Lin, C.H., Chen, N., Wang, Y.: L2-arctic: benchmarking mispronunciation detection for capt systems. In: Proc. ICMI (2022).
15. Pei, H.C., Fang, H., Luo, X., Xu, X.: Gradformer: a framework for multi-aspect multi-granularity pronunciation assessment. IEEE/ACM Trans. Audio Speech Lang. Process. 32 (2023).
16. Povey, D., Ghoshal, A., Boulianne, G., et al.: The kaldi speech recognition toolkit. In: IEEE ASRU (2011).
17. Zhang, J., Liu, H., Wang, Y.: 3hm: a hierarchical heterogeneous multi-modal model for automatic pronunciation scoring. In: Proc. ACM MM (2023).
18. Zhang, J., et al.: speechocean762: an open-source non-native English speech corpus for pronunciation assessment. In: Proc. INTERSPEECH (2021).
