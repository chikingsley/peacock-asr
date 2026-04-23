# Peacock ASR Research & Architecture Scratchpad

This document traces the architectural evolution of Pronunciation Assessment models based on the repository's paper history, diagnosing the specific limitations of recent experiments (like HMamba + HConv) and mapping concrete hypotheses for the next iteration of the Peacock ASR pipeline.

## 1. The Ground Truth: SpeechOcean762 (Zhang et al. 2021)
To understand why models succeed or fail, we must look at what SpeechOcean762 actually measures. It is not a single generic score. It requires a model to successfully decouple and predict:
*   **Phoneme-level:** Accuracy (0-2 scale). Heavily reliant on local phonetic characteristics and alignment.
*   **Word-level:** Accuracy (0-10) and **Stress (0-10)**. Stress requires understanding *relative* changes in pitch, duration, and energy between syllables.
*   **Sentence-level:** Accuracy, Fluency, Completeness, Prosody. Requires global contextual awareness.

## 2. The Evolutionary Tree

### Gen 1: Baseline (GOPT - Gong et al. 2022)
*   **Approach:** Extract Goodness of Pronunciation (GOP) via ASR forced alignment -> Pass to a standard Transformer for multi-task regression across the SpeechOcean metrics.
*   **Limitation:** Transformers scale quadratically and treat all features with global attention, often smearing local phonetic boundaries.

### Gen 2: The Yan et al. Era (ConPCO -> Muffin, 2024-2025)
*   **Approach:** Attempted to fix the "smearing" by preserving phoneme characteristics. ConPCO used Contrastive Ordinal Regularization. Muffin culminated in Interactive Hierarchical Neural Modeling.
*   **Limitation:** While highly accurate (SOTA on SpeechOcean), Muffin became architecturally bloated (requiring multiple SSL models), making it too heavy for fast, interactive inference in a real application.

### Gen 3: The State-Space Shift (HMamba - Chao 2025 & JCAPT Mamba - Yang 2025)
*   **Approach:** Replace the quadratic Transformer with Mamba (Selective State Space Models). Mamba processes sequences in linear time while maintaining infinite context. HMamba specifically decoupled the cross-entropy loss (deXent) for the Mispronunciation Detection (MDD) task, achieving 63.85% F1.
*   **The HConv Failure:** When trying to port hierarchical convolution (HConv) optimizations into the Mamba backbone, performance tanked. 
*   **Diagnosis of Failure:** Mamba is incredible at state-tracking, but standard 1D convolutions inside an HConv block applied to pure SSL/GOP features fail to capture *suprasegmental* prosody. SSL models (like Wav2Vec2/HuBERT) are trained to be speaker- and pitch-invariant. If you feed pitch-invariant features into a Mamba block, it literally cannot learn Word Stress, resulting in the garbage PCC scores observed.

### Gen 4: Recovering Suprasegmental Fidelity (The 2026 Papers)
The most recent papers explicitly solve the exact problem encountered in the HMamba/HConv experiments: capturing Word Stress and Prosody without reverting to Muffin's bloated architecture.

*   **Architectural Fix (Zhao et al. 2026 - Cwacformer):** Instead of standard HConv, Zhao uses **Multi-scale 1D Convolutions** (Kernels 1, 3, and 5) explicitly at the *word-level aggregator*. Stress is relative; kernel 1 captures the point, kernel 3 captures local transition, kernel 5 captures the broad prosodic arc of the word. They achieved a massive jump in Word Stress PCC (0.483).
*   **Feature Fix (Bao et al. 2026):** Bao recognized that SSL features strip out stress. Their solution was to explicitly extract hard DSP features—**Vowel Formants (F1/F2), Spectral Balance, and continuous pitch (PyWORLD)**—into a 6D vector. They then use an Attention-Based Matching (ABM) module to fuse these raw acoustic correlates with the segmental GOP features.

## 3. The "Open Response" Paradigm (Chen et al. 2023 - MultiPA)
For scenarios where there is no exact target text (e.g., a user freely speaking to a chatbot):
*   **Approach:** Run ASR (Whisper) to get a transcript. Treat a larger Whisper model's output as the "target". Use an aligner (Charsiu) and a semantic model (RoBERTa).
*   **Relevance:** This proves that you can run multi-aspect scoring dynamically, but it heavily relies on the ASR pipeline's raw acoustic confidence, not just pre-calculated transcripts.

---

## 4. Hypotheses & Engineering Directions for Peacock ASR

Based on the repos (P003-compact-backbones, P004-training-from-scratch, P013-hmamba-faithful), here are the synthesized research directions:

### Hypothesis 1: The HMamba Word-Stress Rescue
The HMamba architecture is sound, but its word-level aggregator is blind to stress because it only receives localized GOP/SSL features. 
**Action:** Inject Zhao's Multi-scale Convolutions (kernels 1, 3, 5) strictly at the word-level fusion step inside `P013-hmamba-faithful`. This will give the Mamba backbone the varied temporal receptive fields needed to detect stress variations without adding the weight of a full Transformer.

### Hypothesis 2: Explicit Acoustic Correlate Injection (The Bao Method)
We cannot rely solely on HuBERT/Wav2Vec2 for prosody. 
**Action:** Build a parallel, lightweight DSP extraction pipeline (F1/F2 formants + PyWORLD pitch). Concatenate this 6D vector with the GOP features *before* passing them into the HMamba/HConv layers. This explicitly hands the model the acoustic correlates of English stress, bypassing the SSL bottleneck.

### Hypothesis 3: P003/P004 Native Prosody
The long-term goal of training a compact backbone from scratch (Citrinet/Conformer in P003/P004) is structurally superior if we design it to output these acoustic correlates natively.
**Action:** Modify the CTC-encoder in P004 to not just output phoneme posteriors (for GOP), but to have a parallel multi-task head that predicts continuous pitch and spectral balance. This creates a single, ultra-fast ASR backbone that outputs exactly the dense feature vector the pronunciation scoring head needs, eliminating the need for 3rd-party alignment tools or heavy SSLs.
