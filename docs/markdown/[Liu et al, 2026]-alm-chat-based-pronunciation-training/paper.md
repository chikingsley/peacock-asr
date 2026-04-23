---
title: "Unlocking Large Audio-Language Models for Interactive Language Learning"
authors:
  - "Hongfu Liu"
  - "Zhouying Cui"
  - "Xiangming Gu"
  - "Ye Wang"
citation_author: "Liu et al."
year: 2026
doi: null
pages: 24
journal: "ACL 2026"
source_pdf: "paper.pdf"
extraction_method: "Direct LaTeX Transcription"
extracted_at: "2026-04-18"
llm_friendly: true
---

## Abstract

Achieving pronunciation proficiency in a second language (L2) remains a challenge, despite the development of Computer-Assisted Pronunciation Training (CAPT) systems. Traditional CAPT systems often provide unintuitive feedback that lacks actionable guidance, limiting its effectiveness. Recent advancements in audio-language models (ALMs) offer the potential to enhance these systems by providing more user-friendly feedback. In this work, we investigate ALMs for chat-based pronunciation training by introducing **L2-Arctic-plus**, an English dataset with detailed error explanations and actionable suggestions for improvement. We benchmark cascaded ASR+LLMs and existing ALMs on this dataset, specifically in detecting mispronunciation and generating actionable feedback. To improve the performance, we further propose to instruction-tune ALMs on L2-Arctic-plus. Experimental results demonstrate that our instruction-tuned models significantly outperform existing baselines on mispronunciation detection and suggestion generation in terms of both objective and human evaluation, highlighting the value of the proposed dataset. Code is publicly available at [https://github.com/zoeyada/ALMs4Learning](https://github.com/zoeyada/ALMs4Learning).

## 1 Introduction

The acquisition of a second language (L2) is a fundamental necessity in bilingual and multilingual communities. However, attaining a high level of proficiency in pronunciation and language usage remains a considerable challenge for L2 learners. Computer-Assisted Pronunciation Training (CAPT) systems have been developed as effective tools to support L2 learners by detecting, diagnosing, and assessing mispronunciations (Eskenazi, 2009; Rogerson-Revell, 2021). Conventional CAPT systems primarily focus on providing detailed feedback at the phoneme, word, and utterance levels for mispronunciation detection and fluency evaluation (Witt and Young, 2000; Zhang et al., 2021a; El Kheir et al., 2023), thereby facilitating targeted practice and enabling learners to enhance their language skills through systematic error correction.

Despite significant achievements in developing robust models for mispronunciation detection and pronunciation assessment, existing methods primarily provide location-based diagnostic feedback (Xu et al., 2021) and score-based assessment feedback (Gong et al., 2022). However, such feedback is often unintuitive and challenging for L2 learners to interpret, particularly in terms of actionable suggestions for improvement. Recent advances in large-scale speech-language models and audio-language models (ALMs) have demonstrated remarkable performance across various speech and audio-related tasks, including automatic speech recognition (ASR), speech synthesis, and spoken dialogue systems (Chu et al., 2023; Chu et al., 2024; Zhang et al., 2023; Huang et al., 2024; Deshmukh et al., 2023). Nevertheless, their application in interactive language learning, particularly for the complex task of chat-based pronunciation training, remains largely unexplored. The integration of language models presents an opportunity to enhance acoustic analysis by providing user-friendly feedback, such as text-based explanations of pronunciation errors along with actionable suggestions for improvement, as shown in **Figure 1**.

In this work, we investigate the potential of large ALMs as language instructors to enhance language learning, with a particular emphasis on *chat-based pronunciation training*. Our goal is to provide interpretable, text-based feedback that includes detailed error explanations and actionable suggestions. To facilitate this task, we introduce **L2-Arctic-plus**, an extension of the L2-Arctic dataset (Zhao et al., 2018), which incorporates text-based annotations for error explanations and actionable suggestions. Furthermore, we examine the application of the cascaded ASR+LLM framework for chat-based pronunciation training. Our analysis reveals that ASR models often rectify pronunciation errors in the input, yielding an accurate transcription for LLMs and thereby limiting LLMs' ability to detect pronunciation errors from the original audio. Additionally, our evaluation of existing large ALMs on this task indicates their significant limitations in both accurate mispronunciation detection and actionable feedback generation. As a consequence, we propose to improve chat-based pronunciation training by instruction-tuning ALMs using the L2-Arctic-plus training set. Experimental results demonstrate that our instruction-tuned ALM outperforms existing baselines, achieving substantial improvements in chat-based pronunciation training.

Our key contributions are summarized below:

* We construct L2-Arctic-plus, a novel benchmark designed for chat-based pronunciation training in interactive language learning. This dataset is specifically developed for audio-language models and includes text-based annotations on pronunciation error explanations and actionable corrective suggestions.
* We systematically analyze the performance of ASR+LLM cascades and existing ALMs in chat-based pronunciation training. We further improve this novel task by instruction-tuning the ALMs on a curated training set of L2-Arctic-plus, demonstrating significant improvements in both mispronunciation detection and feedback generation.
* This work expands the capability scope of ALMs in the domain of chat-based pronunciation training, addressing an important gap in language learning.

## 2 Related Work

**Audio-Language Modeling.** The development of multimodal large language models has recently expanded beyond vision-based modalities to include audio and video, leading to increased research interest in audio-language models. Prominent models such as Qwen-Audio (Chu et al., 2023), Qwen2-Audio (Chu et al., 2024), SpeechGPT (Zhang et al., 2023), AudioGPT (Huang et al., 2024), Pengi (Deshmukh et al., 2023), and GPT-4o (OpenAI, 2024) demonstrate remarkable versatility, addressing a wide array of downstream tasks, including speech, sound, and music processing. These efforts seek to unify diverse audio-related tasks within a single foundation model. Despite their impressive capabilities, these models have limited applications in pronunciation detection, a critical task in language learning. Notably, prior acoustic models have demonstrated effectiveness in pronunciation detection tasks (Hu et al., 2015; Xu et al., 2021; Korzekwa et al., 2021), highlighting the gap in current audio-language models for educational applications.

**Computer-Assisted Pronunciation Training.** CAPT has become an essential component of modern language learning, leveraging technological advancements to enhance learners' pronunciation proficiency. Early CAPT systems primarily relied on repetitive drills and rudimentary feedback mechanisms, utilizing basic audio playback and recording features (Amrate and Tsai, 2024). The introduction of ASR technology has enabled more interactive and adaptive training environments, facilitating real-time feedback on pronunciation (Arora et al., 2018; Henrichsen, 2021; Liu et al., 2024). More recently, CAPT systems have further integrated machine learning to deliver more sophisticated feedback, encompassing the evaluation of prosodic features such as intonation, stress, and rhythm (Eskenazi, 2009; Rogerson-Revell, 2021). Contemporary CAPT methodologies emphasize detailed assessments at the phoneme, word, and utterance levels (Gong et al., 2022; El Kheir et al., 2023; Liu et al., 2023), enabling learners to accurately distinguish and produce specific consonants and vowels while addressing suprasegmental features like stress patterns, intonation, and rhythm. However, existing CAPT approaches often lack comprehensive and interpretable feedback, underscoring the need for further advancements to enhance the effectiveness of pronunciation training systems.

## 3 Interactive Language Learning

### 3.1 Problem Statement

This study focuses on *chat-based pronunciation training* within the context of interactive language learning. In this framework, the user is instructed to read a canonical text sequence, denoted as $\boldsymbol{W}_{1:N}$, where $N$ represents the total number of words. The user's speech is then recorded as an audio sample, $\boldsymbol{X}_A$. The primary objective of the chat-based pronunciation training system, denoted as $f_{\theta}(\cdot)$, where $\theta$ represents model parameters, is to generate text-based responses: $\boldsymbol{Y}_R=f_{\theta}(\boldsymbol{X}_A)$. This response is designed to identify mispronunciation in the user's speech and provide corresponding actionable suggestions for improvement through an interactive chat-based interface.

### 3.2 Dataset Curation of L2-Arctic-plus

Since no existing datasets are specifically designed for chat-based pronunciation training, especially without ground-truth responses $\boldsymbol{Y}_R$, we introduce **L2-Arctic-plus** as a benchmark for this task. L2-Arctic-plus is built upon the L2-Arctic dataset (Zhao et al., 2018), a non-native English corpus designed for mispronunciation detection with frame-level annotations. The original L2-Arctic dataset consists of speech recordings from 24 non-native English speakers (12 males, 12 females) with diverse native languages including Hindi, Korean, Mandarin, Spanish, Arabic, and Vietnamese.

Following prior practices in Peng et al. (2021), Feng et al. (2020), and Yang et al. (2022), we select the same 900 samples as the evaluation set. Each sample comprises a speech recording $\boldsymbol{X}_A$ along with manual annotations, including canonical word sequences $\{\boldsymbol{W}_n\}^N_{n=1}$, a binary mispronunciation indicator $\boldsymbol{D} \in \{0,1\}$ — where $\boldsymbol{D}(\boldsymbol{W}_n)=1$ denotes that the $n$-th word $\boldsymbol{W}_n$ is mispronounced — and a mispronunciation type indicator $\boldsymbol{E} \subseteq \{S, D, I\}$. Here, $\boldsymbol{E}(\boldsymbol{W}_n)$ represents the set of mispronunciation types (Substitution, Deletion, or Insertion) present in the $n$-th word $\boldsymbol{W}_n$, with $\boldsymbol{D}(\boldsymbol{W}_n) = 0$ if no mispronunciation is detected $\boldsymbol{E}(\boldsymbol{W}_n) = \emptyset$. The annotations are based on phonemes, so a single word may contain multiple phonemic errors which may belong to different types. In these annotations, the mispronounced phonemes and their corresponding error types are clearly marked. Based on these existing annotations, we illustrate how to construct new ground-truth responses $\boldsymbol{Y}_R$ following a coarse-to-fine manner through a two-stage process.

In the first stage, we generate initial responses by formulating a structured prompt and utilizing the existing annotations as input to query GPT-4o (OpenAI, 2024). The model generates feedback that includes both mispronunciation error explanations and corrective suggestions. Specifically, the response is structured as a sequence of word-level error-suggestion pairs $\boldsymbol{Y}_R = \{\boldsymbol{W}^{(l)}\colon[{\boldsymbol{Y}_E}^{(l)}, {\boldsymbol{Y}_S}^{(l)}]\}^L_{l=1}$, where $\boldsymbol{W}^{(l)}$ represents the $l$-th mispronounced word $\boldsymbol{D}(\boldsymbol{W}^{(l)})=1$, ${\boldsymbol{Y}_E}^{(l)}$ refers to a text-based explanation of the mispronunciation type and ${\boldsymbol{Y}_S}^{(l)}$ represents a corrective suggestion on how to improve the pronunciation given this error explanation ${\boldsymbol{Y}_E}^{(l)}$. The total number of pairs, $L$, corresponds to the total number of mispronounced words $L=\sum_{n=1}^N\boldsymbol{D}(\boldsymbol{W}_n)$.

In the second stage, three human annotators are involved to verify GPT-4o-generated responses in terms of the correctness of both error explanation and corrective suggestion $[{\boldsymbol{Y}_E}^{(l)}, {\boldsymbol{Y}_S}^{(l)}]$. If any responses contain incorrect explanations or inappropriate suggestions, we prompt GPT-4o to regenerate new responses, followed by another round of human verification. The final verified responses constitute the ground-truth annotations in L2-Arctic-plus.

### 3.3 Evaluation Protocols

This subsection outlines the evaluation protocols for assessing a chat-based pronunciation training system $f_{\theta}(\cdot)$ on the L2-Arctic-plus dataset. Given a generated response $\hat{\boldsymbol{Y}}_R=\{\hat{\boldsymbol{W}}^{(l)}\colon[\hat{\boldsymbol{Y}}_E^{(l)}\textrm{,}\,\hat{\boldsymbol{Y}}_S^{(l)}]\}_{l=1}^{\hat{L}}$ and a reference response $\boldsymbol{Y}_R$, the evaluation consists of both objective and subjective assessments. Objective evaluation measures performance in mispronunciation detection and feedback generation, while subjective evaluation involves human judgment.

**Mispronunciation Detection Evaluation.**
To evaluate mispronunciation detection, we compute standard classification metrics: True Positives (TP), False Positives (FP), False Negatives (FN), and True Negatives (TN). Unlike prior acoustic-only approaches with frame-level evaluation (Xu et al., 2021), our framework adopts a **word-level** evaluation scheme:
$$ \text{TP} = \sum_{n=1}^{N} \mathbb{I}(\boldsymbol{D}(\hat{\boldsymbol{W}}_n) = 1 \wedge \boldsymbol{D}(\boldsymbol{W}_n) = 1)\textrm{;} $$
$$ \text{FP} = \sum_{n=1}^{N} \mathbb{I}(\boldsymbol{D}(\hat{\boldsymbol{W}}_n) = 1 \wedge \boldsymbol{D}(\boldsymbol{W}_n) = 0)\textrm{;} $$
$$ \text{FN} = \sum_{n=1}^{N} \mathbb{I}(\boldsymbol{D}(\hat{\boldsymbol{W}}_n) = 0 \wedge \boldsymbol{D}(\boldsymbol{W}_n) = 1)\textrm{;} $$
$$ \text{TN} = \sum_{n=1}^{N} \mathbb{I}(\boldsymbol{D}(\hat{\boldsymbol{W}}_n) = 0 \wedge \boldsymbol{D}(\boldsymbol{W}_n) = 0)\textrm{.} $$

We report Precision, Recall, and F1-score, computed across all samples rather than averaging per entry. Additionally, we introduce a new metric **Extra Words Ratio (EWR)** to evaluate the system's tendency to introduce spurious words absent from the canonical text $\boldsymbol{W}_{1:N}$. Specifically, EWR is defined as follows:
$$ \mathrm{EWR} = \frac{1}{M}\sum_{j=1}^{M} \mathbb{I}(\hat{\boldsymbol{W}}_j \notin \{\boldsymbol{W}_n\}^N_{n=1})\textrm{,} $$
where $M$ is the total number of words predicted by the system. A higher EWR indicates a greater tendency to hallucinate non-existent words, reflecting lower system reliability in mispronunciation detection.

**Feedback Generation Evaluation.**
To assess the quality of generated feedback, we compare the system-generated error-suggestion pairs $\{\hat{\boldsymbol{W}}^{(l)}\colon[\hat{\boldsymbol{Y}}_E^{(l)}\textrm{,}\,\hat{\boldsymbol{Y}}_S^{(l)}]\}_{l=1}^{\hat{L}}$ against the referenced ground-truth pairs $\{\boldsymbol{W}^{(l)}\colon[{\boldsymbol{Y}_E}^{(l)}, {\boldsymbol{Y}_S}^{(l)}]\}^L_{l=1}\}$. For objective evaluations, we calculate metrics: **BLEU-2** (Papineni et al., 2002), measuring 2-gram overlap between system outputs and ground truth; **ROUGE-L** (Lin, 2004), measuring the longest common subsequence; and **BERTScore** (Zhang et al., 2019b), calculating semantic similarity leveraging contextual embeddings. Additionally, we conduct subjective human evaluations to assess the suggestion relevance, interpretability, and helpfulness of the generated feedback.

## 4 Investigating ASR+LLMs Cascade

LLMs have been increasingly integrated into speech-related tasks such as ASR (Ma et al., 2024; Geng et al., 2024). Since LLMs cannot directly process audio input, a common approach is to employ a pre-trained ASR model to transcribe speech into text, enabling LLMs to handle downstream tasks. This section explores the potential of the ASR+LLMs cascade for chat-based pronunciation training.

### 4.1 Cascaded ASR+LLM Framework

**ASR-based Transcription.** In this framework, we utilize a pre-trained ASR model to transcribe the given speech recordings $\boldsymbol{X}_{A}$ into text $\hat{\boldsymbol{W}}_{1:\hat{N}}$. We assume that mispronounced words would be transcribed into incorrect words, thus allowing LLMs to infer mispronunciation errors based on these transcription inconsistencies.

**LLM-based In-Context Learning.** To enable LLMs to detect mispronunciation and generate feedback, we prompt LLMs to conduct in-context learning using a one-shot demonstration. LLMs are provided with the canonical text $\boldsymbol{W}_{1:N}$ alongside the ASR transcription $\hat{\boldsymbol{W}}_{1:\hat{N}}$, and one example illustrating how to identify mispronunciations by comparing text discrepancies.

### 4.2 Evaluation Results

We evaluate instruct versions of Mistral-7B (Jiang et al., 2023) and Llama-3.1-8B (Dubey et al., 2024) paired with various sizes of Whisper (Radford et al., 2022) and Wav2vec2 (Baevski et al., 2020).

**Table 1: Performance comparisons of different cascaded ASR+LLM frameworks.**

| ASR Model | LLM | Precision | Recall | F1 | EWR | BLEU-2 | ROUGE-L | BERTScore |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Whisper Small | Mistral-7B | 53.6 | 4.9 | 9.0 | 0.3 | 4.5 | 7.0 | 79.8 |
| Whisper Medium | Mistral-7B | 48.2 | 4.0 | 7.4 | 0.3 | 4.6 | 7.1 | 79.8 |
| Whisper Large | Mistral-7B | 48.9 | 3.4 | 6.4 | 0.1 | 4.1 | 6.1 | 79.5 |
| Wav2vec2 Base | Mistral-7B | 52.8 | 6.8 | 12.1 | 0.4 | 5.0 | 8.5 | 80.5 |
| Wav2vec2 Large | Mistral-7B | 51.2 | 4.5 | 8.3 | 0.3 | 4.7 | 7.2 | 79.9 |
| Whisper Small | Llama-3.1-8B | 53.3 | 12.1 | 19.7 | 0.9 | 6.6 | 12.8 | 82.1 |
| Whisper Medium | Llama-3.1-8B | 51.9 | 10.2 | 17.0 | 1.0 | 5.8 | 11.5 | 81.7 |
| Whisper Large | Llama-3.1-8B | 52.8 | 8.4 | 14.5 | 0.7 | 5.5 | 10.7 | 81.4 |
| Wav2vec2 Base | Llama-3.1-8B | 53.8 | 17.8 | 26.8 | 1.1 | 7.3 | 15.0 | 83.0 |
| Wav2vec2 Large | Llama-3.1-8B | 57.9 | 11.8 | 19.6 | 0.7 | 6.3 | 11.9 | 81.8 |

**Stronger ASR models degrade detection performance.** Surprisingly, smaller ASR models (Whisper Small, Wav2vec2 Base) outperform their larger counterparts in F1 score. We conjecture that stronger ASR models tend to "correct" pronunciation errors during transcription due to their robustness, preventing them from accurately reflecting learners' actual speech errors.

**Stronger LLMs improve performance.** Llama-3.1-8B consistently outperforms Mistral-7B, achieving up to 121.5% relative improvement in F1 score. However, it also displays higher EWR, indicating an increased propensity for hallucination.

> **Limitations:** ASR models discard acoustic information in their text outputs, restricting LLMs from further understanding input speech nuances.

## 5 Investigating Existing ALMs

ALMs integrate an audio encoder and an LLM, projecting audio representations into the text embedding space to preserve acoustic information.

### 5.1 Employed ALMs

We evaluate Pengi, SpeechGPT, Qwen-Audio, Qwen2-Audio, and GPT-4o-Audio.

### 5.2 Evaluation Results

**Failure of Pengi and SpeechGPT.** Only Qwen-Audio, Qwen2-Audio, and GPT-4o-Audio successfully follow instructions. Pengi and SpeechGPT either generate irrelevant responses or misinterpret the task as pure ASR.

**ALMs outperform cascaded ASR+LLM.** End-to-end ALMs outperform cascaded approaches, demonstrating the value of latent audio representations.

**Table 2: Performance comparisons of existing ALMs.**

| ALM | Precision | Recall | F1 | EWR | BLEU-2 | ROUGE-L | BERTScore |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen-Audio | 50.4 | 18.7 | 27.2 | 0.7 | 3.9 | 11.8 | 82.7 |
| Qwen2-Audio | 41.7 | 22.0 | 28.8 | 2.0 | 6.9 | 18.3 | 82.9 |
| GPT-4o-Audio | 52.7 | 41.3 | 46.3 | 0.2 | 10.9 | 22.3 | 86.0 |

> **Limitations:** Open-source ALMs still lag behind GPT-4o-Audio as they are not explicitly trained for mispronunciation detection.

## 6 Instruction Tuning ALMs for Interactive Language Learning

We enable end-to-end ALMs to perform this task by constructing a synthesized training dataset and facilitating modality fusion through two-stage instruction tuning.

### 6.1 Speech Instruction Tuning

**Stage 1: Acoustic feature alignment.** We sample 200k audio-text pairs from CommonVoice (Ardila et al., 2020). Only the projector is trainable at this stage to establish basic speech understanding.

**Stage 2: Task-specific instruction tuning.** We construct 2.7k prompt-response pairs based on L2-Arctic (excluding L2-Arctic-plus samples). We fine-tune both the projector and the LLM backbone using LoRA (Hu et al., 2021).

### 6.2 Evaluation Results

**Instruction-tuned ALMs outperform baseline methods.** Our models surpass the best ASR+LLM cascade and generic ALMs, achieving up to 134.3% and 35.6% relative F1 improvements.

**Table 3: Performance of instruction-tuned ALMs vs SOTA baselines.**

| Audio Encoder | LLM | Precision | Recall | F1 | EWR | BLEU-2 | ROUGE-L | BERTScore |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Wav2vec2+Llama3 | (Cascade) | 53.8 | 17.8 | 26.8 | 1.1 | 7.3 | 15.0 | 83.0 |
| GPT-4o-Audio | (ALM) | 52.7 | 41.3 | 46.3 | 0.2 | 10.9 | 22.3 | 86.0 |
| Whisper Small | Mistral-7B | 50.5 | 65.5 | 57.1 | 0.0 | 17.4 | 25.9 | 85.7 |
| Whisper Medium | Mistral-7B | 51.6 | 78.2 | 62.1 | 0.0 | 19.7 | 30.7 | 87.2 |
| Whisper Large | Mistral-7B | 50.6 | 81.8 | 62.5 | 0.0 | 20.1 | 30.5 | 87.2 |
| Whisper Small | Llama-3.1-8B | 49.7 | 68.2 | 57.5 | 0.0 | 17.2 | 25.4 | 85.5 |
| Whisper Medium | Llama-3.1-8B | 51.2 | 78.3 | 61.9 | 0.0 | 20.4 | 31.9 | 87.4 |
| Whisper Large | Llama-3.1-8B | 48.9 | 87.7 | 62.8 | 0.0 | 20.0 | 30.5 | 87.3 |

**Instruction tuning mitigates hallucination.** EWR is reduced to 0.0, indicating that tuning reinforces focus on the canonical text.

**Larger audio encoders yield improved performance.** Larger encoders provide better embedding spaces for effective fine-tuning.

### 6.3 LLM-as-a-Judge

Using GPT-4o as a reference-guided evaluator:

* **Average Score (1-5):** Ours (2.328) > GPT-4o-Audio (2.145) > Cascade (1.426).
* **Win Rate vs Cascade:** 96.55%.
* **Win Rate vs GPT-4o-Audio:** 80.78%.

### 6.4 Human Evaluation

Seven participants rated models on Suggestion Relevance (SR), User Understandability (UU), and Overall Evaluation (OE). Our model outperformed GPT-4o-Audio by 24.2% in SR, 8.5% in UU, and 21.5% in OE.

## 7 Conclusion

We explored the potential of ALMs for chat-based pronunciation training. By introducing L2-Arctic-plus and performing task-specific instruction tuning, we demonstrated that end-to-end ALMs can provide more accurate and actionable feedback for L2 learners than traditional systems.

### Limitations

1. **Scope:** Currently limited to "reading-aloud" scenarios; future work should include free-form conversation.
2. **Modality:** Feedback is text-only; future models should generate auditory references (synthesized speech).

---

## References

* Amrate, M., and Tsai, P. 2024. Computer-assisted pronunciation training: A systematic review. *ReCALL*.
* Ardila, R., et al. 2020. Common Voice: A Massively-Multilingual Speech Corpus. *LREC*.
* Arora, V., et al. 2018. Phonological feature-based speech recognition system for pronunciation training. *JASA*.
* Baevski, A., et al. 2020. wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations. *NeurIPS*.
* Chu, Y., et al. 2023. Qwen-audio: Advancing universal audio understanding via unified large-scale audio-language models. *arXiv:2311.07919*.
* Chu, Y., et al. 2024. Qwen2-audio technical report. *arXiv:2407.10759*.
* Deshmukh, S., et al. 2023. Pengi: An audio language model for audio tasks. *NeurIPS*.
* Dubey, A., et al. 2024. The llama 3 herd of models. *arXiv:2407.21783*.
* El Kheir, Y., et al. 2023. Automatic Pronunciation Assessment--A Review. *arXiv:2310.13974*.
* Eskenazi, M. 2009. An overview of spoken language technology for education. *Speech Communication*.
* Feng, Y., et al. 2020. SED-MDD: Towards Sentence Dependent End-To-End Mispronunciation Detection and Diagnosis. *ICASSP*.
* Geng, X., et al. 2024. Unveiling the Potential of LLM-Based ASR on Chinese Open-Source Datasets. *ISCSLP*.
* Gong, Y., et al. 2022. Transformer-based multi-aspect multi-granularity non-native English speaker pronunciation assessment. *ICASSP*.
* Hu, E., et al. 2021. LoRA: Low-Rank Adaptation of Large Language Models. *arXiv:2106.09685*.
* Hu, W., et al. 2015. Improved mispronunciation detection with deep neural network trained acoustic models. *Speech Communication*.
* Huang, R., et al. 2024. Audiogpt: Understanding and generating speech, music, sound, and talking head. *AAAI*.
* Jiang, A., et al. 2023. Mistral 7B. *arXiv:2310.06825*.
* Korzekwa, D., et al. 2021. Mispronunciation detection in non-native (L2) English with uncertainty modeling. *ICASSP*.
* Lin, B., and Wang, L. 2021. Deep feature transfer learning for automatic pronunciation assessment. *Interspeech*.
* Lin, C. 2004. Rouge: A package for automatic evaluation of summaries. *Text summarization branches out*.
* Liu, H., et al. 2023. Zero-shot automatic pronunciation assessment. *arXiv:2305.19563*.
* Liu, H., et al. 2024. Advancing test-time adaptation in wild acoustic test settings. *EMNLP*.
* Ma, Z., et al. 2024. An Embarrassingly Simple Approach for LLM with Strong ASR Capacity. *arXiv:2402.08846*.
* OpenAI. 2024. GPT-4o System Card. *arXiv:2410.21276*.
* Papineni, K., et al. 2002. Bleu: a method for automatic evaluation of machine translation. *ACL*.
* Peng, L., et al. 2021. A Study on Fine-Tuning wav2vec2.0 Model for the Task of Mispronunciation Detection and Diagnosis. *Interspeech*.
* Radford, A., et al. 2022. Robust Speech Recognition via Large-Scale Weak Supervision. *arXiv:2212.04356*.
* Rogerson-Revell, P. 2021. Computer-assisted pronunciation training (CAPT): Current issues and future directions. *RELC Journal*.
* Witt, S. 2012. Use of speech recognition in computer-assisted language learning.
* Witt, S., and Young, S. 2000. Phone-level pronunciation scoring and assessment for interactive language learning. *Speech Communication*.
* Xu, X., et al. 2021. Explore wav2vec 2.0 for Mispronunciation Detection. *Interspeech*.
* Yang, M., et al. 2022. Improving mispronunciation detection with wav2vec2-based momentum pseudo-labeling. *arXiv:2203.15937*.
* Zhang, D., et al. 2023. Speechgpt: Empowering large language models with intrinsic cross-modal conversational abilities. *arXiv:2305.11000*.
* Zhang, J., et al. 2021a. speechocean762: An open-source non-native english speech corpus for pronunciation assessment. *arXiv:2104.01378*.
* Zhang, T., et al. 2019b. Bertscore: Evaluating text generation with bert. *arXiv:1904.09675*.
* Zhao, G., et al. 2018. L2-ARCTIC: A non-native English speech corpus.
* Zheng, L., et al. 2023. Judging llm-as-a-judge with mt-bench and chatbot arena. *NeurIPS*.
