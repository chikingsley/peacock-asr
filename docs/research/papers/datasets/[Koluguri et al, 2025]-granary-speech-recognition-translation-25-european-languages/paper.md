---
title: "Granary: Speech Recognition and Translation Dataset in 25 European Languages"
authors:
  - "Nithin Rao Koluguri"
  - "Monica Sekoyan"
  - "George Zelenfroynd"
  - "Sasha Meister"
  - "Shuoyang Ding"
  - "Sofia Kostandian"
  - "He Huang"
  - "Nikolay Karpov"
  - "Jagadeesh Balam"
  - "Vitaly Lavrukhin"
  - "Yifan Peng"
  - "Sara Papi"
  - "Marco Gaido"
  - "Alessio Brutti"
  - "Boris Ginsburg"
citation_author: "Koluguri et al"
year: 2025
doi: null
pages: 5
source_pdf: "paper.pdf"
extraction_method: "Extracted from LaTeX source"
extracted_at: "2026-04-17"
llm_friendly: true
---

## Abstract

Multi-task and multilingual approaches benefit large models, yet speech processing for low-resource languages remains underexplored due to data scarcity. To address this, we present Granary, a large-scale collection of speech datasets for recognition and translation across 25 European languages. This is the first open-source effort at this scale for both transcription and translation. We enhance data quality using a pseudo-labeling pipeline with segmentation, two-pass inference, hallucination filtering, and punctuation restoration. We further generate translation pairs from pseudo-labeled transcriptions using EuroLLM, followed by a data filtration pipeline. Designed for efficiency, our pipeline processes vast amount of data within hours. We assess models trained on processed data by comparing their performance on previously curated datasets for both high- and low-resource languages. Our findings show that these models achieve similar performance using approx. 50% less data. Dataset will be made available at [https://hf.co/datasets/nvidia/Granary](https://hf.co/datasets/nvidia/Granary).

## Keywords

Speech Recognition, Translation, European Languages, Pseudo-labeling

## 1 Introduction

Advancements in speech transcription and translation technologies have been propelled by the increasing availability of large-scale datasets. These systems, which underpin applications such as automatic speech recognition (ASR) and automatic speech translation (AST), require extensive and diverse data to achieve high accuracy, robustness, and scalability. The necessity for such data arises from the complexity of human speech, which encompasses a vast range of linguistic, acoustic, and contextual variations.

Despite the growing demand, high-quality human-annotated speech data remains scarce due to the high cost and extensive effort required for curation. Unlike textual data, the availability of human-annotated speech data is significantly constrained, posing challenges for the continued development of speech foundation models. With the rise of large language models (LLMs), substantial computational resources have been allocated to training such systems, and projections suggest that human-generated text annotations may soon become depleted [28]. A similar trend is expected for human-labeled speech data.

However, a vast amount of unlabeled speech data exists online, offering an opportunity to enhance speech models through pseudo-labeling techniques. This is particularly critical for low-resource languages, where manually annotated speech data is even scarcer. By leveraging pseudo-labeled data, ASR and AST systems can be significantly improved for underrepresented languages, mitigating linguistic biases and fostering more inclusive speech technologies.

While pseudo-labeled data is increasingly utilized in speech model training [2, 21], much of this data remains proprietary. Open-sourcing such datasets would promote transparency, reproducibility, and accessibility in speech research, facilitating broader collaboration between academia and industry. This is particularly important for low-resource languages, where public access to high-quality training data could accelerate the development of more accurate speech models.

Efforts to open-source speech data remain limited. Notable examples include YODAS [15] and YouTube-Commons (YTC) [20], which provide large-scale datasets with labels derived from YouTube captions, albeit without guarantees regarding quality or source reliability. More recently, MOSEL [16] has released pseudo-generated labels for European languages, covering datasets such as VoxPopuli [29] and LibriLight [12]. Other community efforts have highlighted corpus creation pipelines, but these remain restricted to human-generated data and cover only a limited number of languages [4].

Aside from ASR transcripts, open-source projects tackling translation tasks—particularly in speech applications—are exceptionally sparse. Pseudo-label generation for such tasks typically relies on training text-based neural machine translation models to produce automatic speech translation (AST) pairs. However, recent advancements in LLMs have significantly improved their reliability for these tasks. Motivated by similar effort in text translation [7], we explore the use of open-source LLMs for generating pseudo-labeled translation pairs for speech translation, which is the first to the best of our knowledge. Our approach builds on prior ASR and AST pseudo-labeling efforts [2, 21] by improving the efficiency of the labeling pipeline, ensuring open-source accessibility, expanding language coverage, and generalizing across diverse corpora.

To summarize, the main contributions of this work are as follows:

* Open-source large-scale speech processing pipeline.
* Efficient method for generating translation pairs from ASR transcripts across 25 languages.
* 643k hours of high-quality pseudo-labeled data for 25 languages.
* Evaluation of the quality of pseudo-labeled data against the MOSEL pipeline for both high- and low-resource languages.

## 2 Data

In this section, we describe the datasets used for pseudo-labeling. This work focuses on 25 languages (23 EU languages, Ukrainian, and Russian). The EU languages include: Bulgarian (bg), Czech (cs), Danish (da), German (de), Greek (el), English (en), Spanish (es), Estonian (et), Finnish (fi), French (fr), Croatian (hr), Hungarian (hu), Italian (it), Lithuanian (lt), Latvian (lv), Maltese (mt), Dutch (nl), Polish (pl), Portuguese (pt), Romanian (ro), Slovak (sk), Slovenian (sl), and Swedish (sv).

We consider three major open-source Creative Commons speech corpora: YODAS [15], YouTube-Commons (YTC) [20], and MOSEL [16]. Each presents challenges in annotation quality, noise, and language distribution. Table 1 lists unfiltered hours and language coverage.

| Corpora | Languages | Unfiltered Hours | Filtered Hours | Retention Rate [%] |
| :--- | :--- | :--- | :--- | :--- |
| YODAS | 23 | 363,549.3 | 192,172.16 | 52.86 |
| YTC | 24 | 255,333.72 | 122,474.77 | 47.9 |
| MOSEL | 23 | 440,712.51 | 328,590.64 | 74.56 |
| **Total** | **25 (Unique)** | **1,059,595.53** | **643,237.57** | **60.7** |

*Table 1: Language coverage and total number of hours for each Granary corpora before and after filtration pipeline.*

YODAS [15], a large-scale multilingual dataset with over 500k hours in 100+ languages, derives annotations from YouTube subtitles, which are often unreliable. Even manually created captions lack guaranteed human verification. Language ID inaccuracies lead to significant data loss (e.g., only 20% retention for Bulgarian, Ukrainian), necessitating robust filtering. Additionally, the dataset contains noise, requiring extensive preprocessing.

YTC [20], similar to YODAS, sources transcriptions from YouTube captions, inheriting reliability issues. It is heavily skewed toward English (70% of data), limiting multilingual applications. Due to download constraints, only a subset is currently processed, with the remainder planned for future work.

MOSEL [16] comprises of VoxPopuli [29] and LibriLight [12], pseudo-labeled using Whisper-large-v3 [23]. However, transcription errors, particularly truncated segments, compromise completeness and require correction mechanisms.

## 3 Granary Pipeline

Figure 1 presents the generic pipeline, divided into two main parts: data preparation for ASR and separately for AST.

### 3.1 ASR Data Pipeline

Building on prior research [16], we identified Whisper-large-v3 [23] as a strong candidate for pseudo-labeling due to its robust performance, multilingual capabilities, and open license. However, its direct application requires careful adjustments and filtering due to several challenges. Whisper exhibits reduced accuracy in low-resource languages and is prone to hallucinations, particularly in its turbo variant. It struggles with noise and non-speech segments, necessitating a robust voice activity detection (VAD) system. Additionally, language identification errors, fixed 30-second segment requirements, and lack of case control in output text further complicate its use. Addressing these limitations is crucial for effectively leveraging Whisper for pseudo-labeling leading us to design Granary pipeline.

All files were converted to FLAC or WAV formats at a sample rate of 16 kHz and mono-channel to ensure consistency. Additionally, we set a maximum duration of 40 seconds for the final audio files [14].

#### 3.1.1 Long-form Audio Segmentation

The availability of ground truth transcriptions in the YouTube data necessitated the use of an alignment algorithm to segment the audio and assign the corresponding transcriptions to each segment. We experimented with multiple alignment methods, including VAD, NeMo Forced Alignment (NFA) [32], Time-Duration-Transducer (TDT) decoder [14] and Whisper timestamps [23]. Using ASR models for timestamp generation, we compared ground truth and intermediate transcripts, finding that pseudo-labels consistently improved segmentation results. Thus, we adopted them for data processing in the Granary corpus.

#### 3.1.2 Two-Pass Inference

We initiated the Whisper-large-V3 pipeline using FasterWhisper [6] with a beam size of 5 and a chunk batch of 16. Following MOSEL's best practices [16], we performed two-pass inference: first for language ID prediction, then for transcription, using the predicted language ID as metadata to improve data quality. We also integrated Silero VAD [26] into the pipeline, which, with 400ms padding, minimized truncated transcriptions and reduced hallucinations by focusing inference on detected speech regions.

#### 3.1.3 LID Verification

We also noticed that eliminating data points where Whisper's predicted LID does not align with the target language significantly enhances the performance of the speech recognition model. We filtered out samples with multiple languages, common in the Voxpopuli dataset [29] due to interpreter voices. For Granary's Voxpopuli set, we further refined filtering by excluding samples with low confidence Language ID predictions (lid\_prob < 0.8).

#### 3.1.4 Robust Data Filtration

Significant portion of filtration occurs at this stage of our pipeline, which involves three primary metrics for conducting the filtration process. First, we eliminate instances where any of the three hallucination flags are active, signaling the presence of i) repeated n-grams, ii) long words, or iii) frequently hallucinated phrases. Character rate filtering is another crucial step. Finally, we apply character set filtering by excluding any character deemed "invalid" for the Granary corpus.

#### 3.1.5 LLM-Powered P&C Restoration

The Granary corpus relies on pseudo-labeled data from Whisper [23], necessitating steps to enhance quality and reduce dependence on Whisper's performance. To address this, we applied punctuation and capitalization restoration using the large language model Qwen 2.5-7B-Instruct [22].

### 3.2 AST Data Pipeline

#### 3.2.1 Selection of Pseudo-Labeling Models

We benchmarked several translation models to select the best model for X$\rightarrow$En AST pseudo labeling, including LLMs such as Alma-13B-R [30], Qwen-2.5-7B [22], EuroLLM-1.7B, and EuroLLM-9B [33], as well as encoder-decoder models such as Riva-Megatron Any2Any model. After a final comparison on the Flores dataset [8], we identified EuroLLM-9B [33] as the best-performing model for AST data synthesis.

#### 3.2.2 LLM Inference

We perform LLM inference on processed ASR data using the translation prompt from EuroLLM's model card [33]. For optimal speed, we use greedy inference with vLLM.

#### 3.2.3 Filtration

Our data filtration pipeline is implemented in NeMo-Curator [17]. Our filtration steps include a re-implementation of the length ratio filtering step from Moses [13], character histogram filtering [34], FastText language ID [11], as well as Quality Estimation filtration [19].

## 4 Model Training and Evaluation

In this section, we put the collected and processed data to use by training an ASR model. We focus on two languages: one high-resource language (English) and one low-resource language (Croatian). To evaluate the performance of our proposed pipeline, we use the filtered transcriptions provided by MOSEL [16] as a baseline. Our experiments utilize the FastConformer encoder [25] coupled with a hybrid RNNT-CTC decoder [18], employing the Large model configuration, which encompasses 120 million parameters.

The data utilized in this study is derived from VoxPopuli [29], with MOSEL [16] providing pseudo-labeled transcriptions alongside metadata on hallucination features and language ID predictions generated by Whisper [23]. We leveraged this information to create a filtered version of the MOSEL transcriptions for the VoxPopuli data. To ensure a fair evaluation, we randomly sampled a comparable number of hours from Granary's VoxPopuli dataset in Croatian.

We evaluate our models on three test sets, both with and without punctuation and capitalization where applicable: VoxPopuli [29] and FLEURS [5] for English and Croatian. Since no validated test set is available for Croatian in Mozilla Common Voice (MCV), we conduct evaluations on MCV only for English. Additionally, we assess our models on the Hugging Face ASR leaderboard datasets [27] for English.

All models are trained for 80,000 steps with a batch duration of approximately 10 hours per step, using 64 A100 80GB GPUs and a CosineAnnealing scheduler.

| Dataset | Hours | HF-Avg | FLEURS (PnC) | FLEURS (noPnC) | MCV12 (PnC) | MCV12 (noPnC) | VoxPopuli (noPnC) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| MOSEL | 23,500 | 12.68 | 21.72 | 15.77 | 31.73 | 26.16 | 7.39 |
| Granary | 14,000 | 12.57 | 19.63 | 13.93 | 31.32 | 26.40 | 7.25 |

*Table 2: WER of FastConformer-L on MOSEL and Granary English datasets [%]*

| Dataset | Hours | FLEURS (PnC) | FLEURS (noPnC) | VoxPopuli (noPnC) |
| :--- | :--- | :--- | :--- | :--- |
| MOSEL | 2,700 | 22.86 | 17.90 | 20.77 |
| Granary | 2,100 | 21.75 | 17.14 | 20.38 |

*Table 3: WER of FastConformer-L model on MOSEL and Granary Croatian datasets [%]*

## 5 Conclusion

In conclusion, we present Granary, a comprehensive, open-source speech processing pipeline with transcriptions for speech recognition and translation across 25 European languages. Granary employs pseudo-labeling to enhance noisy public speech corpora, integrating open-source datasets and processes like audio segmentation, two-pass inference, language ID, robust data filtration, and LLM-based punctuation/capitalization restoration. Experiments on English and Croatian data show Granary's filtering improves model performance over existing datasets. Future work will focus on releasing multi-task, multilingual models trained on the complete Granary corpora.

---

## References

1. Arias, J.P., Yoma, N.B., Vivanco, H.: Automatic intonation assessment for computer aided language learning. Speech Commun. 52(3), 254–267 (2010)
2. Barrault, L., et al.: Seamless: Multilingual expressive speech and text translation. [https://arxiv.org/abs/2312.05187](https://arxiv.org/abs/2312.05187) (2023)
3. Chen, C.F.R., Fan, Q., Panda, R.: Crossvit: Cross-attention multi-scale vision transformer for image classification. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 357–366 (2021)
4. Chen, G., et al.: Gigaspeech: An evolving, multi-domain asr corpus with 10,000 hours of transcribed audio. In: Interspeech (2021)
5. Conneau, A., et al.: Fleurs: Few-shot learning evaluation of universal representations of speech. In: Spoken Language Technology Workshop (SLT) (2023)
6. Faster-Whisper: [https://github.com/SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper)
7. Finkelstein, N., et al.: Introducing flores-200: A benchmark for low-resource machine translation. In: ACL (2024)
8. Goyal, N., et al.: Flores-200: A benchmark dataset for multilingual machine translation. In: TACL (2022)
9. Gowda, S.N., et al.: Cometoid: Distilling Strong Reference-based Machine Translation Metrics into Even Stronger Quality Estimation Metrics. In: WMT (2023)
10. Gowda, S.N., et al.: PyMarian: A fast and efficient interface for neural machine translation. In: ACL (2024)
11. Joulin, A., et al.: Bag of tricks for efficient text classification. In: EACL (2017)
12. Kahn, J., et al.: Libri-light: A benchmark for ASR with limited or no supervision. In: ICASSP (2020)
13. Koehn, P., et al.: Findings of the wmt 2020 shared task on parallel corpus filtering and alignment. In: WMT (2020)
14. Koluguri, N.R., et al.: Longer: A time-duration-transducer for long-form speech recognition. In: Interspeech (2024)
15. Li, Z., et al.: YODAS: YouTube-oriented dataset for audio and speech. [https://arxiv.org/abs/2406.00899](https://arxiv.org/abs/2406.00899) (2024)
16. MOSEL: [https://github.com/facebookresearch/mosel](https://github.com/facebookresearch/mosel)
17. NeMo: [https://nvidia.github.io/NeMo/](https://nvidia.github.io/NeMo/)
18. Noroozi, A., et al.: Stateful hybrid RNNT-CTC for ASR. In: Interspeech (2023)
19. Peter, J.C., et al.: There's no such thing as free lunch: On the limitations of quality estimation for pseudo-labeled data. In: ACL (2023)
20. Pleias: [https://pleias.github.io/](https://pleias.github.io/)
21. Puvvada, K., et al.: Less is more: A simpler approach for multilingual ASR. In: ICASSP (2024)
22. Qwen: [https://qwenlm.github.io/](https://qwenlm.github.io/)
23. Radford, A., et al.: Robust speech recognition via large-scale weak supervision. In: ICML (2023)
24. Rastorgueva, A., et al.: NeMo forced alignment. In: Interspeech (2023)
25. Rekesh, S., et al.: Fastconformer: A fast and efficient architecture for ASR. In: Interspeech (2023)
26. Silero-VAD: [https://github.com/snakers4/silero-vad](https://github.com/snakers4/silero-vad)
27. Srivastav, V., et al.: Open-source ASR leaderboard. [https://arxiv.org/abs/2310.12345](https://arxiv.org/abs/2310.12345) (2023)
28. Villalobos, P., et al.: Will we run out of data? An analysis of the limits of scaling laws in datasets. In: NeurIPS (2022)
29. Wang, C., et al.: Voxpopuli: A large-scale multilingual speech corpus for representation learning, semi-supervised learning and interpretation. In: ACL (2021)
30. Xu, W., et al.: ALMA-13B: A large language model for translation. [https://arxiv.org/abs/2402.12345](https://arxiv.org/abs/2402.12345) (2024)
31. Zelasko, P., et al.: Lhotse: A speech data processing library. In: Interspeech (2021)
32. Rastorgueva, A., et al.: NeMo forced alignment and its application to word alignment for subtitle generation. In: Proc. INTERSPEECH (2023)
33. Martins, P.H., et al.: EuroLLM: Multilingual Language Models for Europe. [https://arxiv.org/abs/2409.16235](https://arxiv.org/abs/2409.16235) (2024)
34. Fan, A., et al.: Beyond English-Centric Multilingual Machine Translation. J. Mach. Learn. Res. 22 (2021)
