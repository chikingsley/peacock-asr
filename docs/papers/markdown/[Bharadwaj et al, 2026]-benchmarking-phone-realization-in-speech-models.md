url: https://arxiv.org/html/2601.14046v1
title: "PRiSM: Benchmarking Phone Realization in Speech Models"
content: |-
  Shikhar Bharadwaj 1 Chin-Jou Li∗1 Yoonjae Kim∗2 Kwanghee Choi 3 Eunjung Yeo 3

  Ryan Soh-Eun Shim 4 Hanyu Zhou 1 Brendon Boldt 1 Karen Rosero Jacome 1

  Kalvin Chang 5 Darsh Agrawal 1 Keer Xu 1 Chao-Han Huck Yang 6

  Jian Zhu 7 Shinji Watanabe 1 David R. Mortensen 1

  1 CMU 2 Gwangju Institute of Science and Technology 3 UT Austin 4 LMU Munich 

  5 UC Berkeley 6 NVIDIA 7 UBC 

  {sbharad2,chinjoul,dmortens}@andrew.cmu.edu, rladbswo12@gm.gist.ac.kr

  ###### Abstract

  Phone recognition (PR) serves as the atomic interface for language-agnostic modeling for cross-lingual speech processing and phonetic analysis. Despite prolonged efforts in developing PR systems, current evaluations only measure surface-level transcription accuracy. We introduce PRiSM, the first open-source benchmark designed to expose blind spots in phonetic perception through intrinsic and extrinsic evaluation of PR systems. PRiSM standardizes transcription-based evaluation and assesses downstream utility in clinical, educational, and multilingual settings with transcription and representation probes. We find that diverse language exposure during training is key to PR performance, encoder-CTC models are the most stable, and specialized PR models still outperform Large Audio Language Models. PRiSM releases code, recipes, and datasets to move the field toward multilingual speech models with robust phonetic ability 1 1 1 https://github.com/changelinglab/prism.

  PRiSM: Benchmarking Phone Realization in Speech Models

  Shikhar Bharadwaj††thanks: Equal contribution.1 Chin-Jou Li∗1 Yoonjae Kim∗2 Kwanghee Choi 3 Eunjung Yeo 3 Ryan Soh-Eun Shim 4 Hanyu Zhou 1 Brendon Boldt 1 Karen Rosero Jacome 1 Kalvin Chang 5 Darsh Agrawal 1 Keer Xu 1 Chao-Han Huck Yang 6 Jian Zhu 7 Shinji Watanabe 1 David R. Mortensen 1 1 CMU 2 Gwangju Institute of Science and Technology 3 UT Austin 4 LMU Munich 5 UC Berkeley 6 NVIDIA 7 UBC{sbharad2,chinjoul,dmortens}@andrew.cmu.edu, rladbswo12@gm.gist.ac.kr

  1 Introduction
  --------------

  Phone recognition (PR) entails transcribing speech into phonetic units that capture the physical realization of sounds, independent of language-specific phonological constraints. By preserving acoustic nuances often abstracted away by word- or phoneme-level models 2 2 2 For example, tell may be transcribed as \tipaencoding[t h E\textltilde] in Mainstream American English and \tipaencoding[t h El] in Scottish English, while the phonemic form of tell is consistently \tipaencoding/tEl/. , PR provides a robust foundation for cross-lingual speech processing (Li et al., 2022; Yusuyin et al., 2025) and downstream applications in clinical (Shriberg et al., 2025; Choi et al., 2025) and educational settings (Tu et al., 2018; Inceoglu et al., 2023).

  Figure 1: PRiSM is the first open-source benchmark for phone recognition systems, covering intrinsic and extrinsic evaluations, i.e., transcription task and downstream task performance.

  PR models have scaled substantially to cover diverse linguistic settings (see §2.1), yet existing evaluations remain difficult to compare across studies. For example, models often differ in language coverage and phone inventories Zhu et al. (2025), and evaluation metrics are not standardized Li et al. (2025). A common response has been to fix a metric Taguchi et al. (2023); Li et al. (2025) and expand the number of test datasets to mitigate bias Zhu et al. (2025). Yet this approach scales poorly due to the scarcity of phonetically transcribed data. Moreover, transcription error rates do not necessarily reflect a model’s phonetic capabilities or practical utility. Error rates in PR are inherently noisier than in ASR, as phones, unlike lexical units, correspond to a lower-level, articulatorily defined abstraction of the acoustic signal.

  Furthermore, the link between transcription accuracy and downstream performance is often assumed rather than empirically proven. In practice, models leverage phonetic information via two channels: explicit transcriptions and latent internal representations. The latter are especially potent, as they encode rich phonetic cues (see §2.2). Consequently, metrics based solely on transcription error fail to capture the full utility and nuanced quality of these representations.

  Therefore, we propose PRiSM to fairly benchmark P hone R ealization i n S peech M odels. PRiSM assesses PR systems 3 3 3 Any pipeline that converts speech into phonetic units. intrinsically through transcription error, and extrinsically through utility in clinical, educational, and multilingual speech tasks using generated transcriptions and hidden representations. PRiSM applies to PR systems ranging from specialized PR models to general speech-to-text (S2T) systems, including Large Audio Language Models (LALMs), which are increasingly used for general speech tasks despite limited evaluation of their phonetic abilities (Peng et al., 2024; Arora et al., 2025).

  PRiSM is the first open-source benchmark for PR systems, for which code, evaluation recipes, and datasets are released, where licensing permits. With its reproducible and expandable framework, PRiSM supports researchers in understanding model behavior and training strategies, and helps practitioners make informed model choices. We evaluate a broad range of PR systems and find that: (i) language exposure matters: seen languages benefit from familiar patterns, unseen from multilingual training; (ii) data and architecture shape performance: broad, diverse coverage improves results, while encoder-CTC architectures are more stable; (iii) LALMs lag behind specialized PR models.

  Ultimately, our goal is to establish a common evaluation basis to drive progress toward PR systems that capture robust and generalizable phonetic information across resource conditions.

  Abbr.Task Dataset Lang.
  Intrinsic: Core Capability (Metrics ↓\downarrow Lower is better)
  Phone Recognition (PFER)
  PR-tmt Variation of Seen Language TIMIT Garofolo et al. (1993)English
  PR-arc Variation of Seen Language L2-ARCTIC Perceived Zhao et al. (2018b)English
  PR-saa Variation of Seen Language Speech Accent Archive Weinberger (2015)English
  PR-drc Unseen Languages DoReCo Paschen et al. (2020)45 langs
  PR-vox Unseen Languages VoxAngeles Chodroff et al. (2024)95 langs
  PR-tsm Unseen Languages Tusom2021 Mortensen et al. (2021)Tusom
  Extrinsic: Downstream Utility (Metrics ↑\uparrow Higher is better)
  Pathological Speech: Dysarthria Intelligibility Prediction (τ\tau) & Child Speech Disorder Detection (F1)
  DYS-ez Dysarthria Intelligibility Prediction EasyCall Turrisi et al. (2021)Italian
  DYS-ua Dysarthria Intelligibility Prediction UASpeech Kim et al. (2008)English
  CSD-us Child Speech Disorder Detection UltraSuite Eshky et al. (2018)English
  L2 Speech: L1 Classification (F1) & L2 Assessment (τ\tau)
  L1-eda L1 Classification EdAcc Sanabria et al. (2023)English
  L1-arc L1 Classification Kominek and Black (2004)&Zhao et al. (2018b)English
  L2-so L2 Assessment Speechocean762 Zhang et al. (2021a)English
  Multilingual: Lang. ID (F1), Geolocation (Recall@1) & Phone Inventory Induction (F1-PI)
  LID-fl Lang. ID (LID)FLEURS-24 Conneau et al. (2023)24 langs
  GEO-v Speech Geolocation Vaani Ghosh et al. (2025)Hindi Dialects
  PI-drc Phone Inventory Induction DoReCo Paschen et al. (2020)45 langs

  Table 1:  List of evaluation tasks. Blue denotes core capabilities, where lower scores are better. Yellow denotes downstream utility, where higher scores are better. F1-PI is described in §B.1. See Appendix A for license details. 

  2 Background
  ------------

  ### 2.1 Phone Recognition Systems

  PR can be viewed as a variant of the S2T task that maps speech to phonetic symbols such as IPA International Phonetic Association (1999). In this work, we use “PR system” to refer broadly to any system capable of converting speech into IPA in a language-agnostic fashion.

  Modern PR systems are typically fine-tuned from ASR systems Baevski et al. (2020); Radford et al. (2023) or trained from scratch on ASR datasets Zhu et al. (2024) with transcriptions automatically converted to IPA using grapheme-to-phoneme (G2P) tools Mortensen et al. (2018); Zhu et al. (2022). Language-specific approaches Li et al. (2020); Gao et al. (2021) rely on phoneme inventories, while language-agnostic approaches, which we focus on, seek to learn phonetic representations generalized across languages. LALMs have recently become prominent in speech tasks and have shown competitive performance with cascaded systems that combine LLMs with speech processing modules Yang et al. (2025), motivating interest in their application to PR Huang et al. (2025); Wang et al. (2025). We describe the systems investigated in this work in §4.

  ### 2.2 Phonetic information in PR systems

  Explicitly generated phonetic transcriptions are easy for humans to inspect and utilize. For example, faithful phonetic transcriptions of the speech of a child with a speech sound disorder can help a clinician understand the nature of the disorder and design interventions Dodd (2013). Nevertheless, representing continuous speech with discrete symbols inherently incurs information loss, filtering out non-linguistic variation.

  Internal model representations serve as a complement that retains richer information. Speech models trained on S2T tasks produce temporally aligned representations that capture empirically useful acoustic-phonetic Choi et al. (2024), articulatory Cho et al. (2024), and even semantic Ma et al. (2025) features. The most widely used S2T model representations are from end-to-end ASR models such as Whisper Radford et al. (2023) and WavLM Chen et al. (2022). In contrast, LALMs’ representations are often inaccessible or difficult to analyze, as they focus mainly on textual output and lack strict temporal alignment with input speech.

  ### 2.3 Assessing phonetic/phonological ability

  In the text modality, language models are evaluated with text input and output. PhonologyBench Suvarna et al. (2024) evaluates G2P, syllable counting, and rhyme judgment, while Bunzeck et al. (2025) and Goriely and Buttery (2025) probe phonological knowledge using minimal pairs and word segmentation. In the speech modality, models are evaluated with speech (and optionally text) input and representation output. SUPERB Yang et al. (2021) and Dynamic-SUPERB Huang et al. (2025) include phoneme recognition, phonological feature analysis, and pronunciation evaluation. BabySLM Lavechin et al. (2023) and the ZeroSpeech challenges Nguyen et al. (2020) propose metrics that evaluate phonological and acoustic-phonetic contrasts based on minimal pairs. In contrast to previous work, PRiSM evaluates phonetic ability in both text and speech through intrinsic and extrinsic tasks.

  Table 2: Included PR systems. Architecture abbrv.: Encoder (Enc), Decoder (Dec), Audio Transformer (AuT), Mixture-of-Experts (MoE); Loss abbrv.: Consistency Regularized CTC (CR-CTC) Yao et al. (2025), Hybrid CTC/Attention (CTC-Att) Watanabe et al. (2017), Intermediate CTC (Int-CTC) Lee and Watanabe (2021), Autoregressive (AR); Data abbrv.: Multilingual LibriSpeech (MLS), Common Voice (CV), Pseudo-labeled (PL).

  3 Evaluation Framework of PRiSM
  -------------------------------

  PRiSM covers intrinsic (§3.1) and extrinsic (§3.2) evaluations shown in Figure 1. Intrinsic evaluation compares predicted transcriptions to gold labels, while extrinsic evaluation measures transcriptions and internal representations on downstream tasks. In extrinsic evaluation, transcriptions provide a direct and interpretable signal of explicit phonetic content, whereas representations are commonly used in downstream tasks but may encode non-phonetic information. Table 1 summarizes included datasets and metrics.

  ### 3.1 Intrinsic: Core Capability

  We use Phonetic Feature Error Rate (PFER) to measure the distance between reference and predicted transcriptions. Unlike Phone Error Rate (PER), which treats each phone as a token, PFER computes the edit distance D​(⋅,⋅)D(\cdot,\cdot) over articulatory features feat​(⋅)\text{feat}(\cdot) such as roundness or voicing. As shown in Equation 1, where u u denotes an utterance (sequence of phones where i i indexes this sequence) and u∗u^{*} its ground truth, PFER is calculated as the total feature edit distance across all utterances divided by the total number of phones, representing the percentage of incorrect features Mortensen et al. (2016).

  PFER=1∑i|u i∗|​∑i D​(feat​(u i∗),feat​(u i))\text{PFER}=\frac{1}{\sum_{i}|u_{i}^{*}|}\sum_{i}D(\text{feat}(u^{*}_{i}),\text{feat}(u_{i}))(1)

  The tasks comprise two categories: Variation of seen languages includes regional and non-native speech, testing whether PR systems rely excessively on seen patterns rather than the actual input. Unseen languages assess the system’s language-agnostic phonetic knowledge. Details of each task and dataset are in §A.2.

  ### 3.2 Extrinsic: Downstream Utility

  We evaluate PR systems using two downstream probes, namely the transcript probe and the representation probe. For transcript probe (TP), input consists of predicted phonetic transcriptions, and the probe is a text-based bi-GRU. For representation probe (RP), following the setup in Turian et al. (2022), we use the last layer’s hidden representations as input and temporal pooling with attention followed by a Multi-Layer Perceptron as the probe. Metrics for each task are listed in Table 1 and the detailed experimental setup is in Appendix C.

  We consider three categories of downstream tasks where phonetic information is essential. In pathological speech assessment, phonetic transcriptions are used to document patients’ speech and support diagnosis and treatment planning Ball et al. (2009); Nelson et al. (2020). In L2 speech assessment, phonetic cues enable pronunciation feedback Franco et al. (2010) and accent classification Angkititrakul and Hansen (2006). In multilingual speech identification, analyzing phonetic and phonological differences across languages and dialects, such as phone inventories, phonotactics, and phoneme realization, is crucial Schultz and Kirchhoff (2006). We describe each task and dataset in detail in §A.3.

  4 Benchmarked Models
  --------------------

  Table 2 summarizes the studied model families:

  *   •Wav2Vec2Phs: MultiIPA, W2V2P-LV60, and W2V2P-XLSR53 are fine-tuned variants of Wav2Vec2 Baevski et al. (2020), contrastively pre-trained speech SSL models, and differ in pre-training coverage and phone recognition fine-tuning datasets. 
  *   •ZIPAs: ZIPA-CTC and ZIPA-CTC-NS are encoder-CTC models trained from scratch on multilingual data, with ZIPA-CTC-NS further trained on large-scale pseudo-labeled data from ZIPA-CTC. 
  *   •POWSMs: POWSM is an attention-based encoder-decoder (AED) model trained on the same dataset as ZIPAs and augmented for other S2T tasks. Following their framework, we train POWSM-CTC, an encoder-CTC variant for comparison. 
  *   •LALMs: We include Gemini 2.5 Flash (closed-source) and Qwen3-Omni-Instruct (open-weight), both state-of-the-art systems widely used in recent studies Lee et al. (2025). Since their representations are not easy to access or pool, we probe them with zero-shot prompting, which is a form of context-based fine-tuning Petrov et al. (2023). The prompts are in Appendix D. 
  *   •

  5 Results and Discussion
  ------------------------

  Table 3 presents PR performance, and Table 4 presents a comprehensive breakdown of downstream evaluations. In general, ZIPA-CTC-NS performs well in all settings, while Whisper excels in RP. LALMs generally remain less competitive.

  Variation of Seen Language Unseen Languages
  Model PR-tmt PR-arc PR-saa Avg.PR-drc PR-vox PR-tsm Avg.
  MultiIPA∗16.3 15.5 13.8 15.2 18.3 15.2 30.5 21.3
  W2V2P-LV60 13.2 10.9 0 9.4 11.2 17.8 15.7 24.9 19.5
  W2V2P-XLSR53 13.5 0 9.9 0 9.0 10.8 17.3 13.9 31.9 21.0
  ZIPA-CTC 13.1 0 9.7 0 9.0 10.6 18.0 17.0 23.7 19.6
  ZIPA-CTC-NS 13.1 0 9.7 0 8.9 10.6 16.8 17.1 23.1 19.0
  POWSM 13.7 11.3 27.6 17.5 17.1 17.1 22.0 18.7
  POWSM-CTC 13.1 10.3 10.0 11.1 18.1 15.3 32.2 21.9
  Gemini 2.5 Flash∗∗15.2 12.7 13.2 13.7 105.3 0 19.7 36.3 53.8
  Qwen3-Omni-Instruct∗∗15.1 11.9 0 9.1 12.0 150.2 0 49.0 117.1 0 105.4 0

  Table 3: PFER of the intrinsic evaluation (↓\downarrow). ∗English is included during pretraining but not fine-tuning. ∗∗Some of the “unseen languages” may have appeared in the training data. See §5.1 for details.

  Pathological Speech L2 Speech Multilingual Speech Score
  Model DYS-ez DYS-ua CSD-us L1-eda L1-arc L2-so LID-fl GEO-v PI-drc
  Naive Baseline 0 0.7±\pm 1.6-0.8±\pm 0.9 41.8±\pm 1.0 0 6.3±\pm 0.4 14.3±\pm 0.3 0 1.5±\pm 1.0 0 4.3±\pm 0.3 0 3.3±\pm 0.0—8.9
  Transcript Probe (TP)
  MultiIPA 48.2±\pm 0.2 45.6±\pm 1.4 93.6±\pm 1.6 10.0±\pm 1.0 50.5±\pm 0.9 33.3±\pm 1.7 89.3±\pm 0.5 44.5±\pm 0.4 40.9 44.3
  W2V2P-LV60 42.4±\pm 1.3 50.3±\pm 0.9 95.6±\pm 1.4 0 7.6±\pm 0.5 38.0±\pm 0.3 36.1±\pm 1.7 91.4±\pm 0.2 45.7±\pm 0.9 51.3 42.0
  W2V2P-XLSR53 49.2±\pm 0.8 47.6±\pm 0.8 92.3±\pm 2.6 0 9.1±\pm 0.6 43.1±\pm 0.6 37.5±\pm 0.8 94.1±\pm 0.2 44.5±\pm 1.2 56.9 43.8
  ZIPA-CTC 55.0±\pm 0.6 57.0±\pm 0.5 91.7±\pm 2.3 0 6.6±\pm 0.4 30.5±\pm 0.5 36.6±\pm 2.8 95.6±\pm 0.2 44.1±\pm 1.0 55.2 43.5
  ZIPA-CTC-NS 56.6±\pm 0.8 51.1±\pm 1.3 99.4±\pm 0.5 0 6.7±\pm 0.3 30.0±\pm 0.3 40.8±\pm 0.8 95.9±\pm 0.1 44.7±\pm 1.8 56.6 44.2
  POWSM 52.7±\pm 1.7 46.1±\pm 0.8 94.3±\pm 1.3 0 6.5±\pm 0.8 28.0±\pm 0.3 28.4±\pm 2.2 95.1±\pm 0.5 43.7±\pm 1.4 48.7 39.6
  POWSM-CTC 53.3±\pm 0.4 46.5±\pm 0.6 96.9±\pm 0.7 0 6.4±\pm 0.5 29.8±\pm 0.1 26.8±\pm 0.7 90.4±\pm 0.4 42.9±\pm 0.8 57.7 40.2
  Gemini 2.5 Flash 27.9±\pm 0.6 38.5±\pm 0.4 95.0±\pm 1.6 0 6.4±\pm 0.4 22.3±\pm 0.4 20.1±\pm 1.1 91.8±\pm 0.3 33.2±\pm 1.0 39.1 31.6
  Qwen3-Omni-Instruct 52.5±\pm 1.8 49.4±\pm 1.0 98.9±\pm 0.8 0 6.9±\pm 0.6 30.5±\pm 0.3 15.6±\pm 1.4 89.3±\pm 0.2 34.7±\pm 1.2 44.5 39.2
  Representation Probe (RP)
  MultiIPA 65.5±\pm 4.0 77.0±\pm 1.4 98.5±\pm 0.8 11.7±\pm 1.1 53.0±\pm 3.3 46.3±\pm 1.9 78.2±\pm 1.0 24.5±\pm 3.7—56.5
  W2V2P-LV60 67.2±\pm 1.8 79.9±\pm 0.9 98.6±\pm 0.6 12.0±\pm 0.3 60.7±\pm 2.9 49.9±\pm 1.0 76.6±\pm 1.3 24.6±\pm 2.1—59.4
  W2V2P-XLSR53 70.8±\pm 2.2 82.0±\pm 2.2 99.2±\pm 0.7 13.0±\pm 1.1 47.0±\pm 6.0 50.7±\pm 3.1 81.0±\pm 2.3 21.5±\pm 3.1—58.2
  ZIPA-CTC 73.2±\pm 2.2 74.7±\pm 1.2 99.5±\pm 0.3 13.9±\pm 0.9 73.4±\pm 2.5 54.0±\pm 0.8 96.1±\pm 0.7 23.0±\pm 1.2—62.9
  ZIPA-CTC-NS 71.2±\pm 2.2 75.1±\pm 0.8 98.6±\pm 0.9 13.7±\pm 0.6 74.1±\pm 2.8 54.3±\pm 0.5 96.8±\pm 0.3 24.0±\pm 1.5—62.7
  POWSM 73.0±\pm 3.0 70.8±\pm 1.1 99.5±\pm 0.3 10.3±\pm 1.2 68.0±\pm 1.9 53.1±\pm 0.3 96.5±\pm 0.1 21.5±\pm 2.2—60.4
  POWSM-CTC 73.6±\pm 1.5 66.7±\pm 1.6 97.9±\pm 0.9 0 8.0±\pm 0.7 53.0±\pm 0.5 45.7±\pm 3.0 75.4±\pm 1.5 14.1±\pm 2.6—55.2
  WavLM 69.2±\pm 2.0 77.5±\pm 1.4 99.0±\pm 0.5 14.4±\pm 1.0 58.3±\pm 2.0 50.2±\pm 1.4 76.2±\pm 3.2 23.5±\pm 4.6—59.4
  Whisper 74.8±\pm 1.1 79.5±\pm 0.3 99.5±\pm 0.3 24.3±\pm 1.6 84.3±\pm 3.0 57.2±\pm 0.8 96.3±\pm 0.5 35.0±\pm 2.7—68.5
  Zero-shot
  Gemini 2.5 Flash 21.4 50.4 75.3 32.7 43.9 35.8 91.5 0 6.5—41.5
  Qwen3-Omni-Instruct 27.0 61.7 70.9 18.2 31.8 49.8 59.1 0 5.3—41.5

  Table 4: PR system performance on extrinsic tasks (↑\uparrow). Results are reported as mean ±\pm standard deviation across 5 random seeds where applicable. Best numbers are bolded and second-best underlined. See §5.2 for details. The formula for aggregrated score is in §B.2. 

  ### 5.1 Intrinsic Evaluation

  We observe a consistent trend for language variation: CTC-based models generally outperform LALMs, followed by AED models. For MultiIPA, English appears during pretraining but not finetuning, highlighting the importance of language coverage in PR data. On PR-saa, POWSM performs poorly likely due to decoder search on long speech sequences; meanwhile, a text-based G2P model Zhu et al. (2022) achieves a PFER of 10.2, beating Gemini 2.5 Flash despite modeling only canonical pronunciations.

  For unseen languages, AED and CTC models show comparable performance, whereas LALMs perform poorly, sometimes producing repeated or degenerate outputs indicative of limited training exposure Holtzman et al. (2020). Performance also varies across datasets, and language-wise breakdowns reveal heterogeneous behavior. POWSM outperforms POWSM-CTC and exhibits performance comparable to ZIPAs, suggesting that incorporating a degree of language modeling may improve generalization by capturing shared phonological patterns, as further analyzed in §6.1.

  These trends show that variation in seen languages benefits from outputs grounded in known patterns, whereas unseen languages benefit from multilingual training and learned phonological patterns.

  ### 5.2 Extrinsic Evaluations

  For transcript probe, ZIPAs and W2V2P-XLSR53 are generally competitive. ZIPAs perform well on pathological speech, likely due to their normalized, smaller vocabularies, which approximate broad transcription known to be reliable for speech disorders Shriberg and Lof (1991), while W2V2P-XLSR53 benefits from diverse pretraining data. Multilingual training further improves performance, especially on multilingual tasks. We discuss PI-drc in §6.2 as an example. Whisper’s strength in representation probe suggests that large-scale ASR pretraining produces representations that retain phonetic information.

  A trade-off of TP and RP emerges among specialized PR models: for example, Wav2Vec2Phs achieve strong TP results on L2 speech but show limited gains on RP, whereas ZIPAs underperform on TP yet excel on RP. Task category also influences their relative performance: Pathological speech benefits more from RP, L2 speech falls in the middle, and multilingual tasks tend to favor TP. We hypothesize that transcripts act as a structured bottleneck: pathological speech relies on features such as timbre and prosody, whereas multilingual settings benefit less from acoustic detail. We investigate the behavior of TP on GEO-v in §6.3.

  LALMs show task-dependent performance. Notably, Qwen3-Omni-Instruct achieves competitive TP on pathological speech, but they generally perform poorly in zero-shot settings and underperform on languages other than English (DYS-ez). An exception is L2 speech, where the gap is smaller, explored in §6.4.

  Overall, our results highlight the importance of evaluating PR systems with a combination of intrinsic and extrinsic tasks. Intrinsic evaluation alone may not fully capture phonetic capabilities, while extrinsic evaluation reveals that relative performance on TP and RP is task-dependent. Multilingual pretraining and fine-tuning improve performance across model families, and encoder-CTC based architectures provide more stable PR performance in new domains. In contrast, LALMs remain limited in phone recognition and related tasks.

  6 Analysis
  ----------

  We conduct several analyses to anchor our observations. In §6.1, We examine how architectural choices affect the balance between phonotactics and acoustics, echoing with evaluation results. In §6.2, we study multilingual generalization and confirm that encoder-only architectures trained with diverse language coverage at all stages perform well for PR. In §6.3, we analyze TP in detail and show that it effectively captures phone distribution differences across regions. Finally, in §6.4, we assess zero-shot performance of LALMs on challenging tasks, concluding that they remain insensitive to sociophonetic variation.

  ### 6.1 Phonotactics or the Acoustic Signal

  Figure 2: PFER vs Phone masking rate. A PR model that relies only on acoustics should produce a horizontal line. Encoder-only models trained with CTC loss retain acoustic fidelity at high masking levels. See §6.1. 

  Ideally, PR systems would faithfully transcribe the actual pronunciation in the speech signal via acoustic modeling. Instead, model transcriptions often normalize toward standard pronunciations or other probabilistically likely phone patterns (Zhu et al., 2025; Li et al., 2025), essentially relying on (phone-level) language modeling (Pimentel et al., 2020).6 6 6 A common example of such phonotactic knowledge is the intuition that brick [\tipaencoding b⁢rIk] is a valid phone sequence in English while bnick [\tipaencoding bnIk] is not (Chomsky and Halle, 1968). Additionally, models can also overfit phonotactics from the high-resource languages. In this experiment, we investigate the extent to which PR systems rely on such phonotactic patterns present in the training data, as opposed to information derived directly from acoustic signal.

  Using TIMIT Garofolo et al. (1993)’s time-aligned phone transcripts, we replace p%p\% of phones with silence, transcribe the modified speech using PR, and compute PFER against a reference containing only the remaining phones. In Figure 2, we plot the phone masking rate against the PFER for different model families. For a model that only relies on the acoustic waveform for prediction, the curve would be a horizontal line. However, for models that rely on phonotactics, the PFER will increase with greater noise. While all models start at a similar PFER, Wav2Vec2Phs and POWSM-CTC perform better than ZIPAs and POWSM at higher masking levels. Thus Wav2Vec2Phs relies more on the acoustic signal than on phonotactics. While POWSM is an AED model trained with next-token prediction, Wav2Vec2Phs and POWSM-CTC are encoder-only models trained with the CTC objective.

  However, ZIPAs are also encoder-only (Zipformer Yao et al. (2024)) models, but they are trained with a consistency regularized CTC (CR-CTC) loss Yao et al. (2025). The high PFER of ZIPA means that CR-CTC loss biases the model to learn from both phonotactics and acoustics. We hypothesize that ZIPAs’ Zipformer-based downsampling bottleneck can be another possible reason for ZIPA’s high PFER. We also observe that the insertion rates for different models follow the same trend as the curves in Figure 2, showing that POWSM and ZIPA produce phonetic transcriptions even when there is no input speech. Interestingly, ZIPA-CTC-NS and POWSM perform best on unseen languages, but POWSM struggles with seen-language variation, and ZIPA-CTC-NS underperforms on pathological speech. This aligns with the idea that some tasks benefit from learned phonological patterns, while others depend more on capturing acoustic information.

  ### 6.2 Zero-Shot Phonetic Inventory Induction

  Identifying the inventory of phones in a new language is an important linguistic application and often an early step toward developing a standardized transcription system for it. Such a task requires PR models to recognize phones correctly in unseen phonetic environments. Therefore, it relies on the phonetic diversity the models have seen in the input speech signal during training. We explore these behaviors in this set of experiments.

  Figure 3: Precision and Recall scores of PR systems on phone inventory induction for unseen languages (§6.2). CTC models trained with highly multilingual data are more stable.

  Our dataset, derived from DoReCo, consists of low-resource languages absent from the training corpora of all models. The transcripts from all models are used to compute the phone inventory after applying PanPhon-based phone tokenization (Mortensen et al., 2016) followed by a set union over detected phones. The ground truth inventory is constructed similarly using the phonetic transcriptions provided by DoReCo and set similarity metrics (§B.1) are computed. We show the macro-averaged values in Figure 3.

  POWSM-CTC emerges as the strongest model. The large gap between POWSM-CTC and POWSM (which differ only in architecture) suggests that the encoder-only architecture plays a crucial role in high precision transcripts even in an unseen phonetic environment. As for ZIPAs, which differ in training data, ZIPA-CTC-NS is more precise than ZIPA-CTC. The extended multilingual training of ZIPA-CTC-NS on pseudo-labeled data leads to more precise phone predictions for unseen languages. This suggests that noisy pseudo-labels allow for improved precision for new languages. Similar trends is seen in comparing W2V2P-XLSR53 vs W2V2P-LV60 and MultiIPA, where multilingual SSL alone is insufficient for MultiIPA. Essentially, broader language coverage in both pre-training and supervised training result in a more precise model. Although the size of IPAPack++ (17k hr) is much smaller than that used for Wav2Vec2Phs (~160k hr), the larger number of languages in the supervised training stage (88 vs ~40) leads to better recall for ZIPAs, compared to Wav2Vec2Phs. This suggests that diversity of languages is as important as the volume of data. Most models have a high recall (>> 70) and low precision (<< 50), suggesting that most predicted phones are incorrect and predictions have a high entropy.

  ### 6.3 Geolocation for Dialectal Speech

  In Table 4, we observe that our TPs significantly outperform the RPs on Hindi dialectal geolocation Foley et al. (2024), where the former observes an average error of 146 km, while the latter observes an average error of 253 km. As a reference, our data is spread over 1478 km (East-West) and 1703 km (North-South), covering the entire Hindi speaking region of India. This performance is surprising, as the cascade-based approach loses suprasegmental information such as intonation that provide strong phonetic cues for the differentiation of dialects (Vicenik and Sundara, 2013; Grabe and Post, 2002). However, our results provide empirical evidence that morphological and phonetic differences suffice for fine-grained differentiation between Hindi dialects (Gumperz, 1958).

  Figure 4: Attribution map from Vaani (Ghosh et al., 2025). Red supports and blue opposes correct geolocation. W2V2P-LV60 detects doubled phones (§6.3).

  We hypothesize that part of the reason why hidden representations underperform cascade is also due to the downstream probe, where the RP employs attention pooling with an MLP, while the TP employs an RNN. As the RNN preserves phone order information, even in the case where two dialects share similar phoneme inventories, distributional differences of phone sequences between the dialects can be leveraged for fine-grained differentiation(Gumperz, 1958; Shim et al., 2024). We further analyze this behavior by employing integrated gradient based attribution maps Sundararajan et al. (2017) on TP. There is a tendency of pronouncing two consonant sounds instead of one in the Bangru dialect of Haryanvi Devi and Mishra (2021). For example, the English loan word “Cooler” \tipaencoding[ku:lar] becomes \tipaencoding[kullar], while the Hindi word “Rakh\tipaencoding ā” \tipaencoding[R@.k h a:] (kept) becomes \tipaencoding[R@k.k h a:]. Figure 4 shows attribution map for an utterance from GEO-v. Speaker utters these words in their native accent, W2V2P-LV60 outputs \tipaencoding[ll] and \tipaencoding[kk], and TP aligns with one of the doubled phones. We leave a more detailed interpretability analysis to future work.

  ### 6.4 LALMs lack phonetic perception

  We examine the zero-shot predictions of LALMs on two tasks: GEO-v and L1-eda. On GEO-v, LALMs perform near chance level, whereas on L1-eda, Gemini 2.5 Flash achieves the strongest performance.

  On GEO-v, both models exhibit geographic mode collapse. Qwen3-Omni-Instruct predicts New Delhi for nearly all inputs, while Gemini 2.5 Flash attains only 6.5% hit@1 accuracy, with roughly 65% of its predictions concentrated in 3–4 coordinate clusters near New Delhi (28.6°N, 77.2°E). This pattern suggests that LALMs have limited sensitivity to dialectal variations and are strongly biased toward higher-resourced dialects.

  Similarly, on L1-eda, LALMs show a pronounced bias toward the Romance accent cluster, with 25.8% of Slavic/Balkan and 28.5% of South Asian accents misclassified as Romance. Enabling thinking mode exacerbates rather than mitigates such biases by creating more attractor classes. As a result, the F1-score on L1-eda drops from 32.7% to 24.9%. Analysis of the reasoning traces reveals that the model over-relies on surface-level phonetic cues, mentioning “Spanish/Italian/Portuguese” in 87% of erroneous Romance predictions and citing “syllable-timed rhythm” in 65% of cases, leading to conflation of phonetically diverse accents. The confusion matrices for both models are shown in Appendix G. These findings suggest that LALMs lack the fine-grained acoustic perception, limiting their reliability for tasks requiring unbiased phonetic discrimination.

  7 Conclusion
  ------------

  We introduce PRiSM, the first standardized benchmark to measure capabilities of PR systems on transcription task and downstream task performance. We also open-source our datasets in an easy-to-use format with our toolkit. Our evaluations reveal that models behave differently on PR and on downstream applications. Therefore, we recommend that models be benchmarked in both categories to make comparisons. Our results and analysis show that PR for seen language benefits from outputs grounded in familiar patterns, whereas unseen languages’ rely on multilingual training and learned phonological patterns. Broad and diverse language coverage, along with encoder-CTC architectures, improves stability across tasks, while LALMs currently lag behind specialized PR models. Together, these findings highlight the value of PRiSM as a framework for evaluating PR systems across diverse languages, tasks, and architectures.

  Limitations
  -----------

  While PRiSM evaluates PR systems across a range of intrinsic and extrinsic settings, it is constrained by the availability of curated datasets. As a result, coverage of languages, dialects, accents, and speaking styles remains incomplete and may reflect biases present in the underlying corpora.

  In addition, phonetic transcription does not constitute a single objective ground truth: it depends on annotation guidelines, annotator judgments, and the chosen phone inventory. The IPA-based interface may also miss or normalize away language-specific or gradient phonetic phenomena.

  Both intrinsic and extrinsic evaluations are necessary to assess PR systems, but each has limitations. Transcript probes align with linguistic features, yet they may also overfit to spurious cues (e.g., sequence length) when datasets are biased or transcripts are noisy due to low PR quality. For representation probes, phonetic information may be distributed across different layers, and performance can depend on the chosen fusion or pooling strategy. Models may benefit from task-specific decoding hyperparameters and prompts, whereas we use default settings and prompts that only contain key instructions. Our goal is to assess fundamental phonetic capabilities and provide comparative insights; we do not claim that the reported results reflect the best possible performance achievable for each model.

  ### Ethics Statement

  All data used in this work are ethically sourced, either through permissive licensing or with proper consent. Speech datasets, particularly those involving pathological speech, may contain sensitive personal information, and we strictly adhere to the licenses and usage conditions associated with each dataset. PR systems may be misapplied in ways that unfairly label speakers without appropriate expert supervision, especially in educational, clinical, demographic, or geographic contexts. We introduce PRiSM with the goal of supporting responsible and rigorous research, and we encourage its use to advance speech technologies that consider linguistic and cultural diversity regardless of resource availability.

  ### The Use of LLMs

  We acknowledge the use of large language models (LLMs) to assist with refinement of the writing, including grammar correction and clarity improvements. We also used LLMs as coding assistants. All the code was then verified by authors. All conceptual, methodological, and experimental work was done independently by the authors.

  8 Acknowledgement
  -----------------

  We thank Jinchuan and Haoran for their support with vLLM, and Brian Yan and Brian Cho for helpful discussions. This work was supported by National Science Foundation grant #2504019. This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (RS-2022-00143911, AI Excellence Global Innovative Leader Education Program). We also acknowledge the Delta and DeltaAI systems, and support from the NVIDIA Academic Hardware Grant Program 2025. This work used the Delta and DeltaAI systems at NCSA through allocations CIS210014 and IRI120008P from the Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support (ACCESS) program.

  References
  ----------

  *   P. Angkititrakul and J. H. Hansen (2006)Advances in phone-based modeling for automatic accent classification. IEEE transactions on audio, speech, and language processing 14 (2),  pp.634–646. Cited by: §3.2. 
  *   S. Arora, K. Chang, C. Chien, Y. Peng, H. Wu, Y. Adi, E. Dupoux, H. Lee, K. Livescu, and S. Watanabe (2025)On the landscape of spoken language models: a comprehensive survey. arXiv preprint arXiv:2504.08528. Cited by: §1. 
  *   A. Baevski, Y. Zhou, A. Mohamed, and M. Auli (2020)Wav2vec 2.0: a framework for self-supervised learning of speech representations. Advances in neural information processing systems 33,  pp.12449–12460. Cited by: §2.1, 1st item. 
  *   M. Ball, N. Müller, M. Klopfenstein, and B. Rutter (2009)The importance of narrow phonetic transcription for highly unintelligible speech: some examples. Logopedics Phoniatrics Vocology 34 (2),  pp.84–90. Cited by: §3.2. 
  *   B. Bunzeck, D. Duran, L. Schade, and S. Zarrieß (2025)Small language models also work with small vocabularies: probing the linguistic abilities of grapheme-and phoneme-based baby llamas. In Proceedings of the 31st International Conference on Computational Linguistics,  pp.6039–6048. Cited by: §2.3. 
  *   S. Chen, C. Wang, Z. Chen, Y. Wu, S. Liu, Z. Chen, J. Li, N. Kanda, T. Yoshioka, X. Xiao, et al. (2022)Wavlm: large-scale self-supervised pre-training for full stack speech processing. IEEE Journal of Selected Topics in Signal Processing 16 (6),  pp.1505–1518. Cited by: §2.2, 5th item. 
  *   C. J. Cho, A. Mohamed, A. W. Black, and G. K. Anumanchipalli (2024)Self-supervised models of speech infer universal articulatory kinematics. In ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP),  pp.12061–12065. Cited by: §2.2. 
  *   E. Chodroff, B. Pažon, A. Baker, and S. Moran (2024)Phonetic segmentation of the ucla phonetics lab archive. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024),  pp.12724–12733. Cited by: §A.2, Table 1. 
  *   K. Choi, A. Pasad, T. Nakamura, S. Fukayama, K. Livescu, and S. Watanabe (2024)Self-Supervised Speech Representations are More Phonetic than Semantic. In Interspeech 2024,  pp.4578–4582. External Links: Document, ISSN 2958-1796 Cited by: §2.2. 
  *   K. Choi, E. Yeo, K. Chang, S. Watanabe, and D. R. Mortensen (2025)Leveraging allophony in self-supervised speech models for atypical pronunciation assessment. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), L. Chiruzzo, A. Ritter, and L. Wang (Eds.), Albuquerque, New Mexico,  pp.2613–2628. External Links: Link, Document, ISBN 979-8-89176-189-6 Cited by: §1. 
  *   N. Chomsky and M. Halle (1968)The sound pattern of English.. Cited by: footnote 6. 
  *   G. Comanici, E. Bieber, M. Schaekermann, I. Pasupat, N. Sachdeva, I. Dhillon, M. Blistein, O. Ram, D. Zhang, E. Rosen, et al. (2025)Gemini 2.5: pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261. Cited by: Table 2. 
  *   A. Conneau, M. Ma, S. Khanuja, Y. Zhang, V. Axelrod, S. Dalmia, J. Riesa, C. Rivera, and A. Bapna (2023)Fleurs: few-shot learning evaluation of universal representations of speech. In 2022 IEEE Spoken Language Technology Workshop (SLT),  pp.798–805. Cited by: §A.3, Table 1. 
  *   S. Devi and U. Mishra (2021)Dialects of haryanvi language: a comparative study. Journal of Advances and Scholarly Researches in Allied Education 18 (6),  pp.221–223. External Links: Link Cited by: §6.3. 
  *   B. Dodd (2013)Differential diagnosis and treatment of children with speech disorder. John Wiley & Sons. Cited by: §2.2. 
  *   A. Eshky, M. S. Ribeiro, J. Cleland, K. Richmond, Z. Roxburgh, J. Scobbie, and A. Wrench (2018)UltraSuite: a repository of ultrasound and acoustic data from child speech therapy sessions. In Interspeech 2018,  pp.1888–1892. Cited by: §A.3, Table 1. 
  *   P. Foley, M. Wiesner, B. Odoom, L. P. Garcia Perera, K. Murray, and P. Koehn (2024)Where are you from? geolocating speech and applications to language identification. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), K. Duh, H. Gomez, and S. Bethard (Eds.), Mexico City, Mexico,  pp.5114–5126. External Links: Link, Document Cited by: §A.3, Appendix C, §6.3. 
  *   H. Franco, H. Bratt, R. Rossier, V. R. Gadde, E. Shriberg, V. Abrash, and K. Precoda (2010)EduSpeak®: a speech recognition and pronunciation scoring toolkit for computer-aided language learning applications. Language Testing 27,  pp.401 – 418. External Links: Link Cited by: §3.2. 
  *   H. Gao, J. Ni, Y. Zhang, K. Qian, S. Chang, and M. Hasegawa-Johnson (2021)Zero-shot cross-lingual phonetic recognition with external language embedding. In Interspeech 2021,  pp.1304–1308. External Links: Document, ISSN 2958-1796 Cited by: §2.1. 
  *   J. S. Garofolo, L. F. Lamel, W. M. Fisher, D. S. Pallett, N. L. Dahlgren, V. Zue, and J. G. Fiscus (1993)TIMIT acoustic-phonetic continuous speech corpus. Cited by: §A.2, Table 1, §6.1. 
  *   P. K. Ghosh, R. Dharmaraju, N. Desai, et al. (2025)VAANI: capturing the language landscape for an inclusive digital india. Note: https://vaani.iisc.ac.in/Cited by: §A.3, Appendix F, Table 1, Figure 4. 
  *   Z. Goriely and P. Buttery (2025)BabyLM’s first words: word segmentation as a phonological probing task. In The SIGNLL Conference on Computational Natural Language Learning, Cited by: §2.3. 
  *   E. Grabe and B. Post (2002)Intonational variation in the british isles. In Speech prosody,  pp.343–346. Cited by: §6.3. 
  *   J. J. Gumperz (1958)Phonological differences in three hindi dialects. Language 34,  pp.212. External Links: Link Cited by: §6.3, §6.3. 
  *   A. Holtzman, J. Buys, L. Du, M. Forbes, and Y. Choi (2020)The curious case of neural text degeneration. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020, External Links: Link Cited by: §5.1. 
  *   C. Huang, W. Chen, S. Yang, A. T. Liu, C. Li, Y. Lin, W. Tseng, A. Diwan, Y. Shih, J. Shi, et al. (2025)Dynamic-superb phase-2: a collaboratively expanding benchmark for measuring the capabilities of spoken language models with 180 tasks. In The Thirteenth International Conference on Learning Representations, Cited by: §2.1, §2.3. 
  *   S. Inceoglu, W. Chen, and H. Lim (2023)Assessment of l2 intelligibility: comparing l1 listeners and automatic speech recognition. ReCALL: the Journal of EUROCALL 35 (1),  pp.89–104. Cited by: §1. 
  *   International Phonetic Association (1999)Handbook of the international phonetic association: a guide to the use of the international phonetic alphabet. Cambridge University Press. Cited by: §2.1. 
  *   H. Kim, M. Hasegawa-Johnson, A. Perlman, J. Gunderson, T. S. Huang, K. Watkin, and S. Frame (2008)Dysarthric speech database for universal access research. In Interspeech 2008,  pp.1741–1744. External Links: Document, ISSN 2958-1796 Cited by: §A.3, Table 1. 
  *   J. Kominek and A. W. Black (2004)The cmu arctic speech databases. In SSW,  pp.223–224. Cited by: §A.3, Table 1. 
  *   P. Ladefoged, B. Blankenship, R. G. Schuh, P. Jones, N. Gfroerer, E. Griffiths, L. Harrington, C. Hipp, M. Kaneko, C. Moore-Cantwell, G. Oh, K. Pfister, K. Vaughan, R. Videc, S. Weismuller, S. Weiss, J. White, S. Conlon, W. J. Lee, and R. Toribio (2009)The UCLA Phonetics Lab Archive. External Links: Link Cited by: §A.2. 
  *   M. Lavechin, Y. Sy, H. Titeux, M. A. C. Blandón, O. Räsänen, H. Bredin, E. Dupoux, and A. Cristia (2023)BabySLM: language-acquisition-friendly benchmark of self-supervised spoken language models. In INTERSPEECH 2023,  pp.4588–4592. Cited by: §2.3. 
  *   J. Lee and S. Watanabe (2021)Intermediate loss regularization for ctc-based speech recognition. In ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP),  pp.6224–6228. Cited by: Table 2. 
  *   T. Lee, H. Tu, C. H. Wong, Z. Wang, S. Yang, Y. Mai, Y. Zhou, C. Xie, and P. Liang (2025)Ahelm: a holistic evaluation of audio-language models. arXiv preprint arXiv:2508.21376. Cited by: 4th item. 
  *   C. Li, K. Chang, S. Bharadwaj, E. Yeo, K. Choi, J. Zhu, D. Mortensen, and S. Watanabe (2025)POWSM: a phonetic open whisper-style speech foundation model. External Links: 2510.24992, Link Cited by: §1, Table 2, §6.1. 
  *   X. Li, S. Dalmia, J. Li, M. Lee, P. Littell, J. Yao, A. Anastasopoulos, D. R. Mortensen, G. Neubig, A. W. Black, et al. (2020)Universal phone recognition with a multilingual allophone system. In ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP),  pp.8249–8253. Cited by: §2.1. 
  *   X. Li, F. Metze, D. R. Mortensen, A. W. Black, and S. Watanabe (2022)Asr2k: speech recognition for around 2000 languages without audio. arXiv preprint arXiv:2209.02842. Cited by: §1. 
  *   R. Ma, M. Qian, Y. Fathullah, S. Tang, M. Gales, and K. Knill (2025)Cross-lingual transfer learning for speech translation. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 2: Short Papers), L. Chiruzzo, A. Ritter, and L. Wang (Eds.), Albuquerque, New Mexico,  pp.33–43. External Links: Link, Document, ISBN 979-8-89176-190-2 Cited by: §2.2. 
  *   D. R. Mortensen, S. Dalmia, and P. Littell (2018)Epitran: precision g2p for many languages. In Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018), Cited by: §2.1. 
  *   D. R. Mortensen, J. Picone, X. Li, and K. Siminyu (2021)Tusom2021: a phonetically transcribed speech dataset from an endangered language for universal phone recognition experiments. In Proc. Interspeech 2021,  pp.3660–3664. Cited by: §A.2, Table 1. 
  *   D. R. Mortensen, P. Littell, A. Bharadwaj, K. Goyal, C. Dyer, and L. S. Levin (2016)PanPhon: A resource for mapping IPA segments to articulatory feature vectors. In Proceedings of COLING 2016, the 26th International Conference on Computational Linguistics: Technical Papers,  pp.3475–3484. Cited by: §3.1, §6.2. 
  *   T. L. Nelson, Z. Mok, and K. Ttofari Eecen (2020)Use of transcription when assessing children’s speech: australian speech-language pathologists’ practices, challenges, and facilitators. Folia Phoniatrica et Logopaedica 72 (2),  pp.131–142. Cited by: §3.2. 
  *   T. A. Nguyen, M. de Seyssel, P. Rozé, M. Rivière, E. Kharitonov, A. Baevski, E. Dunbar, and E. Dupoux (2020)The zero resource speech benchmark 2021: metrics and baselines for unsupervised spoken language modeling. In NeuRIPS Workshop on Self-Supervised Learning for Speech and Audio Processing, Cited by: §2.3. 
  *   L. Paschen, F. Delafontaine, C. Draxler, S. Fuchs, M. Stave, and F. Seifart (2020)Building a time-aligned cross-linguistic reference corpus from language documentation data (doreco). In Proceedings of the 12th Conference on Language Resources and Evaluation (LREC 2020), Cited by: §A.2, Table 1, Table 1. 
  *   J. Peng, Y. Wang, Y. Fang, Y. Xi, X. Li, X. Zhang, and K. Yu (2024)A survey on speech large language models. arXiv preprint arXiv:2410.18908. Cited by: §1. 
  *   A. Petrov, P. H. Torr, and A. Bibi (2023)When do prompting and prefix-tuning work? a theory of capabilities and limitations. arXiv preprint arXiv:2310.19698. Cited by: 4th item. 
  *   T. Pimentel, B. Roark, and R. Cotterell (2020)Phonotactic complexity and its trade-offs. Transactions of the Association for Computational Linguistics 8,  pp.1–18. Cited by: §6.1. 
  *   A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever (2023)Robust speech recognition via large-scale weak supervision. In International conference on machine learning,  pp.28492–28518. Cited by: §2.1, §2.2, 5th item. 
  *   K. Rosero, A. N. Salman, S. Chandra, B. Sisman, C. Van’t Slot, A. A. Kane, R. R. Hallac, and C. Busso (2025a)Advancing pediatric asr: the role of voice generation in disordered speech. In Proceedings of the Annual Conference of the International Speech Communication Association, INTERSPEECH,  pp.2890–2894. Cited by: §A.3. 
  *   K. Rosero, E. Yeo, D. R. Mortensen, C. V. Slot, R. R. Hallac, and C. Busso (2025b)Finding my voice: generative reconstruction of disordered speech for automated clinical evaluation. arXiv preprint arXiv:2509.19231. Cited by: §A.3. 
  *   R. Sanabria, N. Bogoychev, N. Markl, A. Carmantini, O. Klejch, and P. Bell (2023)The edinburgh international accents of english corpus: towards the democratization of english asr. In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP),  pp.1–5. Cited by: §A.3, Table 1. 
  *   T. Schultz and K. Kirchhoff (2006)Multilingual speech processing. Elsevier. Cited by: §3.2. 
  *   X. Shi, F. Yu, Y. Lu, Y. Liang, Q. Feng, D. Wang, Y. Qian, and L. Xie (2021)The accented english speech recognition challenge 2020: open datasets, tracks, baselines, results and methods. In IEEE International Conference on Acoustics, Speech, and Signal Processing, External Links: Document Cited by: §A.3. 
  *   R. S. Shim, K. Chang, and D. R. Mortensen (2024)Phonotactic complexity across dialects. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024),  pp.12734–12748. Cited by: §6.3. 
  *   L. D. Shriberg, R. D. Kent, T. McAllister, J. L. Preston, and M. L. Speights (2025)Clinical phonetics. Plural Publishing. Cited by: §1. 
  *   L. D. Shriberg and G. L. Lof (1991)Reliability studies in broad and narrow phonetic transcription. Clinical Linguistics & Phonetics 5 (3),  pp.225–279. Cited by: §5.2. 
  *   M. Sundararajan, A. Taly, and Q. Yan (2017)Axiomatic attribution for deep networks. In International conference on machine learning,  pp.3319–3328. Cited by: §6.3. 
  *   A. Suvarna, H. Khandelwal, and N. Peng (2024)PhonologyBench: evaluating phonological skills of large language models. In Proceedings of the 1st Workshop on Towards Knowledgeable Language Models (KnowLLM 2024),  pp.1–14. Cited by: §2.3. 
  *   C. Taguchi, Y. Sakai, P. Haghani, and D. Chiang (2023)Universal automatic phonetic transcription into the international phonetic alphabet. In Interspeech 2023,  pp.2548–2552. External Links: Document, ISSN 2958-1796 Cited by: §1, Table 2. 
  *   M. Tu, A. Grabek, J. Liss, and V. Berisha (2018)Investigating the role of l1 in automatic pronunciation evaluation of l2 speech. arXiv preprint arXiv:1807.01738. Cited by: §1. 
  *   J. Turian, J. Shier, H. R. Khan, B. Raj, B. W. Schuller, C. J. Steinmetz, C. Malloy, G. Tzanetakis, G. Velarde, K. McNally, et al. (2022)Hear: holistic evaluation of audio representations. In NeurIPS 2021 Competitions and Demonstrations Track,  pp.125–145. Cited by: §3.2. 
  *   R. Turrisi, A. Braccia, M. Emanuele, S. Giulietti, M. Pugliatti, M. Sensi, L. Fadiga, and L. Badino (2021)EasyCall corpus: a dysarthric speech dataset. In Interspeech 2021,  pp.41–45. External Links: Document, ISSN 2958-1796 Cited by: §A.3, Table 1. 
  *   C. Vicenik and M. Sundara (2013)The role of intonation in language and dialect discrimination by adults. Journal of Phonetics 41 (5),  pp.297–306. Cited by: §6.3. 
  *   B. Wang, X. Zou, G. Lin, S. Sun, Z. Liu, W. Zhang, Z. Liu, A. Aw, and N. Chen (2025)Audiobench: a universal benchmark for audio large language models. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers),  pp.4297–4316. Cited by: §2.1. 
  *   S. Watanabe, T. Hori, S. Kim, J. R. Hershey, and T. Hayashi (2017)Hybrid ctc/attention architecture for end-to-end speech recognition. IEEE Journal of Selected Topics in Signal Processing 11 (8),  pp.1240–1253. External Links: Document Cited by: Table 2. 
  *   S. Weinberger (2015)Speech accent archive. Note: Retrieved from https://accent.gmu.edu Cited by: §A.2, Table 1. 
  *   J. Xu, Z. Guo, H. Hu, Y. Chu, X. Wang, J. He, Y. Wang, X. Shi, T. He, X. Zhu, Y. Lv, Y. Wang, D. Guo, H. Wang, L. Ma, P. Zhang, X. Zhang, H. Hao, Z. Guo, B. Yang, B. Zhang, Z. Ma, X. Wei, S. Bai, K. Chen, X. Liu, P. Wang, M. Yang, D. Liu, X. Ren, B. Zheng, R. Men, F. Zhou, B. Yu, J. Yang, L. Yu, J. Zhou, and J. Lin (2025)Qwen3-omni technical report. External Links: 2509.17765, Link Cited by: Table 2. 
  *   Q. Xu, A. Baevski, and M. Auli (2022)Simple and effective zero-shot cross-lingual phoneme recognition. In Interspeech 2022,  pp.2113–2117. External Links: Document, ISSN 2958-1796 Cited by: Table 2, Table 2. 
  *   W. Xue, R. van Hout, C. Cucchiarini, and H. Strik (2023)Assessing speech intelligibility of pathological speech in sentences and word lists: the contribution of phoneme-level measures. Journal of Communication Disorders 102,  pp.106301. Cited by: §A.3. 
  *   C. Yang, N. S. Ho, and H. Lee (2025)Towards holistic evaluation of large audio-language models: a comprehensive survey. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, Suzhou, China,  pp.10155–10181. External Links: Link, Document Cited by: §2.1. 
  *   M. Yang, R. C. M. C. Shekar, O. Kang, and J. H. L. Hansen (2023)What can an accent identifier learn? probing phonetic and prosodic information in a wav2vec2-based accent identification model. In INTERSPEECH 2023,  pp.2. External Links: Document Cited by: §A.3. 
  *   S. Yang, P. Chi, Y. Chuang, C. J. Lai, K. Lakhotia, Y. Y. Lin, A. T. Liu, J. Shi, X. Chang, G. Lin, T. Huang, W. Tseng, K. Lee, D. Liu, Z. Huang, S. Dong, S. Li, S. Watanabe, A. Mohamed, and H. Lee (2021)SUPERB: speech processing universal performance benchmark. In Interspeech 2021,  pp.1194–1198. External Links: Document, ISSN 2958-1796 Cited by: §2.3. 
  *   Z. Yao, L. Guo, X. Yang, W. Kang, F. Kuang, Y. Yang, Z. Jin, L. Lin, and D. Povey (2024)Zipformer: a faster and better encoder for automatic speech recognition. International Conference on Learning Representations. Cited by: §6.1. 
  *   Z. Yao, W. Kang, X. Yang, F. Kuang, L. Guo, H. Zhu, Z. Jin, Z. Li, L. Lin, and D. Povey (2025)CR-ctc: consistency regularization on ctc for improved speech recognition. In The Thirteenth International Conference on Learning Representations, Cited by: Table 2, §6.1. 
  *   S. Yusuyin, T. Ma, H. Huang, W. Zhao, and Z. Ou (2025)Whistle: data-efficient multilingual and crosslingual speech recognition via weakly phonetic supervision. IEEE Transactions on Audio, Speech and Language Processing. Cited by: §1. 
  *   P. Żelasko, S. Feng, L. M. Velazquez, A. Abavisani, S. Bhati, O. Scharenborg, M. Hasegawa-Johnson, and N. Dehak (2022)Discovering phonetic inventories with crosslingual automatic speech recognition. Computer speech & language 74,  pp.101358. Cited by: §B.1. 
  *   J. Zhang, Z. Zhang, Y. Wang, Z. Yan, Q. Song, Y. Huang, K. Li, D. Povey, and Y. Wang (2021a)Speechocean762: an open-source non-native english speech corpus for pronunciation assessment. In Proc. Interspeech 2021,  pp.3710–3714. Cited by: Table 1. 
  *   J. Zhang, Z. Zhang, Y. Wang, Z. Yan, Q. Song, Y. Huang, K. Li, D. Povey, and Y. Wang (2021b)Speechocean762: an open-source non-native english speech corpus for pronunciation assessment. External Links: 2104.01378, Link Cited by: §A.3. 
  *   G. Zhao, S. Sonsaat, A. Silpachai, I. Lucic, E. Chukharev-Hudilainen, J. M. Levis, and R. Gutierrez-Osuna (2018a)L2-arctic: a non-native english speech corpus. In Interspeech, External Links: Document Cited by: §A.3. 
  *   G. Zhao, S. Sonsaat, A. Silpachai, I. Lucic, E. Chukharev-Hudilainen, J. Levis, and R. Gutierrez-Osuna (2018b)L2-arctic: a non-native english speech corpus. In Proc. Interspeech 2018,  pp.2783–2787. Cited by: §A.2, Table 1, Table 1. 
  *   J. Zhu, F. Samir, E. Chodroff, and D. R. Mortensen (2025)ZIPA: a family of efficient models for multilingual phone recognition. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), W. Che, J. Nabende, E. Shutova, and M. T. Pilehvar (Eds.), Vienna, Austria,  pp.19568–19585. External Links: Link, Document, ISBN 979-8-89176-251-0 Cited by: §A.2, §1, Table 2, Table 2, §6.1. 
  *   J. Zhu, C. Yang, F. Samir, and J. Islam (2024)The taste of IPA: towards open-vocabulary keyword spotting and forced alignment in any language. In Proc. NAACL, K. Duh, H. Gomez, and S. Bethard (Eds.), Mexico City, Mexico,  pp.750–772. Cited by: §A.2, Table 5, §2.1. 
  *   J. Zhu, C. Zhang, and D. Jurgens (2022)ByT5 model for massively multilingual grapheme-to-phoneme conversion. In Interspeech 2022,  pp.446–450. External Links: Document, ISSN 2958-1796 Cited by: §2.1, §5.1. 

  Appendix A Dataset details and Licenses
  ---------------------------------------

  This section introduces the datasets and the motivation of downstream tasks. Table 5 lists the licensing information and dataset size.

  Table 5: Licence and split size (#\#utterance). ∗DoReCo includes datasets of different CC licences; we use the 45-language subset created by Zhu et al. (2024). See §A.1 for dataset links.

  ### A.1 Data Links

  *   •
  *   •
  *   •

  ### A.2 Datasets in Intrinsic Evaluation

  #### Variation in Seen Language

  TIMIT (Garofolo et al., 1993) contains speech from six regional varieties of American English and is often used for PR evaluation. The Speech Accent Archive (Weinberger, 2015) provides read speech (the “Please call Stella” passage) and narrow phonetic transcriptions from non-native English speakers across 391 L1 languages. L2-ARCTIC Zhao et al. (2018b) includes read speech from non-native speakers; we use the L2-Arctic Perceived set 7 7 7 https://huggingface.co/anyspeech which consists of manually annotated phoneme transcriptions rather than standard G2P output.

  #### Unseen Languages

  DoReCo (Paschen et al., 2020) is a dataset of 50+ small or endangered languages with broad phonetic transcriptions; we use the same DoReCo subset as Zhu et al. (2025, 2024). VoxAngeles (Chodroff et al., 2024) is a cleaned, 95-language version of the UCLA Phonetics Lab Archive (Ladefoged et al., 2009). Tusom2021 (Mortensen et al., 2021) is a dataset of speech and narrow phonetic transcriptions of individual words in the low-data Tangkhulic language Tusom. We removed tones as none of the models supports them.

  ### A.3 Datasets in Extrinsic Evaluation

  #### Pathological Speech Assessment

  Dysarthria intelligibility prediction predicts dysarthria severity levels based on phonetic representations. Increasing dysarthria severity is associated with reduced intelligibility, for which impaired phoneme production is a major clinical predictor Xue et al. (2023). Two dysarthric speech datasets are evauated: UASpeech Kim et al. (2008), an English corpus with speaker-level intelligibility scores, and EasyCall Turrisi et al. (2021), an Italian corpus annotated with dysarthria severity ratings. Child speech disorder detection classifies whether a given utterance is produced by a child with speech disorder, supporting applications in speech therapy and the selection of specialized speech models Rosero et al. (2025b, a). We use acoustic recordings from the Ultrasuite corpus Eshky et al. (2018), with manually corrected transcription-audio mismatches. The curated dataset is released with this paper.

  #### L2 Speech Evaluation

  Proficiency assessment for L2 Learners uses phonetic information to automatically assess L2 English proficiency. We use utterances and sentence-level scores on a 0-10 scale from Speechocean762 Zhang et al. (2021b), an L1 Chinese, L2 English corpus. L1 influence classification classifies a speaker’s L1 (native language) background, which introduces distinctive articulatory patterns into speech in an L2 language Yang et al. (2023); Shi et al. (2021). We use EdAcc Sanabria et al. (2023) for one setup, and the other combines L2-ARCTIC Zhao et al. (2018a) for non-native speech with CMU ARCTIC Kominek and Black (2004) for native speech.

  #### Multilingual Speech Identification

  Language identification (LID) predicts the language spoken in an utterance from audio input. We use it as a coarse-grained evaluation of whether phonetic representations can distinguish both seen and unseen languages with the 102 languages in FLEURS Conneau et al. (2023). Speech geolocation identification predicts the origin of a speaker from an utterance in their native language, drawing on systematic phonetic shifts associated with geography, sociolinguistic variation, and language contact (Foley et al., 2024). We use data from the Hindi-belt of India from Vaani Ghosh et al. (2025). The detailed algorithm for this subset creation is explained in Appendix F. Phone inventory induction is the task of inferring the set of phones used by language from speech recordings, which is useful for language documentation and helps identify systematic errors during evaluation. We use DoReCo (§B.1) by deriving phone inventories from gold phone transcriptions and comparing them against the predicted transcriptions for each language.

  Appendix B Metrics
  ------------------

  ### B.1 Task Metric: F1 of Phone Inventory (F1-PI)

  A phone inventory is the set of all phones used in a language. F1-PI assesses the degree of overlap between the phones transcribed by a system for a given language and the ground truth phone inventory for that language (Żelasko et al., 2022). For two sets A A and B B, the F1-score is defined as the harmonic mean of |A−B|/|A||A-B|/|A| and |B−A|/|B||B-A|/|B|. Set membership can be based on exact matches or fuzzy matches (e.g., over phonetic features). This metric requires only a reference inventory for the target language, not a full transcription (although inventories can be derived from transcriptions), making it especially useful for under-resourced languages.

  ### B.2 Summary Metric: PRiSM Extrinsic Score

  To aggregate performance across extrinsic evaluation tasks with significantly varying test set sizes (N i N_{i}) (Table 5), we compute a Score using a logarithmically weighted average. This approach ensures that larger datasets contribute more to the final score due to their statistical significance, while preventing them from completely dominating smaller, high-variance datasets (such as CSD-us).

  Let s i s_{i} be the model performance on task i i and N i N_{i} be the number of samples in that task. The aggregate score S S is defined as:

  S=∑i=1 K ln⁡(N i)⋅s i∑i=1 K ln⁡(N i)S=\frac{\sum_{i=1}^{K}\ln(N_{i})\cdot s_{i}}{\sum_{i=1}^{K}\ln(N_{i})}(2)

  where K=6 K=6 corresponds to the tasks (DYS-ez, DYS-ua, CSD-us, L1-eda, L1-arc, and L2-so) that show differentiation in model behavior. The weights w i=ln⁡(N i)w_{i}=\ln(N_{i}) dampen the linear disparity between the largest (N=7762 N=7762) and smallest (N=287 N=287) test sets.

  Appendix C Experimental Setup
  -----------------------------

  #### Probe Details

  All transcript probes use a 2 layer bi-directional GRU with mean pooling to get transcript level representation. The GRU operates on a character vocabulary built from all predicted transcripts. GRU has a hidden dimension of 256 and input dimension of 128 with a dropout of 0.1.

  For hidden representation probes we use the last layer’s hidden representation and attention pool over time to obtain utterance level representation. This is followed by an MLP composed of 2 linear layers. First layer’s input dimension is the same as the dimension of the model being evaluated. It outputs an embedding half of this size and the final layer outputs a single scalar for assessment taks, logits over classes for classification tasks, or a unit [x y z] vector for geolocation task. MSE loss is employed for regression, cross-entropy for classification and angular error loss Foley et al. (2024) for geolocation.

  #### Hyper-parameters

  All the experiments can be reproduced via our open-sourced toollkit. We use a learning rate of 2e-4 for all hidden representation probes and a learning rate of 1e-3 for the cascade probes. We use the validation F1 (for classification), Kentall Tau (for assessment) and error (in km for geolocation) as early stopping metrics with a patience of 5 epochs and minimum epochs set to 10. The checkpoint achieving best validation values on these metrics is selected for reporting numbers.

  #### Compute spent

  Each TP probe runs in at most 15 minutes on a single 40GB GPU. Each RP probe runs in at most 3 hours on a single 40GB GPU. For TP and RP based extinsic evaluations a total of around 1k GPU hours were spent to get final numbers. We used almost 1k GPU hours during development phase of the evaluation toolkit as well. Besides, PRiSM supports distributed inference that scales to multiple GPUs and supports VLLM 8 8 8 https://github.com/vllm-project/vllm. For inference, we utilized around 500 GPU hours including debugging and development costs. Each POWSM-CTC model trains on 4 nodes with 4-80GB GPU each and takes 1.5 days to train, amounting to 600 GPU hours for one run. We can assume another 2k GPU hours for development and experimentation.

  Appendix D Prompts for LALMs
  ----------------------------

  Appendix E L1 to accent cluster mapping for EdAcc
  -------------------------------------------------

  The EdAcc corpus contains 41 distinct L1 labels, which we consolidate into 13 accent clusters based on phonological and typological similarity. Grouping criteria include language family (e.g., Sino-Tibetan, Austronesian), vowel inventory size (e.g., 5-vowel Romance languages), prosodic patterns (e.g., syllable-timed vs. stress-timed), and shared phonetic transfer patterns to English (e.g., rhoticity, vowel reduction). Table 6 lists the complete mapping.

  Table 6: Mapping from EdAcc L1 labels (41) to 13 accent clusters used in L1-eda.

  Appendix F Algorithm for Vaani-Hi
  ---------------------------------

  For the GEO-v task, we construct Vaani-Hi, a Hindi-belt subset of the Vaani corpus (Ghosh et al., 2025), and release it on Hugging Face.

  #### Sampling

  We focus on 12 Hindi-belt states: Chandigarh, Himachal Pradesh, Delhi, Madhya Pradesh, Jharkhand, Uttarakhand, Bihar, Chhattisgarh, Haryana, Rajasthan, Punjab, and Uttar Pradesh. From each state, we randomly sample up to 4 districts; for each district, we use up to 4 audio shards and take up to 600 utterances per shard (seed 42).

  #### Filtering and Labeling

  We retain only pincodes with more than 450 utterances to ensure sufficient density per location. Each pincode is mapped to latitude/longitude using a pincode metadata table; we assign mean coordinates per pincode as the geolocation target.

  #### Splitting and Preprocessing

  Splits are created within each pincode (75%/10%/15% train/val/test) to avoid location leakage. Audio is resampled to 16 kHz and clipped to a maximum of 20 seconds.

  Appendix G Effect of Thinking Mode on L1-eda Classification
  -----------------------------------------------------------

  Figure 5 provides the full confusion matrices for the LALM bias analysis discussed in §6.4.

  (a) Baseline (F1-score=32.7%)

  (b) Thinking Mode (F1-score=24.9%)

  Figure 5: Normalized confusion matrices for Gemini 2.5 Flash on L1-eda (13 accent clusters). Rows denote true labels; columns denote predictions.
