# K-Function: Joint Pronunciation Transcription and Feedback for Evaluating Kids Language Function

###### Abstract

Evaluating young children's language is challenging for automatic speech recognizers due to high-pitched voices, prolonged sounds, and limited data. We introduce K-Function, a framework that combines accurate sub-word transcription with objective, Large Language Model (LLM)-driven scoring. Its core, Kids-Weighted Finite State Transducer (K-WFST), merges an acoustic phoneme encoder with a phoneme-similarity model to capture child-specific speech errors while remaining fully interpretable. K-WFST achieves a 1.39% phoneme error rate on MyST and 8.61% on Multitudes—an absolute improvement of 10.47% and 7.06% over a greedy-search decoder. These high-quality transcripts are used by an LLM to grade verbal skills, developmental milestones, reading, and comprehension, with results that align closely with human evaluators. Our findings show that precise phoneme recognition is essential for creating an effective assessment framework, enabling scalable language screening for children. Our K-Function framework demo is available at https://chenxukwok.github.io/K-function/.

Index Terms— Phoneme Recognition, WFST, Content Feedback, Language Function

## 1 Introduction

1 in 14 children in the U.S. has a speech or language disorder, such as dyslexia or Autism, that can hinder reading, vocabulary growth, and communication [1, 2, 3, 4]. These delays can affect quality of life, academic success, and future opportunities. Thus, developing a reliable and automatic framework for evaluating kids language function, covering verbal skills, developmental milestones, reading, and comprehension, is essential. It enables early detection of delays and supports educators and specialists in early identification and intervention.

The bottleneck in developing such a framework lies in the accurate transcription of kids speech, which is significantly more challenging than that of adults. This difficulty arises from several unique characteristics of child vocalizations, including higher fundamental frequency, longer phone durations, greater articulatory variability, and persistent data sparsity [5, 6, 7, 8, 9, 10].

Although recent work has produced increasingly robust child-oriented Automatic Speech Recognition (ASR) models [11, 12, 13, 14, 15, 16, 17, 18, 19], most developmentally significant pronunciation errors arise at the sub-word level and are consequently hidden by word-level metrics such as Word Error Rate (WER). Consider the prompt bird /b@rd/: a child who utters /bed/ (“bed”) exhibits a phoneme substitution, while /pb@rd/ (“p-bird”) contains an intrusive plosive. State-of-the-art end-to-end ASR systems often output the canonical word bird for both productions, masking the underlying articulatory deviation. A dedicated sub-word recognizer that reliably recovers phoneme sequences is therefore indispensable for fine-grained kids language function evaluation, where phoneme recognition is the most widely used approach.

Traditional work has primarily focused on fine-tuning children's speech data on pretrained self-supervised speech learning (SSL) models [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]. However, these approaches exhibit poor generalizability to open-domain children's speech. Moreover, directly adopting adult-oriented ASR or phoneme recognition pipelines fails to address allophonic variations and disrupted phoneme-level language structures.

Unlike purely data-driven approaches, the real breakthrough in modeling atypical speech transcription lies in the development of disfluent speech transcription pipelines that incorporate human and linguistic priors [31, 32, 33, 34, 35, 36]. A state-of-the-art example is Dysfluent Weighted Finite State Transducer (D-WFST), which leverages WFST to model disrupted phoneme structures [37]. However, these WFST-based approaches lack robustness in detecting subtle phonetic variations, particularly substitutions and deletions, an issue especially pronounced in downstream applications involving audio with moderate to severe disfluency.

In this work, we present K-Function, a comprehensive framework for evaluating children’s language function. It consists of our newly proposed Kids-Weighted Finite State Transducer (K-WFST), a lightweight and interpretable kids phoneme transcription model—an LLM-boosted automatic language function scoring system, and a feedback component. Together, these modules form a unified assessment framework for generalized language function evaluation in children.

K-Function is evaluated on two child speech datasets MyST [38] and Multitudes [39]. Our framework consistently achieves superior phoneme recognition, enables more accurate language function scoring via Large Language Models (LLMs), and aligns closely with real proctor scores.

Crucially, the high-fidelity transcriptions from K-WFST provide the foundation for a complete assessment-feedback loop, powering sophisticated downstream analyses. These results collectively highlight that accurate sub-word recognition is a crucial, missing cornerstone for kids language function evaluation. Our K-Function makes the following key contributions:

(1) K-WFST: We present a WFST decoder designed specifically for children’s speech, which achieves state-of-the-art phoneme error rates (PER) on both the MyST [38] and Multitudes [39] corpora.

(2) A Unified Assessment Framework: We propose a complete system that integrates our state-of-the-art sub-word transcription (K-WFST) with an advanced LLM-based scoring module to create a robust pipeline for automated language function assessment.

(3) Align with Proctor Scores and Go Beyond: Our framework's automated scores show high consistency with the manual scores given by expert proctors on the Multitudes corpus. Critically, our system also provides detailed, phoneme-level error analysis, offering insights that are difficult to achieve at scale through manual scoring alone.

(4) Demonstrated Downstream Utility: We show significant value in practical applications, including achieving higher consistency in LLM-assisted language function assessment.

## 2 Method

Our proposed K-Function framework is an end-to-end pipeline designed for child speech assessment, converting raw audio into a comprehensive evaluation report. The architecture, depicted in Figure 1, consists of three primary stages: audio input, K-WFST transcription, and automated scoring.

Our pipeline consists of three main stages, each corresponding to a subsection below. First, in Datasets and Preprocessing, we describe the data sources and preparation steps needed to adapt the system to child speech. Second, in K-WFST Phonetic Recognition, we introduce our enhanced phonetic recognition framework that improves robustness to child-specific variations. Finally, in LLM-based Scoring, we detail how the recognized phoneme sequences are used with a large language model to produce automated evaluation scores aligned with expert assessments.

### 2.1 Datasets and Preprocessing

Our study utilizes two distinct datasets for model fine-tuning and evaluation: the My Science Tutor (MyST) [38] dataset for adapting our acoustic model, and the UCSF California Multitudes [39] corpus for evaluating downstream performance.

#### 2.1.1 MyST Dataset

For fine-tuning our base acoustic model to the unique characteristics of child speech, we use the MyST dataset, which contains conversational speech from students in third to fifth grade (8-10 years old) interacting with a virtual tutor. To align with the shorter vocalizations typically produced by younger children, we select transcribed utterances with a duration of under 20 seconds.

To generate phoneme-level labels required for end-to-end model training, we process the reference texts using the nltk [40] toolkit to convert them into phoneme sequences. The resulting dataset is partitioned into 61.5 hours for training and 11.4 hours for testing.

#### 2.1.2 UCSF California Multitudes Corpus

To evaluate our framework's practical utility in downstream assessment tasks, we use data from the UCSF California Multitudes corpus. This corpus is sourced from a digital universal screener administered to a representative sample of K-2 California public school children.

Our study specifically utilizes data from the Oral Reading Fluency (ORF) task, in which a child reads a passage aloud for two minutes. The passages read by the children are drawn from nine different reading materials: Grizzly, Banana, Quail, Raccoon, Shark, Lizard, Condor, Fox, and Sealion. A key feature of this dataset is that each ORF performance is manually scored by trained proctors, providing a ground-truth expert score for our downstream scoring evaluation. To establish a ground-truth for our transcription model's performance on this data, we performed manual annotation to obtain the reference phoneme sequences for all audio samples used from the ORF task.

### 2.2 K-WFST: Phonetic Recognizer for Disorder Screening

Weighted Finite State Transducer (WFST) is a foundational tool in speech recognition, representing a finite-state machine that maps input sequences to output sequences. A WFST consists of a set of states and weighted arcs, where each arc is a tuple defined by a start state, an end state, an input label, an output label, and a weight. A Finite State Acceptor (FSA) is a special case of a WFST where the input and output labels on each arc are identical, effectively accepting or rejecting an input sequence rather than transducing it.

A core limitation of traditional WFST-based decoders, especially in downstream applications involving child speech, is their reduced robustness in detecting subtle sub-word variations like substitutions and deletions. To address this, we introduce K-WFST, a framework that enhances the standard Dysfluent-WFST [37] by integrating a novel phoneme similarity-based substitution structure. The central innovation is to augment the WFST graph with additional, weighted paths that represent phonetically plausible substitutions. This allows the decoder to align an audio signal to a sequence that may be slightly erroneous but is phonetically similar to the reference, which is critical for accurately transcribing variable child speech.

The process for generating this augmented graph is formally detailed in Algorithm 1. The algorithm's inputs are the reference phoneme ID sequence , a hyperparameter that controls the penalty for errors, and the SimMatrix [41]. The SimMatrix is a pre-calculated matrix where is the number of phonemes in the vocabulary. We construct SimMatrix using a heuristic approach based on eight phonological features like vowel height, voicing. The similarity between and is computed as the normalized weighted sum of their matching features, yielding a score in that captures linguistically grounded proximity. The algorithm initializes an empty set of arcs, , and iterates through all start states and end states in the reference sequence, skipping cases where to prevent self-loops. From , it derives weights for correct transitions () and errors (). It then populates with arcs for correct paths, substitutions, deletions, and repetitions, with weights determined by , , and the similarity scores.

Furthermore, to control the flexibility of this substitution mechanism, we employ a Task-dependent K-Selection strategy. This allows the decoder to operate in two distinct modes depending on the input speech characteristics and the base acoustic model's performance:

When , the model is constrained to only consider the most similar phoneme, which is the phoneme itself. When , the model is allowed to consider the top two most similar phonemes, adding the extra substitution paths to enhance robustness in more challenging scenarios. This task-dependent approach ensures optimal performance across different conditions without sacrificing efficiency.

### 2.3 LLM-based Scoring

To validate the practical utility of our high-fidelity transcriptions, we designed an experiment to assess if a Large Language Model (LLM) could replicate the nuanced scoring of human experts on the Multitudes Oral Reading Fluency (ORF) task. For this task, we evaluated a state-of-the-art instruction-tuned model, meta-llama/Llama-3.1-70B-Instruct [42].

We employed a few-shot prompting strategy to guide the model in simulating the evaluation process of a human proctor. For each ORF sample to be scored, the LLM was provided with a comprehensive set of inputs111Prompt is available at https://chenxukwok.github.io/K-function/:

(1) The official Multitudes scoring guidelines, detailing the criteria for evaluation.

(2) The original reference text the child was asked to read.

(3) The detailed phoneme-level transcription generated by our K-WFST model.

(4) Four distinct, manually-scored examples to serve as in-context demonstrations of the scoring process.

The LLM synthesized these inputs to produce a single quantitative score reflecting each child’s reading performance. To account for the probabilistic nature of the model, we set the decoding temperature to 0.5 and performed the prediction five consecutive times for each sample. All five scores generated for each sample were then included in the final evaluation. We compared this entire set of predictions against the ground-truth expert scores from the Multitudes corpus to calculate the overall MAE and MSE.

## 3 Experiments

### 3.1 Model Fine-tuning and Evaluation

Our experimental process is designed to first adapt a pre-trained model for child speech and then rigorously evaluate its performance across different conditions and decoding strategies.

#### 3.1.1 Fine-tuning and Evaluation on MyST

| Model | Method | PER SD (%) |
|---|---|---|
| Base | Greedy | 40.2666.92 |
| WFST (K=1) | 3.7227.90 | |
| WFST (K=2) | 6.9129.00 | |
| Kids-FT | Greedy | 11.8665.89 |
| WFST (K=1) | 1.399.83 | |
| WFST (K=2) | 8.3114.67 |

| Multitudes Materials | Base Model PER (%) | Kids-FT Model PER (%) | ||||
|---|---|---|---|---|---|---|
| Greedy | WFST (K=1) | WFST (K=2) | Greedy | WFST (K=1) | WFST (K=2) | |
| Grizzly | 35.61 | 22.92 | 7.95 | 7.95 | 1.85 | 1.77 |
| Banana | 45.13 | 48.16 | 37.72 | 23.21 | 15.47 | 11.41 |
| Quail | 42.05 | 31.10 | 20.14 | 14.31 | 9.72 | 6.01 |
| Raccoon | 37.48 | 28.06 | 19.36 | 11.19 | 7.10 | 4.80 |
| Shark | 49.71 | 24.20 | 11.82 | 11.07 | 7.88 | 5.63 |
| Lizard | 43.64 | 24.20 | 11.82 | 11.07 | 7.88 | 5.63 |
| Condor | 43.35 | 28.06 | 19.36 | 11.19 | 7.10 | 4.80 |
| Fox | 54.35 | 48.16 | 37.72 | 23.21 | 15.47 | 11.41 |
| Sealion | 37.48 | 28.06 | 19.36 | 11.19 | 7.10 | 4.80 |

First, we fine-tuned the pre-trained Phoneme-based Wav2Vec2.0 model [43] using the 61.5-hour training partition of the MyST dataset. This process created our child-speech-adapted models, hereafter referred to as ``Kids-FT". We then conducted a comprehensive evaluation on the 11.4-hour MyST test set, which primarily consists of relatively fluent child speech. The performance of our baseline (``Base") and fine-tuned (``Kids-FT") models, measured by Phoneme Error Rate (PER), is presented in Table 1.

The results clearly show the substantial benefit of fine-tuning, with the Kids-FT models significantly outperforming the Base models across all decoding methods. Notably, on this fluent speech dataset, the constrained setting of our decoder, WFST (K=1), achieves the optimal performance with a PER of only 1.39%. This finding supports our hypothesis that for less variable speech, a more constrained decoding path prevents potential error propagation and yields higher accuracy.

#### 3.1.2 Evaluation on Multitudes

To assess model performance in a more challenging and realistic downstream scenario, we evaluated all models on the 1.87-hour Multitudes corpus. This dataset contains a higher degree of disfluent child speech from the Oral Reading Fluency (ORF) task. The performance of each model configuration across the nine different reading passages of the ORF task is detailed in Table 2.

As shown in the table, the fine-tuned ``Kids-FT Model" consistently outperforms the ``Base Model" across all reading passages, reaffirming the importance of adaptation to child speech. Crucially, on this more challenging disfluent dataset, the flexible WFST (K=2) configuration consistently yields the lowest Phoneme Error Rate (PER) for the fine-tuned model on every single passage. This result strongly supports our Task-dependent K-Selection strategy, demonstrating that allowing greater flexibility for phonetically plausible substitutions is essential for enhancing model robustness and accuracy in complex downstream scenarios.

### 3.2 LLM-Assisted Scoring on the Multitudes Corpus

To validate the practical utility of our high-fidelity transcriptions, we assessed if a Large Language Model (LLM) could replicate the nuanced scoring of human experts. Using the Llama-3.1-70B-Instruct model, we employed a few-shot prompt that included four scored examples and the official scoring guidelines. The LLM then generated a score based on the phoneme transcriptions produced by each of our six model configurations.

The results, measured in Mean Absolute Error (MAE) and Mean Squared Error (MSE) against the ground-truth expert scores, are presented in Table 3. The data reveals a clear and consistent trend: higher transcription quality directly leads to a more accurate automated score from the LLM, demonstrating a stronger agreement with human evaluators.

Specifically, the transcriptions from the fine-tuned ``Kids-FT'' models consistently result in lower MAE and MSE than those from the ``Base'' models. The optimal performance is achieved when the LLM is provided with the transcription from our best-performing model: the Kids-FT model paired with the flexible WFST (K=2) decoder. This configuration yielded the lowest error rates, with an MAE of 8.43% and an MSE of 0.2224. This confirms that the detailed and accurate phoneme-level information captured by our proposed K-WFST framework provides the most robust and practically useful representation for downstream assessment tasks.

| Model | Method | MAE (%) | MSE |
|---|---|---|---|
| Base | Greedy | 14.82 | 0.2876 |
| WFST (K=1) | 11.78 | 0.2662 | |
| WFST (K=2) | 8.71 | 0.2371 | |
| Kids-FT | Greedy | 10.29 | 0.2504 |
| WFST (K=1) | 11.47 | 0.2581 | |
| WFST (K=2) | 8.43 | 0.2224 |

## 4 Conclusion and future work

K-Function unifies robust, child-oriented phoneme recognition via our K-WFST framework with sophisticated LLM reasoning into an end-to-end pipeline. This system converts children's speech into objective scores and valuable insights for language assessment. Its modular architecture shows strong promise for broad deployment across educational and interventional settings. Notably, K-WFST is inherently language-agnostic, as the SimMatrix relies on universal articulatory features adaptable to any language with defined phonetics. Future work will focus on expanding the framework to multilingual contexts, refining the analysis to finer linguistic units such as syllables, and verifying its long-term impact and fairness through large-scale field studies.

## Acknowledgements

Thanks for support from UC Noyce Initiative, Society of Hellman Fellows, NIH/NIDCD, and the Schwab Innovation fund.

## References

- [1] National Institute on Deafness and Other Communication Disorders, ``Percentage of children ages 3–17 years with a communication or swallowing disorder,'' 2023, Accessed: 2025-06-25.
- [2] California Department of Education, ``California announces new special education funding allocations for 2024–2025,'' 2024.
- [3] Cross River Therapy, ``Dyslexia statistics,'' 2024.
- [4] M. L. Gorno-Tempini et al., ``Classification of primary progressive aphasia and its variants,'' Neurology, 2011.
- [5] S. Lee et al., ``Analysis of children's speech: Duration, pitch and formants,'' in Proc. Eurospeech, 1997.
- [6] Trang Tran, Morgan Tinkler, Gary Yeung, et al., ``Analysis of disfluency in children's speech,'' arXiv preprint arXiv:2010.04293, 2020.
- [7] Tilda Neuberger and Mária Gósy, ``A cross-sectional study of disfluency characteristics in children’s spontaneous speech,'' Govor, vol. 31, no. 1, pp. 3–27, 2014.
- [8] M. Gerosa et al., ``A review of ASR technologies for children's speech,'' in Proc. WOCCI, 2009.
- [9] G. Yeung et al., ``On the difficulties of automatic speech recognition for kindergarten-aged children,'' Interspeech, 2018.
- [10] N. B. Shankar et al., ``Selective attention merging for low resource tasks: A case study of child ASR,'' arXiv:2501.08468, 2025.
- [11] R. Fan et al., ``Benchmarking children's asr with supervised and self-supervised speech foundation models,'' arXiv:2406.10507, 2024.
- [12] V. Bhardwaj et al., ``Automatic speech recognition (ASR) systems for children: A systematic literature review,'' Appl. Sci., vol. 12, no. 9, 2022.
- [13] S Shahnawazuddin et al., ``Developing children's asr system under low-resource conditions using end-to-end architecture,'' Digital Signal Processing, vol. 146, pp. 104385, 2024.
- [14] V. P. Singh et al., ``Causal analysis of ASR errors for children: Quantifying the impact of physiological, cognitive, and extrinsic factors,'' arXiv:2502.08587, 2025.
- [15] D. K. Singh et al., ``Data augmentation using CycleGAN for end-to-end children ASR,'' in EUSIPCO, 2021.
- [16] H. K. Kathania et al., ``A formant modification method for improved ASR of children's speech,'' Speech Communication, 2022.
- [17] R. Fan et al., ``Towards better domain adaptation for self-supervised models: A case study of child ASR,'' IEEE J. Sel. Top. Signal Process., 2022.
- [18] A. Xu et al., ``Exploring speech foundation models for speaker diarization in child-adult dyadic interactions,'' in Interspeech, 2024.
- [19] J. Kim et al., ``Conditional variational autoencoder with adversarial learning for end-to-end text-to-speech,'' in ICML, 2021.
- [20] J. Li et al., ``Enhancing child vocalization classification with phonetically-tuned embeddings for assisting autism diagnosis,'' arXiv:2309.07287, 2024.
- [21] J. Li et al., ``Analysis of self-supervised speech models on children's speech and infant vocalizations,'' 2024.
- [22] X. Shi et al., ``Direct articulatory observation reveals phoneme recognition performance characteristics of a self-supervised speech model,'' JASA Express Lett., 2024.
- [23] H. Gao et al., ``G2PU: Grapheme-to-phoneme transducer with speech units,'' in ICASSP, 2024.
- [24] Y. Peng et al., ``A study on the integration of pre-trained SSL, ASR, LM and SLU models for spoken language understanding,'' in SLT, 2023.
- [25] Nicholas Mehlman and Shri Narayanan, ``Adversarial robustness of self-supervised learning features,'' IEEE Open Journal of Signal Processing, vol. 6, pp. 468–477, 2025.
- [26] J. Zhu et al., ``Phone-to-audio alignment without text: A semi-supervised approach,'' in ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2022, pp. 8167–8171.
- [27] W. Zhao et al., ``MuTIS: Enhancing reasoning efficiency through multi turn intervention sampling in reinforcement learning,'' in EMNLP, 2025.
- [28] A. Mohamed et al., ``Self-supervised speech representation learning: A review,'' IEEE J. Sel. Top. Signal Process., 2022.
- [29] W.-N. Hsu et al., ``HuBERT: Self-supervised speech representation learning by masked prediction of hidden units,'' IEEE/ACM TASLP, 2021.
- [30] S. Chen et al., ``WavLM: Large-scale self-supervised pre-training for full stack speech processing,'' IEEE JSTSP, 2022.
- [31] Jiachen L., Xuanru Z., Zoe E., et al., ``Ssdm: Scalable speech dysfluency modeling,'' in Advances in Neural Information Processing Systems, 2024, vol. 37.
- [32] C. Zwilling et al., ``The speech accessibility project: Best practices for collection and curation of disordered speech,'' in Interspeech, 2025.
- [33] Z. Ye, J. Lian, X. Zhou, et al., ``Seamless dysfluent speech text alignment for disordered speech analysis,'' Interspeech, 2025.
- [34] Jinming Z., Xuanru Z., Jiachen L., et al., ``Analysis and evaluation of synthetic data generation in speech dysfluency detection,'' Interspeech, 2025.
- [35] J. Lian et al., ``Deep Neural Convolutive Matrix Factorization for Articulatory Representation Decomposition,'' in Interspeech, 2022.
- [36] J. Lian et al., ``Articulatory representation learning via joint factor analysis and neural matrix factorization,'' in ICASSP, 2023.
- [37] C. Guo, J. Lian, X. Zhou, et al., ``Dysfluent wfst: A framework for zero-shot speech dysfluency transcription and detection,'' Interspeech, 2025.
- [38] W. Ward et al., ``My science tutor: A conversational multimedia virtual tutor for elementary school science,'' ACM Trans. Speech Lang. Process., vol. 7, pp. 1–29, 2011.
- [39] ``Multitudes universal screening platform,'' https://multitudesinfo.ucsf.edu/, 2025.
- [40] Steven Bird, Edward Loper, and Ewan Klein, Natural Language Processing with Python, O'Reilly Media Inc., 2009.
- [41] X. Zhou, J. Lian, C. Cho, et al., ``Towards accurate phonetic error detection through phoneme similarity modeling,'' Interspeech, 2025.
- [42] G. Aaron et al., ``The llama 3 herd of models,'' arXiv:2407.21783, 2024.
- [43] B. Alexei et al., ``wav2vec 2.0: A framework for self-supervised learning of speech representations,'' NeurIPS, 2020.
