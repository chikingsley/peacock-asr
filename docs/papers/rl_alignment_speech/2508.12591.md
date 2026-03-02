# Beyond Modality Limitations: A Unified MLLM

Approach to Automated Speaking Assessment with
Effective Curriculum Learning
Yu-Hsuan Fang, Tien-Hong Lo, Yao-Ting Sung, Berlin Chen
National Taiwan Normal University
{andyfang, teinhonglo, sungtc, berlin}@ntnu.edu.tw
Abstract—Traditional Automated Speaking Assessment (ASA)
systems exhibit inherent modality limitations: text-based approaches lack acoustic information while audio-based methods
miss semantic context. Multimodal Large Language Models
(MLLM) offer unprecedented opportunities for comprehensive
ASA by simultaneously processing audio and text within unified
frameworks. This paper presents a very first systematic study
of MLLM for comprehensive ASA, demonstrating the superior
performance of MLLM across the aspects of content and language use . However, assessment on the delivery aspect reveals
unique challenges, which is deemed to require specialized training
strategies. We thus propose Speech-First Multimodal Training
(SFMT), leveraging a curriculum learning principle to establish
more robust modeling foundations of speech before cross-modal
synergetic fusion. A series of experiments on a benchmark dataset
show MLLM-based systems can elevate the holistic assessment
performance from a PCC value of 0.783 to 0.846. In particular,
SFMT excels in the evaluation of the delivery aspect, achieving an
absolute accuracy improvement of 4% over conventional training
approaches, which also paves a new avenue for ASA.
Index Terms—Multimodal large language model (MLLM),
automated speaking assessment (ASA), multimodal training, L2
proficiency, cross-modal learning
## I INTRODUCTION
Recent advances in Multimodal Large Language Models
(MLLM) have ushered in an unprecedented era of technological transformation, fundamentally reshaping the paradigm of
human-machine interaction by jointly integrating information
across multiple modalities [1]–[4]. Pioneering efforts such
as GPT-4o [5] have demonstrated remarkable capabilities in
seamlessly handling text, audio, and visual inputs within an
unified framework. Particularly noteworthy is the emergence
of open-source MLLM such as Phi-4-multimodal [3] that
has demonstrated superior performance over traditional unimodal approaches after model fine-tuning on domain-specific
data [6]–[10] for used in specialized language assessment
tasks. Such excellent multimodal capabilities also open new
avenues for addressing complex real-world applications previously beyond the reach of conventional approaches.
Within the domain of Computer-Assisted Language Learning (CALL), Automated Speaking Assessment (ASA) represents one of the most challenging and multifaceted tasks [11],
[12]. The complexity of evaluating L2 (second-language)
speaking proficiency stems from the need to assess multiple aspects of speaking proficiency simultaneously, including delivery (e.g., pronunciation accuracy, fluency, prosodic
features), content appropriateness (e.g., topic relevance and
coherence), and language use (e.g., vocabulary richness and
grammatical correctness) [11], [13]. These evaluation criteria
encompass both quantifiable linguistic elements and subtle
acoustic characteristics such as stress patterns, intonation
contours, and speech rhythm [14], [15]. The multidimensional
nature of speaking assessment, combined with the variability
inherent in L2 speech production, establishes ASA systems as
indispensable components in modern language learning environments, providing objective, consistent, and scalable evaluation capabilities that complement human assessment [12].
However, traditional ASA approaches suffer from fundamental modality-specific limitations that constrain their
effectiveness. Text-based classifiers, exemplified by BERTbased systems [6], [8], excel in semantic comprehension
and contextual understanding but remain critically dependent
on ASR transcription quality and inherently lack access to
acoustic features essential for delivery and prosodic evaluation.
Conversely, audio-based approaches utilizing self-supervised
learning models like wav2vec 2.0 [7], [9] directly process
speech signals to capture rich acoustic information for delivery assessment, yet sacrifice semantic context and linguistic
content analysis crucial for evaluating language use sophistication and grammatical accuracy. While previous research
has explored fusion strategies combining both modalities [13],
these approaches typically fuse the outputs of separate unimodal systems, rather than achieving the genuine cross-modal
information synchronization found in unified architectures.
This fundamental limitation motivates our investigation into
whether MLLM can transcend traditional modality boundaries
and achieve more effective multimodal integration for comprehensive ASA.
This paper presents a very first systematic study of MLLM
for comprehensive ASA, investigating three critical questions:
1) Can multimodal large language models effectively resolve
the information fusion challenges encountered in traditional
ASA systems, and what performance levels can be achieved?
2) Despite MLLM advances, does the audio modality remain
irreplaceable for delivery assessment tasks? 3) Do there exist
simple yet cost-effective training strategies that can significantly enhance ASA performance across different aspects
of speaking proficiency evaluations? To this end, we design
arXiv:2508.12591v1 [cs.CL] 18 Aug 2025
thorough experiments using the TEEMI dataset and propose Speech-First Multimodal Training (SFMT), a curriculum
learning approach [16] that progressively transitions from
speech foundations to cross-modal integration, achieving an
absolute improvement of 4% in terms of the assessment
accuracy for the delivery aspect.
Fig. 1. ASA systems have evolved from handcrafted feature engineering
through self-supervised learning approaches to unified multimodal frameworks
capable of comprehensive assessment and feedback generation (adapted
from [14]).
## II RELATED WORK
A. Evolution of Automated Speaking Assessment Systems
Automated speaking assessment (ASA) has evolved through
three distinct paradigms, each marking fundamental advances
in the automation of evaluations on speaking proficiency for
L2 learners. Figure 1 illustrates this progression from handcrafted feature-based systems, through self-supervised models,
to unified multimodal frameworks.
1) Handcrafted Feature-based Systems: Early ASA systems typically rely on explicit feature engineering pipelines
(Figure 1 (1)), extracting handcrafted acoustic features (spectral, prosodic, temporal) from speech and linguistic features
from ASR transcripts [17]. Traditional machine learning algorithms process these features for proficiency prediction, with
Educational Testing Service (ETS) pioneered foundational
approaches via extensive feature engineering research [18]–
[20]. More recently, Wu et al. [21] showed that expert-defined
knowledge clues (delivery/language use criteria) significantly
enhanced assessment performance. Despite interpretability,
these systems have limited generalization and require substantial domain expertise.
2) Self-Supervised Learning Paradigm: Self-supervised
learning tackles ASA via either text-based or audio-based pretrained models (Figure 1 (2) and (3)).
Text-based Models: BERT-based models enables sophisticated semantic evaluation (grammar, language use, content)
from ASR transcripts [6], [8], but are limited by ASR quality
and lack acoustic information for assessing the aspect of
delivery.
Audio-based Models: Self-supervised speech models like
wav2vec 2.0 process raw speech to capture acoustic patterns [7], [9]. Lo et al. [11] found wav2vec 2.0 inherently
encodes syntactic information, revealing the potential of crossmodal feature extraction. Yet, they lack semantic context for
comprehensive evaluation.
Both approaches have achieved some success on various
ASA tasks, but remain limited by modality constraints. To get
around this limitation, prior fusion strategies typically operated
at the model level, which would fail to achieve genuine crossmodal synchronization [13].
3) Multimodal Large Language Models: Contemporary
MLLM mark a paradigm shift to unified multimodal
processing (Figure 1(4)). Models like Qwen-Audio [2],
SALMONN [1], and Phi-4-multimodal [3] simultaneously
process speech and text in single frameworks, enabling true
multimodal integration via cross-modal attention.
MLLM transcend traditional assessment limitations by providing comprehensive educational feedback beyond scores.
Nevertheless, how to design optimal training strategies for
multimodal integration, particularly for the evaluation on the
aspect of delivery that requires fine-grained acoustic analysis,
remains largely underexplored.
B. Curriculum Learning for Multimodal Training
Curriculum learning posits that structured progression from
simple to complex tasks enhances model performance [16].
Recent multimodal speech applications, such as WavLLM [22]
and SALMONN [1], have also confirmed the effectiveness
of progressive training in speech-text joint modeling. Furthermore, Zhang et al. [23] applied curriculum learning to
speaking assessment via strategic data ordering, showing
improvements in limited-data scenarios. However, existing
approaches focus on data-level curriculum (ordering samples
by difficulty), rather than addressing fundamental challenges
in multimodal integration.
Our research extends the notion of curriculum learning to
modality-level progression, investigating the relative importance of acoustic versus textual information for MLLM-based
ASA tasks. We propose SFMT, a simple-to-complex learning
approach that first establishes robust acoustic foundations
before processing cross-modal integration. This modality-level
curriculum approach specifically addresses optimizing MLLM
performance for fine-grained assessment tasks where acoustic
and semantic information must be effectively integrated, while
preserving discriminative capabilities essential for accurate
proficiency evaluation.
## III METHODOLOGY
A. Multimodal Large Language Model Architecture for ASA
We leverage Phi-4-multimodal [3] for comprehensive automated speaking assessment. This model employs a mixture-ofLoRAs architecture enabling efficient multimodal fine-tuning
while preserving base language capabilities. As illustrated in
Fig. 2. The proposed MLLM architecture processes both audio and text inputs
through specialized pathways to generate multi-aspect proficiency scores
across content, delivery, language use, and holistic assessment aspects.
Figure 2, the system processes both raw audio and ASRgenerated transcripts through modality-specific pathways before integration, comprising: (1) a 3.8B parameter decoderonly Transformer as the reasoning backbone, (2) an audio
processing pipeline with 460M-parameter encoder using conformer blocks and audio projector for shared embedding
space mapping, and (3) a modality-specific audio adapter
(LoRAaudio, 460M parameters) enabling learning of targeted
acoustic traits without language capability interference.
For comprehensive assessment on the spoken responses of
the TEEMI dataset, we train three specialized models targeting
aspects of Content (C), Delivery (D), and Language Use (L),
respectively. Each model receives aspect-specific instructions
during training, allowing focused optimization. The Holistic
(H) score integrates assessment results gathered from all three
aspects, providing an overall proficiency indicator aligned with
CEFR standards.
B. Speech-First Multimodal Training (SFMT) Strategy
Standard multimodal training approach to ASA encounters a fundamental challenge: modality imbalance. Models of
these approaches tend to exhibit systematic preference for
textual features due to their structured representations and
computational efficiency, consequently underutilizing acoustic
information critical for delivery assessment [24]. This imbalance impairs the model’s capacity to learn fine-grained
acoustic patterns—including pronunciation accuracy, fluency
variations, and prosodic characteristics—that text representations inherently cannot encode.
Our empirical investigation through systematic ablation
studies (Section V-B) reveals a counterintuitive finding: the
audio modality demonstrates superior learning efficiency for
MLLM-based graders compared to text, particularly for the
assessment on the delivery aspect. This observation of audio’s
stronger initial performance and faster convergence under
identical training conditions motivates our speech-first strategy.
This empirical superiority of acoustic learning stems from
three fundamental factors:
(1) Information Completeness: Raw audio signals preserve
the complete spectrum of speech information—from phonetic
details to prosodic contours—providing MLLM with unfiltered
access to all acoustic evidence necessary for proficiency assessment. In contrast, ASR-transcribed text represents a lossy
transformation that discards paralinguistic features critical for
delivery evaluation.
(2) Direct Signal Access: Audio inputs bypass the error
propagation inherent in text-based approaches, offering direct
access to ground-truth acoustic patterns. This eliminates the
cascading effects of ASR transcription errors and systematic
biases from ASR systems trained predominantly on native
speech.
(3) Preferential Learning Patterns: When exposed to both
modalities simultaneously, models demonstrate preferential
optimization toward text-based features as computationally
efficient pathways [25], particularly for content and language
use assessment. This preference inhibits the development of
acoustic discrimination capabilities, as models converge on
solutions that underutilize acoustic information.
Building upon these insights, we propose Speech-First
Multimodal Training (SFMT), a two-stage curriculum learning strategy that exploits the discovered learning hierarchy.
By establishing robust acoustic feature extraction capabilities
before introducing textual information, SFMT ensures that
models develop strong delivery assessment abilities that persist
through subsequent multimodal integration(Figure 3):
Stage 1 - Acoustic Foundation (Fig. 3(a)): Given training
data Daudio = {(ai,Ii,yi)}N
i=1 where ai is audio input vector,
Ii ∈ {IC,ID,IL} is aspect-specific instruction, and yi is the
target score, we optimize:
θ1
LoRA = argmin
θLoRA
X
(a,I,y)∈Daudio
L(fPhi-4(a,I;θLoRA),y), (1)
where fPhi-4 denotes the MLLM and L is the loss function.
Only the LoRA audio adapter parameters θLoRA are updated.
Stage 2 - Cross-Modal Integration (Fig. 3(b)): Using
multimodal data Dmulti = {(ai,ti,Ii,yi)}N
i=1 with additional
transcript vector ti, we continue optimization from Stage 1:
θ2
LoRA = argmin
θ1
LoRA
X
(a,t,I,y)∈Dmulti
L(fPhi-4(a,t,I;θ1
LoRA),y),
(2)
where θLoRA is the pre-trained adapter. This progression ensures robust acoustic specialization before multimodal integration, particularly enhancing the performance of the assessment
on the delivery aspect.
Fig. 3. SFMT employs a two-stage curriculum learning approach that first establishes acoustic foundations through audio-only training before introducing
cross-modal integration with textual information.
## IV EXPERIMENTS
A. Datasets
We evaluate our proposed models on two distinct datasets:
the proprietary TEEMI corpus and the publicly available the
Speak & Improve Corpus.
1) TEEMI Corpus: The TEEMI corpus (Test for EnglishMedium Instruction) [26] is a comprehensive L2 proficiency
dataset designed for EMI research in higher education contexts. The corpus features spontaneous English speech from
undergraduate and graduate L2 learners, with each response
evaluated across four aspects—holistic, content, language use,
and delivery—using an eight-level CEFR-aligned scale (PreA1 to B2). TEEMI is equipped with triple-rater annotation
with majority voting to ensure scoring reliability.
The speaking assessment of TEEMI includes three task
formats: general listen and answer (A), situational question
and answer (B), and thematic question and answer (C). In
this paper, we focus on a subset consisting of tasks A01,
A02, yielding a total of 8,214 responses. Model training and
validation are performed solely on A01, which contains 6,152
responses from 1,231 speakers. The A02 task is held out to
evaluate the model’s ability to generalize to previously unseen
prompts. The detailed CEFR-level distributions for the A01
and A02 tasks utilized in this study are illustrated in Table I.
TABLE I
STATISTICAL INFORMATION FOR SELECTED CEFR PROFICIENCY LEVELS
(A01, A02) IN THE TEEMI DATASET.
Task Usage Pre-A A1 A1+ A2 A2+ B1 B1+ B2
A01
Train 34 61 76 156 150 169 79 65
Valid 8 16 19 38 39 43 23 12
Test 11 20 23 49 50 48 32 15
A02 Unseen 9 7 12 19 12 26 23 15
Total - 62 104 130 262 251 286 157 107
2) SLaTE 2025 Speak & Improve Corpus: We utilize the
Speak & Improve Corpus 2025 [27], containing 315 hours
of L2 English speech with CEFR proficiency levels from
A2 to C1+. The corpus includes four task types: Interview,
Opinion, Presentation, and Communication Activity, equipped
with holistic scores averaged across different aspects. We follow official data splits to construct the corresponding training,
development, and test sets.
B. Implementation Details
Model configurations were initialized using the
Phi-4-multimodal-instruct1
with LoRA
adaptation [29] (rank=320) applied to the audio encoder.
Training employed the AdamW optimizer (lr=4e-5) for 3
epochs with batch size 32 (gradient accumulation steps: 16)
and bfloat16 mixed precision on a single NVIDIA RTX 3090.
Flash attention [30] was utilized for memory efficiency.
For speech recognition, we compare Whisper large v2
(14.75% WER) against the integrated ASR module of Phi-
4 (18.25% WER) on the TEEMI corpus. Output generation
is constrained to 10 tokens to prevent hallucination. SFMT
training follows the prescribed two-stage curriculum: Stage 1
processes audio-only inputs with null text placeholders, while
Stage 2 incorporates full multimodal inputs. Aspect-specific
prompts guide targeted assessment during inference.
Model performance is evaluated using Pearson Correlation
Coefficient (PCC) for prediction consistency, Absolute Accuracy for exact CEFR-level classification, Adjacent Accuracy
for predictions within ±0.5 levels, and Macro Accuracy for
balanced cross-level performance measurement accounting for
dataset class imbalance. Additionally, for the evaluation of
regression-based scoring tasks, Root Mean Squared Error
(RMSE) is utilized to assess the average magnitude of the
error between predicted and actual continuous scores.
1https://huggingface.co/microsoft/Phi-4-multimodal-instruct
TABLE II
MODEL PERFORMANCE ON THE TEEMI TEST SET.
Models
Content (C) Delivery (D) Language Use (L) Holistic (H)
PCC↑ ABS↑ ADJ↑ PCC↑ ABS↑ ADJ↑ PCC↑ ABS↑ ADJ↑ PCC↑ ABS↑ ADJ↑
Baseline Models
W2V [7] 0.755 35.08 81.85 0.768 39.92 83.06 0.740 36.29 79.03 0.771 34.67 83.87
BERT [6] 0.774 33.47 84.68 0.794 38.31 84.68 0.759 36.29 80.24 0.781 35.48 82.66
W2V-BERT [13] 0.735 35.08 81.45 0.794 38.71 87.10 0.798 41.13 82.66 0.771 38.71 84.68
W2V-PT [11] 0.733 30.65 79.84 0.796 39.11 83.06 0.779 42.74 81.45 0.785 34.68 83.07
BERT-PT [11] 0.756 29.44 79.84 0.783 40.73 83.06 0.788 35.08 81.85 0.777 33.87 81.85
Multi-Aspect [28] 0.760 37.10 80.24 0.810 41.94 85.48 0.785 39.92 81.45 0.783 38.31 84.27
Our Approach
Phi-4 0.826 41.93 87.90 0.831 42.34 89.11 0.840 41.53 89.52 0.846 42.34 90.32
Phi-4 (SFMT) 0.821 39.11 88.31 0.848 46.77 89.11 0.835 40.73 88.31 0.838 41.13 90.73
(a) Content (Phi-4) (b) Delivery (Phi-4) (c) Language use (Phi-4) (d) Holistic (Phi-4)
(e) Content (SFMT) (f) Delivery (SFMT) (g) Language use (SFMT) (h) Holistic (SFMT)
Fig. 4. Confusion matrices comparing standard Phi-4 and SFMT performance on CEFR scale, demonstrating enhanced diagonal concentration particularly
for delivery assessment.
To facilitate reproducibility and promote community advancement in multimodal ASA research, we will make all
source code and fine-tuning implementations publicly available upon publication2
.
## V RESULTS
A. Overall MLLM Performance
Table II demonstrates substantial MLLM (viz. Phi-4) superiority over current state-of-the-art models across all aspects of
assessment. Standard Phi-4 achieves PCC scores consistently
above 0.82, representing significant improvements over the
compared models which all have PCC results below 0.80. This
seems to validate MLLM multimodal integration capabilities
for comprehensive ASA.
The confusion matrices in Figure 4 provide visual confirmation of enhanced classification precision when performing
2https://github.com/ntnuYuhsuan/asa-grader.git
ASA with the MLLM-based models; MLLM-based models exhibits superior diagonal concentration compared to traditional
ones, suggesting MLLM as an all-around workhorse capable
of transcending inherent modality limitations.
B. Modality Analysis and SFMT Effectiveness
The ablation studies, with Pearson Correlation Coefficient
(PCC) and Macro Accuracy (Macro Acc) as key performance
indicators (Table III), reveal fundamental insights into modality contributions for MLLM-based ASA graders and validates
the efficacy of our SFMT strategy.
Modality Contributions: Table III reports on the performance levels of MLLM-based models that operate on different
modalities and their combination. Audio-only configurations
demonstrate strong overall performance, particularly excelling
in the assessment on the delivery aspect. In contrast, textonly models exhibit a general decline in performance, with
TABLE III
ABLATION STUDY COMPARING MODALITY CONTRIBUTIONS TO MLLM-BASED ASA PERFORMANCE.
Training Configuration Audio Text
Content (C) Delivery (D) Language Use (L) Holistic (H)
PCC↑ Macro Acc↑ PCC↑ Macro Acc↑ PCC↑ Macro Acc↑ PCC↑ Macro Acc↑
Phi-4 ✓ ✓ 0.826 82.00 0.831 82.48 0.840 85.27 0.841 84.27
Phi-4 (Text-Only) × ✓ 0.784 74.76 0.776 75.83 0.768 72.80 0.776 73.35
Phi-4 (Audio-Only) ✓ × 0.811 82.33 0.835 82.94 0.830 86.16 0.836 86.83
Phi-4 (SFMT) ✓ ✓ 0.821 83.41 0.848 84.01 0.835 83.67 0.838 86.75
the most significant drop observed in the assessment on the
delivery aspect. This underscores the challenges facing textonly models, which is partly due to their reliance on ASR
transcripts alone (achieving 14.75% WER with Whisper large
v2 on TEEMI) and the inherent lack of direct acoustic cues
for the assessment on the delivery aspect.
SFMT Validation: The strategic emphasis of SFMT on
establishing robust speech processing foundations before introducing textual information yields significant enhancements,
particularly in the assessment on the delivery aspect—whose
success is most critically dependent on fine-grained acoustic
discrimination. This is clearly demonstrated by improvements
over the Phi-4 baseline (Table III): the assessment on the
delivery aspect shows a pronounced PCC advantage (a value
of 0.848 for SFMT vs. 0.831 for the Phi-4 baseline). Furthermore, SFMT improves on the Macro Accuracy for this
aspect from 82.48% to 84.01%. These results validate SFMT
as an effective curriculum learning approach, highlighting the
benefits of establishing robust acoustic representations prior to
cross-modal integration.
## C Generalization to Unseen Tasks
Evaluation on the unseen tasks of TEEMI (cf. Table IV)
confirms the robust generalization capablilty of fine-tuned Phi-
4 across all aspects. The assessment on the delivery aspect
exhibits the strongest transfer performance, indicating effective
learning of transferable acoustic features. The results on the
content and language use aspects also show strong correlations despite semantic variations in task prompts. This again
validates the MLLM’s capability to develop generalizable
multimodal representations for cross-task ASA applications.
TABLE IV
MODEL PERFORMANCE ON THE UNSEEN TEEMI DATASET.
Aspect PCC↑ ABS Acc↑ ADJ Acc↑
Content (C) 0.851 32.52 78.86
Delivery (D) 0.863 44.72 86.18
Language Use (L) 0.855 33.33 78.86
Holistic (H) 0.846 32.52 78.86
D. Cross-corpus evaluation
Cross-corpus evaluation on the Speak & Improve Corpus
(Table V) further confirms the effectiveness of our model
across diverse L2 populations and assessment tasks. The
SFMT strategy consistently outperforms both the traditional
baselines and the standard Phi-4 implementation across all
evaluation metrics, demonstrating superior prediction accuracy
and correlation with human judgments. This cross-corpus
success validates that the proposed model and training regime
generalize beyond the specific characteristics of the TEEMI
corpus to broader international assessment contexts. The consistent performance improvements across different datasets
and learner populations establish the practical applicability of
SFMT for real-world ASA deployment scenarios.
TABLE V
PERFORMANCE ON THE SPEAK & IMPROVE CORPUS.
Method RMSE↓ PCC↑ Acc±0.5↑ Acc±1.0↑
BERT [6] 0.445 0.727 76.0 96.3
W2V [7] 0.394 0.790 81.3 99.3
Phi-4 0.412 0.796 74.7 98.0
Phi-4 (SFMT) 0.387 0.800 79.7 99.2
## VI CONCLUSION AND FUTURE WORK
This paper presents a very first systematic study of MLLM
for comprehensive automated speaking assessment (ASA),
addressing three fundamental research questions. Our findings
demonstrate that MLLM effectively resolve traditional challenges facing information fusion, achieving superior performance across all assessment aspects compared to uni-modality
based models. The ablation studies confirm the irreplaceability
of the audio modality for delivery assessment, while the
proposed SFMT strategy considerably promotes performance
through speech-first curriculum learning, particularly benefiting fine-grained acoustic discrimination. A series of experimental validation on TEEMI and the Speak & Improve Corpus
confirm the robust generalization capability of our model
across diverse L2 populations and assessment contexts. These
results also suggest MLLM-based models as the transformative
backbone for ASA, enabling more accurate, comprehensive,
and generalizable evaluation systems. Future research will
explore multi-task learning frameworks for multi-aspect assessment and integrate comprehensive feedback generation
into ASA, advancing towards the broader goal of creating
intelligent, adaptive language learning environments that can
provide personalized, real-time guidance for L2 learners in
various contexts of computer-assisted language learning.
## REFERENCES
[1] C. Tang, W. Yu, G. Sun, X. Chen, T. Tan, W. Li, L. Lu,
Z. Ma, and C. Zhang, “SALMONN: Towards generic hearing
abilities for large language models,” in The Twelfth International
Conference on Learning Representations, 2024. [Online]. Available:
https://openreview.net/forum?id=Vti B5p1l6
[2] Y. Chu, J. Xu, Q. Yang, H. Wei, X. Wei, Z. Guo, Y. Leng, Y. Lv,
J. He, J. Lin et al., “Qwen2-audio technical report,” arXiv preprint
arXiv:2407.10759, 2024.
[3] Microsoft and Others, “Phi-4-mini technical report: Compact yet powerful multimodal language models via mixture-of-loras,” arXiv preprint
arXiv:2503.01743, 2025.
[4] A. Rouditchenko, S. Bhati, E. Araujo, S. Thomas, H. Kuehne, R. Feris,
and J. Glass, “Omni-r1: Do you really need audio to fine-tune your
audio llm?” arXiv preprint arXiv:2505.09439, 2025.
[5] OpenAI, J. Achiam, S. Adler, and ..., “Gpt-4 technical report,” 2024.
[Online]. Available: https://arxiv.org/abs/2303.08774
[6] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: Pre-training
of deep bidirectional transformers for language understanding,” in
Proceedings of the 2019 Conference of the North American Chapter
of the Association for Computational Linguistics: Human Language
Technologies, Volume 1 (Long and Short Papers). Minneapolis,
Minnesota: Association for Computational Linguistics, Jun. 2019, pp.
4171–4186. [Online]. Available: https://aclanthology.org/N19-1423
[7] A. Baevski, Y. Zhou, A. Mohamed, and M. Auli, “wav2vec 2.0: A
framework for self-supervised learning of speech representations,” in
Advances in Neural Information Processing Systems 33, 2020, pp.
12449–12460.
[8] X. Wang, K. Evanini, Y. Qian, and M. Mulholland, “Automated
scoring of spontaneous speech from young learners of english using
transformers,” in 2021 IEEE Spoken Language Technology Workshop,
SLT 2021, Shenzhen, China, January 19-22, 2021. IEEE, 2021, pp.
705–712. [Online]. Available: https://doi.org/10.1109/SLT48900.2021.
9383501
[9] S. Banno and M. Matassoni, “Proficiency assessment of l2 spoken
english using wav2vec 2.0,” in 2022 IEEE Spoken Language Technology
Workshop (SLT). Doha, Qatar: IEEE, 2023, pp. 1088–1095.
[10] H. Nguyen and S. Park, “Providing automated feedback on formative
science assessments: Uses of multimodal large language models,”
in Proceedings of the 15th International Learning Analytics and
Knowledge Conference, ser. LAK ’25. New York, NY, USA:
Association for Computing Machinery, 2025, p. 803–809. [Online].
Available: https://doi.org/10.1145/3706468.3706480
[11] T.-H. Lo, F.-A. Chao, T.-I. Wu, Y.-T. Sung, and B. Chen, “An
effective automated speaking assessment approach to mitigating data
scarcity and imbalanced distribution,” in Findings of the Association
for Computational Linguistics: NAACL 2024. Mexico City, Mexico:
Association for Computational Linguistics, 2024, pp. 1352–1362.
[Online]. Available: https://aclanthology.org/2024.findings-naacl.86
[12] N. H. de Jong, “Assessing second language speaking proficiency,”
Annual Review of Linguistics, vol. 9, pp. 541–560, 2023.
[13] S. Park and R. Ubale, “Multitask learning model with text and speech
representation for fine-grained speech scoring,” in 2023 IEEE Automatic
Speech Recognition and Understanding Workshop (ASRU). Taipei,
Taiwan: IEEE, 2023, pp. 1–7.
[14] S. Bannò, K. M. Knill, M. Matassoni, V. Raina, and M. Gales, “Assessment of l2 oral proficiency using self-supervised speech representation
learning,” in 9th Workshop on Speech and Language Technology in
Education (SLaTE). ISCA, 2023, pp. 126–130. [Online]. Available:
https://www.isca-speech.org/archive/slate 2023/banno23 slate.html
[15] E. Kim, J.-J. Jeon, H. Seo, and H. Kim, “Automatic Pronunciation
Assessment using Self-Supervised Speech Representation Learning,” in
Proc. Interspeech 2022, 2022, pp. 1411–1415.
[16] Y. Bengio, J. Louradour, R. Collobert, and J. Weston, “Curriculum
learning,” in International Conference on Machine Learning, 2009.
[Online]. Available: https://api.semanticscholar.org/CorpusID:873046
[17] S. B. Davis and P. Mermelstein, “Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences,” IEEE Transactions on Acoustics, Speech and Signal Processing,
vol. 28, no. 4, pp. 357–366, Aug. 1980.
[18] A. Loukina, K. Zechner, L. Chen, and M. Heilman, “Feature selection
for automated speech scoring,” in Proceedings of the Tenth Workshop
on Innovative Use of NLP for Building Educational Applications (BEA).
Association for Computational Linguistics, 2015, pp. 12–19.
[19] X. Xi, D. Higgins, K. Zechner, and D. M. Williamson, “Automated
scoring of spontaneous speech using speechratersm v1.0,” ETS Research
Report Series, vol. 2008, no. 2, pp. i–47, 2008.
[20] S. Xie, K. Evanini, and K. Zechner, “Exploring content features for
automated speech scoring,” in Proceedings of the 2012 Conference of the
North American Chapter of the Association for Computational Linguistics: Human Language Technologies. Association for Computational
Linguistics, 2012, pp. 103–111.
[21] T.-I. Wu, T.-H. Lo, F.-A. Chao, Y.-T. Sung, and B. Chen, “A
preliminary study on automated speaking assessment of English as
a second language (ESL) students,” in Proceedings of the 34th
Conference on Computational Linguistics and Speech Processing
(ROCLING 2022), Y.-C. Chang and Y.-C. Huang, Eds. Taipei, Taiwan:
The Association for Computational Linguistics and Chinese Language
Processing (ACLCLP), Nov. 2022, pp. 174–183. [Online]. Available:
https://aclanthology.org/2022.rocling-1.22/
[22] W. Chen, H. Liu, and X. Wang, “wavllm: Hierarchical curriculum
learning for multimodal speaking assessment,” IEEE/ACM Transactions
on Audio, Speech, and Language Processing, vol. 32, pp. 1024–1036,
2024.
[23] C. Zhang, Y. Wang, Y. Zhang, B. Li, Y. B. Zhao, Y. Lu, Y. Li, and Z. Liu,
“Oversampling, augmentation and curriculum learning for speaking
assessment with limited training data,” in Proc. INTERSPEECH 2024,
Kos Island, Greece, September 2024, pp. 506–510.
[24] Y. Fan, W. Xu, H. Wang, J. Wang, and S. Guo, “Pmr: Prototypical modal
rebalance for multimodal learning,” in 2023 IEEE/CVF Conference on
Computer Vision and Pattern Recognition (CVPR), 2023, pp. 20029–
20038.
[25] T. Yu, X. Liu, Z. Hou, L. Ding, D. Tao, and M. Zhang, “Selfpowered llm modality expansion for large speech-text models,” in
Proceedings of the 2024 Conference on Empirical Methods in Natural
Language Processing (EMNLP). Miami, Florida, USA: Association
for Computational Linguistics, November 2024, pp. 12401–12417.
[Online]. Available: https://aclanthology.org/2024.emnlp-main.690/
[26] S.-Y. Chen, T.-H. Lo, Y.-T. Sung, C.-Y. Tseng, and B. Chen, “A speaking
practice tool on teemi for automated english-speaking assessment of
chinese learners,” in Proceedings of the Annual Conference of the
International Speech Communication Association (INTERSPEECH),
Kos, Greece, September 2024, pp. 2048–2049. [Online]. Available: https:
//www.isca-archive.org/interspeech 2024/chen24aa interspeech.pdf
[27] K. Knill, D. Nicholls, M. Gales, M. Qian, and P. Stroinski, “Speak
& improve corpus 2025: an l2 english speech corpus for language
assessment and feedback,” ArXiv, vol. abs/2412.11986, 2024. [Online].
Available: https://api.semanticscholar.org/CorpusID:274789386
[28] W.-H. Peng, S. Chen, and B. Chen, “Enhancing automatic speech
assessment leveraging heterogeneous features and soft labels for ordinal
classification,” in 2024 IEEE Spoken Language Technology Workshop
(SLT). Macao: IEEE, 2024, pp. 945–952.
[29] E. J. Hu, yelong shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang,
and W. Chen, “LoRA: Low-rank adaptation of large language models,”
in International Conference on Learning Representations, 2022.
[Online]. Available: https://openreview.net/forum?id=nZeVKeeFYf9
[30] T. Dao, “Flashattention-2: Faster attention with better parallelism
and work partitioning,” in The Twelfth International Conference
on Learning Representations, 2024. [Online]. Available: https:
//openreview.net/forum?id=mZn2Xyh9Ec
