# SpeechJudge: Towards Human-Level Judgment for Speech Naturalness

1]The Chinese University of Hong Kong, Shenzhen 2]ByteDance Seed 3]DataBaker Technology

# SpeechJudge: Towards Human-Level Judgment for Speech Naturalness

###### Abstract

Aligning large generative models with human feedback is a critical challenge. In speech synthesis, this is particularly pronounced due to the lack of a large-scale human preference dataset, which hinders the development of models that truly align with human perception. To address this, we introduce SpeechJudge, a comprehensive suite comprising a dataset, a benchmark, and a reward model centered on naturalness—one of the most fundamental subjective metrics for speech synthesis. First, we present SpeechJudge-Data, a large-scale human feedback corpus of 99K speech pairs. The dataset is constructed using a diverse set of advanced zero-shot text-to-speech (TTS) models across diverse speech styles and multiple languages, with human annotations for both intelligibility and naturalness preference. From this, we establish SpeechJudge-Eval, a challenging benchmark for speech naturalness judgment. Our evaluation reveals that existing metrics and AudioLLMs struggle with this task; the leading model, Gemini-2.5-Flash, achieves less than 70% agreement with human judgment, highlighting a significant gap for improvement. To bridge this gap, we develop SpeechJudge-GRM, a generative reward model (GRM) based on Qwen2.5-Omni-7B. It is trained on SpeechJudge-Data via a two-stage post-training process: Supervised Fine-Tuning (SFT) with Chain-of-Thought rationales followed by Reinforcement Learning (RL) with GRPO on challenging cases. On the SpeechJudge-Eval benchmark, the proposed SpeechJudge-GRM demonstrates superior performance, achieving 77.2% accuracy (and 79.4% after inference-time scaling @10) compared to a classic Bradley-Terry reward model (72.7%). Furthermore, SpeechJudge-GRM can be also employed as a reward function during the post-training of speech generation models to facilitate their alignment with human preferences.

Zhizheng Wu at wuzhizheng@cuhk.edu.cn\checkdata[Project Page]https://speechjudge.github.io/ \checkdata[Dataset, Model, and Code]https://github.com/AmphionTeam/SpeechJudge

## 1 Introduction

The collection and integration of human feedback corpora for model alignment has become a critical stage in the development of modern large-scale generative models, proving indispensable in domains such as text [openai-rm, instructgpt, RLHF-anthoropic], image [imagereward, pick-a-pic], and video generation [visionreward, videoreward].

1

1footnotetext: Equal Contribution.

In the field of speech synthesis, naturalness has long been a cornerstone subjective metric for quality assessment [naturalspeech3, seedtts, cosyvoice, qwen2.5-omni, kimi-audio], representing one of the most general-purpose indicators of performance [tts-book-paul-taylor, tts-book-tanxu]. Prior research has explored automated speech assessment through MOS predictors [utmos, voicemos-challenge-2024] and constructed the human feedback corpora for specific attributes like the low-level acoustic quality [qualispeech]. However, a large-scale human feedback corpus centered on the holistic quality of naturalness—and a corresponding reward model trained to capture these preferences—remains a notably underexplored area. To fill this void, this paper focuses on the dimension of speech naturalness and present a three-part contribution:

1. A Large-scale Human Feedback Dataset: SpeechJudge-Data. We recruit human annotators to provide feedback on synthesized speeches, with a focus on assessing two fundamental speech aspects: intelligibility and naturalness. For data synthesis, we employ a diverse set of advanced, open-source zero-shot TTS models with varying architectures (such as CosyVoice2 [cosyvoice2], Ints [intp], F5-TTS [f5tts], and MaskGCT [maskgct]) to produce the compared speech pairs. We prepare speech references in both regular and expressive styles, construct multilingual target texts, and cover both monolingual and cross-lingual synthesis scenarios to ensure data diversity (Section 3.1). We instruct human annotators to perform two tasks based on a speech pair (Figure 1): (a) pointwise annotation of text accuracy to assess intelligibility, and (b) pairwise preference annotation to judge relative speech naturalness. This extensive effort, involving 69 labelers over two months, results in 99K annotated pairs, with each pair receiving an average of 2.49 annotations from different labelers. We believe the SpeechJudge-Data can serve as a valuable corpus for alignment research in speech synthesis (e.g., DPO alignment [dpo] or reward modeling [openai-rm, instructgpt, RLHF-anthoropic] in Section 5).

2. An Evaluation Benchmark for Speech Naturalness Judgment: SpeechJudge-Eval. We design a dedicated evaluation benchmark for the task of speech naturalness judgment. The task is structured as follows: given a target text and two corresponding speech samples, a model needs to judge which one is more natural. To construct the evaluation set, we select a subset from the SpeechJudge-Data where human annotators demonstrated high inter-annotator agreement, ensuring a high-quality ground truth. We assess the naturalness judgment capabilities of a wide range of metrics and models, including Word Error Rate (WER) [whisper, funasr], Fréchet Audio Distance (FAD) [fad], MOS predictors [utmos, dnsmos, meta-audiobox-aesthetics], Deepfake Detectors [aasist, audio-deepfake-verification], and AudioLLMs [qwen2.5-omni, kimi-audio, mimoaudio, gemini2.5, gpt4o]. Our evaluations reveal that even the most capable model—specifically, Gemini-2.5-Flash [gemini2.5] in our experiments—achieved less than 70% agreement with human preferences. This finding highlights a significant performance gap and underscores the substantial room for research and improvement in automated speech naturalness judgment.

3. A Generative Reward Model for Speech Naturalness: SpeechJudge-GRM. To develop a reward model that more effectively captures human preferences, we develop SpeechJudge-GRM, a generative reward model (GRM) [google-grm, deepseek-grm] trained on the SpeechJudge-Data. Specifically, we base our model on Qwen2.5-Omni-7B [qwen2.5-omni] and design a two-stage post-training process. During the first stage, we perform Supervised Fine-Tuning (SFT) as the “cold start” to improve the model’s instruction-following and rationale-based reasoning capabilities. To achieve this, we leverage Gemini-2.5-Flash [gemini2.5] to generate Chain-of-Thought (CoT) data for speech naturalness judgment task. In the second stage, we focus on more challenging cases of SpeechJudge-Data, which we define as instances where Gemini-2.5-Flash fails to make the correct judgment. Treating the human-annotated labels as the verifiable reward [deepseek-r1, deepseek-grm], we apply the GRPO-based Reinforcement Learning (RL) stage [grpo]. Our experiments demonstrate that when trained on the same data, SpeechJudge-GRM significantly outperformed the classic Bradley-Terry reward model (BTRM) [bt-rm, dpo], achieving a higher accuracy in predicting human preferences (77.2% for SpeechJudge-GRM vs. 72.7% for SpeechJudge-BTRM, Table 3). Besides, SpeechJudge-GRM also supports inference-time scaling and offers explainability through its CoT outputs. Furthermore, SpeechJudge-GRM can also be employed as an objective naturalness metric for sample selection (Figure 5) or as a reward function in RL algorithms to enhance the quality of existing speech generation models (Figure 6).

We will release all resources at https://github.com/AmphionTeam/SpeechJudge to facilitate future research in human-aligned speech synthesis. Audio samples are available at https://speechjudge.github.io/.

## 2 Related Work

Human Alignment for Speech Generation Aligning generative models with human feedback has proven crucial, a process also known as RLHF in LLMs [instructgpt, RLHF-anthoropic]. In the vision domain, many similar human preference datasets exist, such as Pick-a-Pic [pick-a-pic], ImageReward [imagereward], and VideoReward [videoreward]. The speech synthesis field, pioneering efforts to construct human corpora involved MOS datasets [utmos, voicemos-challenge-2024]. However, these datasets often did not use advanced TTS models for data generation, provided only the pointwise labels rather than the direct pairwise human preference, and were limited in scale. More recently, efforts have focused on building human feedback corpora centered on specific speech attributes, such as low-level acoustic quality [qualispeech], intelligibility [intp], or the instruction-following capabilities of spoken dialogue systems [wavreward, sagelm]. Despite this progress, a large-scale human feedback corpus built specifically around naturalness—one of the most general-purpose and fundamental metrics for speech synthesis [tts-book-paul-taylor, tts-book-tanxu]—has remained a critical missing piece.

AudioLLM as a Judge Using LLMs as automated quality evaluators is a prominent topic in the textual LLM field, popularized by the “LLM-as-a-judge” paradigm [llm-as-a-judge]. This idea has recently been extended to the audio domain. A concurrent work, AudioJudge [audiojudge], evaluates the capabilities and limitations of using AudioLLMs for speech quality assessment and paralinguistic understanding via prompt engineering. Furthermore, many studies have focused on fine-tuning AudioLLMs to better expose their understanding capabilities for specific tasks, such as discriminating the human-likeness of audio [audio-turing-test], modeling low-level acoustic qualities [chenchen-quality-audiollm, qualispeech], unifying multiple speech quality evaluation tasks into a single AudioLLM [sq-llm], and enhancing the assessment of instruction-following in spoken dialogue systems [wavreward, sagelm]. However, how to improve the ability of AudioLLMs to understand and judge speech naturalness, and how to use their quality-assessment capabilities as a reward to improve the post-training of speech generation models themselves, remain significantly underexplored.

## 3 SpeechJudge-Data

Our work is grounded in SpeechJudge-Data, a large-scale human feedback corpus for assessing the intelligibility and naturalness of synthesized speech. Formally, we aim to construct a dataset , where each triplet comprises a pair of synthesized speech samples and the corresponding target text . We instruct annotators to provide pointwise intelligibility and pairwise naturalness preference annotations based on (Figure 1).

### 3.1 Dataset Construction

We employ a diverse set of recent advanced zero-shot TTS models to prepare the dataset . Formally, for each sample , we denote the synthesized speech as being produced by the model , i.e., , where is the reference speech.

Model Selection For , we select the following six models of three architectures to enrich the distribution of the synthetic data (Figure 2(a)): (1) AR-based: ARS [maskgct], CosyVoice2 [cosyvoice2], CosyVoice2-INTP [intp], and Ints-INTP [intp]. The latter two are released by intp as intelligibility-enhanced models. (2) FM-based: F5-TTS. (3) MGM-based: MaskGCT [maskgct].

Prompt Construction To build diverse prompts for TTS, for , we adopt both regular and expressive speech samples. The regular samples are randomly selected from the Emilia-Large dataset [emilia-large]. The expressive samples are sourced from corpora rich in paralinguistics, including the emotional corpora: ParaSpeechCaps [paraspeechcaps], the accented corpora: L2-Arctic [l2arctic] and KeSpeech [kespeech], the whisper samples from an in-house corpus, and the character voices from video games Genshin Impact [genshin]. We display the detailed distribution of speech references in Figure 2(b).

The target text paired with each is constructed as follows: For regular samples, we randomly sample transcriptions from the Emilia-Large dataset [emilia-large]. These are then refined using DeepSeek-V3 [deepseek-v3] to correct typos and normalize punctuations. For expressive samples, we instruct DeepSeek-V3 to generate several scripts in different writing styles, tailored to the topic of (see Appendix 8.1 for more details). The languages of the target texts included Chinese (zh), English (en), and Chinese-English code-switching (mixed). For the combinations , we include both monolingual settings (en2en and zh2zh) and cross-lingual settings (zh2en, en2zh, zh2mixed, and en2mixed), where zh2en denotes Chinese with English , and similarly for others. The distribution of the language settings of is shown in Figure 2(c).

Speech Pair Construction To ensure the diversity of the pairs being compared, we follow intp and adopt both intra-model (i.e., and being generated by the same model) and inter-model pairs (i.e., and being generated by the different models). The distribution of the speech pair is shown in Figure 7.

### 3.2 Human Annotation

Given a sample , human annotators are instructed to perform both pointwise intelligibility and pairwise naturalness annotations (Figure 1). For intelligibility, annotators perform a binary classification to determine whether the speech ( and ) accurately reads the text without any content insertion, omission, or mispronunciation. For naturalness, they perform a five-scale Comparative Mean Opinion Score (CMOS) annotation to determine which of the two audio clips ( or ) sounds more natural and human-like.

We recruited professional annotators from a specialized data annotation firm in China and provided them with training for speech naturalness judgement. All annotators assigned to Chinese data were native speakers. For the English and code-switching datasets, annotators were required to have a proficiency level equivalent to at least CET-6. All personnel underwent standardized training based on a detailed annotation manual. Initially, we conducted a pilot study among researchers to refine the guidelines for clarity and unambiguity. To ensure annotation quality, each sample was independently annotated by two individuals. A third annotator was introduced if any disagreements. The detailed annotation guidelines are provided in Appendix 9.

Statistics We recruit 69 annotators and conduct annotations over two months. The resulting constructed dataset , which we denote as SpeechJudge-Data (raw), contains 99K samples, with each sample receiving an average of 2.49 annotations from different labelers. The market value of this annotation scale is estimated at over 500K RMB (about 70K USD). Based on the raw dataset, we also construct several subsets for analysis and reward model training. We provide detailed descriptions of each subset and its applications in the following sections and in Appendix 8.2.

Human Agreement Analysis We analyze the human annotations for naturalness in this section; discussions regarding intelligibility are provided in Appendix 9.3. For naturalness annotations, we evaluate the inter-annotator agreement across our constructed dataset. To simplify the analysis, given the sample , we transform the five-scale naturalness scale (CMOS) into a ternary classification system: either is better, is better, or their quality is a Tie. Based on this simplified classification, we categorize the annotation results into four distinct levels of agreement111Note: Each sample of SpeechJudge-Data is independently annotated by a minimum of two and a maximum of three annotators (Appendix 9).: (1) Full Agreement (FA): A consensus is reached among all annotators, with all ratings pointing to the same outcome (e.g., “2A”, “3A”, “2B”, “3B”). We use “2A” to indicate that two annotators both rated as better, while “3B” denotes three annotators all rating as better. (2) Weak Agreement (WA): This level captures cases where two annotators agree on a specific polarity, while the third annotator marks a Tie (e.g., “2A+1T”, “2B+1T”). We also include the “2T+1A” and “2T+1B” cases in this level. (3) Weak Disagreement (WD): This occurs when two annotators’ ratings share the same polarity, but the third’s rating is the opposite (e.g., “2A+B”, “2B+A”). (4) Full Disagreement (FD): This represents a complete lack of consensus, where all three annotators provide different classifications, denoted as “1A+1B+1T”.

In Figure 3, we demonstrate the distribution of these human agreement levels for the SpeechJudge-Data and its two subsets, regular and expressive (which are defined by their speech references). The figure shows that about 70% of the entire dataset falls into the Full Agreement (51.5%) or Weak Agreement (17.2%) levels. Furthermore, we observe that the expressive subset has a lower agreement level than the regular subset, which suggests that human evaluation of expressive speech generation is inherently a more challenging problem. Besides this sample-level agreement analysis, we also analyze the reliability of individual annotators, and we will discuss this in the Appendix 9.1.

## 4 SpeechJudge-Eval

To evaluate speech naturalness, existing studies typically organize their own listening tests, which often have inconsistent settings across different papers [naturalspeech3, seedtts, cosyvoice, qwen2.5-omni, kimi-audio]. Alternatively, previous researchers use proxy MOS predictors, such as UTMOS [utmos], as an objective metric. However, it remains an underexplored problem whether these metrics can accurately judge the naturalness of more advanced speech generation models [maskgct, cosyvoice2, f5tts, intp] and align with human preferences. Motivated by this, we construct a benchmark, SpeechJudge-Eval, specifically for the speech naturalness judgment task.

| Type | Protocol | |||
|
WER , Naturalness | |||
| SIM , Naturalness | ||||
| FAD , Naturalness | ||||
|
MOS , Naturalness | |||
|
|
|||
| AudioLLMs | Score , Naturalness |

| Prompts of AudioLLMs |

-
*
We instruct AudioLLMs using two modes of prompt: plain and CoT. The text in blue is only employed during the CoT mode.

| Model | Regular | Expressive | Total |
| Objective Metrics | |||
| WER | 59.3 | 57.0 | 57.9 |
| SIM | 47.5 | 42.5 | 44.5 |
| FAD | 50.3 | 47.5 | 48.6 |
| MOS Predictor | |||
| DNSMOS | 61.0 | 55.8 | 57.9 |
| UTMOS | 54.0 | 53.5 | 53.7 |
| Content Enjoyment (CE) | 69.3 | 55.2 | 60.8 |
| Content Usefulness (CU) | 61.3 | 54.7 | 57.3 |
| Production Complexity (PC) | 39.3 | 48.7 | 44.9 |
| Production Quality (PQ) | 61.3 | 54.3 | 57.1 |
| Deepfake Detectors | |||
| AASIST | 40.5 | 50.8 | 46.7 |
| ADV | 35.3 | 40.3 | 38.3 |
| AudioLLMs (Open-source) | |||
| Phi-4-Multimodal | 54.8 | 58.5 | 57.0 |
| Qwen2.5-Omni-7B | 62.0 | 59.7 | 60.6 |
| Kimi-Audio-7B-Instruct | 65.5 | 68.0 | 67.0 |
| Gemma-3n-E4B-it | 49.0 | 47.7 | 48.2 |
| Voxtral-Mini-3B-2507 | 60.0 | 53.3 | 56.0 |
| MiDashengLM | 58.8 | 63.5 | 61.6 |
| MiMo-Audio-7B-Instruct | 61.3 | 49.3 | 54.1 |
| AudioLLMs (Closed-source) | |||
| Gemini-2.5-Flash | 73.5 | 66.2 | 69.1 |
| Gemini-2.5-Pro | 73.0 | 62.2 | 66.5 |
| GPT-4o mini Audio | 56.3 | 46.7 | 50.5 |
| GPT-4o Audio | 71.5 | 64.7 | 67.4 |

- *

### 4.1 Task Description

Task Formulation We formulate the naturalness judgment task as a pairwise comparison, specifically a win-or-lose binary classification task: Given a target text and a corresponding audio pair , a model needs to determine which audio has better naturalness. This results in a binary choice: either is better or is better. We use the human answer as the ground truth, and use Accuracy to measure the judgment performance of a model on the evaluation set :

| (1) |

where is the total number of samples in the evaluation set, and represent the answers of the model and human for the sample , respectively. is the indicator function.

Evaluation Data We sample a subset of the SpeechJudge-Data to create the evaluation set for SpeechJudge-Eval. Specifically, we first select a subset that contains only preference data (i.e., we filter out samples with the “Tie” annotation), and then choose only those with full-agreement-level (FA) samples to ensure a high-quality ground truth. We perform sampling from both the regular and expressive subsets of SpeechJudge-Data and proportionally cover the three target text languages (zh, en, and mixed) within each subset. The final SpeechJudge-Eval dataset consists of 1,000 samples. The construction details of SpeechJudge-Eval and its distribution can be found in Appendix 8.2.

### 4.2 Benchmark for Different Models

We test the naturalness judgment capability of various models based on SpeechJudge-Eval. We consider four different categories of models, whose evaluation protocols are shown in Table 1:

-
1.
Objective metrics, such as WER [whisper, funasr], SIM [wavlm], and FAD [fad] in audio generation tasks. We assume that a better value of these metrics (e.g., lower for WER and FAD; higher for SIM) indicates better naturalness.

-
2.
MOS Predictors, including DNSMOS [dnsmos], UTMOS [utmos], and predictors from audiobox-aesthetics (CE, CU, PC, and PQ) [meta-audiobox-aesthetics]. We assume that a higher MOS score corresponds to better naturalness.

-
3.
Deepfake detectors, which are typically pre-trained on a binary classification task to predict whether an audio is fake or not [aasist, audio-deepfake-verification]. We assume that an audio with a lower fake probability should have better naturalness.

-
4.
AudioLLMs, which are employed to test their speech naturalness understanding capabilities in a zero-shot manner

222We assume that the adopted AudioLLMs have not been directly trained on the speech naturalness judgment task. Their performance on this benchmark is therefore considered a zero-shot capability.. We include the open-source Phi-4-Multimodal [phi4-mm], Qwen2.5-Omni [qwen2.5-omni], Kimi-Audio [kimi-audio], Gemma-3n [gemma3], Voxtral [voxtral], MiDashengLM [midashenglm], Mimo-Audio [mimoaudio], and the closed-source Gemini-2.5 [gemini2.5] and GPT-4o [gpt4o]. We use the plain prompt of Table 1 to instruct the model to pairwise score the naturalness of two audios. We use their grading to determine the naturalness preference.

The performance of different models on SpeechJudge-Eval is presented in Table 2. A key observation is that speech naturalness judgment is a highly challenging task. The leading model, Gemini-2.5-Flash, still only achieves less than 70% agreement with human preferences. When comparing different models, we find that: (1) common objective metrics and MOS predictors show only a weak correlation with human preferences, often achieving less than 60% accuracy and sometimes performing at the level of a random guess (around 50%). (2) While deepfake detectors are highly effective at distinguishing between machine-generated and human-recorded speech [audio-deepfake-verification, aasist], their ability to do so is not well-aligned with the naturalness objective when comparing two generated samples. (3) AudioLLMs demonstrate significant potential for this task. While some models, such as Gemma-3n and GPT-4o mini Audio, perform at a chance level, a number of others achieve an accuracy exceeding 60%. This promising performance motivates us to further leverage these AudioLLMs for the design of a reward model for speech naturalness.

## 5 SpeechJudge-GRM

Based on the proposed SpeechJudge-Data, we further explore how to train a reward model capable of accurately capturing human preferences. Specifically, we propose SpeechJudge-GRM, where we leverage the inherent audio understanding capabilities of AudioLLMs (specifically, Qwen2.5-Omni-7B [qwen2.5-omni]) to elicit their speech naturalness judgment capability. Compared to the classic BTRM [bt-rm], the key strengths of GRM are its ability to enable Chain-of-Thought (CoT) reasoning and its support for test-time computation via majority voting, which ultimately leads to improved preference judgment performance [google-grm].

### 5.1 Methodology

We develop SpeechJudge-GRM based on Qwen2.5-Omni-7B (Thinker) [qwen2.5-omni]. Inspired by the powerful capabilities of RL with the verifiable reward (RLVR) [grpo, deepseek-r1], our natural initial approach is to treat the human preference for the pair as a verifiable reward, and launch a RLVR training based on Qwen2.5-Omni. However, in practice, we find that the instruction-following reasoning capabilities of Qwen2.5-Omni are very weak (more detailed discussions can be found in Appendix 11). Therefore, we adopt a two-stage post-training process (“SFT + RL”) to develop SpeechJudge-GRM (Figure 4). We describe the details as follows.

SFT Stage We consider SFT as a “cold start” stage to improve the Qwen2.5-Omni’s instruction-following, reasoning, and speech naturalness understanding capabilities. We select Gemini-2.5-Flash [gemini2.5]—one of the leading closed-source models on SpeechJudge-Eval (Table 2)—to serve as a teacher model, and instruct it to generate the CoT data. Specifically, for each sample from SpeechJudge-Data, we use the CoT prompt from Table 1 (denoted as ) to instruct Gemini-2.5-Flash to generate a rationale-based output (denoted as ). We then extract the preference judgment () from this output. For samples where Gemini-2.5-Flash’s preference is consistent with the human (i.e., ), we concatenate the CoT prompt and the model’s output, , to create a data point for our SFT dataset. Conversely, we consider the sample a challenging case and reserve the prompt for the second-stage RL dataset. During the SFT stage, for each training sample , we perform the next token prediction only on the segment .

RL Stage We treat the annotated human preference as a verifiable reward, and, building on the SFT model, we further trained it using the GRPO algorithm [grpo]. Specifically, for each sample in the RL dataset, we adopt the CoT prompt to instruct the policy model to conduct multiple rollouts during each iteration. For the -th rollout, we parse the model’s preference for , denoted as . Following [deepseek-grm], we use an accuracy-based rule to calculate the reward: the reward is 1 if , and -1 otherwise. In other words, during the RL stage, we only constrain the model’s final naturalness judgment to align with human preferences, allowing the model to autonomously optimize its reasoning and rationale generation capabilities.

We denote the training dataset of SpeechJudge-GRM as SpeechJudge-Data (train). Its construction process is as follows (see Appendix 8.2 for more details). Based on the raw SpeechJudge-Data, we first filter out all samples at the Full Disagreement (FD) level. For the other samples—at the FA, WA, and WD levels—we apply a majority voting principle among annotators to determine the final label for each. We then further exclude samples with a “Tie” label, using only the remaining preference data to form the SpeechJudge-Data (train). We use LoRA [lora] to fine-tune the GRM during both the SFT and RL stages. Other experimental setup details are provided in Appendix 12.

| Model | Regular | Expressive | Total |
| Qwen2.5-Omni-7B | 62.0 | 59.7 | 60.6 |
| Gemini-2.5-Flash | 73.5 | 66.2 | 69.1 |
| SpeechJudge-BTRM | 77.5 | 69.5 | 72.7 |
| SpeechJudge-GRM (SFT) | 77.8 | 73.7 | 75.3 |
| w/ Voting@10 | 77.4 | 77.6 | 77.6 |
| SpeechJudge-GRM (SFT+RL) | 79.0 | 76.0 | 77.2 |
| w/ Voting@10 | 80.5 | 78.7 | 79.4 |

- *

### 5.2 Effectiveness of SpeechJudge-GRM on Naturalness Judgement

To verify the effectiveness of SpeechJudge-GRM for naturalness judgment, we evaluate it on the SpeechJudge-Eval benchmark. We develop SpeechJudge-BTRM as a baseline, which utilizes the BTRM paradigm [bt-rm, dpo] by adding a linear layer on Qwen2.5-Omni-7B (Thinker) to produce a single scalar reward prediction. SpeechJudge-BTRM also uses LoRA fine-tuning and uses the same training data as SpeechJudge-GRM.

From the results of Table 3, we can observe that: (1) The SpeechJudge-BTRM achieves a 72.7% agreement with human preferences on SpeechJudge-Eval, a level of performance comparable to the initial development of BTRMs in the textual LLM RLHF field [openai-rm, RLHF-anthoropic, instructgpt]. (2) After conducting SFT training with the CoT data, the accuracy of SpeechJudge-GRM (SFT) reaches 75.3%. Besides, further RLVR training improves the final model SpeechJudge-GRM (SFT+RL) to an accuracy of 77.2%. (3) Due to the generative nature of the GRM, we can further enhance the accuracy of SpeechJudge-GRM using inference-time scaling. For example, by using majority voting across 10 outputs instead of just one, the accuracy is improved by approximately 2 percentage points (75.3% 77.6%; 77.2% 79.4%). These results collectively verify the effectiveness of our proposed SpeechJudge-GRM for judging speech naturalness.

### 5.3 High-Quality Sample Selection based on SpeechJudge-GRM

We investigate the effect of SpeechJudge-based reward models for high-quality sample selection. We use the hard cases from SeedTTS-Eval [seedtts] and the code-switching cases from Amphion-TTS-Eval [amphion] as target texts. For each text, we instruct the Qwen2.5-Omni-7B (Talker) [qwen2.5] to generate 100 speeches. We then ask human subjects to compare the best-of-100 output—as selected by either SpeechJudge-BTRM or SpeechJudge-GRM—against a randomly sampled output. The evaluation measures the win/lose/tie ratios based on speech naturalness. From Figure 5, we observe that the best-of-100 samples selected by both SpeechJudge-BTRM and SpeechJudge-GRM are more likely to outperform a randomly selected sample from the same set. This finding demonstrates the advantage of using the SpeechJudge-Data corpus for training human-aligned reward model. Furthermore, SpeechJudge-GRM exhibits better performance than SpeechJudge-BTRM, which highlights the superiority of the proposed GRM.

### 5.4 Post-Training of Zero-Shot TTS based on SpeechJudge-GRM

We investigate the effect of using SpeechJudge-GRM as a reward function for post-training of TTS model. Specifically, we develop a new zero-shot TTS model, Qwen2.5-0.5B-TTS, to serve as the base model, which was not involved in the construction of the SpeechJudge-Data. This model is based on Qwen2.5-0.5B [qwen2.5], adopts the classic two-stage “AR+Diffusion” architecture [seedtts, cosyvoice], uses the speech tokenizer from DualCodec [dualcodec], and is pre-trained on the Emilia dataset [emilia-large].

| Model | T-ACC | N-CMOS |
| Qwen2.5-0.5B-TTS | 84.0% | 0.00 |
| w/ INTP | 87.0% | 0.18 |
| w/ SpeechJudge-Data | 91.0% | 0.16 |
| w/ SpeechJudge-GRM (offline) | 91.0% | 0.21 |
| w/ SpeechJudge-GRM (online) | 90.0% | 0.25 |

Based on this pre-trained model, we design four comparative methods: (1) w/ INTP: We use the intelligibility preference dataset, INTP [intp], to perform offline DPO alignment [dpo]. (2) w/ SpeechJudge-Data: We use the SpeechJudge-Data (train) to perform offline DPO alignment. (3) w/ SpeechJudge-GRM (offline): We use SpeechJudge-GRM as an offline preference data annotator. We take all speech pairs from the INTP dataset and re-annotate their preference labels using SpeechJudge-GRM, then perform offline DPO alignment on the resulting data. (4) w/ SpeechJudge-GRM (online): We use SpeechJudge-GRM as a reward function for the online DPO algorithm [online-dpo]. The training data consists of only the prompts from INTP (i.e., the target texts and speech references for zero-shot TTS).

We use SeedTTS-Eval [seedtts] and Amphion-TTS-Eval [intp, amphion, vevo2] as evaluation sets. We present the objective results (WER and SIM) in Table 14 and the subjective results in Figure 6. We observe that both intelligibility and naturalness are enhanced for all the four methods after post-training. Additionally, the post-training method based on SpeechJudge-GRM achieves a greater improvement in naturalness (Figure 6(a)). Besides, the SpeechJudge-based methods could match or lead to a slight improvement in speaker similarity (Figure 6(b)).

## 6 Limitations and Future Work

While SpeechJudge-Data and SpeechJudge-GRM represent a step toward human-aligned speech naturalness judges, several limitations remain and open up directions for future work.

Scope of data and annotators. Our corpus is constructed entirely from synthetic TTS outputs in Chinese, English, and Chinese–English code-switching, and our annotators are professional raters in China (native Mandarin speakers with high but still L2 English proficiency). As shown in Appendix 9.2, inter-annotator agreement is noticeably higher on Chinese than on English and mixed subsets, indicating that the current dataset primarily reflects the preferences of Chinese and Chinese–English bilingual listeners, and is tailored to TTS-style read speech rather than spontaneous conversation. Extending SpeechJudge-Data to more languages, speaking styles, and listener populations (including native speakers of other languages and more diverse cultural backgrounds) is an important direction for building more universal naturalness judges.

Residual failure cases. Our error analysis in Appendix 13.3 shows that SpeechJudge-GRM’s remaining mistakes on SpeechJudge-Eval concentrate on some specific trade-offs, such as clean but robotic vs. slightly noisy but lively speech, prosody vs. articulation, and extreme expressive styles like very high-F0 emotional speech or whispers. In these regimes the model can over-weight cleanness, under-weight style-appropriate prosody, or become effectively indifferent when preference gaps are extremely small. Future work could incorporate explicit modeling of recording conditions (e.g., background noise), style-aware priors, and targeted augmentation of expressive and cross-lingual examples to better capture these nuanced preferences.

CoT quality and teacher bias. The CoT capability of SpeechJudge-GRM is bootstrapped from a proprietary teacher (Gemini-2.5-Flash) via an SFT stage. Although our analyses in Appendix 13.2 suggest that GRM’s CoT is largely self-consistent and moderately faithful, the reasoning style still inherits biases from the teacher, and we do not perform large-scale human verification of the intermediate explanations. An interesting direction is to involve humans more directly in assessing and curating CoT rationales—similar in spirit to the concurrent SQ-LLM work [sq-llm], which incorporates human involvement in CoT annotation—for example, by collecting human-written or human-edited analyses, or learning from explicit feedback on explanation quality, and to explore alternative, fully open-source teachers for bootstrapping.

From coarse-grained to fine-grained naturalness. In this work, naturalness is annotated and modeled at the utterance level: for each sentence, annotators choose which of two speeches is more natural, and SpeechJudge-GRM outputs a single decision per pair. However, in real-world speech, naturalness is often highly non-uniform within an utterance—some segments sound very natural while others contain local artifacts, disfluencies, or prosodic issues. Our current formulation does not explicitly localize such fine-grained phenomena. A promising future direction is to collect segment-level or time-aligned human feedback and to train reward models that can produce not only utterance-level judgments but also fine-grained scores or rationales over time.

## 7 Conclusion

In this work, we tackle the challenge of aligning speech synthesis with human perception of naturalness by introducing SpeechJudge: a suite consisting of a large-scale human preference dataset (SpeechJudge-Data), a challenging benchmark (SpeechJudge-Eval), and a generative reward model (SpeechJudge-GRM). Our benchmark shows that even strong AudioLLMs struggle at naturalness judgment, reaching under 70% agreement with human preferences. In contrast, the proposed SpeechJudge-GRM achieves 77.2% accuracy on SpeechJudge-Eval (up to 79.4% with inference-time scaling @10), outperforming a classic Bradley–Terry reward model (72.7%). We further demonstrate that SpeechJudge-GRM serves as an effective reward function for post-training TTS models, leading to improved perceived naturalness in downstream evaluations. By releasing our data, benchmark, and models, we hope to enable further research on human-aligned speech generation and more reliable evaluation of speech naturalness.

## Ethics Statement

Our dataset was constructed with feedback from paid professional annotators under fair labor conditions, and the data itself consists of synthesized speech from properly licensed corpora, safeguarding the privacy of all individuals. All human annotations were collected from professional annotators recruited by a third-party data annotation company under written informed consent, on synthetic TTS audio only, with no collection of personally identifying information. We acknowledge that our models may reflect linguistic biases present in the English and Chinese source data and recognize that generative speech technology has dual-use potential. We do not condone any malicious use of our work, such as the creation of misleading deepfakes.

## 8 Details of SpeechJudge-Data

### 8.1 Details of Prompt Construction

For the target texts paired with the regular speech references, we use DeepSeek-V3 [deepseek-v3] to fix typos and normalize punctuations, the prompt used is listed below.

For the target texts paired with the expressive speech references, we use DeepSeek-V3 to generate several scripts in different writing styles based on the speech reference’s text, the prompt used is listed below.

### 8.2 Subsets of SpeechJudge-Data

We construct several subsets based on SpeechJudge-Data (Figure 8). We begin with the SpeechJudge-Data (raw) corpus, containing 99K pairs, where each pair is annotated by multiple labelers as a five-scale naturalness CMOS. We aggregate these annotations via a majority vote for each pair, and subsequently discard all “Tie” pairs, yielding the 79K-pair human preference data, denoted as SpeechJudge-Data (pref).

During our preliminary analysis based on SpeechJudge-Data (pref), we observe that a significant disparity in intelligibility between two speech samples can overshadow the subtler quality of naturalness, biasing human preference toward the more comprehensible sample. To mitigate this confounding factor and create a more high-quality dataset focused specifically on naturalness, we further refine the data. Specifically, we retain only pairs where the absolute WER gap of those is below 12%. This process results in the 44K-pair high-quality SpeechJudge-Data (hq) subset, ensuring that its preference labels are more reflective of genuine differences in naturalness.

From SpeechJudge-Data (hq), we construct our benchmark, SpeechJudge-Eval, by applying stratified sampling to FA-level pairs, resulting in 1,000 pairs; its composition is detailed in Table 4. Similarly, we use the same strategy to construct a validation set of the same size, SpeechJudge-Data (dev). The remaining 42K pairs, SpeechJudge-Data (train), constitute the training set for our reward models.

| Subset | Source of Speech References | Languages of Target Texts | # Pairs | ||
| Regular | Emilia-Large | en | 200 | ||
| zh | 200 | ||||
| Expressive |
|
en | 200 | ||
| zh | 200 | ||||
| mixed | 200 |

## 9 Human Annotation Details

The complete annotation guidelines are attached below:

### 9.1 Individual Annotator Reliability

To assess the reliability of individual annotators, we computed agreement rate for each participant. This rate measures the extent to which an annotator’s judgments align with those of their peers on the same sample .

For a given sample annotated by a group of annotators, the agreement score for annotator is calculated as the fraction of the other annotators who assigned the exact same label. An annotator’s final reliability score is the average of these scores across all samples they evaluated. We excluded participants who annotated fewer than 10 samples from this analysis.

Formally, for an annotator who labeled samples, the agreement rate for sample is defined as:

The overall agreement rate for annotator , denoted as , is then:

where is the label assigned by annotator to sample . The label (i.e., Tie) is treated as a distinct category, and it’s agreement is counted only on exact matches.

Figure 9 illustrates the distribution of these agreement rates for our 69 annotators for SpeechJudge-Data (raw). The distribution is generally unimodal with a peak in the 60–70% range333We have noted that one annotator’s agreement with the others is less than 30%, so we ultimately removed his data from SpeechJudge-Data., which indicates a consistent and reliable level of performance across the annotation pool.

### 9.2 Inter-Annotator Agreement

Complementary to the per-annotator reliability analysis in Appendix 9.1, we now quantify the overall level of agreement among annotators at the dataset level. Following common practice in RLHF-style preference datasets, we compute inter-annotator agreement as the probability that two annotators chosen at random assign the same preference label to the same pair [openai-rm, instructgpt, imagereward]. Table 5 summarizes the results.

Ternary vs. binary preferences. SpeechJudge-Data (raw) contains ternary labels (i.e., A better / B better / Tie), and the corresponding inter-annotator agreement is over the whole corpus. After removing “Tie” cases and restricting to clear binary preferences (SpeechJudge-Data (pref), labels A better / B better only), the agreement rises to . This level is comparable to well-established RLHF datasets in text and vision: ImageReward reports an agreement of [imagereward], and InstructGPT reports on binary A/B comparisons [instructgpt]. These numbers indicate that, despite the inherent subjectivity of the task, our binary preference data are on par with existing human-feedback corpora.

Effect of style. Within each language, expressive prompts are slightly harder than regular ones: for example, on SpeechJudge-Data (pref), regular zh reaches agreement while expressive zh reaches . A similar, mildly lower agreement is observed for expressive en and mixed subsets. Overall, however, regular vs. expressive speech remains in a similar agreement range, suggesting that our guidelines allow annotators to make consistent naturalness judgments.

Effect of language and code-switching. A more pronounced pattern emerges across languages. In both the raw ternary data and the binary preference subset, agreement on Chinese (zh) is noticeably higher than on English (en) and especially on code-switched (mixed) samples (e.g., vs. and in SpeechJudge-Data (pref)). We attribute this gap to the annotator population: our annotators are predominantly native Mandarin speakers; while English and code-switched items are assigned only to annotators who pass a high English proficiency bar, they are still L2 or bilingual listeners. Similar phenomena—lower agreement for L2 or cross-lingual annotations compared to L1 ones—have also been discussed in recent text RLHF studies [RLHF-anthoropic].

In the current work, we therefore interpret SpeechJudge-Data as primarily reflecting the preferences of Chinese and Chinese–English bilingual listeners. In future work, we plan to augment the corpus by recruiting native English speakers for the English subset and stronger bilingual/native speakers for code-switched data, so as to improve agreement on these subsets and broaden the cultural and linguistic coverage of the dataset.

|
|
Regular | Expressive |
|
||||||
| en | zh | en | zh | mixed | ||||||
| SpeechJudge-Data (raw) | Ternary | 54.9 | 55.5 | 50.2 | 49.5 | 36.3 | 50.7 | |||
| SpeechJudge-Data (pref) | Binary | 61.7 | 74.4 | 59.9 | 73.0 | 62.5 | 69.0 | |||
| ImageReward [imagereward] | – | – | – | – | – | 65.3 | ||||
| InstructGPT [instructgpt] | – | – | – | – | – | 72.6 |

-
*
SpeechJudge-Data (raw) uses ternary labels (A better / B better / Tie); SpeechJudge-Data (pref) removes ties and uses binary labels (A better / B better). Results of SpeechJudge-Data are further broken down by style and language.

### 9.3 Intelligibility Annotation Analysis

We provide a detailed analysis of the relationship between the mostly common used objective intelligibility metric, Word Error Rate (WER), and the subjective human judgments of intelligibility. Our goal is to determine the extent to which WER can serve as a reliable proxy for human perception.

We use all the speech samples from SpeechJudge-Data (raw) for this analysis. We visualize the relationship between WER and the subjective text accuracy in Figure 10. For the regular speeches (the orange curve), we observe a consistent negative correlation: as the WER increases, its perceived text accuracy steadily declines. For the expressive speeches (the green curve), the similar trend holds for expressive speech when WER is under about 12%. When WER is over the threshold, however, the correlation between WER and the subjective text accuracy weakens significantly. We think this divergence is sourced from that the greater stylistic variations in expressive speech pose a substantial challenge to the robustness of ASR systems compared to the regular samples.

## 10 Details of Evaluation on the SpeechJudge-Eval Benchmark

During the evaluation on the SpeechJudge-Eval Benchmark of Table 2, we adopt the following protocol for each model:

-
•
WER [whisper, funasr]: We employ Whisper-large-v3

4 https://huggingface.co/openai/whisper-large-v3 [whisper] for English texts, and Paraformer-zh https://huggingface.co/funasr/paraformer-zh [paraformer, funasr] for Chinese and code-switching texts. -
•
SIM [wavlm]: We compute the cosine similarity between the WavLM TDNN

6 https://github.com/microsoft/UniSpeech/tree/main/downstreams/speaker_verification [wavlm] speaker embeddings of generated samples and the prompt samples. -
•
FAD [fad]: We use the officially released checkpoint, VGGish, to obtain the FADs of audios.

- •
- •
-
•
AASIST [aasist]: It is a common baseline model for audio deepfake detection, employs a heterogeneity-aware approach to integrate spectral and temporal sub-graphs. We use a large-scale in-house corpus to train the model.

-
•
ADV [audio-deepfake-verification]: It is a state-of-the-art (SOTA) deepfake detection model built upon the pre-trained w2v-bert-2.0

9 https://huggingface.co/facebook/w2v-bert-2.0. It utilizes a multi-task training approach involving deepfake source tracing to extract robust audio deepfake features. We use the same corpus of AASIST to train the model. -
•
AudioLLMs: We use the plain prompt of Table 1 to instruct AudioLLMs to pairwise score the naturalness of two audios. For the closed-source models, we use the official API released by Google

1 https://ai.google.dev/gemini-api/docs/models for Gemini and OpenAI https://platform.openai.com/docs/models for GPT. We use the model variants gemini-2.5-flash, gemini-2.5-pro, gpt-4o-mini-audio-preview-2024-12-17, and gpt-4o-audio-preview-2025-06-03 for Gemini-2.5-Flash, Gemini-2.5-Pro, GPT-4o mini Audio, and GPT-4o Audio.

## 11 More Evaluation Results of Existing AudioLLMs

Using AudioLLM as a judge models, prompt engineering strategies are usually believed crucial for improving the performance [llm-as-a-judge, audiojudge]. Some common prompt engineering strategies include using the CoT prompts to activate the model’s thinking and reasoning abilities [google-grm, deepseek-grm, audiojudge], or employing few-shot evaluation formats [llm-as-a-judge, mimoaudio].

### 11.1 Chain-of-thought prompting for AudioLLM judges

In this study, we investigate whether using the CoT from Table 1 helps AudioLLMs better judge speech naturalness. We display the results in Table 6. Interestingly, we find that some closed-source AudioLLMs, such as Gemini-2.5-Flash, improve their performance on SpeechJudge-Eval through this thinking and reasoning process. However, this strategy often does not work for existing open-source AudioLLMs. For example, the results in Table 6 show that Qwen2.5-Omni-7B and Kimi-Audio-7B-Instruct, which is already the leading open-source models on SpeechJudge-Eval (Table 2), actually sees a decline in performance when using the CoT prompt.

| Model | Regular | Expressive | Total |
| Qwen2.5-Omni-7B | 62.0 | 59.7 | 60.6 |
| w/ CoT prompt | 54.4 | 50.6 | 52.9 |
| Kimi-Audio-7B-Instruct | 65.5 | 68.0 | 67.0 |
| w/ CoT prompt | 67.4 | 66.1 | 66.5 |
| Gemini-2.5-Flash | 73.5 | 66.2 | 69.1 |
| w/ CoT prompt | 75.0 | 67.5 | 70.5 |

Based on our preliminary qualitative analysis, we believe the reason why the open-source AudioLLMs do not work well with the CoT prompt is that their foundational capabilities are relatively weak. These weaknesses include instruction-following (such as format-following), multiple-audio understanding, long-context processing, and reasoning abilities. This is also why, when we developed SpeechJudge-GRM, we did not directly apply RLVR on top of Qwen2.5-Omni-7B. Instead, we used an initial SFT stage as a cold start.

### 11.2 Few-Shot Prompting for AudioLLM Judges

Motivated by the common belief that in-context examples can improve LLM judging ability [llm-as-a-judge, audiojudge], we also investigate few-shot prompting for Qwen2.5-Omni-7B on SpeechJudge-Eval.

| Model | Regular | Expressive | Total |
| Qwen2.5-Omni-7B (0-shot) | 62.0 | 59.7 | 60.6 |
| w/ 2-shot | 50.9 | 46.6 | 48.2 |
| w/ 4-shot | 46.1 | 52.0 | 49.6 |
| w/ 6-shot | 50.3 | 53.0 | 51.9 |
| w/ 8-shot | 51.0 | 54.8 | 53.3 |
| w/ 16-shot | 48.6 | 53.0 | 51.3 |

We start from the plain zero-shot prompt in Table 1, which asks the model to decide which of two audios is more natural. For the -shot setting, we prepend preference exemplars to this prompt. Each exemplar consists of: (i) a target text, (ii) an audio pair associated with this text, and (iii) the corresponding human naturalness label (which of the two audios is preferred). The model is then queried on a new SpeechJudge-Eval pair with the same instruction and output format. We evaluate Qwen2.5-Omni-7B with ; results are reported in Table 7.

Contrary to the usual expectation that few-shot prompting should help, we observe that none of the -shot configurations improves over the zero-shot baseline. On the contrary, the overall accuracy drops from in the 0-shot setting to – with few-shot prompts. The degradation is particularly pronounced on regular speech, while expressive speech shows small, inconsistent fluctuations.

These findings are consistent with our observations in Appendix 11.1 on chain-of-thought prompting: current open-source AudioLLMs such as Qwen2.5-Omni-7B still have limited instruction-following, multi-audio understanding, and long-context handling capabilities. Although few-shot prompts provide more information in principle, the model struggles to reliably associate multiple text-audio pairs with their labels in the context. As a result, increasing mainly adds complexity to the input without yielding better judgments. This further motivates our choice to move beyond pure prompt engineering, and instead train dedicated reward models (BTRM / GRM) on human preference data for robust naturalness evaluation.

## 12 Training Details of SpeechJudge-GRM

SFT Stage We use Gemini-2.5-Flash [gemini2.5] to generate the CoT data for SpeechJudge-Data (train). For the total 42K samples, Gemini-2.5-Flash’s judgments agree with human feedback on 25K samples, while they disagree on 17K samples. During the SFT stage, we fine-tune Qwen2.5-Omni-7B (Thinker) [qwen2.5-omni] on the 25K CoT data using LoRA [lora] with a rank of 128. We use Adam [adam, adamw] as the optimizer and set the learning rate to 5e-5. The maximum number of tokens per batch is 4000. We select the best checkpoint on SpeechJudge-Data (dev) as the SFT model, SpeechJudge-GRM (SFT).

RL Stage We use the 17K samples (as described above) to conduct DAPO [dapo], which is an enhanced variant of GRPO [grpo]. We utilize the ms-swift https://github.com/modelscope/ms-swift toolkit to launch the training process. We initialize the policy model with the SFT model and use LoRA training with a rank of 64. The number of rollouts for each prompt is set to 8, and the batch size is 32. The learning rate is 5e-6. We select the best checkpoint on SpeechJudge-Data (dev) as the final SpeechJudge-GRM model, i.e., SpeechJudge-GRM (SFT+RL).

## 13 More Evaluation Results for SpeechJudge-GRM

### 13.1 Performance under Different Data Distributions

|
Regular | |||||
| en2en | zh2en | zh2zh | en2zh | Avg | ||
| Qwen2.5-Omni-7B | 48.1 | 58.5 | 75.7 | 66.0 | 61.0 | |
| Gemini-2.5-Flash | 59.4 | 62.8 | 81.6 | 87.6 | 72.8 | |
| SpeechJudge-BTRM | 66.0 | 71.3 | 86.4 | 86.6 | 77.5 | |
| SpeechJudge-GRM (SFT) | 67.0 | 74.5 | 84.5 | 85.6 | 77.8 | |
| w/ Voting@10 | 65.1 | 75.5 | 83.5 | 85.6 | 77.3 | |
| SpeechJudge-GRM (SFT+RL) | 69.8 | 77.7 | 85.4 | 83.5 | 79.0 | |
| w/ Voting@10 | 75.5 | 79.8 | 80.6 | 86.6 | 80.5 |

|
Expressive | |||||||
| en2en | zh2en | zh2zh | en2zh | en2mixed | zh2mixed | Avg | ||
| Qwen2.5-Omni-7B | 43.6 | 51.7 | 61.1 | 70.9 | 64.5 | 69.6 | 59.7 | |
| Gemini-2.5-Flash | 53.6 | 68.3 | 73.3 | 76.4 | 64.5 | 67.1 | 66.2 | |
| SpeechJudge-BTRM | 62.9 | 56.7 | 72.2 | 85.5 | 68.6 | 67.1 | 69.5 | |
| SpeechJudge-GRM (SFT) | 61.4 | 66.7 | 89.1 | 77.8 | 74.4 | 73.4 | 73.7 | |
| w/ Voting@10 | 69.3 | 75.0 | 88.9 | 90.9 | 71.1 | 74.7 | 77.8 | |
| SpeechJudge-GRM (SFT+RL) | 71.4 | 65.0 | 81.1 | 86.4 | 70.2 | 81.0 | 76.0 | |
| w/ Voting@10 | 75.0 | 66.7 | 82.2 | 89.1 | 72.7 | 84.8 | 78.7 |

In Table 2 of the main paper, we reported accuracies on SpeechJudge-Eval aggregated over all languages and styles. Here we provide a more fine-grained view of how different judges behave under different data distributions, and we additionally evaluate on a completely out-of-distribution (OOD) test set involving real human speech versus commercial TTS clones.

Languages on the regular subset. Table 8 breaks down performance on the regular subset of SpeechJudge-Eval by language setting: en2en, zh2en, zh2zh, and en2zh. Across all models we observe that pairs involving Chinese (zh2zh and en2zh) are consistently easier than purely English pairs (en2en). For example, SpeechJudge-GRM (SFT+RL) reaches / accuracy on zh2zh/en2zh, but only on en2en, and the same trend holds for BTRM and Gemini-2.5-Flash. We believe several factors contribute to this gap. On the data side, as shown in Appendix 9.2, Chinese (zh) subsets exhibit higher inter-annotator agreement than English and mixed subsets, so supervision for English-like conditions is more varied and harder to fit. On the modeling side, current TTS systems may also produce relatively high-quality English outputs compared to Chinese, making the naturalness differences between two English samples more subtle and thus more difficult to judge reliably. Even under these challenges, SpeechJudge-GRM (SFT+RL) with Voting@10 still achieves the strongest average accuracy () among all open judges.

Languages on the expressive subset. Table 9 shows the same breakdown for the expressive subset. As expected, expressive speech is generally harder: all models are a few points lower than on their regular counterparts. The language pattern persists: Chinese-involving settings (zh2zh, zh2mixed) tend to be easier than en2en. For instance, SpeechJudge-GRM (SFT+RL) with Voting@10 attains on zh2zh and on zh2mixed, compared to on en2en. This is consistent with the inter-annotator statistics in Appendix 9.2, where expressive and code-switched English subsets show lower human agreement, and also reflects the increased linguistic and prosodic complexity of expressive and code-switched speech. Overall, expressive data are slightly more challenging than regular data, but the relative ranking of judges is stable and GRM maintains a clear advantage over BTRM and the AudioLLM baselines across all language settings.

| Model | Regular | Emotional | Accented | Whisper | Game |
| Qwen2.5-Omni-7B | 61.0 | 56.7 | 64.4 | 57.1 | 61.3 |
| Gemini-2.5-Flash | 72.8 | 63.7 | 66.7 | 74.6 | 66.0 |
| SpeechJudge-BTRM | 77.5 | 69.8 | 71.3 | 76.2 | 66.8 |
| SpeechJudge-GRM (SFT) | 77.8 | 75.3 | 80.5 | 71.4 | 70.2 |
| w/ Voting@10 | 77.3 | 76.7 | 80.5 | 74.6 | 78.7 |
| SpeechJudge-GRM (SFT+RL) | 79.0 | 74.4 | 81.6 | 79.4 | 74.5 |
| w/ Voting@10 | 80.5 | 78.1 | 85.1 | 82.5 | 75.7 |

Different prompt styles. Table 10 compares performance across prompt styles: regular, emotional, accented, whisper, and game-character speech. For SpeechJudge-BTRM we see that regular prompts are the easiest (), while emotional and game styles are notably harder ( and ). In contrast, SpeechJudge-GRM benefits more from the expressive settings: with SFT+RL and Voting@10, GRM achieves on regular, but rises to on accented and on whisper prompts, and narrows the gap on emotional and game prompts (from about points for BTRM to only a few points). We believe this reflects the advantage of the generative reward model and its CoT-based training: by explicitly reasoning about artifacts, prosody, and style, GRM is better able to handle challenging conditions such as accented speech and whisper, rather than overfitting to the most frequent regular style.

OOD evaluation on real speech vs. commercial TTS clones. Finally, Table 11 presents a new, fully
out-of-distribution evaluation designed to stress-test generalization to
real speech and unseen TTS architectures.
We select two native English voice actors, each recording 250 utterances
(500 in total), and use a commercial SeedTTS voice-cloning API https://www.volcengine.com/product/voicecloning to synthesize
clones for each utterance.
SeedTTS is a state-of-the-art proprietary system whose output quality is
typically higher than that of the open-source TTS models used to construct
SpeechJudge-Data, so this benchmark effectively probes the gap
between very strong modern TTS and human recordings.
For every sentence, we form a pair (human recording vs. SeedTTS clone) and
treat the human recording as the ground-truth more natural sample.
Neither the human recordings nor the SeedTTS outputs are present in
SpeechJudge-Data, making this a challenging OOD test.
The results show three interesting trends:

| Model | Character1 | Character2 | Avg |
| Deepfake Detectors | |||
| AASIST | 97.2 | 100 | 98.6 |
| ADV | 99.6 | 100 | 99.8 |
| AudioLLMs | |||
| Qwen2.5-Omni-7B | 48.0 | 44.8 | 46.4 |
| Kimi-Audio-7B-Instruct | 85.2 | 85.6 | 85.4 |
| Gemini-2.5-Flash | 52.8 | 48.8 | 50.8 |
| Naturalness Reward Model | |||
| SpeechJudge-BTRM | 55.6 | 45.2 | 50.4 |
| SpeechJudge-GRM (SFT) | 37.6 | 44.0 | 40.8 |
| w/ Voting@10 | 36.0 | 41.4 | 38.7 |
| SpeechJudge-GRM (SFT+RL) | 57.6 | 67.2 | 62.4 |
| w/ Voting@10 | 59.8 | 67.5 | 63.7 |

-
•
First, deepfake detectors (AASIST and ADV) achieve almost perfect accuracy (about ) on this task, even though they perform at chance level on SpeechJudge-Eval (Table 2), confirming that they mainly learn to discriminate real vs. synthetic rather than judge naturalness between two synthetic samples.

-
•
Second, among AudioLLMs, Kimi-Audio-7B-Instruct performs strongly ( on average), possibly because its training includes tasks or signals related to authenticity detection. In contrast, Gemini-2.5-Flash attains only accuracy, roughly at chance level and similar to Qwen2.5-Omni-7B (). This indicates that Gemini’s strong performance on SpeechJudge-Eval (69.1% in Table 2) does not automatically transfer to the human–vs–clone setting: it appears to be a good judge for synthetic-vs-synthetic naturalness comparisons, but it is not explicitly biased toward humans when facing a high-quality commercial clone.

-
•
Third, for naturalness reward models, SpeechJudge-BTRM is close to random guessing (), suggesting that the classical Bradley-Terry training may overfit more heavily to the specific synthetic generators in SpeechJudge-Data. Interestingly, SpeechJudge-GRM (SFT) alone performs even worse than BTRM, which we hypothesize is due to SFT encouraging the model to over-memorize our prepared CoT patterns and thus hurting OOD generalization [gem, sft-rl-mayi]. Once we add the RLVR stage, however, SpeechJudge-GRM (SFT+RL) improves substantially to (and with Voting@10), outperforming BTRM as well as generic AudioLLMs such as Qwen2.5-Omni-7B and Gemini-2.5-Flash on this benchmark. Given that the SeedTTS clones are already very close to human quality, this performance indicates that SpeechJudge-GRM has the potential to provide useful feedback not only for open-source TTS models, but also for strong proprietary systems that were never seen during training. This suggests that the generative reward modeling paradigm is more robust to distribution shift than classical BTRM and off-the-shelf AudioLLM judges, and that RL on human preferences is crucial for recovering and enhancing generalization beyond the synthetic training distribution.

### 13.2 Quality Analysis of Chain-of-Thought Reasoning

Beyond scalar accuracy, we further analyze the quality of the Chain-of-Thought (CoT) rationales produced by different judges. Specifically, we compare Gemini-2.5-Flash (teacher), SpeechJudge-GRM (SFT), and SpeechJudge-GRM (SFT+RL) along three aspects: (i) logical consistency between reasoning and conclusion, (ii) faithfulness and hallucination rate as judged by human experts, and (iii) differences in reasoning style.

Consistency between reasoning and conclusion. We first ask whether a model’s CoT reasoning is logically compatible with its final decision. Using DeepSeek-V3 [deepseek-v3] as a meta-judge, we prompt it with the instruction like: given the CoT analysis of A and B (over prosody, pacing, articulation, and overall naturalness) and the final scores assigned to A and B, decide whether the conclusion is consistent with the reasoning. DeepSeek-V3 returns a binary label (consistent / not consistent) and a brief justification. The specific instruction is as follows:

Table 12 summarizes the results on CoT outputs for SpeechJudge-Eval across the three models. All three models exhibit very high internal consistency: Gemini-2.5-Flash, SpeechJudge-GRM (SFT), and SpeechJudge-GRM (SFT+RL) achieve , , and consistency, respectively. This indicates that, at least at the coarse level captured by this automatic check, the CoT analyses are not arbitrary narratives but align well with the final naturalness preference.

| Model |
|
|
| Gemini-2.5-Flash | 97.9% | |
| SpeechJudge-GRM (SFT) | 97.6% | |
| SpeechJudge-GRM (SFT + RL) | 98.2% |

| Model |
|
|
|
Avg | ||||||
| Gemini-2.5-Flash | 1.90 | 2.00 | 2.10 | 2.00 | ||||||
| SpeechJudge-GRM (SFT) | 2.00 | 2.15 | 1.95 | 2.03 | ||||||
| SpeechJudge-GRM (SFT+RL) | 2.10 | 2.00 | 2.40 | 2.17 |

Human evaluation of CoT faithfulness. Logical consistency does not guarantee that the reasoning is correct or grounded in the audio. To measure faithfulness and hallucination, we conduct a human evaluation with experienced speech researchers (the background of these subjects are detailed in Appendix 14.1). For each model and each sampled SpeechJudge-Eval pair, the experts examine the CoT and assign a 1–3 score on three dimensions—(1) Prosody and Intonation, (2) Pacing and Rhythm, and (3) Articulation and Clarity—which match the dimensions specified in our CoT prompt (Table 1) and used by the models in their explanations. A score of 3 means “highly sensible (e.g., the CoT cites concrete audio details and the analysis is correct)”, 2 means “partially sensible (e.g., the overall good/bad tendency is right but details are coarse or partially off)”, and 1 means “not sensible / mostly hallucinatory”.

The results are reported in Table 13. On average, all three models obtain scores around 2.0, indicating that their CoT rationales are generally meaningful rather than dominated by hallucination. Importantly, SpeechJudge-GRM does not lose CoT quality compared to the Gemini teacher even though it is initialized from Gemini-generated rationales: SpeechJudge-GRM (SFT) slightly improves the average score to , and after RL, SpeechJudge-GRM (SFT+RL) further increases it to . The largest gain appears in the “Articulation and Clarity” dimension ( vs. for Gemini), suggesting that RLVR encourages the model to focus more accurately on pronunciation errors and intelligibility-related artifacts when explaining its decisions. Overall, the human study suggests that the RL stage not only improves preference alignment but also mildly enhances the faithfulness of the CoT reasoning.

Differences in reasoning style. Finally, we examine whether the three models share the same reasoning style or
develop distinct emphases.
We randomly sample 20 SpeechJudge-Eval cases on which all three models predict
the correct preference label, and collect their CoT outputs.
We then submit these triplets of CoTs (anonymized as “Model 1/2/3”) to three
strong text LLMs—GPT 5. https://chatgpt.com/, Gemini 3 Pro https://aistudio.google.com/, and DeepSeek-V https://chat.deepseek.com/, asking them to compare the similarities
and systematic differences among the models. The specific instruction is as follows:

The three analyzers broadly agree on the following qualitative picture: all models follow a similar structural template (prosody pacing articulation overall naturalness, followed by a 1–10 score), but they differ in what they emphasize:

-
•
Gemini-2.5-Flash: Described as the most sophisticated at linking prosody to semantic meaning and discourse structure, often explaining why an intonation pattern distorts the intended message.

-
•
SpeechJudge-GRM (SFT): Viewed as emotionally and narratively focused, with stable formatting and slightly more generous tone; it emphasizes human-likeness and expressiveness but is somewhat less detailed on low-level signal artifacts.

-
•
SpeechJudge-GRM (SFT+RL): Characterized as more critical and technically oriented: it pays more attention to mispronunciations, noise, and clarity, sometimes with blunter wording. Some analyzers note that its formatting is slightly less uniform than SpeechJudge-GRM (SFT), but its focus on error severity and technical correctness is stronger.

Taken together, these analyses suggest that SpeechJudge-GRM does not simply imitate the teacher’s explanation style. Instead, SFT initializes a shared analytical framework, while the RLVR stage shifts the model toward rationales that are more tightly coupled to human preferences and to concrete acoustic evidence (especially articulation and clarity), without sacrificing internal consistency. We will release the full set of CoT examples and meta-analyses in our open-source release to facilitate further research on CoT quality and reasoning in audio reward models.

### 13.3 Error Analysis of SpeechJudge-GRM

Although SpeechJudge-GRM performs well on SpeechJudge-Eval, it still disagrees with the human on a subset of it. We manually inspected these errors and found several recurring patterns.

Over-weighting cleanness vs. liveliness. In many errors the human-preferred sample contains mild background noise but clearly more human-like prosody and articulation, while the alternative is cleaner yet more robotic or over-smoothed. Annotators consistently favor the livelier sample, whereas GRM sometimes chooses the cleaner one, indicating that it can over-emphasize acoustic cleanness when the trade-off is subtle.

Prosody-articulation trade-offs. Another frequent pattern is a trade-off between expressive prosody and technical correctness. Humans often prefer speech with natural rhythm and intonation despite minor pronunciation issues, while GRM occasionally favors the perfectly articulated but flatter reading. These cases reveal that the relative weighting between prosody and articulation is still imperfectly captured.

Extreme expressive styles. Errors also concentrate in highly expressive styles. For emotional speech with very high F0 or strong emphasis, humans interpret the exaggerated prosody as appropriate for the style, but GRM sometimes penalizes it as “unnatural” and prefers a neutral reading. For whispers, the lack of voicing makes prosody judgments difficult; GRM occasionally fails to distinguish the more fluent whisper when both samples sound degraded.

Very small preference gaps. A small number of mistakes arise when both clips are high-quality and differ only in subtle cues (micro-pauses, breathing, slight emphasis shifts). In these cases GRM’s predictions are effectively close to random, which is unsurprising given the weak supervision signal.

In summary, SpeechJudge-GRM’s errors concentrate on nuanced trade-offs (clean vs. lively, prosody vs. articulation) and on challenging expressive styles, suggesting future work on modeling recording conditions, style-aware priors, and more targeted training examples.

## 14 High-Quality Sample Selection and Post-Training based on SpeechJudge-GRM

### 14.1 Details of Subjective Evaluation

During the construction of SpeechJudge-Data, we hired human labelers from a data crowdsourcing company. To verify the effectiveness of our training for them and to ensure the high quality of both the dataset and the resulting SpeechJudge-GRM, the human subjects for the final sample selection and TTS post-training experiments (Section 5.3 and 5.4) were all experienced speech generation researchers. All these researchers had extensive audio backgrounds, with a minimum of two years of experience in speech synthesis.

We randomly selected the subjective evaluation samples from both SeedTTS-Eval [seedtts] https://github.com/BytedanceSpeech/seed-tts-eval and Amphion-TTS-Eval [intp, vevo2] https://huggingface.co/datasets/amphion/Amphion-TTS-Eval. The evaluation set for each system in Figure 6 consists of 70 samples, while the set for each system in Figure 5 contains 100 samples. Each audio sample in these evaluations received at least three independent ratings. These subjective evaluation results show that the annotation quality of SpeechJudge-Data largely aligns with the judgments of professional researchers.

### 14.2 Objective Results

| Model | Regular | Articulatory | Code-switching | Cross-lingual | Expressive | Avg | ||||||
| WER | SIM | WER | SIM | WER | SIM | WER | SIM | WER | SIM | WER | SIM | |
| Qwen2.5-0.5B-TTS | 2.63 | 0.698 | 10.53 | 0.679 | 23.87 | 0.666 | 10.51 | 0.593 | 11.10 | 0.706 | 11.73 | 0.668 |
| w/ INTP | 2.06 | 0.697 | 8.62 | 0.694 | 18.37 | 0.663 | 7.12 | 0.588 | 9.80 | 0.708 | 9.19 | 0.670 |
| w/ SpeechJudge-Data | 2.12 | 0.698 | 8.92 | 0.678 | 19.01 | 0.657 | 7.72 | 0.583 | 9.97 | 0.707 | 9.55 | 0.664 |
| w/ SpeechJudge-GRM (offline) | 2.31 | 0.698 | 7.83 | 0.681 | 15.36 | 0.662 | 7.84 | 0.593 | 9.72 | 0.709 | 8.51 | 0.668 |
| w/ SpeechJudge-GRM (online) | 2.35 | 0.696 | 8.45 | 0.674 | 15.87 | 0.653 | 7.82 | 0.580 | 9.79 | 0.702 | 8.85 | 0.661 |

We present the objective results (WER and SIM) of the Qwen2.5-0.5B-TTS post-training in Table 14. The results show that all four post-training methods significantly improve the WER. This trend is similar to the subjective intelligibility results shown in Figure 6(a).

Regarding the SIM metric, both w/ INTP and w/ SpeechJudge-GRM (offline) either match or slightly outperform the baseline model, while the other two methods show a slight decline. However, the objective SIM results appear to be in slight conflict with the subjective speaker similarity results in Figure 6(b). For instance, in the subjective evaluation, w/ INTP actually shows a decrease in speaker similarity (Win: 24.30%, Lose: 32.90%).

Through follow-up interviews with the subjects who participated in our subjective evaluation, we gathered additional qualitative insights. Participants consistently reported that the synthesized samples, both before and after post-training, demonstrated excellent speaker similarity, closely matching the reference speaker’s timbre and style. In most cases, participants found it challenging to distinguish any significant differences in similarity, leading them to prefer selecting “Tie". For example, in Figure 6(b), all four methods have the highest “Tie" proportion, each exceeding 40%. This demonstrates that post-training methods centered on naturalness (SpeechJudge-based) or intelligibility (INTP-based) are not yet fully aligned with speaker similarity, which requires further research into speaker similarity alignment.
