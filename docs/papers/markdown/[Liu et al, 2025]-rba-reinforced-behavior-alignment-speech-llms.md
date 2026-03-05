# Enhancing Speech Large Language Models through Reinforced

Behavior Alignment

###### Abstract

The recent advancements of Large Language Models (LLMs) have spurred considerable research interest in extending their linguistic capabilities beyond text to other modalities, which leads to emergence of speech-based LLMs (SpeechLMs) with capability of processing user request in either speech or textual formats. However, owing to inter-modal discrepancies, these SpeechLMs still exhibit a significant performance gap compared to their text-based LLM counterparts in instruction-following, particularly when confronted with the dynamic and variable nature of user speech. To address this challenge, this paper introduces a framework termed Reinforced Behavior Alignment (RBA), designed to bolster the language generation proficiency of SpeechLMs. Instead of relying on supervised fine-tuning from human annotations, RBA employs a self-synthesis methodology to generate extensive, high-fidelity alignment data by a powerful teacher LLM. Then SpeechLMs is aligned its behavior with that of a teacher using a reinforcement learning-based approach. Experimental results demonstrate that this method effectively enhances the instruction-following capabilities of SpeechLMs that outperform conventional distillation baselines. Crucially, we demonstrate that RBA can be seamlessly extended to tasks such including spoken question answering and speech-to-text translation, attaining state-of-the-art performance on open benchmarks with only self-generated data.

## 1 Introduction

Large Language Models (LLMs), such as GPT (Achiam et al. 2023) and LlaMa (Touvron et al. 2023), have fundamentally reshaped the landscape of Natural Language Processing (NLP). These models, benefiting from training on vast quantities of data, showcase unparalleled proficiency in language understanding and generation, establishing themselves as general NLP task solvers. Concurrently, existing research efforts have focused on extending the capabilities of LLMs to encompass a broader range of modalities (Alayrac et al. 2022; Lin et al. 2023; Zhang et al. 2024), including human speech, which represent an intuitive and fundamental form of human-computer interaction (Fang et al. 2024).

In contrast to text, human speech is inherently more dynamic and complex, characterized by a rich tapestry of paralinguistic information (Lin et al. 2024). The same semantic content within user utterances can manifest as widely divergent auditory patterns, influenced by factors such as speaker identity, emotional state, and prosodic variations. This intrinsic variability significantly complicates the task of cross-modal perception for Speech Language Models (SpeechLMs) (Kim et al. 2024). Mainstream solution typically utilizes a pre-trained speech encoder as their primary perception module (Chu et al. 2023). Through subsequent Supervised Fine-Tuning (SFT), these models can be adapted to perform various downstream speech tasks through a followed LLM backbone, including Automatic Speech Recognition (ASR) and speech-to-text translation (S2TT). However, a key challenge persists: such a backbone LLM has never seen speech during its initial pre-training. Consequently, the overall system’s proficiency in instruction-following often remains considerably lower than that of advanced text-based LLMs (Wang et al. 2023b), suggesting a disparity in their effective ”intelligence” or general cognitive abilities when processing spoken instructions.

We clarify this topic with following metaphor: We conceptualize the LLM as an intelligent ”teacher”, despite lacking auditory perception, it can generate high-quality responses to text-based queries. Conversely, SpeechLM can handle variable speech input while the sub-intelligent content can not match the teacher’s generations. Then, two fundamental questions arise to bridge this intelligence gap. (1) What form of knowledge should the LLM teacher provide to ensure its accuracy, high quality, and scalability? (2) In what manner should the SpeechLM learn to more effectively align with the teacher’s behavior?

In this paper, we answer these two questions by proposing a novel learning paradigm called Reinforced Behavior Alignment (RBA). Firstly, inspired by textual self-synthesis method MAGPIE (Xu et al. 2024), we constructs a large-scale, high-quality instruction dataset for bi-modalities. This is achieved by prompting teacher LLMs with a pre-defined query template to sample diverse instructions. Concurrently, a teacher LLM, which has undergone post-training alignment, generates accurate reference responses corresponding to these instructions. Furthermore, a zero-shot Text-to-Speech (TTS) model (Du et al. 2024) is employed to synthesize multi-speaker spoken versions of these instructions from their textual form. Secondly, these multi-speaker instructions are fed into the SpeechLM for sampling, where the different paralinguistic information inherent in these spoken instructions significantly enhances the diversity of the generated responses. This increased variability is instrumental in mitigating potential data biases during the subsequent learning phase. Finally, these generated samples, in conjunction with the reference responses from the teacher LLM, are then utilized to compute a reward signal and update SpeechLM using a reinforcement learning-based optimization algorithm. Our experimental results indicate that the RBA pipeline substantially outperforms conventional ”TTS-SFT” learning strategies by showing superior distillation efficacy and model performance. Furthermore, our approach exhibits flexible adaptability to downstream tasks such as spoken question answering (SQA) and S2TT. Furthermore, RBA attains state-of-the-art (SOTA) results on open benchmarks for these applications, evaluated by accuracy and BLEU metrics, without requiring any annotated supervised data.

Our contributions can be summarized as follows:

-
•
With a self-synthesis methodology, we constructed a high-quality instruction dataset for SpeechLM learning, which comprises 1 million of audio and text instructions and corresponding text response generated by aligned teacher LLMs. Notably, each textual instruction has been synthesized into speech using 4 distinct speaker voices, thereby enriching the auditory diversity of the training corpus.

-
•
We introduce RBA that aims to bridge the intelligence gap between SpeechLMs and advanced LLMs. RBA leverages a self-synthesized dataset and employs a reinforcement learning-based optimization strategy to align the response generation behavior of SpeechLMs with that of proficient LLMs, achieving remarkable performance gain on instruction-following capability.

-
•
We demonstrate the effective extension of RBA to downstream tasks, specifically SQA and S2TT. Despite not utilizing any external annotation for these tasks, our approach attains SOTA performance on public benchmarks.

## 2 Related Work

### 2.1 Speech-based Large Language Models

The integration of speech processing capabilities into large language models has catalyzed significant advances in multimodal understanding (Shu et al. 2023; Chen et al. 2023). Pioneering works (Chu et al. 2023; Wu et al. 2023) demonstrated that augmenting LLMs with neural audio encoders enables direct perception of speech signals, creating unified frameworks for diverse audio-text tasks (Das et al. 2024). Another approach utilize audio codec model to discrete the speech (Yang et al. 2023; Zhang et al. 2023), providing audio tokens (Ji et al. 2024) for LLMs to perceive or generate. These architectures support applications ranging from speech recognition, translation, and spoken language understanding (Fathullah et al. 2023; Ye et al. 2025; Hu et al. 2025), traditionally addressed through cascaded ASR-LLM pipelines.

Architectural innovations have converged on multi-task training paradigms, as exemplified by AudioPaLM (Rubenstein et al. 2023), SALMONN (Tang et al. 2023) for speech processing, Qwen2-Audio (Chu et al. 2024) for general audio understanding, and WavLLM (Hu et al. 2024) for cross-modal alignment. These systems jointly optimize acoustic feature extraction and linguistic reasoning through unified attention mechanisms, achieving state-of-the-art performance on both modalities.

### 2.2 Reinforcement Learning Finetuning

Reinforcement learning (RL) has become pivotal in refining sequence generation tasks by directly optimizing task-specific rewards, including vision (Rennie et al. 2017), speech (Prabhavalkar et al. 2018), and NLP (Wu et al. 2018). In recent years, RL offers a solution to align pre-trained models with human feedback (Bai et al. 2022; Dai et al. 2023; Dong et al. 2024; Feng et al. 2025), as known as reinforcement learning from human feedback (RLHF). Recent advances such as DPO (Rafailov et al. 2024) advocate for closed-form loss functions that act directly on preference data, offering a simpler alternative to traditional preference-based reinforcement learning approaches (Jain et al. 2013; Busa-Fekete et al. 2014; Liu et al. 2025). Unlike conventional methods that require explicit reward model learning, DPO-style algorithms bypass this step while achieving comparable alignment performance to standard RLHF techniques (Zhou et al. 2023; Amini, Vieira, and Cotterell 2024; Zeng et al. 2024b; Liu, Sun, and Zheng 2024). In parallel, other efforts (Chen et al. 2024; Yuan et al. 2024; Zhang et al. 2025) explore “self-rewarding” strategies to calibrate LLMs without external supervision.

## 3 RBA Method

We present Reinforced Behavior Alignment (RBA), a systematic approach for enhancing speech-based language models through teacher-student paradigm optimization. The framework operates on the principle that proficient text-based LLMs can serve as instructional guides for SpeechLMs, despite their inability to process auditory inputs directly. Our methodology encompasses two primary phases: (1) large-scale synthetic data generation, and (2) reinforcement learning-based behavioral alignment.

### 3.1 Self-Synthesis Data Generation

The initial phase focuses on constructing a comprehensive instruction dataset without relying on human annotations. We leverage a high-capacity aligned LLM (specifically, Llama-3.1-70B-Instruct https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct) as our teacher model to generate both instruction and corresponding response as in Figure 1. For an aligned LLM, the input sequence can be represented as . Here, is the user query (e.g., ”Can you introduce the history of pyramid?”), while and are pre-query and post-query templates. As shown in Figure 1, the is set as <|start_header_id|>user<|end_header_id|>, and the is further annotated by <|eot_id|>. They are both designed to ensure the correct prompting.

Instruction Sampling. The objective of this step is to generate diverse instructions that leverage the extensive pre-training knowledge embedded within LLMs. To this end, we craft a pre-query template that conforms to the predefined instruction format of the teacher LLM. Since LLMs acquire knowledge through autoregressive learning from instruction data during their SFT phase, when presented with our crafted pre-query template, the LLM autonomously generates instructions without requiring any seed questions, thereby ensuring the diversity of generated instructions. Generation continues until the model produces an end-of-sequence token, ensuring complete instruction formation. By repeating this process multiple times, we obtain a large-scale instruction corpus that reflects the knowledge distribution captured during the teacher LLM’s training.

Response Completion. Subsequently, the teacher LLM generates corresponding responses to the instructions produced in Step 1. Given that the teacher LLM has undergone SFT and post-training alignment, the resulting instruction-response pairs typically exhibit high quality and behavioral consistency with human preferences. We implement a filtering strategy to ensure the suitability of the generated data for speech-based interactions. Specifically, we exclude instructions that are excessively lengthy or involve complex mathematical computations. This filtering criterion is motivated by practical considerations: users rarely pose such technical queries through speech modality due to the inherent difficulty in verbalizing complex mathematical expressions or lengthy technical specifications. By removing these categories, we ensure that our training data better reflects realistic speech-based interaction patterns, where users typically engage with more conversational and direct queries rather than highly technical or verbose instructions.

User Speech Generation. To simulate diverse user speech patterns, we employ the pre-trained CosyVoice TTS system (Du et al. 2024) for multi-speaker instruction generation. Our implementation follows three key steps: (1) Speaker Bank Construction. We leverage the LibriTTS dataset (Zen et al. 2019) to create a speaker repository containing 2,456 distinct voices. Each speaker provides a unique vocal fingerprint while maintaining professional-grade articulation. (2) Reference Selection: Utterances exceeding 3 seconds in duration are selected as reference prompts for zero-shot synthesis. This duration threshold ensures sufficient acoustic information for stable voice cloning (Wang et al. 2023a). (3) Each textual instruction generates 4 distinct spoken sampled from speaker bank, exponentially increasing the SpeechLM’s exposure to vocal variations. While LibriTTS speakers exhibit relatively neutral prosody, this characteristic ensures clear enunciation for instruction clarity. Totally, 1 million of samples are generated.

Data Analysis. We use LlaMa-3-8B-Instruct https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct to categorize the generated topic, as shown in Table 1. The predominant topic is ”information seeking” with a proportion of 60.3%, followed by ”advice seeking” and ”creative writing”. This distribution approximately aligns with practical request from human users (Li et al. 2023). To characterize our instruction dataset’s complexity distribution, we employ the same model as a difficulty rater. The model assesses each instruction on a 5-level scale (’very easy’ to ’very hard’) through chain-of-thought reasoning. Our filtered dataset exhibits the following distribution: [26.1%, 39.2%, 28.7%, 52%, 0.8%].

Discussion on Teacher Model Selection. While multi-teacher distillation frameworks have demonstrated potential in some scenarios, multiple teachers may introduce conflicting guidance signals (e.g., divergent reasoning patterns) that complicate behavior alignment, particularly detrimental for speech-based tasks requiring stable acoustic-textual mapping. Future work could explore hybrid approaches combining our reinforcement framework with curated multi-teacher knowledge.

| Topic Category | Percentage (%) |
|---|---|
| Information Seeking | 60.3 |
| Advice Seeking | 18.5 |
| Creative Writing | 12.7 |
| Planning | 4.2 |
| Reasoning | 2.8 |
| Brainstorming | 0.2 |
| Role Playing | 0.1 |
| Others | 1.2 |
| Total | 100.0 |

### 3.2 Reinforcement Learning for SpeechLMs

Given a speech instruction from speaker , the SpeechLM predicts response tokens . We minimize the cross-entropy loss between predictions and reference responses :

| (1) |

Consequently, the SpeeechLM is anticipated to generate responses that closely resemble those from data distribution . However, since the entire SFT process employs teacher-forcing – where SpeechLM predicts the current token based on the reference sequence – this creates a discrepancy between training and inference. During inference, the model must rely on its own generated sequence for the first time, leading to the exposure bias problem. RL-based fine-tuning effectively mitigates this issue by optimizing sequences generated through model self-sampling, transforming the supervision signal from SFT’s ”rote memorization” to encouraging the model to explore and produce higher-quality sequences.

Reward Modeling.
Given candidate responses generated by SpeechLM for spoken instruction, we employ a pre-trained reward model https://huggingface.co/RLHFlow/ArmoRM-Llama3-8B-v0.1 to evaluate SpeechLM’s responses, where a group of reward is provided in an online mode. For preference data construction, we propose two distinct positive-negative sampling strategies as follows.

RBA-Group: Within each generated sample group containing multi-speaker inputs, the positive example is selected as the with highest reward. Despite inherent speaker variations in audio inputs, we enforce speaker-invariant response generation through DPO-based optimization:

| (2) | ||||
| (3) | ||||
| (4) |

where is the reference model to stabilize the optimization. Empirically, we observe that explicit reference model parameterization can be omitted by directly utilizing reference responses for auxiliary supervision as shown in Eq.(1). Then a weight is employed to balance the and :

| (5) |

RBA-Single: For each generated sample , the corresponding from teacher LLMs typically exhibit better quality. Therefore, the reward evaluation process can be skipped by directly treating the reference response as the positive sample, while generated by SpeechLM are used as negative samples:

| (6) | ||||
| (7) | ||||
| (8) |

Although the outputs from SpeechLM are not necessarily always undesired, existing work (Chen et al. 2024) has theoretically verified the feasibility of this approach and demonstrated that it achieves comparable or even superior performance to SFT.

### 3.3 Extension on SQA and S2TT

Besides instruction-following, we take SQA and S2TT as examples to demonstrate how RBA can substantially enhance the task-specific performance of SpeechLMs.

We developed a rapid Q&A dataset called RBA-QA to enhance SpeechLM’s real-world knowledge. A total of 20,000 questions are generated by teacher LLMs and paired with answers. This dataset exhibits two key distinctions from conventional RBA dataset: (1) Single-Speaker Design: Each spoken question is synthesized using only one speaker profile. (2) Concise Answers: Responses are strictly constrained to 1-5 words. During optimization, we exclusively employ the RBA-Single strategy because the baseline SpeechLM often fails to produce valid answers for these questions, making positive sample selection impractical.

Given a source language in spoken form, the S2TT task aims to predict the text in the target language. In public datasets, the source language text is usually available, and the teacher LLM generates the reference text prediction based on this transcript. After sampling with SpeechLM, the BLEU score between its output and the reference text is used as the reward to select positive and negative samples for RBA-Group optimization. RBA-Single continues to follow the original setting. In general, this approach does not require supervised labels, and SpeechLM can effectively aligns its behavior with the teacher LLM when only source speech is available.

| Evaluation Topic | #Num. | RBA-G v.s. Base. | RBA-G v.s. Ref. | RBA-S v.s. Base. | RBA-S v.s. Ref. | ||||
|---|---|---|---|---|---|---|---|---|---|
| WR(%) | LC(%) | WR(%) | LC(%) | WR(%) | LC(%) | WR(%) | LC(%) | ||
| Information Seeking | 603k | 79.5 | 55.0 | 42.0 | 47.5 | 95.0 | 76.5 | 46.5 | 48.0 |
| Advice Seeking | 185k | 77.0 | 56.5 | 41.0 | 44.5 | 85.5 | 72.5 | 40.5 | 44.0 |
| Creative Writing | 127k | 74.5 | 60.0 | 46.5 | 45.5 | 89.0 | 70.0 | 48.5 | 46.0 |
| Planning | 42k | 72.5 | 61.0 | 45.0 | 45.5 | 88.0 | 80.0 | 47.5 | 44.0 |
| Reasoning | 28k | 81.0 | 55.5 | 40.5 | 41.0 | 95.5 | 70.5 | 43.0 | 44.5 |
| Brainstorming | 2k | 73.0 | 71.0 | 44.0 | 45.5 | 89.0 | 73.5 | 45.0 | 46.0 |
| Role Playing | 1k | 72.0 | 66.5 | 39.0 | 43.0 | 78.5 | 73.5 | 41.0 | 43.5 |
| Others | 12k | 73.5 | 65.5 | 41.0 | 42.0 | 79.0 | 70.0 | 42.0 | 43.5 |
| Spoken-Alpaca | N.a. | 65.5 | 60.5 | 43.0 | 39.5 | 74.0 | 63.0 | 44.5 | 46.0 |

| Model | Web-Q | Llama-Q | TriviaQA |
|---|---|---|---|
| GSLM (Lakhotia et al. 2021) | 1.5 | 4.0 | - |
| TWIST (Hassid et al. 2023) | 1.1 | 0.5 | |
| SpeechGPT (Zhang et al. 2023) | 6.5 | 21.6 | 14.8 |
| Spectron (Nachmani et al. 2023) | 6.1 | 21.9 | - |
| Moshi (Défossez et al. 2024) | 26.6 | 62.3 | 22.8 |
| GLM-voice (Zeng et al. 2024a) | 32.2 | 64.7 | 39.1 |
| Baseline Model | 10.1 | 33.4 | 19.7 |
| BRA-S | 40.7 | 77.0 | 55.1 |

| Response by | User Input: Propose an ethical solution to the problem of data privacy. |
|---|---|
| SpeechLM (before RBA) | Rely on user agreements and opt-in consent mechanisms. By clearly informing users and letting them choose whether or not to share data, the solution respects autonomy. However, it may be insufficient in practice due to information asymmetry and low user engagement. (GPT score: 6.0) |
| SpeechLM (after RBA) | Use data minimization and purpose limitation principles: collect only the data strictly necessary for a task and ensure it is not repurposed without user consent. This approach is simpler than differential privacy but still supports ethical handling of personal data. (GPT score: 8.0) |
| Teacher LLM | Implement differential privacy across all data collection and processing pipelines. This mathematical framework ensures that individual user data cannot be reverse-engineered from aggregate outputs, even by internal actors. Combined with strict access control and transparency reporting, this provides a high standard for ethical data privacy. (GPT score: 9.5) |

## 4 Experimental Setting

### 4.1 Dataset and Metric

For the RBA dataset, we select 200 samples from each topic category to construct a test set, totaling 1,600 samples. To validate the generalization capability of our method, we sample 200 out-domain questions from the Spoken-Alpaca dataset (GSQA/spoken-alpaca-gpt4) as an out-of-domain test set. Furthermore, Web-Questions (Berant et al. 2013), Llama-Questions (Nachmani et al. 2023), and TriviaQA (Joshi et al. 2017) are employed to demonstrate improvements in real-world knowledge comprehension. Our primary experiments utilize Qwen2-Audio (Chu et al. 2024) (8B) as the SpeechLM backbone, where its performance is viewed as baseline. To verify the efficacy of RBA, we adopt the following evaluation metrics from GPT-4o (2024-08-06):

-
•
Win-Rate (WR): the fraction of responses that are favored by the GPT evaluator.

- •
-
•
Accuracy (Acc): the correct rate of responses.

For S2TT, we select the source language from FLEURS (Conneau et al. 2023), CoVoST2 (Wang et al. 2021), and MuST-C (Di Gangi et al. 2019), resulting in 327K XEn and EnX pairs. Their training set is employed for training while the original test sets with labels are used for evaluation in terms of BLEU score.

### 4.2 Training Details

During the data generation phase, we configure the teacher LLM with temperature settings ranging from 1.0 to 1.2 and Top-P values between 0.99 and 1.00 to encourage the diversity. The computational infrastructure utilizes 128 NVIDIA A100 (80GB) GPUs for and 64 A40 (48GB) GPUs for data synthesis and model training.

The learning rate follows a 3,000-step warmup schedule to achieve max value of 1e-4, the weight decay is set as 0.98. For validation, we randomly select 128 data points from each domain to construct validation sets, which guide checkpoint selection and early stopping criteria. Hyperparameters are configured with = 0.2 and = 0.1 to balance alignment strength and distribution preservation. In practice, both hyper-parameters are not sensitive to experimental results.

## 5 Result

In this section, we design experiments to address the following questions and demonstrate the efficacy of RBA approach: (1) Does RBA enhance general instruction-following capabilities of SpeechLM across different topics and external evaluation set? (2) Does RBA improve SpeechLM’s factual knowledge and lead other models in SQA accuracy? (3) How does the RBA-enhanced model perform on S2TT capability with unlabeled speech, and can it compete with task-specific models? (4) Does our proposed reinforcement learning optimization strategy outperform SFT?

### 5.1 Result on Instruction-Following

| Test Set | RBA-G v.s. SFT | RBA-S v.s. SFT | ||
|---|---|---|---|---|
| WR(%) | LC(%) | WR(%) | LC(%) | |
| In-domain | 52.7 | 50.6 | 56.0 | 51.3 |
| Out-domain | 55.0 | 52.6 | 57.1 | 53.4 |

| Model (XEn) | Ar | Cy | De | El | Es | Fa | Fr | Hi | It | Ja | Pt | Ta | Uk | Vi | Zh | Avg. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Whisper-large V2 | 25.5 | 13.0 | 34.6 | 23.7 | 23.3 | 19.6 | 32.2 | 22.0 | 23.6 | 18.9 | 38.1 | 9.2 | 29.4 | 20.4 | 18.4 | 23.5 |
| SeamlessM4T-L | 32.8 | 31.7 | 35.8 | 25.6 | 25.0 | 28.2 | 33.1 | 26.3 | 25.0 | 17.0 | 38.9 | 16.0 | 30.2 | 21.6 | 19.8 | 27.1 |
| AudioPaLM2 | 29.0 | 7.2 | 38.7 | 18.8 | 26.9 | 25.7 | 36.5 | 21.7 | 27.8 | 11.1 | 38.4 | 15.0 | 26.9 | 15.6 | 21.3 | 24.0 |
| SeamlessM4T-L V2 | 34.7 | 34.9 | 37.1 | 27.3 | 25.4 | 30.3 | 33.7 | 28.5 | 26.5 | 19.5 | 38.5 | 22.1 | 33.2 | 25.7 | 23.0 | 29.4 |
| RBA-S | 35.1 | 36.8 | 37.3 | 25.2 | 40.1 | 36.1 | 34.4 | 31.6 | 37.1 | 19.9 | 39.1 | 26.2 | 31.3 | 27.9 | 30.0 | 32.5 |
| RBA-G | 36.3 | 36.1 | 37.6 | 26.9 | 40.3 | 36.9 | 34.4 | 32.7 | 39.7 | 17.7 | 38.2 | 28.0 | 33.6 | 28.8 | 30.8 | 33.2 |

To evaluate whether RBA improves SpeechLM’s instruction-following performance, we conduct comprehensive experiments across multiple domains and assess generalization on external datasets in Table 2. It is observed that both RBA variants substantially improve SpeechLM’s instruction-following capabilities, with RBA-Single consistently outperforming RBA-Group across all evaluated domains.

RBA v.s. Baseline. For in-domain performance, RBA-S achieves win rates of 78.7%-95.5% against the baseline, with particularly strong performance on Information Seeking (95.1%) and Reasoning (95.5%) tasks. RBA-G shows more modest improvements, ranging from 71.8%-80.8% win rates. On the out-of-domain Spoken-Alpaca dataset, RBA-S maintains strong performance (74.1% win rate vs. baseline), demonstrating robust cross-domain transfer. RBA-G shows reduced but still significant improvement (65.5% win rate), indicating that both methods generalize beyond training domains. An case study is visualized in Table 4 for comparison. After RBA, the generation quality is obviously improved, supported by GPT score.

RBA v.s. Reference. When compared against teacher LLM references, both variants achieve 40-48% win rates across different domains, representing substantial progress toward teacher-level performance despite processing various speech inputs rather than clean text.

RBA-G v.s. RBA-S. RBA-Single consistently outperforms RBA-Group by 6-15 percentage points across all domains, though the former eliminates the reward evaluation process. This performance gap suggests that: (1) Due to the teacher LLM’s significant performance advantage over SpeechLM, its generated responses inherently exhibit superior quality. This inherent quality gap allows reward modeling to be partially omitted in practice. (2) RBA-S demonstrates better sampling efficiency by fully utilizing every generated sample, whereas group-based approaches like RBA-G discard potentially useful intermediate-quality samples. (3) RBA-G faces inherent limitations where an entire speaker group might fail to explore high-quality samples (e.g., all four speaker versions generating suboptimal responses), thereby constraining its maximum achievable performance.

RBA v.s. SFT. We demonstrate the superiority of reinforcement learning-based optimization over SFT by comparing RBA-generated outputs with SFT results. As shown in the tables 5, both RBA-Group and RBA-Single exhibit advantages over baselines across in-domain and out-of-domain evaluations. This validates the effectiveness of using self-sampled training data for model alignment, which aligns with recent findings in instruction tuning literature (Chen et al. 2024).

| Model | En X | |||||
|---|---|---|---|---|---|---|
| De | Zn | Es | Fr | It | Avg. | |
| Base. | 25.9 | 45.2 | 21.4 | 37.1 | 22.0 | 30.3 |
| SFT | 26.0 | 45.0 | 21.4 | 37.4 | 22.7 | 30.5 |
| RBA-S | 27.7 | 46.0 | 26.2 | 35.9 | 27.0 | 32.6 |
| RBA-G | 30.3 | 47.3 | 23.8 | 38.1 | 25.1 | 33.0 |
| Ref. | 36.0 | 49.7 | 33.6 | 47.7 | 34.4 | 40.3 |

### 5.2 Result on SQA

The accuracy result of SQA task is reported in Table 3. Given that the RBA-QA dataset contains only single-speaker samples, we exclusively employed the RBA-Single optimization strategy. The RBA-QA data was combined with the original training set and fine-tuned for 2 additional epochs to prevent catastrophic forgetting while incorporating new knowledge. We observe that RBA successfully enhances SpeechLM’s factual knowledge comprehension and establishes new state-of-the-art results on standard SQA benchmarks, confirming its effectiveness for knowledge-intensive spoken language understanding tasks.

### 5.3 Result on S2TT

We first report the En X translation results in Table 7. SFT achieves only marginal gains over the baseline model (30.5 vs. 30.3 average BLEU), representing a mere 0.2-point improvement. This limited effectiveness stems from SFT’s fundamental approach of fitting the model to a new distribution without adequately addressing the modality gap between speech and text representations. Contrary to instruction-following tasks, RBA-G achieves superior performance (33.0 average BLEU) compared to RBA-S (32.6 average BLEU). This reversal occurs because: (1) Quality of SpeechLM Translation Samples: The SpeechLM inherently generates reasonable translation outputs, providing a solid foundation for group-based optimization. (2) Self-Generated Sample Exploration: RBA-G leverages purely self-generated training samples to explore superior translation strategies through contrastive learning across speaker variations. Despite significant improvements, both RBA variants maintain substantial gaps compared to references.

Then the XEn translation with other competitive models are report in Table 6. The 3.8 BLEU improvement over SeamlessM4T-Large V2 is particularly significant, given that SeamlessM4T is specifically designed and optimized for multilingual translation tasks. The results validate RBA’s contribution that effective behavioral alignment can enable general-purpose SpeechLMs to match or exceed specialized architectures without task-specific engineering. This finding is particularly impressive given that competing models (Whisper (Radford et al. 2023), SeamlessM4T (Barrault et al. 2023), AudioPaLM (Rubenstein et al. 2023)) are explicitly optimized for translation tasks, while RBA maintains its instruction-following capabilities.

### 5.4 Analysis of Cross-Speaker Output Consistency

A critical question arises from our methodology: since both the SFT baseline and the RBA-G model are trained on the same multi-speaker data, what is the precise source of RBA’s enhanced robustness? We posit that the key difference lies not in the data exposure itself, but in the fundamental nature of the optimization objective. This experiment is therefore designed to empirically validate that RBA-G’s intra-group contrastive learning, rather than SFT’s independent supervision, is the primary driver of speaker-invariant behavior.

To test this hypothesis, we conducted an output consistency analysis. The premise is that a truly robust model should produce semantically consistent responses to the same instruction, regardless of which speaker’s voice is used for the spoken input. We sampled 500 instructions from our in-domain test set and fed their four corresponding spoken versions (each from a different speaker) into both our final RBA-G model and the SFT baseline. This process generated a group of four responses per model for each instruction. We then quantified model stability by measuring the Average Pairwise Semantic Similarity within each group of four responses, using a pre-trained sentence-transformer (paraphrase-mpnet-base-v2) to compute cosine similarity.

| Model | Output Consistency Score (↑) |
|---|---|
| Baseline | 0.826 |
| SFT | 0.893 |
| RBA-G | 0.945 |

The results in Table 8 provide direct evidence supporting our hypothesis. The RBA-G model achieves a remarkably high consistency score of 0.945, confirming that its outputs are semantically stable across different speaker inputs. In contrast, the SFT model scores significantly lower at 0.893. This variability persists despite SFT being trained on all four speakers because the standard cross-entropy loss treats each pair as an independent sample; it lacks an explicit mechanism to enforce consistency across the samples within an instruction group. RBA-G’s superior consistency stems from its inherently relational objective. By rewarding the best response and penalizing the worst from within the group of self-generated samples, it creates a powerful optimization pressure for the model to converge on a single, robust policy that performs well for all speakers.

## 6 Conclusion

In this paper, we explore how to leverage a teacher LLM to enhance the reasoning and knowledge capabilities of SpeechLMs. To this end, we construct a knowledge transfer dataset via a self-synthesis approach, and propose a reinforcement learning-based alignment method to effectively bridge the modality gap. Experimental results demonstrate significant improvements in both instruction-following ability and factual knowledge acquisition. Moreover, we show that the proposed method can be seamlessly extended to S2TT tasks, achieving competitive performance on public benchmarks without relying on any labeled data.

## References

-
Achiam et al. (2023)
Achiam, J.; Adler, S.; Agarwal, S.; Ahmad, L.; Akkaya, I.; Aleman, F. L.; Almeida, D.; Altenschmidt, J.; Altman, S.; Anadkat, S.; et al. 2023.
Gpt-4 technical report.
*arXiv preprint arXiv:2303.08774*. -
Alayrac et al. (2022)
Alayrac, J.-B.; Donahue, J.; Luc, P.; Miech, A.; Barr, I.; Hasson, Y.; Lenc, K.; Mensch, A.; Millican, K.; Reynolds, M.; et al. 2022.
Flamingo: a visual language model for few-shot learning.
*Advances in neural information processing systems*, 35: 23716–23736. -
Amini, Vieira, and Cotterell (2024)
Amini, A.; Vieira, T.; and Cotterell, R. 2024.
Direct Preference Optimization with an Offset.
*arXiv preprint arXiv:2402.10571*. -
Bai et al. (2022)
Bai, Y.; Jones, A.; Ndousse, K.; Askell, A.; Chen, A.; DasSarma, N.; Drain, D.; Fort, S.; Ganguli, D.; Henighan, T.; et al. 2022.
Training a helpful and harmless assistant with reinforcement learning from human feedback.
*arXiv preprint arXiv:2204.05862*. -
Barrault et al. (2023)
Barrault, L.; Chung, Y.-A.; Meglioli, M. C.; Dale, D.; Dong, N.; Duquenne, P.-A.; Elsahar, H.; Gong, H.; Heffernan, K.; Hoffman, J.; et al. 2023.
SeamlessM4T: Massively Multilingual & Multimodal Machine Translation.
*arXiv preprint arXiv:2308.11596*. -
Berant et al. (2013)
Berant, J.; Chou, A.; Frostig, R.; and Liang, P. 2013.
Semantic parsing on freebase from question-answer pairs.
In
*Proceedings of the 2013 conference on empirical methods in natural language processing*, 1533–1544. -
Busa-Fekete et al. (2014)
Busa-Fekete, R.; Szörényi, B.; Weng, P.; Cheng, W.; and Hüllermeier, E. 2014.
Preference-based reinforcement learning: evolutionary direct policy search using a preference-based racing algorithm.
*Machine learning*, 97: 327–351. -
Chen et al. (2023)
Chen, C.; Hu, Y.; Yang, C.-H. H.; Siniscalchi, S. M.; Chen, P.-Y.; and Chng, E.-S. 2023.
Hyporadise: An open baseline for generative speech recognition with large language models.
*Advances in Neural Information Processing Systems*, 36: 31665–31688. -
Chen et al. (2024)
Chen, Z.; Deng, Y.; Yuan, H.; Ji, K.; and Gu, Q. 2024.
Self-play fine-tuning converts weak language models to strong language models.
*arXiv preprint arXiv:2401.01335*. -
Chu et al. (2024)
Chu, Y.; Xu, J.; Yang, Q.; Wei, H.; Wei, X.; Guo, Z.; Leng, Y.; Lv, Y.; He, J.; Lin, J.; et al. 2024.
Qwen2-audio technical report.
*arXiv preprint arXiv:2407.10759*. -
Chu et al. (2023)
Chu, Y.; Xu, J.; Zhou, X.; Yang, Q.; Zhang, S.; Yan, Z.; Zhou, C.; and Zhou, J. 2023.
Qwen-audio: Advancing universal audio understanding via unified large-scale audio-language models.
*arXiv preprint arXiv:2311.07919*. -
Conneau et al. (2023)
Conneau, A.; Ma, M.; Khanuja, S.; Zhang, Y.; Axelrod, V.; Dalmia, S.; Riesa, J.; Rivera, C.; and Bapna, A. 2023.
Fleurs: Few-shot learning evaluation of universal representations of speech.
In
*2022 IEEE Spoken Language Technology Workshop (SLT)*, 798–805. IEEE. -
Dai et al. (2023)
Dai, J.; Pan, X.; Sun, R.; Ji, J.; Xu, X.; Liu, M.; Wang, Y.; and Yang, Y. 2023.
Safe rlhf: Safe reinforcement learning from human feedback.
*arXiv preprint arXiv:2310.12773*. -
Das et al. (2024)
Das, N.; Dingliwal, S.; Ronanki, S.; Paturi, R.; Huang, Z.; Mathur, P.; Yuan, J.; Bekal, D.; Niu, X.; Jayanthi, S. M.; et al. 2024.
Speechverse: A large-scale generalizable audio language model.
*arXiv preprint arXiv:2405.08295*. -
Défossez et al. (2024)
Défossez, A.; Mazaré, L.; Orsini, M.; Royer, A.; Pérez, P.; Jégou, H.; Grave, E.; and Zeghidour, N. 2024.
Moshi: a speech-text foundation model for real-time dialogue.
*arXiv preprint arXiv:2410.00037*. -
Di Gangi et al. (2019)
Di Gangi, M. A.; Cattoni, R.; Bentivogli, L.; Negri, M.; and Turchi, M. 2019.
Must-c: a multilingual speech translation corpus.
In
*Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*, 2012–2017. Association for Computational Linguistics. -
Dong et al. (2024)
Dong, H.; Xiong, W.; Pang, B.; Wang, H.; Zhao, H.; Zhou, Y.; Jiang, N.; Sahoo, D.; Xiong, C.; and Zhang, T. 2024.
Rlhf workflow: From reward modeling to online rlhf.
*arXiv preprint arXiv:2405.07863*. -
Du et al. (2024)
Du, Z.; Chen, Q.; Zhang, S.; Hu, K.; Lu, H.; Yang, Y.; Hu, H.; Zheng, S.; Gu, Y.; Ma, Z.; et al. 2024.
Cosyvoice: A scalable multilingual zero-shot text-to-speech synthesizer based on supervised semantic tokens.
*arXiv preprint arXiv:2407.05407*. -
(19)
Dubois, Y.; Galambosi, B.; Liang, P.; and Hashimoto, T. B. ????
Length-controlled alpacaeval: A simple way to debias automatic evaluators, 2024.
*URL https://arxiv. org/abs/2404.04475*. -
Fang et al. (2024)
Fang, Q.; Guo, S.; Zhou, Y.; Ma, Z.; Zhang, S.; and Feng, Y. 2024.
Llama-omni: Seamless speech interaction with large language models.
*arXiv preprint arXiv:2409.06666*. -
Fathullah et al. (2023)
Fathullah, Y.; Wu, C.; Lakomkin, E.; Li, K.; Jia, J.; Shangguan, Y.; Mahadeokar, J.; Kalinli, O.; Fuegen, C.; and Seltzer, M. 2023.
Audiochatllama: Towards general-purpose speech abilities for llms.
*arXiv preprint arXiv:2311.06753*. -
Feng et al. (2025)
Feng, X.; Jiang, Z.; Kaufmann, T.; Xu, P.; Hüllermeier, E.; Weng, P.; and Zhu, Y. 2025.
DUO: Diverse, Uncertain, On-Policy Query Generation and Selection for Reinforcement Learning from Human Feedback.
In
*Proceedings of the AAAI Conference on Artificial Intelligence*, volume 39, 16604–16612. -
Hassid et al. (2023)
Hassid, M.; Remez, T.; Nguyen, T. A.; Gat, I.; Conneau, A.; Kreuk, F.; Copet, J.; Defossez, A.; Synnaeve, G.; Dupoux, E.; et al. 2023.
Textually pretrained speech language models.
*Advances in Neural Information Processing Systems*, 36: 63483–63501. -
Hu et al. (2025)
Hu, K.; Chen, Z.; Yang, C.-H. H.; Żelasko, P.; Hrinchuk, O.; Lavrukhin, V.; Balam, J.; and Ginsburg, B. 2025.
Chain-of-thought prompting for speech translation.
In
*ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 1–5. IEEE. -
Hu et al. (2024)
Hu, S.; Zhou, L.; Liu, S.; Chen, S.; Meng, L.; Hao, H.; Pan, J.; Liu, X.; Li, J.; Sivasankaran, S.; et al. 2024.
Wavllm: Towards robust and adaptive speech large language model.
*arXiv preprint arXiv:2404.00656*. -
Jain et al. (2013)
Jain, A.; Wojcik, B.; Joachims, T.; and Saxena, A. 2013.
Learning trajectory preferences for manipulators via iterative improvement.
*Advances in neural information processing systems*, 26. -
Ji et al. (2024)
Ji, S.; Jiang, Z.; Wang, W.; Chen, Y.; Fang, M.; Zuo, J.; Yang, Q.; Cheng, X.; Wang, Z.; Li, R.; et al. 2024.
Wavtokenizer: an efficient acoustic discrete codec tokenizer for audio language modeling.
*arXiv preprint arXiv:2408.16532*. -
Joshi et al. (2017)
Joshi, M.; Choi, E.; Weld, D. S.; and Zettlemoyer, L. 2017.
Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension.
*arXiv preprint arXiv:1705.03551*. -
Kim et al. (2024)
Kim, H.; Seo, S.; Jeong, K.; Kwon, O.; Kim, S.; Kim, J.; Lee, J.; Song, E.; Oh, M.; Ha, J.-W.; et al. 2024.
Paralinguistics-Aware Speech-Empowered Large Language Models for Natural Conversation.
*arXiv preprint arXiv:2402.05706*. -
Lakhotia et al. (2021)
Lakhotia, K.; Kharitonov, E.; Hsu, W.-N.; Adi, Y.; Polyak, A.; Bolte, B.; Nguyen, T.-A.; Copet, J.; Baevski, A.; Mohamed, A.; et al. 2021.
On generative spoken language modeling from raw audio.
*Transactions of the Association for Computational Linguistics*, 9: 1336–1354. - Li et al. (2023) Li, X.; Zhang, T.; Dubois, Y.; Taori, R.; Gulrajani, I.; Guestrin, C.; Liang, P.; and Hashimoto, T. B. 2023. Alpacaeval: An automatic evaluator of instruction-following models.
-
Lin et al. (2023)
Lin, B.; Ye, Y.; Zhu, B.; Cui, J.; Ning, M.; Jin, P.; and Yuan, L. 2023.
Video-llava: Learning united visual representation by alignment before projection.
*arXiv preprint arXiv:2311.10122*. -
Lin et al. (2024)
Lin, G.-T.; Shivakumar, P. G.; Gandhe, A.; Yang, C.-H. H.; Gu, Y.; Ghosh, S.; Stolcke, A.; Lee, H.-y.; and Bulyko, I. 2024.
Paralinguistics-enhanced large language modeling of spoken dialogue.
In
*ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 10316–10320. IEEE. -
Liu et al. (2025)
Liu, S.; Fang, W.; Hu, Z.; Zhang, J.; Zhou, Y.; Zhang, K.; Tu, R.; Lin, T.-E.; Huang, F.; Song, M.; et al. 2025.
A survey of direct preference optimization.
*arXiv preprint arXiv:2503.11701*. -
Liu, Sun, and Zheng (2024)
Liu, Z.; Sun, X.; and Zheng, Z. 2024.
Enhancing LLM Safety via Constrained Direct Preference Optimization.
*arXiv preprint arXiv:2403.02475*. -
Nachmani et al. (2023)
Nachmani, E.; Levkovitch, A.; Hirsch, R.; Salazar, J.; Asawaroengchai, C.; Mariooryad, S.; Rivlin, E.; Skerry-Ryan, R.; and Ramanovich, M. T. 2023.
Spoken question answering and speech continuation using spectrogram-powered llm.
*arXiv preprint arXiv:2305.15255*. -
Prabhavalkar et al. (2018)
Prabhavalkar, R.; Sainath, T. N.; Wu, Y.; Nguyen, P.; Chen, Z.; Chiu, C.-C.; and Kannan, A. 2018.
Minimum word error rate training for attention-based sequence-to-sequence models.
In
*2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 4839–4843. IEEE. -
Radford et al. (2023)
Radford, A.; Kim, J. W.; Xu, T.; Brockman, G.; McLeavey, C.; and Sutskever, I. 2023.
Robust speech recognition via large-scale weak supervision.
In
*International conference on machine learning*, 28492–28518. PMLR. -
Rafailov et al. (2024)
Rafailov, R.; Sharma, A.; Mitchell, E.; Manning, C. D.; Ermon, S.; and Finn, C. 2024.
Direct preference optimization: Your language model is secretly a reward model.
*Advances in Neural Information Processing Systems*, 36. -
Rennie et al. (2017)
Rennie, S. J.; Marcheret, E.; Mroueh, Y.; Ross, J.; and Goel, V. 2017.
Self-critical sequence training for image captioning.
In
*Proceedings of the IEEE conference on computer vision and pattern recognition*, 7008–7024. -
Rubenstein et al. (2023)
Rubenstein, P. K.; Asawaroengchai, C.; Nguyen, D. D.; Bapna, A.; Borsos, Z.; Quitry, F. d. C.; Chen, P.; Badawy, D. E.; Han, W.; Kharitonov, E.; et al. 2023.
Audiopalm: A large language model that can speak and listen.
*arXiv preprint arXiv:2306.12925*. -
Shu et al. (2023)
Shu, Y.; Dong, S.; Chen, G.; Huang, W.; Zhang, R.; Shi, D.; Xiang, Q.; and Shi, Y. 2023.
Llasm: Large language and speech model.
*arXiv preprint arXiv:2308.15930*. -
Tang et al. (2023)
Tang, C.; Yu, W.; Sun, G.; Chen, X.; Tan, T.; Li, W.; Lu, L.; Ma, Z.; and Zhang, C. 2023.
Salmonn: Towards generic hearing abilities for large language models.
*arXiv preprint arXiv:2310.13289*. -
Touvron et al. (2023)
Touvron, H.; Lavril, T.; Izacard, G.; Martinet, X.; Lachaux, M.-A.; Lacroix, T.; Rozière, B.; Goyal, N.; Hambro, E.; Azhar, F.; et al. 2023.
Llama: Open and efficient foundation language models.
*arXiv preprint arXiv:2302.13971*. -
Wang et al. (2023a)
Wang, C.; Chen, S.; Wu, Y.; Zhang, Z.; Zhou, L.; Liu, S.; Chen, Z.; Liu, Y.; Wang, H.; Li, J.; et al. 2023a.
Neural codec language models are zero-shot text to speech synthesizers.
*arXiv preprint arXiv:2301.02111*. -
Wang et al. (2023b)
Wang, C.; Liao, M.; Huang, Z.; Lu, J.; Wu, J.; Liu, Y.; Zong, C.; and Zhang, J. 2023b.
Blsp: Bootstrapping language-speech pre-training via behavior alignment of continuation writing.
*arXiv preprint arXiv:2309.00916*. -
Wang et al. (2021)
Wang, C.; Wu, A.; Gu, J.; and Pino, J. 2021.
CoVoST 2 and massively multilingual speech translation.
In
*Interspeech*, volume 2021, 2247–2251. -
Wu et al. (2023)
Wu, J.; Gaur, Y.; Chen, Z.; Zhou, L.; Zhu, Y.; Wang, T.; Li, J.; Liu, S.; Ren, B.; Liu, L.; et al. 2023.
On decoder-only architecture for speech-to-text and large language model integration.
In
*2023 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU)*, 1–8. IEEE. -
Wu et al. (2018)
Wu, L.; Tian, F.; Qin, T.; Lai, J.; and Liu, T.-Y. 2018.
A study of reinforcement learning for neural machine translation.
*arXiv preprint arXiv:1808.08866*. -
Xu et al. (2024)
Xu, Z.; Jiang, F.; Niu, L.; Deng, Y.; Poovendran, R.; Choi, Y.; and Lin, B. Y. 2024.
Magpie: Alignment data synthesis from scratch by prompting aligned llms with nothing.
*arXiv preprint arXiv:2406.08464*. -
Yang et al. (2023)
Yang, D.; Tian, J.; Tan, X.; Huang, R.; Liu, S.; Chang, X.; Shi, J.; Zhao, S.; Bian, J.; Wu, X.; et al. 2023.
Uniaudio: An audio foundation model toward universal audio generation.
*arXiv preprint arXiv:2310.00704*. -
Ye et al. (2025)
Ye, Z.; Zhu, X.; Chan, C.-M.; Wang, X.; Tan, X.; Lei, J.; Peng, Y.; Liu, H.; Jin, Y.; DAI, Z.; et al. 2025.
Llasa: Scaling Train-Time and Inference-Time Compute for Llama-based Speech Synthesis.
*arXiv preprint arXiv:2502.04128*. -
Yuan et al. (2024)
Yuan, W.; Pang, R. Y.; Cho, K.; Sukhbaatar, S.; Xu, J.; and Weston, J. 2024.
Self-rewarding language models.
*arXiv preprint arXiv:2401.10020*. -
Zen et al. (2019)
Zen, H.; Dang, V.; Clark, R.; Zhang, Y.; Weiss, R. J.; Jia, Y.; Chen, Z.; and Wu, Y. 2019.
Libritts: A corpus derived from librispeech for text-to-speech.
*arXiv preprint arXiv:1904.02882*. -
Zeng et al. (2024a)
Zeng, A.; Du, Z.; Liu, M.; Zhang, L.; Jiang, S.; Dong, Y.; and Tang, J. 2024a.
Scaling speech-text pre-training with synthetic interleaved data.
*arXiv preprint arXiv:2411.17607*. -
Zeng et al. (2024b)
Zeng, Y.; Liu, G.; Ma, W.; Yang, N.; Zhang, H.; and Wang, J. 2024b.
Token-level Direct Preference Optimization.
*arXiv preprint arXiv:2404.11999*. -
Zhang et al. (2023)
Zhang, D.; Li, S.; Zhang, X.; Zhan, J.; Wang, P.; Zhou, Y.; and Qiu, X. 2023.
Speechgpt: Empowering large language models with intrinsic cross-modal conversational abilities.
*arXiv preprint arXiv:2305.11000*. -
Zhang et al. (2024)
Zhang, J.; Huang, J.; Jin, S.; and Lu, S. 2024.
Vision-language models for vision tasks: A survey.
*IEEE Transactions on Pattern Analysis and Machine Intelligence*. -
Zhang et al. (2025)
Zhang, S.; Liu, X.; Zhang, X.; Liu, J.; Luo, Z.; Huang, S.; and Gong, Y. 2025.
Process-based self-rewarding language models.
*arXiv preprint arXiv:2503.03746*. -
Zhou et al. (2023)
Zhou, Z.; Liu, J.; Yang, C.; Shao, J.; Liu, Y.; Yue, X.; Ouyang, W.; and Qiao, Y. 2023.
Beyond one-preference-for-all: Multi-objective direct preference optimization.
*arXiv preprint arXiv:2310.03708*.
