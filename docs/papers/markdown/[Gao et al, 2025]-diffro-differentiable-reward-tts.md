# Differentiable Reward Optimization for LLM based TTS system

###### Abstract

This paper proposes a novel Differentiable Reward Optimization (DiffRO) method aimed at enhancing the performance of neural codec language models based text-to-speech (TTS) systems. In contrast to conventional reinforcement learning from human feedback (RLHF) approaches applied to TTS, DiffRO directly compute the rewards based on neural codec tokens, rather than relying on synthesized audio. Furthermore, we employ the Gumbel-Softmax technique to render the reward function differentiable, thereby streamlining the RLHF training process. Additionally, we introduce a multi-task reward (MTR) model which can provide feedback from different perspectives and find that it can augment the system’s capability to follow instructions effectively. Experimental results indicate that DiffRO significantly improves the pronunciation accuracy of the TTS system, achieving state-of-the-art (SOTA) WER results on the seed-tts-eval benchmark. Moreover, with the integration of the MTR model, we demonstrate the ability to control emotional and quality attributes in a zero-shot manner.

Index Terms: Text-to-speech (TTS) synthesis, large language models, RLHF

## 1 Introduction

Inspired by the success of text-based large language models (LLMs), neural codec token language modeling (LM) has emerged as a leading approach for text-to-speech (TTS) generation. This methodology employs a pre-trained speech tokenizer to encode speech into a discrete sequence of tokens, which are subsequently modeled using a decoder-only LM that predicts these tokens based on textual input. Ultimately, a flow matching (FM)[1] model and a vocoder are applied to transform the tokens into audible speech. Leveraging the capabilities of zero-shot and in-context learning inherent in LLMs, neural codec-based TTS systems [2, 3] have demonstrated enhanced quality and naturalness, enabling applications such as zero-shot TTS and instruction-based TTS.

In the training of LLMs, Reinforcement Learning from Human Feedback (RLHF) [4] represents a critical step in aligning LLMs with human preferences, contributing to the success of numerous state-of-the-art models. Recent TTS systems have also endeavored to incorporate RLHF to enhance the quality of generated speech; however, a widely applicable methodology remains elusive. The challenges can be summarized as follows:

Firstly, unlike traditional natural language processing (NLP) tasks, TTS systems require additional backend FM and vocoder models to convert discrete neural codec tokens into waveform audio. The computational demands posed by these backend models are substantial, which hinders the large-scale production of RLHF data.

Secondly, the diversity of generated TTS speech is often insufficient. Although the neural codec LLM can generate different token sequences from the same text and speech prompt by employing various sampling strategies and random seeds, the resultant voices consistently exhibit high similarity. Consequently, it becomes challenging to differentiate between positive and negative feedback for training the reward model.

Finally, the evaluation methods for TTS systems are inherently complex. Typically, the mean opinion score (MOS) serves as the gold standard for evaluating TTS quality. However, more critical aspects such as accuracy, naturalness, and speaker similarity are ultimately determined by listener perception. This issue is exacerbated in instruction-based TTS scenarios, where certain instructions may conflict with the need for precise pronunciation accuracy.

Several existing studies [5, 6, 7, 8, 9] have attempted to address one or more of the aforementioned challenges through various approaches. For instance, Chen et al. [6] introduced a uncertainty-aware optimization (UNO) with unpaired negative and positive samples which can mitigate the impact of insufficient diversity within TTS systems. They further proposed a Reverse Input Output (RIO) strategy to generate additional positive samples by reversing the prompts and their corresponding responses [7]. Anastassiou et al. [8] utilized automatic speech recognition (ASR) and speaker recognition systems to measure word error rate (WER) and speaker similarity as reinforcement learning (RL) rewards. Moreover, Gao et al. [9] demonstrated that employing the accuracy of the speech emotion recognition (SER) model as a reward can enhance emotional controllability. Collectively, these studies illustrate that leveraging objective indicators can yield significant improvements in TTS systems across various dimensions.

In this work, we propose a novel Differentiable Reward Optimization (DiffRO) method specifically designed for neural codec-based TTS systems. Unlike existing approaches, DiffRO predicts rewards directly from the neural codec tokens rather than from the synthesized speech, thereby reducing the computational burden associated with backend frame-matching and vocoder models. Furthermore, we utilize the Gumbel-Softmax technique to ensure that the loss function of the reward model is differentiable with respect to the input text of the TTS system. Finally, we construct a multiple downstream tasks reward (MTR) model, including ASR, speech emotion recognition (SER), speech quality assessment (SQA), and age and gender prediction. Our findings indicate that the MTR model can effectively regulate speech attributes in accordance with preset preferences or input instructions.

## 2 Related work

### 2.1 Neural Codec LLM based TTS systems

Most neural codec LLM based TTS systems comprise four primary building blocks: a pre-trained speech tokenizer, a neural codec token LM, a FM model, and an acoustic vocoder. Given a paired input of speech and text , the speech tokenizer encodes the speech into a token sequence . Subsequently, the LM is trained using a next-token prediction task:

| (1) |

where is the code-book size of the tokenizer, stand for the LM model.

In general, the token sequence carries most information of the text and plays a crucial role in the TTS system. The TTS system will recover the audios from with the FM and the vocoder.

| (2) |

where and are the parameters of the FM model and the vocoder.

### 2.2 RLHF for TTS

To obtain human or AI feedback, one input should be inputted into the LM more than once, then different sample strategy and random seed are used to obtain different audios . The positive and negative and are chosen from the and their neural codec tokens and are used to train a reward model , which can provide feedback to the LM during the RL phase.

| (3) |

| (4) |

Proximal policy optimization (PPO) [10] are used to optimize the eq 4. Direct Preference Optimization (DPO) [11] can merge the reward modeling and RL into one phase and optimize the eq 3 and 4 with and directly.

| (5) |

where is the optimized LM model and denotes a frozen reference model.

Although DPO can simplify RLHF into a single phase, constructing a preferred and dispreferred training set remains a complex task for TTS systems. On one hand, to derive meaningful feedback, the predicted token sequence must be decoded into waveform audio via the FM model and the vocoder. On the other hand, synthesized audio samples often exhibit high similarity, making it challenging to accurately distinguish between positive and negative examples based solely on two samples. In addition to this complexity, another limitation of DPO is that categorizing audio into good and bad may not be adequate for TTS. The quality of audio output should be assessed from multiple perspectives, including pronunciation, rhythm, speaker similarity, emotional expression, and other relevant attributes.

## 3 Differentiable Reward Optimization

This section will introduce the DiffRO, which can further simplify the RL training process and can provide feedback on different aspects. Figure 1 shows the difference between the DiffRO and the existing RL method like DPO.

### 3.1 Token2Reward Prediction

Unlike previous work, DiffRO directly predicts the reward from the speech tokens rather than the raw audio. As a TTS system should read the text correctly, the predicted codec token should catch all information from the text. So we can predict the input text from the code with a neural network in ASR way [12]:

| (6) |

| (7) |

The post-probability can be regarded as the reward model, as it can encourage the catch more information from the text. We further use the Gumbel-Softmax operation to replace the argmax operation in eq 6 to sample the predict token . Then the reward function is differentiable and the LM model can be directly optimized to maximize the reward score without PPO or DPO strategy.

| (8) |

### 3.2 Multi-Task Reward Model

In addition to utilizing ASR rewards, we can incorporate additional downstream tasks to ensure that the predicted token sequence encompasses all the requisite information. Initially, we train a codec-based speech understanding model using a multi-task training approach, enabling it to perform tasks such as SER, SQA, AED and other audio understanding functionalities. Subsequently, we employ this model as a MTR model to guide the TTS system in generating audio that adheres to specific instructions or exhibits particular characteristics. The structure of the MTR model is illustrated in Figure 2.

If the MTR model is accessible, we can control the synthesized audio’s attribute or make the TTS system follow some instruction by maxmize the post-probably predicted by the MTR model.

| (9) |

Where is the text, is the predict token and is the target attribute and is the task id.

## 4 Experiments

### 4.1 Experimental Setup.

#### 4.1.1 DataSet and Baseline TTS system

We use the Cosyvoice2.0-0.5B [13] model as the baseline system and fine-tune it with our in-home target speaker data as the SFT baseline. The SFT data contains 4000 audio samples from 5 speakers (4 female and 1 male) and all of them speak in Mandarin. For the RL data, we collect ten thousand texts from the Internet, 90% of which are Chinese and English, and others are Japanese and Korean. Then we use the Cosyvoice2.0 to synthesize corresponding audios.

#### 4.1.2 Training of the Reward Model

For the reward model, we replace the front-end CNN module of the SenseVoice with an embedding layer to accept the speech token input and add an attention pooling layer for each downstream task.
And for the multi-task training data, we generate pseudo labels by the existing models for the inhome industry ASR training corpus. The final data obtained contains more than 13000 hours audios, and each audio has their transcription, emotion https://modelscope.cn/models/iic/emotion2vec_plus_large/,
MOS[14],
speaker age&gender https://huggingface.co/audeering/wav2vec2-large-robust-24-ft-age-gender, and event labels[15].
The emotion and gender are predicted by cross entropy (CE) loss, the age are predicted by mean squared error (MSE) loss and the events are predicted by binary CE loss. While for MOS prediction, we divide the audio quality into 5 levels and also predict it by CE loss.

After training, we evaluate this token2reward model on the downstream tasks [16, 17, 18, 19] and show the result in Table 1. We can find that for the ASR task, the WER is a little worse than the SenseVoice[20]; this is reasonable because compared with the raw audio, the information in the codec token is incomplete. However, when it comes to the MTR, our reward model has better SQA, Age&Gender prediction capability than the reference[21, 22, 23]. This proves that the codec tokens contain sufficient emotion, speaker and noise information, which means that the LM based TTS system has the potential to synthesize rich expression audio. This also proves that we can also train a multi-task speech understand model under the traditional classification framework rather than the LLM based generative framework [23, 24].

| Task&Data | Ref-Source | Value | |
|---|---|---|---|
| refer | result | ||
| ASR-CMV(en)-WER(%) | SenseVoice | 8.67 | 11.3 |
| ASR-CMV(zh)-WER(%) | SenseVoice | 7.59 | 7.70 |
| SQA-BVCC-LCC | DNSMOS | 0.61 | 0.65 |
| SER-IEMOCAP-WA(%) | EmoBox | 72.8 | 66.0 |
| Age-AirBench-Acc(%) | Qwen-Audio | 58.8 | 65.8 |
| Gender-AirBench-Acc(%) | Qwen-Audio | 82.5 | 92.5 |

#### 4.1.3 Reinforcement Learning Setup

For the RL, we compare the DPO and the proposed DiffRO. For the DPO, we synthesize the text five times and select the and according to the WER and speaker similarity. For DiffRO, we utilize the ASR reward to enhance pronunciation, while employing the MTR to control the audio properties. is set to 0.1 for all experiments and the learning rate is fixed to . 4 A800 GPU are used for the training.

### 4.2 Results for ASR-based DiffRO

We evaluate the TTS system on the seed-tts-eval benchmark by the WER of the ASR models. For the Chinese zh and hard subset, we use the Paramformer-zh [25] and for the en, we use the whisper-large-v3[26]. We further also evaluate the Japanese and Korean in CV3-Eval 3 https://github.com/FunAudioLLM/CV3-Eval by whisper. Results are shown in Table 2.

| Model | Seed-TTS | CMV | |||
|---|---|---|---|---|---|
| zh | en | hard | ja | ko | |
| CosyVoice2.0 | 1.56 | 2.75 | 6.91 | 9.13 | 7.43 |
| + DiffRO-ASR | 0.78 | 1.89 | 5.58 | 6.36 | 5.41 |
| CosyVoice2.0-SFT | 1.50 | 4.26 | 7.90 | 20.2 | 11.1 |
| + DPO | 1.27 | 3.28 | 6.74 | 10.4 | 9.12 |
| + DiffRO-ASR | 1.09 | 2.57 | 5.83 | 8.38 | 6.35 |
| + DiffRO-MTR | 1.05 | 3.43 | 6.41 | 14.8 | 10.7 |

According to the table, we can find that after speaker fine-tuning (SFT), the TTS system makes some improvement on the zh subset but becomes a little worse on the en and hard. This is reasonable as the SFT data are mostly composed of Mandarin and in lack of hard case. And the decrease in ja and ko could be more significant due to catastrophic forgetting. While RL can improve the TTS performance on all sets, and the DiffRO-ASR is the best one. It can even beat the base model on ja and ko, although only a small amount of Japanese and Korean text can be utilized during RL. Because the TTS can learn to generate multilingual codec-tokens by the ASR model. The DiffRO can also be applied to the base model and achieves state-of-the-art result for zh and hard. However, in the ASR task, the MTR-based DiffRO demonstrates inferior performance compared to the ASR-only DiffRO model. This discrepancy arises because the MTR approach encourages the predicted tokens to carry additional information that is not utilized by the ASR system. Nevertheless, this extra information can facilitate audio style control, a topic we will discuss further in later subsections.

### 4.3 Results for MTR-based DiffRO

#### 4.3.1 Emotion Control

We can utilize the SER reward to the emotion of the output audio, and the data are structured according to the following instruction template:

Then is used for the SER reward. Additionally, the ASR reward is retained. After training, we compare the performance of the DiffRO-MTR model against the CosyVoice 2.0, GPT-SoVits, and F5-TTS, to evaluate their emotional expression capabilities. The test text and prompt audio also comes from CV3-Eval. Each emotion category consists of 100 samples, with Chinese and English evenly represented. We employ the emo2vec-plus-large model as the classifier and present the accuracy results in Table 3.

| Model | HAPPY | SAD | ANGRY | |||
|---|---|---|---|---|---|---|
| zh | en | zh | en | zh | en | |
| F5-TTS[27] | 0.92 | 0.64 | 0.68 | 0.92 | 0.40 | 0.76 |
| GPT-SoVITS | 0.87 | 0.39 | 0.52 | 0.88 | 0.68 | 0.48 |
| CosyVoice2.0 | 0.92 | 0.80 | 0.52 | 0.84 | 0.76 | 0.80 |
| + DiffRO-MTR | 1.00 | 0.92 | 0.76 | 0.96 | 0.84 | 0.92 |

The results demonstrate that the proposed DiffRO significantly enhances the emotional expressiveness of the TTS system. Notably, there are no emotion-labeled data available during the RL phase; in other words, emotional knowledge is acquired from the MTR model rather than from explicit data. Another intriguing finding is that the TTS system can learn to synthesize laughter, sobs, breaths, and other audio events to convey emotion. Examples of these findings are illustrated in Figure 3.

#### 4.3.2 Other Attribute Control

We also try to control other voice attributes such as MOS, age and gender. Experiments prove that DiffRO can influence the pronunciation, but, however, cannot play a decisive role. We show the results of the MOS control in Table 4, which uses as a reward. We can find the DiffRO do influence the audio quality, however, the MOS predicted from codec tokens can be close to the target, while the final generated audio will be higher (expected the ). Because the DiffRO is only applied on the LM, which can learn to generate noisy tokens in an effort to minimize the MOS. However, the ultimate quality of the audio output is more heavily influenced by the FM and the vocoder, both of which are trained on clean audio and possess denoising capabilities. This issue becomes particularly pronounced when controlling speaker attributes such as age and gender, especially since the speaker information is integrated during the FM phase in the CosyVoice 2.0 framework.

| Model | MOS | ||
|---|---|---|---|
| Codec | Audio | Recodec | |
| Baseline | 3.82 | 3.84 | 3.88 |
| 2.20 | 3.77 | 3.47 | |
| 3.14 | 3.81 | 3.66 | |
| 3.96 | 3.86 | 3.92 |

## 5 Conclusions

In this paper, we propose a novel DiffRO based reinforcement learning strategy for neural codec LM based text-to-speech systems. Compared to other reinforcement learning methods, DiffRO is capable of directly predicting reward scores from speech tokens rather than from synthesized audio. Moreover, it allows for the direct optimization of LM parameters through backpropagation. Additionally, we have developed a MTR model that not only enhances pronunciation accuracy but also improves emotional expression and other acoustic attributes through various downstream task rewards. For future work, we plan to incorporate additional downstream tasks into the MTR model and explore the application of DiffRO to the FM module, which plays a crucial role in speaker-related downstream tasks.

## References

-
[1]
J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” in
*Advances in Neural Information Processing Systems*, H. Larochelle, M. Ranzato, R. Hadsell, M. Balcan, and H. Lin, Eds., vol. 33. Curran Associates, Inc., 2020, pp. 6840–6851. -
[2]
S. Chen, C. Wang, Y. Wu, Z. Zhang, L. Zhou, S. Liu, Z. Chen, Y. Liu, H. Wang, J. Li, L. He, S. Zhao, and F. Wei, “Neural codec language models are zero-shot text to speech synthesizers,”
*IEEE Transactions on Audio, Speech and Language Processing*, vol. 33, pp. 705–718, 2025. - [3] Z. Zhang, L. Zhou, C. Wang, S. Chen, Y. Wu, S. Liu, Z. Chen, Y. Liu, H. Wang, J. Li, L. He, S. Zhao, and F. Wei, “Speak foreign languages with your own voice: Cross-lingual neural codec language modeling,” 2023. [Online]. Available: https://arxiv.org/abs/2303.03926
- [4] D. M. Ziegler, N. Stiennon, J. Wu, T. B. Brown, A. Radford, D. Amodei, P. Christiano, and G. Irving, “Fine-tuning language models from human preferences,” 2020. [Online]. Available: https://arxiv.org/abs/1909.08593
-
[5]
H. Liao, H. Han, K. Yang, T. Du, R. Yang, Z. Xu, Q. Xu, J. Liu, J. Lu, and X. Li, “Baton: Aligning text-to-audio model with human preference feedback,”
*ArXiv*, vol. abs/2402.00744, 2024. [Online]. Available: https://api.semanticscholar.org/CorpusID:267365170 - [6] C. Chen, Y. Hu, W. Wu, H. Wang, E. S. Chng, and C. Zhang, “Enhancing Zero-shot Text-to-Speech Synthesis with Human Feedback,” Jun. 2024, arXiv:2406.00654. [Online]. Available: http://arxiv.org/abs/2406.00654
- [7] Y. Hu, C. Chen, S. Wang, E. S. Chng, and C. Zhang, “Robust zero-shot text-to-speech synthesis with reverse inference optimization,” 2024. [Online]. Available: https://arxiv.org/abs/2407.02243
- [8] P. Anastassiou, J. Chen, J. Chen, Y. Chen, Z. Chen, Z. Chen, J. Cong, L. Deng, C. Ding, L. Gao, M. Gong, P. Huang, Q. Huang, Z. Huang, Y. Huo, D. Jia, C. Li, F. Li, H. Li, J. Li, X. Li, X. Li, L. Liu, S. Liu, S. Liu, X. Liu, Y. Liu, Z. Liu, L. Lu, J. Pan, X. Wang, Y. Wang, Y. Wang, Z. Wei, J. Wu, C. Yao, Y. Yang, Y. Yi, J. Zhang, Q. Zhang, S. Zhang, W. Zhang, Y. Zhang, Z. Zhao, D. Zhong, and X. Zhuang, “Seed-TTS: A Family of High-Quality Versatile Speech Generation Models,” Jun. 2024, arXiv:2406.02430. [Online]. Available: http://arxiv.org/abs/2406.02430
- [9] X. Gao, C. Zhang, Y. Chen, H. Zhang, and N. F. Chen, “Emo-DPO: Controllable Emotional Speech Synthesis through Direct Preference Optimization,” Sep. 2024, arXiv:2409.10157 [eess]. [Online]. Available: http://arxiv.org/abs/2409.10157
- [10] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal Policy Optimization Algorithms,” Aug. 2017, arXiv:1707.06347 [cs]. [Online]. Available: http://arxiv.org/abs/1707.06347
- [11] R. Rafailov, A. Sharma, E. Mitchell, S. Ermon, C. D. Manning, and C. Finn, “Direct Preference Optimization: Your Language Model is Secretly a Reward Model,” May 2023. [Online]. Available: https://arxiv.org/abs/2305.18290v3
-
[12]
L. Ye, C. Gao, G. Cheng, L. Luo, and Q. Zhao, “Asq: An ultra-low bit rate asr-oriented speech quantization method,”
*IEEE Signal Processing Letters*, vol. 31, pp. 221–225, 2024. - [13] Z. Du, Y. Wang, Q. Chen, X. Shi, X. Lv, T. Zhao, Z. Gao, Y. Yang, C. Gao, H. Wang, F. Yu, H. Liu, Z. Sheng, Y. Gu, C. Deng, W. Wang, S. Zhang, Z. Yan, and J. Zhou, “Cosyvoice 2: Scalable streaming speech synthesis with large language models,” 2024. [Online]. Available: https://arxiv.org/abs/2412.10117
-
[14]
A. Kumar, K. Tan, Z. Ni, P. Manocha, X. Zhang, E. Henderson, and B. Xu, “Torchaudio-squim: Reference-less speech quality and intelligibility measures in torchaudio,”
*ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, pp. 1–5, 2023. [Online]. Available: https://api.semanticscholar.org/CorpusID:257921409 -
[15]
Q. Kong, Y. Cao, T. Iqbal, Y. Wang, W. Wang, and M. D. Plumbley, “Panns: Large-scale pretrained audio neural networks for audio pattern recognition,”
*IEEE/ACM Transactions on Audio, Speech, and Language Processing*, vol. 28, pp. 2880–2894, 2019. [Online]. Available: https://api.semanticscholar.org/CorpusID:209444382 -
[16]
R. Ardila, M. Branson, K. Davis, M. Henretty, M. Kohler, J. Meyer, R. Morais, L. Saunders, F. M. Tyers, and G. Weber, “Common voice: A massively-multilingual speech corpus,” in
*Proceedings of the 12th Conference on Language Resources and Evaluation (LREC 2020)*, 2020, pp. 4211–4215. -
[17]
W. C. Huang, E. Cooper, Y. Tsao, H.-M. Wang, T. Toda, and J. Yamagishi, “The voicemos challenge 2022,” in
*Interspeech 2022*, 2022, pp. 4536–4540. -
[18]
C. Busso, M. Bulut, C.-C. Lee, A. Kazemzadeh, E. Mower, S. Kim, J. N. Chang, S. Lee, and S. S. Narayanan, “Iemocap: Interactive emotional dyadic motion capture database,”
*Language resources and evaluation*, vol. 42, pp. 335–359, 2008. - [19] J. Chen, N. Wang, C. Li, B. Wang, S. Xiao, H. Xiao, H. Liao, D. Lian, and Z. Liu, “Air-bench: Automated heterogeneous information retrieval benchmark,” 2024. [Online]. Available: https://arxiv.org/abs/2412.13102
- [20] K. An, Q. Chen, C. Deng, Z. Du, C. Gao, Z. Gao, Y. Gu, T. He, H. Hu, K. Hu, S. Ji, Y. Li, Z. Li, H. Lu, H. Luo, X. Lv, B. Ma, Z. Ma, C. Ni, C. Song, J. Shi, X. Shi, H. Wang, W. Wang, Y. Wang, Z. Xiao, Z. Yan, Y. Yang, B. Zhang, Q. Zhang, S. Zhang, N. Zhao, and S. Zheng, “Funaudiollm: Voice understanding and generation foundation models for natural interaction between humans and llms,” 2024. [Online]. Available: https://arxiv.org/abs/2407.04051
-
[21]
C. K. Reddy, V. Gopal, and R. Cutler, “Dnsmos p.835: A non-intrusive perceptual objective speech quality metric to evaluate noise suppressors,” in
*ICASSP 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*. IEEE, 2022. - [22] Z. Ma, M. Chen, H. Zhang, Z. Zheng, W. Chen, X. Li, J. Ye, X. Chen, and T. Hain, “Emobox: Multilingual multi-corpus speech emotion recognition toolkit and benchmark,” 2024. [Online]. Available: https://arxiv.org/abs/2406.07162
-
[23]
Y. Chu, J. Xu, X. Zhou, Q. Yang, S. Zhang, Z. Yan, C. Zhou, and J. Zhou, “Qwen-audio: Advancing universal audio understanding via unified large-scale audio-language models,”
*arXiv preprint arXiv:2311.07919*, 2023. - [24] Q. Chen, Y. Chen, Y. Chen, M. Chen, Y. Chen, C. Deng, Z. Du, R. Gao, C. Gao, Z. Gao, Y. Li, X. Lv, J. Liu, H. Luo, B. Ma, C. Ni, X. Shi, J. Tang, H. Wang, H. Wang, W. Wang, Y. Wang, Y. Xu, F. Yu, Z. Yan, Y. Yang, B. Yang, X. Yang, G. Yang, T. Zhao, Q. Zhang, S. Zhang, N. Zhao, P. Zhang, C. Zhang, and J. Zhou, “Minmo: A multimodal large language model for seamless voice interaction,” 2025. [Online]. Available: https://arxiv.org/abs/2501.06282
-
[25]
Z. Gao, S. Zhang, I. McLoughlin, and Z. Yan, “Paraformer: Fast and accurate parallel transformer for non-autoregressive end-to-end speech recognition,” in
*Proc. Interspeech 2022*, 2022, pp. 2063–2067. - [26] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, “Robust speech recognition via large-scale weak supervision,” 2022. [Online]. Available: https://arxiv.org/abs/2212.04356
-
[27]
Y. Chen, Z. Niu, Z. Ma, K. Deng, C. Wang, J. Zhao, K. Yu, and X. Chen, “F5-tts: A fairytaler that fakes fluent and faithful speech with flow matching,”
*arXiv preprint arXiv:2410.06885*, 2024.
