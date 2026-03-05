# Aligning Generative Speech Enhancement with Perceptual Feedback

###### Abstract

Language Model (LM)-based speech enhancement (SE) has recently emerged as a promising direction, but existing approaches predominantly rely on token-level likelihood objectives that weakly reflect human perception. This mismatch limits progress, as optimizing signal accuracy does not always improve naturalness or listening comfort. We address this gap by introducing a perceptually aligned LM-based SE approach. Our method applies Direct Preference Optimization (DPO) with UTMOS, a neural MOS predictor, as a proxy for human ratings, directly steering models toward perceptually preferred outputs. This design directly connects model training to perceptual quality and is broadly applicable within LM-based SE frameworks. On the Deep Noise Suppression Challenge 2020 test sets, our approach consistently improves speech quality metrics, achieving relative gains of up to 56%. To our knowledge, this is the first integration of perceptual feedback into LM-based SE and the first application of DPO in the SE domain, establishing a new paradigm for perceptually aligned enhancement with SE.

Index Terms— Speech Enhancement, Language Models, Direct Preference Optimization, Perceptual Quality, Human-AI Alignment

## 1 Introduction

Speech enhancement (SE) is a cornerstone technology for robust human communication and human-machine interaction, supporting applications such as hearing aids, telecommunications, and voice-driven AI systems. Existing deep neural network (DNN)-based SE approaches are broadly categorized as discriminative or generative. Discriminative methods [5, 11, 6, 7] minimize the distance between noisy and clean speech but often generalize poorly to unseen conditions and may introduce perceptual artifacts [25, 8, 2]. Generative methods [29, 31] instead model the distribution of clean speech to synthesize enhanced signals, improving robustness and enabling solutions to inherently generative challenges such as packet loss concealment.

Recent advances in language models (LMs) for image and audio generation [1, 24] have inspired LM-based generative SE frameworks [26, 28, 13, 29, 9], which show promising performance. These systems are typically trained with paired data under token-level prediction objectives, reconstructing clean or high-quality speech from noisy or degraded inputs. However, such objectives emphasize signal accuracy rather than perceptual aspects like naturalness and comfort, creating a fundamental misalignment with human listening preferences.

Research on perceptual alignment for SE remains limited. MetricGAN [5] introduced adversarial training to optimize perceptual metrics, such as PESQ and STOI. Nonetheless, these metrics do not always correlate strongly with human ratings [22, 10]. More recent studies incorporated MOS predictors [15, 10], providing valuable insights but at the cost of added complexity and modest gains. RLHF methods, such as Proximal Policy Optimization (PPO) [23], also represent a promising avenue, yet those methods typically involve complex pipelines and may face stability challenges [16]. Direct Preference Optimization (DPO) [18] has recently emerged as a simpler and more stable alternative. Originally developed in NLP, DPO bypasses reward modeling and reinforcement learning, directly aligning outputs with human preferences. Its strong performance in dialogue generation and summarization suggests that DPO is a compelling candidate to bridge the gap between SE objectives and perceptual quality.

In this work, we propose Generative Speech Enhancement with Perceptual Feedback (GSEPF), the first LM-based SE approach explicitly aligned with human auditory preferences. We built on a state-of-the-art generative SE model following [29], and fine-tune it with DPO guided by UTMOS [21], a neural MOS predictor that serves as a proxy for human ratings. This approach steers the model toward perceptually preferred outputs without the overhead of traditional RLHF pipelines. Experiments on the 2020 Deep Noise Suppression Challenge test sets [19] show that GSEPF delivers consistent gains in objective metrics (DNSMOS [20], UTMOS, NISQA [14]) and subjective listening tests, achieving relative improvements of up to 56%.

To the best of our knowledge, this is the first study to introduce DPO into speech enhancement and the first to incorporate proxy perceptual feedback into LM-based SE. Our results establish a simple yet powerful framework for perceptually aligned enhancement, signaling a paradigm shift toward enhancement systems that optimize for human auditory preference.

## 2 Generative Speech Enhancement with Perceptual Feedback

### 2.1 Two-Stage Generative SE Framework

We build on a state-of-the-art generative SE framework [29], formulating enhancement as a two-stage language modeling problem that integrates both semantic and acoustic representations.

#### 2.1.1 Stage 1: Noise-to-Semantic (N2S) LM.

Given a noisy waveform , the first six layers of WavLM-Large [3] extract continuous latent features. A pre-trained K-means model then quantizes these frame-level features into semantic tokens, , where is the number of frames, and each token is the index of one of the 1024 clusters in K-means. An autoregressive language model, denoted as the N2S LM, takes as input and produces predicted clean semantic tokens , aligned frame by frame with the noisy tokens.

#### 2.1.2 Stage 2: Semantic-to-Speech (S2S) LM.

SimCodec [29], a neural audio codec with a single codebook, encodes into acoustic tokens , where is the number of frames. A second autoregressive language model, denoted as the S2S LM, takes the concatenated token sequence as context and generates the enhanced acoustic sequence . The enhanced waveform is then reconstructed using SimCodec’s decoder.

#### 2.1.3 Token level training objective.

During training, teacher forcing is applied by replacing and with ground-truth and obtained from the ground-truth clean speech . The S2S LM is trained with the cross-entropy loss

| (1) |

This two-stage formulation leverages the representational power of LMs at both semantic and acoustic levels, providing a strong backbone for our perceptual alignment framework.

### 2.2 Alignment with Perceptual Feedback

#### 2.2.1 Direct Preference Optimization

While cross-entropy maximization improves likelihood, it does not guarantee perceptual quality, since human preference may diverge from token-level accuracy. DPO [18] directly aligns outputs with preference signals via a contrastive objective.

Given preference pairs, where one sequence is favored over another under the same context , DPO maximizes the relative preference margin:

| (2) |

where is the trainable target model parameterized by , is a frozen reference model to stabilize learning, controls preference sharpness, and is the logistic sigmoid. Here, are acoustic token sequences generated by the S2S LM, and denotes the conditioning context from Eq. 1.

#### 2.2.2 Preference Pairs Construction

The effectiveness of DPO depends heavily on the quality of the preferred and rejected pairs. Figure 1 illustrates our method to obtain these preference pairs.

Candidates Generation. Given prompt tokens sequences (i.e. ), a pretrained S2S LM produces the logits of the enhanced acoustic sequence under teacher forcing following Eq 1. We denote the generated logits as , where and each denotes the logits at timestamp , with the vocabulary size. Then, a top- filter is applied to , retaining the highest-probability logits at each timestamp, producing . From , we independently sample candidate acoustic sequences. Each sequence is generated by sampling one token per timestep from the softmax-normalized probability distribution over the top logits. The resulting sequences are denoted as , with each sequence .

Perceptual Scoring. To rank the sampled sequences according to human perceptual preferences, each is decoded into waveform using SimCodec, then evaluated by UTMOS [21], a neural MOS estimator. UTMOS provides scalable, reference-free perceptual scores without the cost of extensive human evaluations, producing annotated pairs .

Preference Selection. From the set , the top- sequences by MOS are designated as preferred outputs , while the bottom- form the rejected set , with by construction.

#### 2.2.3 Perceptual level training objective.

The target S2S LM , initialized from , receives the same prompt (i.e. ) to compute logits under teacher forcing. From and , sequence probabilities can be computed as

Using and , in Eq. 2 is evaluated. The final training objective combines token and perceptual objectives:

Here is updated while remains frozen. No scaling is applied to either loss term since their magnitudes were reasonably close.

## 3 Experiments

### 3.1 Dataset

Trainset. For the N2S and S2S LMs, we follow prior LM-based SE studies [26, 13, 29] by dynamically generating noisy speech using clean speech, noise clips, and room impulse responses (RIRs). Clean speech consists of a subset of LibriTTS [30], VCTK and the read speech partition of the 2022 DNS Challenge [4], totaling 530 hours. Noise clips (175 hours) are drawn from AudioSet and Freesound (DNS 2022), while RIRs ( 17 hours) come from OpenSLR26 and OpenSLR28 (DNS 2022). With 40% probability, reverberation is added using a random RIR; one noise source is mixed with 80% probability and two sources with 20% probability, using SNRs sampled uniformly from dB. SimCodec and the -means tokenizer are pretrained on 960h LibriSpeech [17] and LibriTTS, respectively. All audio is resampled to 16 kHz.

### 3.2 Evaluation Metrics

We evaluate speech quality using DNSMOS [20], NISQA [14], and UTMOS [21]. DNSMOS, a neural network-based metric, has become a standard for evaluating speech quality in LM-based speech enhancement. Unlike PESQ, which is sensitive to time misalignment [26, 13, 12], DNSMOS robustly estimates quality without a reference. Similarly to DNSMOS, NISQA and UTMOS are reference-free and correlate well with human ratings.

To assess speaker preservation, we report speaker embedding cosine similarity (SECS), computed between ReDimNet [27] embeddings of the enhanced and ground-truth speech.

Finally, to directly validate perceptual gains, we conduct an A/B preference test on 30 utterances from the DNS Challenge test set (w/o Reverb), where 20 volunteers compare enhanced samples before and after DPO optimization in randomized order and indicate their preference based on naturalness and listening comfort under headphone playback in quiet environments.

### 3.3 Implementation Details

The N2S and S2S LMs are decoder-only LMs consisting of 12 transformer layers, a hidden size of 1024, and 8 attention heads. The N2S LM is trained with AdamW (peak LR , warmup 1k steps, cosine decay) on a single A40 GPU for 510k steps with batch size 8. The same N2S LM is used in all experiments.

For the reference S2S LM (), we use the same schedule but train on four A40 GPUs for 44k steps with batch size 128, stopping once DNSMOS scores saturate. This is the S2S LM used for our baseline GenSE model, denoted as GenSE* in table 1.

Finally, the target S2S LM () is initialized from and optimized with AdamW (LR ) for 400 steps with batch size 128 on a single A40 GPU, with the DPO temperature (Eq. 2) fixed at 0.1 in all experiments. Unless otherwise specified, we set for top- filtering, sample candidate sequences per prompt, and construct preference pairs.

## 4 Results and Discussions

*denotes the reproduced performances of GenSE on our dataset. Bold = best, underline = second best.

| System | w/o Reverb | w/ Reverb | ||||||||||
| DNSMOS | UTMOS | NISQA | SECS | DNSMOS | UTMOS | NISQA | SECS | |||||
| SIG | BAK | OVL | SIG | BAK | OVL | |||||||
| Noisy | 3.39 | 2.62 | 2.48 | - | - | - | 1.76 | 1.50 | 1.39 | - | - | - |
| GenSE [29] | 3.65 | 4.18 | 3.43 | - | - | - | 3.49 | 3.73 | 3.19 | - | - | - |
GenSE*
|
3.65 | 4.16 | 3.41 | 3.91 | 3.916 | 0.691 | 3.50 | 3.96 | 3.16 | 2.03 | 2.505 | 0.445 |
GenSE*CE
|
3.64 | 4.15 | 3.40 | 3.91 | 3.912 | 0.691 | 3.48 | 3.96 | 3.14 | 2.10 | 2.509 | 0.452 |
GSEPFDPO
|
3.66 | 4.18 | 3.44 | 4.21 | 4.070 | 0.651 | 3.64 | 4.13 | 3.37 | 3.18 | 2.984 | 0.454 |
GSEPFCE+DPO
|
3.67 | 4.18 | 3.44 | 4.17 | 4.021 | 0.667 | 3.60 | 4.10 | 3.32 | 2.86 | 2.815 | 0.477 |

| System | w/o Reverb | w/ Reverb | ||||||||||
| DNSMOS | UTMOS | NISQA | SECS | DNSMOS | UTMOS | NISQA | SECS | |||||
| SIG | BAK | OVL | SIG | BAK | OVL | |||||||
GenSE*
|
3.65 | 4.16 | 3.41 | 3.91 | 3.916 | 0.691 | 3.50 | 3.96 | 3.16 | 2.03 | 2.505 | 0.445 |
| Z=1 (Ground-truth) | 3.65 | 4.17 | 3.42 | 3.92 | 3.913 | 0.688 | 3.48 | 3.95 | 3.14 | 2.06 | 2.498 | 0.456 |
| Z=1 | 3.67 | 4.17 | 3.44 | 4.17 | 4.052 | 0.666 | 3.62 | 4.11 | 3.35 | 2.95 | 2.873 | 0.469 |
| Z=4 | 3.67 | 4.18 | 3.44 | 4.17 | 4.021 | 0.667 | 3.60 | 4.10 | 3.32 | 2.86 | 2.815 | 0.477 |

### 4.1 Main Comparison with Baselines

We explored three loss combination schemes to train the target S2S LM : 1. GenSE*CE. We continue to use the cross entropy (CE) loss in Eq 1, which is the original training objective of GenSE. 2. GSEPFDPO. We use the DPO training objective in Eq 2. 3. GSEPFCE+DPO. We used both the CE and DPO losses. The results in Table 1 show that DPO loss consistently improves all speech quality metrics, with up to 56% increase in UTMOS (from 2.03 to 3.18) and 19% increase in NISQA (from 2.505 to 2.984) on the w/ Reverb partition, demonstrating its effectiveness in enhancing perceptual quality rather than overfitting to UTMOS only. In addition, we observe that the DPO optimization achieves better improvement on the w/ Reverb partition than the w/o Reverb partition, which is likely because recovering clean speech from reverberant+noisy speech is more challenging and hence leaving more room for improvement. In contrast to DPO-related optimizations, we observe from the GenSE*CE experiment that using CE loss alone does not appear to improve the target S2S LM .

For speaker similarity (SECS), DPO optimization results in degradation on the w/o Reverb partition but improves on the w/ Reverb partition. Combining CE loss with DPO loss improves SECS across both test partitions over only DPO loss. These results suggest that incorporating CE loss provides an anchoring effect during DPO training, discouraging over-optimization toward speech quality that compromises speaker similarity, while slightly tempering the speech quality improvements achieved by the DPO loss.

### 4.2 Subjective Evaluation

To further validate perceptual alignment, we conducted A/B preference tests with 20 human listeners on 30 utterances. As shown in Fig. 2, listeners generally preferred GSEPF-enhanced samples (GSEPFCE+DPO) over the GenSE baseline ( GenSE*) in 23 of 30 samples, citing greater naturalness and reduced listening fatigue. This strong subjective preference confirms that our DPO-based optimization aligns well with human auditory perception, beyond what can be captured by automated metrics alone.

Figure 3 additionally presents spectrogram comparisons for an utterance with notable noise and reverberation. The GenSE baseline reduces noise but introduces artifacts in voiced regions, whereas GSEPF better preserves harmonic structures. This qualitative evidence highlights the strength of perceptual alignment: By optimizing towards human-preferred outputs, GSEPF produces speech that is both cleaner and more natural.

### 4.3 Ablation Experiment: Preference Pair Selection

In Table 2, we conduct ablation studies on how different preference pair construction strategies impact the training of the target S2S LM . For all experiments, we optimize using both CE and DPO loss. In the =1 (Ground-truth) experiment, we directly use the ground-truth acoustic tokens of the target clean speech as A+ instead of sampling from the reference S2S LM . We compare this setup with the =1 experiment, where the sampling of A+ follows the pipeline described in section 2.2.2, with , the number of preference pairs set to 1. The results show that DPO training is ineffective when using the =1 (Ground-truth) setup. We attribute this to the model being pushed in the same direction as cross-entropy loss, thereby diminishing the unique contribution of DPO. We also investigated whether the number of preference pairs for A+ and A- have on performance, by altering the value of . From the =1 and =4 experiments, we found that while both setups effectively improved speech quality, using more preference pairs does not lead to better speech quality.

## 5 Conclusion

We presented GSEPF, the first framework to apply Direct Preference Optimization (DPO) to speech enhancement. By leveraging UTMOS as a proxy for human judgments and constructing preference pairs from a reference LM, GSEPF directly aligns model outputs with perceptual feedback. Despite its simplicity, it achieves up to 56% improvement on UTMOS and clear gains on unseen metrics such as NISQA, all within only 400 training steps. This work marks a paradigm shift for LM-based SE: moving beyond token-level likelihood toward preference-driven optimization that better reflects human listening experience. Future directions include extending preference alignment to speaker similarity, controllability, and ultimately multi-objective alignment in audio and multimodal generation.

## References

- [1] (2023) Muse: text-to-image generation via masked generative transformers. arXiv preprint arXiv:2301.00704. Cited by: §1.
- [2] (2022) Noise-robust speech recognition with 10 minutes unparalleled in-domain data. In ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Vol. pp. 4298–4302. External Links: Cited by: §1.
- [3] (2022) Wavlm: large-scale self-supervised pre-training for full stack speech processing. IEEE Journal of Selected Topics in Signal Processing 16 (6), pp. 1505–1518. Cited by: §2.1.1.
- [4] (2022) Icassp 2022 deep noise suppression challenge. In ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 9271–9275. Cited by: §3.1.
- [5] (2019) MetricGAN: generative adversarial networks based black-box metric scores optimization for speech enhancement. In International Conference on Machine Learning (ICML), Cited by: §1, §1.
- [6] (2019) Domain adversarial training for speech enhancement. In 2019 Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC), Vol. pp. 667–672. External Links: Cited by: §1.
- [7] (2021) Learning disentangled feature representations for speech enhancement via adversarial training. In ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Vol. pp. 666–670. External Links: Cited by: §1.
- [8] (2022) Mismatch problem in deep-learning based speech enhancement. Cited by: §1.
- [9] (2025) LLaSE-g1: incentivizing generalization capability for llama-based speech enhancement. arXiv preprint arXiv:2503.00493. Cited by: §1.
- [10] (2025) Using rlhf to align speech enhancement approaches to mean-opinion quality scores. In ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 1–5. Cited by: §1.
- [11] (2025) From KAN to GR-KAN: advancing speech enhancement with kan-based methodology. In Interspeech, Cited by: §1.
- [12] (2025) Speech enhancement using continuous embeddings of neural audio codec. In ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 1–5. Cited by: §3.2.
- [13] (2024) Masksr: masked language model for full-band speech restoration. arXiv preprint arXiv:2406.02092. Cited by: §1, §3.1, §3.2.
- [14] (2021) NISQA: a deep cnn-self-attention model for multidimensional speech quality prediction with crowdsourced datasets. arXiv preprint arXiv:2104.09494. Cited by: §1, §3.2.
- [15] (2023) Attention-based speech enhancement using human quality perception modeling. IEEE/ACM Transactions on Audio, Speech, and Language Processing 32, pp. 250–260. Cited by: §1.
- [16] (2022) Training language models to follow instructions with human feedback. Advances in neural information processing systems 35, pp. 27730–27744. Cited by: §1.
- [17] (2015) Librispeech: an asr corpus based on public domain audio books. In 2015 IEEE international conference on acoustics, speech and signal processing (ICASSP), pp. 5206–5210. Cited by: §3.1.
- [18] (2023) Direct preference optimization: your language model is secretly a reward model. Advances in Neural Information Processing Systems 36, pp. 53728–53741. Cited by: §1, §2.2.1.
- [19] (2020) The interspeech 2020 deep noise suppression challenge: datasets, subjective testing framework, and challenge results. arXiv preprint arXiv:2005.13981. Cited by: §1, §3.1.
- [20] (2021) DNSMOS: a non-intrusive perceptual objective speech quality metric to evaluate noise suppressors. In ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 6493–6497. Cited by: §1, §3.2.
- [21] (2022) Utmos: utokyo-sarulab system for voicemos challenge 2022. arXiv preprint arXiv:2204.02152. Cited by: §1, §2.2.2, §3.2.
- [22] (2014) An improved non-intrusive intelligibility metric for noisy and reverberant speech. In 2014 14th International Workshop on Acoustic Signal Enhancement (IWAENC), pp. 55–59. Cited by: §1.
- [23] (2017) Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347. Cited by: §1.
- [24] (2023) Neural codec language models are zero-shot text to speech synthesizers. arXiv preprint arXiv:2301.02111. Cited by: §1.
- [25] (2019) Bridging the gap between monaural speech enhancement and recognition with distortion-independent acoustic modeling. IEEE/ACM Transactions on Audio, Speech, and Language Processing 28, pp. 39–48. Cited by: §1.
- [26] (2024) SELM: speech enhancement using discrete tokens and language models. In ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 11561–11565. Cited by: §1, §3.1, §3.2.
- [27] (2024) Reshape dimensions network for speaker recognition. arXiv preprint arXiv:2407.18223. Cited by: §3.2.
- [28] (2024) Genhancer: high-fidelity speech enhancement via generative modeling on discrete codec tokens. In Proc. Interspeech 2024, pp. 1170–1174. Cited by: §1.
- [29] (2025) GenSE: generative speech enhancement via language models using hierarchical modeling. arXiv preprint arXiv:2502.02942. Cited by: §1, §1, §1, §2.1.2, §2.1, §3.1, Table 1.
- [30] (2019) Libritts: a corpus derived from librispeech for text-to-speech. arXiv preprint arXiv:1904.02882. Cited by: §3.1.
- [31] (2025) AnyEnhance: a unified generative model with prompt-guidance and self-critic for voice enhancement. arXiv preprint arXiv:2501.15417. Cited by: §1.
