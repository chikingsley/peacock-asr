# DIRECT PREFERENCE OPTIMIZATION FOR SPEECH AUTOREGRESSIVE DIFFUSION

MODELS
Zhijun Liu1
, Dongya Jia4
, Xiaoqiang Wang4
, Chenpeng Du4
, Shuai Wang3,†
, Zhuo Chen4
, Haizhou Li2
1
School of Data Science, The Chinese University of Hong Kong, Shenzhen
2
School of Artificial Intelligence, The Chinese University of Hong Kong, Shenzhen
3
Nanjing University 4
ByteDance Seed
## ABSTRACT
Autoregressive diffusion models (ARDMs) have recently been applied to speech generation, achieving state-of-the-art (SOTA) performance in zero-shot text-to-speech. By autoregressively generating
continuous speech tokens with next-token diffusion, these models
offer a promising alternative to next-token prediction, avoiding the
technical complexities associated with discrete speech tokenization.
As a relatively new paradigm, research on reinforcement learning
(RL)-based fine-tuning of speech ARDMs remains limited. In this
paper, we propose Autoregressive Diffusion-Direct Preference Optimization (ARDM-DPO) to advance this research. By fine-tuning the
recently proposed zero-shot text-to-speech model DiTAR with DPO,
we achieve significant improvements in terms of speech expressiveness and robustness for long texts.
Index Terms— zero-shot text-to-speech, preference alignment
## 1 INTRODUCTION
A growing body of work in multimodal generation, including audio [1–3], image [4–6], and video synthesis [7–9], now employs
autoregressive diffusion models (ARDMs) as the underlying architecture. ARDMs encode continuous modalities into sequences of
continuous latent vectors (“continuous tokens”) and synthesize sequences by autoregressively predicting the next token with a diffusion model. Compared to next-token prediction over discrete token sequences [10,11], this next-token diffusion approach [12] preserves fine details while avoiding excessively long sequences, thanks
to the compactness of continuous latent representations. Notably, recent studies [2,12–14] applying ARDMs to speech generation report
state-of-the-art zero-shot text-to-speech (TTS) performance.
Modern TTS systems trained on large datasets can sample from
the data distribution with high fidelity. However, the generated
speech may not always align with human preferences after pretraining. For example, when prompted with emotional speech, TTS
models can still produce undesirable monotone outputs, requiring
additional post-filtering by users [15]. Preference alignment algorithms [16–18] bias the output distribution of generative models
toward human-preferred samples, making them an essential posttraining step for powerful speech generation systems [19–29].
Direct Preference Optimization [17] (DPO) was originally proposed for aligning language models and was later adapted for
fine-tuning diffusion models [30]. In this work, we extend DPO
to ARDMs (ARDM-DPO) and apply it to fine-tune the recently
proposed DiTAR model [2], which achieves state-of-the-art performance in zero-shot TTS. DiTAR features a highly efficient ARDM
†: Corresponding author
architecture that separates computation for encoding the generation
history from denoising the next token. To our knowledge, this is the
first preference-alignment method tailored to ARDMs for TTS.
We evaluate ARDM-DPO on two benchmarks: (A) F0 variance
for expressiveness and (B) text likelihood for robustness on hard
texts challenging autoregressive TTS. ARDM-DPO nearly doubles
F0 variance with minimal speaker-similarity loss and reduces CER
by25%. Audiosamplesandfurtherdetailscanbefoundintheonline
supplement1
.
## 2 BACKGROUND
## 2.1 Diffusion Models
Suppose q(x0) is the data distribution. For each diffusion time index
t ∈ {1,...,T}, define the positive decreasing sequence (αt)T
t=1
and the positive increasing sequence (σt)T
t=1. For each t define the
Gaussian perturbation distribution q(xt|x0) := N(xt;αtx0,σ2
t Id).
Define the noise perturbed distribution for each t ∈ {1,...,T} as
q(xt) :=
Z
q(x0)q(xt|x0)dx0. (1)
A well-trained diffusion model can be considered as a score estimator ∇xt logq(xt) trained with denoising score matching [31]. In
the DDPM [32,33] sampler, the diffusion model generates samples
starting from random noise xT ∼ p(xT ) = N(xT ;0,I). Then it
samples xt−1 given xt iteratively.
xt−1 = atxt + bt∇xt logq(xt) + ctϵ, (2)
where ϵ ∼ N(0,I) is an independent noise, (at)T
t=1, (bt)T
t=1 and
(ct)T
t=1 are positive sequences depending on (αt)T
t=1 and (σt)T
t=1.
## 2.2 Autoregressive Diffusion Models
For simplicity, we assume that the ARDM always generates N continuous tokens. The ARDM sampling process can be viewed as a
Markov chain, as illustrated in Fig. 1. Each state st
n is indexed by
the pair (n,t) ∈ {1,...,N} × {1,...,T}, where n is the token
index and t is the diffusion time index. Each state st
n contains the
history tokens already denoised x0
<n and the current noisy token xt
n
that is being denoised, where we define x0
<n := x0
1..n−1.
Given the state st
n, an ARDM estimates the conditional score
∇logq(xt
n|x0
<n) and transitions from st
n to st−1
n when t > 1. Upon
reaching state s0
n, where a token is fully denoised, a random Gaussian noise xT
n+1 is sampled, and a new DDPM sampling process
starts from state sT
n+1 = (x0
≤n,xT
n+1).
1https://zjlww.github.io/ardm-dpo/
arXiv:2509.18928v1 [eess.AS] 23 Sep 2025
Token-Level Autoregressive Generation
Step-Level Diffusion
Fig. 1. ARDM sampling viewed as a Markov chain. Each state contains both the history-generated tokens and the current noisy token. In this
work, we define xa:b := {xn : a ≤ n < b,n ∈ Z} and xa..b := {xn : a ≤ n ≤ b,n ∈ Z}.
## 3 ARDM-DPO
LetxdenoteanARDMsamplingtrajectorythatcontainsallintermediate states in Fig. 1. The reward function r(x0
1..N) is defined over
the terminal states. For a trajectory x, we define r(x) := r(x0
1..N).
Consider the following KL divergence-constrained policy optimization problem:
max
π

Eπ(x) [r(x)] − βDKL(π(x)∥µ(x))

, (3)
where β > 0 is the weight of the KL constraint. The KL-divergence
constraint ensures that the learned policy π(x) does not deviate too
far from the reference distribution µ(x). The unique optimal policy
πr given reward r is [17,34]:
πr(x) :=
1
Zr
µ(x)exp

1
β
r(x)

, (4)
where Zr is defined as Zr :=
R
µ(x)exp

1
β
r(x)

dx. Then, we
have:
r(x) = β log
πr(x)
µ(x)
+ β logZr. (5)
Marginalizing over all ARDM intermediate states gives:
r(x0
1..N) = βEπr(x|x0
1..N
)

log
πr(x)
µ(x)

+ β logZr. (6)
As in DPO, we assume that the log probability of human listeners
preferring sample x0
1..N over y0
1..N is modeled by a Bradley-Terry
model [17]:
logP(x0
1..N ≻ y0
1..N) = logσ r(x0
1..N) − r(y0
1..N)

. (7)
Given a dataset D of preference pairs, we can estimate the reward
model parameters with maximum likelihood estimation:
max
r
Ex0
1..N
,y0
1..N
∼D

logP(x0
1..N ≻ y0
1..N)

. (8)
As a result of Eq. (6), optimizing the reward model r with maximum
likelihood can be conducted without an explicit reward model by
optimizing:
Ex0
1..N
,y0
1..N
∼D

J(x0
1..N,y0
1..N)

, (9)
where
J(x0
1..N,y0
1..N) :=
logσ

TNβ · E U(t),U(n)
π(xt
n,xt−1
n |x0
≤n)
π(yt
n,yt−1
n |y0
≤n)

ℓt
n(x) − ℓt
n(y)

, (10)
with ℓt
n(x) defined as:
ℓt
n(x) := log
π(xt−1
n |xt
n,x0
<n)
µ(xt−1
n |xt
n,x0
<n)
. (11)
As proposed in Diffusion-DPO [30], we assume that the distribution π(xt
n,xt−1
n |x0
≤n) can be approximated with q(xt
n|x0
n) ·
q(xt−1
n |xt
n,x0
n). WealsomovetheexpectationoverU(t), q(xt
n|x0
n),
and q(yt
n|y0
n) from the inside of logσ to the outside with Jensen’s
inequality to obtain the approximate lower bound L for J. The
constant factor TN is absorbed into β in Eq. (12). U(t),U(n) are
uniform distributions over all possible values.
J(x0
1..N,y0
1..N) ≥ L(x0
1..N,y0
1..N) :=
E U(t)
q(xt
n|x0
n)
q(yt
n|y0
n)
h
logσ

βE U(n)
q(xt−1
n |xt
n,x0
n)
q(yt−1
n |yt
n,y0
n)

ℓt
n(x) − ℓt
n(y)
i
. (12)
Notice that the expectation Eq(xt−1
n |xt
n,x0
n)
[ℓt
n(x)] can be written as
the difference of KL divergences:
DKL q(xt−1
n |xt
n,x0
n) µ(xt−1
n |xt
n,x0
<n)

−DKL q(xt−1
n |xt
n,x0
n) π(xt−1
n |xt
n,x0
<n)

.
(13)
Without loss of generality, suppose that the ARDM is trained with
the denoising objective. Then Eq. (13) is equivalent to:
ωt −∥vθ(xt
n,x0
<n) − x0
n∥2
2 + ∥vref(xt
n,x0
<n) − x0
n∥2
2

, (14)
where ωt is a time-dependent weight. Finally, we arrive at the
ARDM-DPO training objective:
L(x0
1..N,y0
1..N) := E U(t)
q(xt
n|x0
n)
q(yt
n|y0
n)
h
logσ

βωtEU(n)
h
− vθ(xt
n,x0
<n) − x0
n
2
2
+ vref(xt
n,x0
<n) − x0
n
2
2
+ vθ(yt
n,y0
<n) − y0
n
2
2
− vref(yt
n,y0
<n) − y0
n
2
2
ii
.
(15)
The DiTAR model used in our experiments is based on v-prediction,
with continuous time t ∈ [0,1] where t = 1 corresponds to pure
noise. We can derive the ARDM-DPO training objective for DiTAR
as follows:
L(x0
1..Nx
,y0
1..Ny
) := E t∼U(0,1)
x1
1..Nx
∼i.i.d.N(0,I)
y1
1..Ny
∼i.i.d.N(0,I)
h
logσ

d−1
βEn
h
vref(xt
n,x0
<n) − ẋt
n
2
2
− vθ(xt
n,x0
<n) − ẋt
n
2
2
i
−d−1
βEn
h
vref(yt
n,y0
<n) − ẏt
n
2
2
− vθ(yt
n,y0
<n) − ẏt
n
2
2
ii
.
(16)
where xt
n = αtx0
n + σtx1
n and ẋt
n = α̇tx0
n + σ̇tx1
n. yt
n and ẏt
n
are defined similarly. We dropped the time-dependent weight ωt
following the practice in Diffusion-DPO. In Eq. (16) we compute En
separately for x and y, since the two trajectories can have different
lengths Nx and Ny. We normalized β with d−1
= 1/256 in our
experiments, with d the dimensionality of each token in DiTAR.
## 4 EXPERIMENTS
## 4.1 Common Setup
Base Model. We fine-tuned a base DiTAR model with 0.4B parameters, pretrained on an internal corpus of around 280,000 hours of
Chinese and English audio. The LM in the base model has 24 Transformer blocks, and the diffusion head contains 4 blocks. Each Transformer block has 1024 hidden dimensions and 16 attention heads.
Inference. We used the same diffusion sampler for training sample
generation and model evaluation. We use a 16-step DDPM sampler
with a linear time schedule. We enabled LM Guidance [2], which is
similar to classifier-free guidance (CFG), with weight w = 2.
DPO Training. All experiments were carried out on 32 A100 GPUs,
with a local batch size of 1 pair and gradient accumulation of 32
steps. The effective batch size is 1024 pairs. We used the AdamW
optimizer, with a fixed learning rate of 2×10−6
, weight decay 0.01,
β1 = 0.9,β2 = 0.95.
Objective Evaluations. In all experiments, we report the word error
rate (WER) using Whisper-large-v3 for English and the character
error rate (CER) using Paraformer-zh for Chinese. Additionally,
we calculate the cosine similarity of speaker embeddings (SIM)
between the prompt and the generated audio using the WavLMTDCNN model. All metrics were computed using Seed-TTS-Eval2
.
We also report the token average KL divergence on the test set,
which is defined as
d−1
E π(x),U(n)
U(t),q(xt
n|x0
n)
vθ(xt
n,x0
<n) − vref(xt
n,x0
<n)
2
2
, (17)
as a measure of the divergence between the fine-tuned model πθ and
the reference model µ. For each objective metric, we report the average value across 8 random runs.
Subjective Evaluations. Twenty listeners participated in pairwise
listening tests, comparing audio from two TTS systems (e.g. A and
B) on specific aspects (e.g., naturalness). For each pair, listeners
chose “A wins”, “B wins”, or “tie”.
## 4.2 Task A: Improving F0 Variance
Task. The fundamental frequency variance (F0V) is strongly correlated with the perceived expressiveness of the generated speech.
Optimizing F0V can effectively prevent the model from producing
monotone responses.
Preference Dataset. We randomly sampled speech prompts and
texts from the LibriTTS [35] dataset. For each prompt-text pair, we
generated 32 candidate responses using the base model. We then
measured the F0V of these responses and selected the best and worst
in terms of F0V to form a preference pair. In total, we collected 256k
preference pairs, amounting to approximately 1,000 hours of speech.
Evaluation Dataset. For evaluation, we randomly selected 38
prompt audios and target texts from different speakers in the LibriTTS test-clean subset.
2https://github.com/BytedanceSpeech/seed-tts-eval
20
30
F0V (Hz)
0.70
0.75
SIM
4
5
6
WER (%)
0 200 400 600 800 1000
# training steps
0.00
0.02
0.04
KL
β = 200
β = 400
β = 800
Base
Bo16
Bo64
Fig. 2. Trajectories of F0V, SIM, and WER of various models over
1000 training steps for Task A.
Rejection Sampling Fine-Tuning. We compare ARDM-DPO with
rejection sampling fine-tuning (RAFT) [36]. RAFT performs supervised fine-tuning (SFT) iteratively. In each RAFT iteration, we
collect approximately 1,000 hours of speech continuations from the
best policy found in the previous iteration. For each prompt, we
sample 32 candidate responses and retain the one with the highest
F0V. RAFT experiments use a batch size of 512 and a learning rate
of 1 × 10−5
.
Method F0V ↑ SIM ↑ WER ↓ KL ↓
Base Model 14.2 0.770 5.17 —
Best-of-16 22.5 0.770 4.74 —
Best-of-64 26.6 0.770 4.93 —
RAFT 300 steps
iter 1 18.3 0.763 5.97 0.057
RAFT 300 steps
iter 2 19.7 0.758 5.91 0.230
RAFT 300 steps
iter 3 20.1 0.756 5.99 0.237
DPO200 steps
β = 200 29.2 0.765 3.73 0.010
Table 1. Selected objective evaluation results for Task A.
Grid Search for Optimal β. In DPO training, β controls the
strength of KL regularization. We report results for β ∈ {200,400,
800}. The evaluation results can be found in Tab. 1 and Fig. 2.
We observe that SIM gradually decreases throughout the training
process for all values of β. This decrease is not caused by changes in
prosody, as the best-of-K (BoK) sampling results in Tab. 1 indicate
that an increase in F0V does not significantly reduce SIM. For larger
values of β, the KL constraint is stronger, resulting in less degradation in SIM; however, the improvement in F0V is also smaller. We
recommend applying early stopping to prevent significant quality
degradation.
Diffusion Loss During DPO. We visualize the changes in diffusion
loss for the winning and losing samples (∆+,∆−) during training
with β = 200 in Fig. 3. Although the training objective in Eq. (16) is
supposed to decrease the diffusion loss of the winning samples while
increasing the diffusion loss of the losing samples, we observe that
the model tends to increase both losses during training. A similar
phenomenon has been observed in LLM DPO training [37]. Investigating this behavior is left as future work.
0 100 200 300 400 500
0.000
0.025
0.050
Diffusion Loss
∆+
∆−
Fig. 3. Change in diffusion loss of the winning and losing samples
in Task A DPO with β = 200.
Subjective Evaluations. For each of the 38 test cases, we generated three random responses from both the base model and the DPO
model (200 steps, β = 200) for comparison. Evaluators assessed the
response pairs based on three criteria: naturalness, speaker similarity
to the prompt, and expressiveness. As shown in Fig. 4, DPO training
slightly reduces naturalness and speaker similarity but significantly
enhances perceived expressiveness.
Lose Tie Win
Speaker Similarity 16.9%
Expressiveness
Naturalness
15.3%
13.5% 81.4% 5%
84.7%
69.5% 13.6%
Fig. 4. Results of the subjective evaluation: DPO vs. base model.
## 4.3 Task B: Improving Text Likelihood
Task. When evaluated on out-of-domain complex texts containing repetitions, autoregressive TTS models often make mistakes in
audio-text alignment, such as missing or inserting words. Following
prior work, we trained a phoneme-based CTC model [38] and used
its negative log likelihood per phoneme (NLL) as a proxy for speech
intelligibility. The CTC model consists of 6 transformer blocks, each
with a hidden dimension of 1024 and 16 attention heads.
Preference Dataset. The prompts were randomly sampled from
DidiSpeech-2 [39], a Chinese speech corpus consisting of 227 hours
of recordings from 1,500 speakers. The texts were selected from
a dataset of 100,000 long Chinese sentences, with randomly introduced repetitive phrases and clauses. For each prompt-text pair, we
generated 16 candidate responses using the base model. We then
calculated the CTC loss with our CTC model and selected the best
and worst responses to form a preference pair. In total, we collected
430,000 preference pairs, totaling approximately 3,500 hours.
Evaluation Dataset. We utilized the hard test set proposed in SeedTTS [19], which contains 400 challenging test cases in Chinese with
complex text. We excluded all speakers in the Seed-TTS-Eval hard
set from the preference dataset, guaranteeing that they remained unseen during training.
Method NLL ↓ SIM ↑ CER ↓ KL ↓
Base Model 0.55 0.711 8.37 —
Best-of-8 (CER) 0.39 0.713 4.99 —
Best-of-8 (NLL) 0.27 0.712 6.79 —
DPO9000 steps
β = 1600 0.32 0.712 6.32 0.009
Table 2. Selected objective evaluation results for Task B.
Grid Search for Optimal β. We initially experimented with β ∈
{200,400,...,3200,6400} and observed that β ≤ 400 led to increases in NLL and CER within 300 steps, while β ≥ 6400 resulted
in very slow optimization. Consequently, we focused on β ∈ {800,
1600,3200} for further analysis. The trajectories of NLL, SIM, and
CER during training are shown in Fig. 5. We found that the DPO
model trained for 9000 steps with β = 1600 achieved the best performance, with a 25% reduction in CER. Detailed results are provided in Tab. 2.
0.3
0.4
0.5
NLL (nats)
0.710
0.712
SIM
7
8
WER (%) 0 5000 10000
# training steps
0.000
0.005
0.010
KL
β = 800
β = 1600
β = 3200
Base
NLL Bo8
Fig. 5. Trajectories of NLL, SIM, and CER of various models over
12,000 training steps for Task B.
Subjective evaluations. We randomly sampled 40 test cases from
the test set and generated three random response pairs for each
test case from the base model and the DPO model (9,000 steps,
β = 1600). Evaluators assessed all pairs for naturalness and speaker
similarity. We find that the DPO model performs similarly to the
base model. For naturalness, the lose/tie/win probabilities are 4.4%,
88.7%, and 6.9%, respectively; for speaker similarity, they are 2.1%,
94.3%, and 3.6%, respectively. This indicates good prior preservation of ARDM-DPO on Task B.
## 5 CONCLUSIONS AND LIMITATIONS
In this work, we introduced ARDM-DPO, the first direct preference optimization method tailored for autoregressive diffusion TTS.
Through comprehensive experiments on DiTAR, we demonstrate
that ARDM-DPO achieves significant improvements in speech
expressiveness and robustness on challenging long texts while maintaining speaker similarity and speech naturalness.
We observed that ARDM-DPO training on Task A is unstable
and requires early stopping to avoid speech quality degradation. The
underlying cause warrants further investigation. It is well known that
the construction of the preference dataset in DPO plays a critical role
in ensuring good performance [20,21,40]. We leave these directions
for future work.
## 6 REFERENCES
[1] Zhijun Liu, Shuai Wang et al., “Autoregressive diffusion
transformer for text-to-speech synthesis,” arXiv preprint
arXiv:2406.05551, 2024.
[2] Dongya Jia, Zhuo Chen et al., “DiTAR: Diffusion transformer
autoregressive modeling for speech generation,” in ICML,
2025.
[3] Chenyu Yang, Shuai Wang et al., “SongBloom: Coherent song
generation via interleaved autoregressive sketching and diffusion refinement,” arXiv preprint arXiv:2506.07634, 2025.
[4] Tianhong Li, Yonglong Tian et al., “Autoregressive image generation without vector quantization,” in NeurIPS, 2024.
[5] Siqi Kou, Jiachun Jin et al., “Orthus: Autoregressive interleaved image-text generation with modality-specific heads,” in
ICML, 2025.
[6] NextStep Team, Chunrui Han et al., “NextStep-1: Toward autoregressive image generation with continuous tokens at scale,”
arXiv preprint arXiv:2508.10711, 2025.
[7] Tianwei Yin, Qiang Zhang et al., “From slow bidirectional to
fast autoregressive video diffusion models,” in CVPR, 2025.
[8] Haoge Deng, Ting Pan et al., “Autoregressive video
generation without vector quantization,” arXiv preprint
arXiv:2412.14169, 2024.
[9] HansiTeng, HongyuJiaetal., “MAGI-1: Autoregressivevideo
generation at scale,” arXiv preprint arXiv:2505.13211, 2025.
[10] Yiwei Guo, Zhihan Li et al., “Recent advances in discrete
speech tokens: A review,” arXiv preprint arXiv:2502.06490,
2025.
[11] Tomoki Hayashi, Shinji Watanabe, “DiscreTalk: Text-tospeech as a machine translation problem,” arXiv preprint
arXiv:2005.05525, 2020.
[12] Yutao Sun, Hangbo Bao et al., “Multimodal latent language modeling with next-token diffusion,” arXiv preprint
arXiv:2412.08635, 2024.
[13] Zhiliang Peng, Jianwei Yu et al., “VibeVoice technical report,”
arXiv preprint arXiv:2508.19205, 2025.
[14] Yanqing Liu, Ruiqing Xue et al., “Next tokens denoising for
speech synthesis,” arXiv preprint arXiv:2507.22746, 2025.
[15] Xiaoxue Gao, Chen Zhang et al., “Emo-DPO: Controllable
emotional speech synthesis through direct preference optimization,” in ICASSP, 2025.
[16] Long Ouyang, Jeffrey Wu et al., “Training language models to
follow instructions with human feedback,” NeurIPS, 2022.
[17] Rafael Rafailov, Archit Sharma et al., “Direct Preference Optimization: Your language model is secretly a reward model,”
in NeurIPS, 2023.
[18] Jiazheng Xu, Xiao Liu et al., “ImageReward: Learning and
evaluating human preferences for text-to-image generation,” in
NeurIPS, 2023.
[19] Philip Anastassiou, Jiawei Chen et al., “Seed-TTS: A family of high-quality versatile speech generation models,” arXiv
preprint arXiv:2406.02430, 2024.
[20] Jinchuan Tian, Chunlei Zhang et al., “Preference alignment
improves language model-based TTS,” in ICASSP, 2025.
[21] Xueyao Zhang, Yuancheng Wang et al., “Advancing zero-shot
text-to-speech intelligibility across diverse domains via preference alignment,” in ACL, 2025.
[22] Zhihao Du, Changfeng Gao et al., “CosyVoice 3: Towards inthe-wild speech generation via scaling-up and post-training,”
arXiv preprint arXiv:2505.17589, 2025.
[23] YinghaoAaronLi, RitheshKumaretal., “DMOSpeech: Direct
metric optimization via distilled diffusion model in zero-shot
speech synthesis,” in ICML, 2025.
[24] Yuchen Hu, Chen Chen et al., “Robust zero-shot text-to-speech
synthesis with reverse inference optimization,” arXiv preprint
arXiv:2407.02243, 2024.
[25] Dong Zhang, Zhaowei Li et al., “SpeechAlign: Aligning
speech generation to human preferences,” NeurIPS, 2024.
[26] Chen Chen, Yuchen Hu et al., “Enhancing zero-shot textto-speech synthesis with human feedback,” arXiv preprint
arXiv:2406.00654, 2024.
[27] Jingyi Chen, Ju Seung Byun et al., “Fine-tuning text-tospeech diffusion models using reinforcement learning with human feedback,” in Interspeech, 2025.
[28] Jixun Yao, Yuguang Yang et al., “Fine-grained preference optimization improves zero-shot text-to-speech,” arXiv preprint
arXiv:2502.02950, 2025.
[29] Kangxiang Xia, Xinfa Zhu et al., “MPO: Multidimensional
preference optimization for language model-based text-tospeech,” arXiv preprint arXiv:2509.00685, 2025.
[30] Bram Wallace, Meihua Dang et al., “Diffusion model alignment using direct preference optimization,” in CVPR, 2024.
[31] Yang Song, Jascha Sohl-Dickstein et al., “Score-based generative modeling through stochastic differential equations,” in
ICLR, 2021.
[32] Jonathan Ho, Ajay Jain et al., “Denoising diffusion probabilistic models,” in NeurIPS, 2020.
[33] Jiaming Song, Chenlin Meng et al., “Denoising diffusion implicit models,” in ICLR, 2021.
[34] Xue Bin Peng, Aviral Kumar et al., “Advantage-weighted regression: Simple and scalable off-policy reinforcement learning,” arXiv preprint arXiv:1910.00177, 2019.
[35] Heiga Zen, Viet Dang et al., “LibriTTS: A corpus derived from
librispeech for text-to-speech,” in Interspeech, 2019.
[36] Hanze Dong, Wei Xiong et al., “RAFT: Reward rAnked FineTuning for generative foundation model alignment,” TMLR,
2023.
[37] Yuzi Yan, Yibo Miao et al., “3D-Properties: Identifying challenges in dpo and charting a path forward,” in ICLR, 2025.
[38] Alex Graves, Santiago Fernández et al., “Connectionist Temporal Classification: labelling unsegmented sequence data with
recurrent neural networks,” in ICML, 2006.
[39] Tingwei Guo, Cheng Wen et al., “DiDiSpeech: A large scale
mandarin speech corpus,” in ICASSP, 2021.
[40] Navonil Majumder, Chia-Yu Hung et al., “Tango 2: Aligning
diffusion-based text-to-audio generations through direct preference optimization,” arXiv preprint arXiv:2404.09956, 2024.
