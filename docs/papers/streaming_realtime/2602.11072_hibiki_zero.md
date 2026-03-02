# Simultaneous Speech-to-Speech Translation Without Aligned Data

###### Abstract

Simultaneous speech translation requires translating source speech into a target language in real-time while handling non-monotonic word dependencies. Traditional approaches rely on supervised training with word-level aligned data, which is difficult to collect at scale and thus depends on synthetic alignments using language-specific heuristics that are suboptimal. We propose Hibiki-Zero, which eliminates the need for word-level alignments entirely. This fundamentally simplifies the training pipeline and enables seamless scaling to diverse languages with varying grammatical structures, removing the bottleneck of designing language-specific alignment heuristics. We first train on sentence-level aligned data to learn speech translation at high latency, then apply a novel reinforcement learning strategy using GRPO to optimize latency while preserving translation quality. Hibiki-Zero achieves state-of-the-art performance in translation accuracy, latency, voice transfer, and naturalness across five X-to-English tasks. Moreover, we demonstrate that our model can be adapted to support a new input language with less than 1000h of speech. We provide examples, model weights, inference code111github.com/kyutai-labs/hibiki-zero and we release a benchmark containing 45h of multilingual data for speech translation evaluation.222huggingface.co/collections/kyutai/hibiki-zero

## 1 Introduction

We introduce Hibiki-Zero, a system for simultaneous and expressive speech-to-speech (S2ST) and speech-to-text (S2TT) translation that does not require aligned data for training. Unlike offline speech translation systems that access the full source utterance before translating, simultaneous translation must produce output incrementally while maintaining both translation accuracy and speech naturalness. This requires learning a fine-grained translation policy that determines when to listen and when to speak. The most straightforward approach to learning such a policy is through supervised training on aligned data. However, human interpretation data with word-level alignments is virtually non-existent, forcing state-of-the-art systems to rely on synthetic data with automatic alignments (Labiausse et al., 2025). These automatic alignments are inherently limited, as they depend on hand-crafted heuristics rather than being learned from data.

Hibiki-Zero is a decoder-only model that synchronously receives source speech and generates translated speech leveraging a multistream architecture originally introduced by Défossez et al. (2024). Unlike Hibiki (Labiausse et al., 2025), Hibiki-Zero is not trained with supervised learning on synthetic interpretation data but rather casts joint optimization of translation quality and latency as a reinforcement learning (RL) problem. While we still require a base model before the RL phase, it is trained using sentence-level aligned data which can be more easily constructed independently of the language compared to word-level aligned data. During RL, we exploit the sentence-level aspect of our data to design a simple reward system based on BLEU score (Papineni et al., 2002) only. To achieve this, we compute rewards at multiple intermediate instants during the translation of an input speech utterance by leveraging the simultaneous text translation also produced by our model. Using these process rewards, we obtain fine-grained local advantages across multiple translations from the same input. We then adapt GRPO (Shao et al., 2024) to our multistream architecture, using these advantages to optimize the model.

In a multilingual-to-English translation task, Hibiki-Zero outperforms previous state-of-the-art work in translation quality, latency, speaker identity preservation, and speech naturalness. We also retain all the benefits of multistream modeling such as batching and real-time inference on GPU while removing the necessity to build interpretation-like training data thus considerably simplifying the development of such models. We even demonstrate that Hibiki-Zero can adapt to a new input language with less than 1000h of training data marking an important step to make high quality speech translation (ST) available in more languages. We will release our data preparation code, model weights as well as a 45h multilingual speech benchmark for ST evaluation.

## 2 Related Work

### 2.1 Simultaneous end-to-end speech translation

While speech translation was initially performed using cascaded systems combining automatic speech recognition (ASR), machine translation (MT) and text-to-speech synthesis (TTS) (Wahlster, 2000; Nakamura et al., 2006), it recently evolved in fully end-to-end systems (Jia et al., 2019; Lee et al., 2022a; Jia et al., 2022; Rubenstein et al., 2023) reducing error propagation and enabling transfer of non-linguistic information such as the speaker voice identity or prosody to the generated speech. At first trained with auxiliary text of phoneme translation tasks (Jia et al., 2022; Zhang et al., 2024a), most recent works (Barrault et al., 2023; Labiausse et al., 2025; Cheng et al., 2025; Misiunas and Ablavatski, 2025) train directly on simultaneous S2TT and S2ST tasks so they can use the predicted text translation as a scaffolding for speech generation at inference time. Among direct ST training methods, those who achieve better speech naturalness are duplex audio systems that require to build a simultaneous ST dataset. They either rely on a synthetic data generation pipeline which includes a fine word-level text-to-translation alignment method (Labiausse et al., 2025; Misiunas and Ablavatski, 2025) or use a text LLM to split text into semantic chunks (a few words) that are individually translated thus providing chunk-level translation alignment (Cheng et al., 2025) before collecting human-annotated interpretation data for finetuning purposes. Hibiki-Zero removes most of the complexity from synthetic data generation as it only requires sentence-level translation alignment easily obtained from punctuation. Thanks to an efficient RL process, it is then possible to reduce the translation latency of the model so it achieves state-of-the-art quality/latency trade-off in multiple input languages.

### 2.2 Self-improvement of real-time translation systems

RL methods to improve simultaneous translation systems were first explored in the context of text translation. Some works used preference-based approaches (Yu et al., 2025) with preferences established in the context of simultaneous ST by prompting a text LLM while others applied online reinforcement procedures (Yu et al., 2025; Xu et al., 2025) with sequence-level rewards as a combination of translation quality and latency metrics. Because they lack sub-sentence granularity in their preference or reward signals, it is difficult for these methods to find an appropriate balance between translation quality and latency during the RL process. More recently, Seed LiveInterpret 2.0 (Cheng et al., 2025) applied PPO (Schulman et al., 2017) with a combination of intermediate evaluations of the generated sequences (process rewards) and overall evaluation of the translation (outcome rewards). Starting from a base supervised ST model trained with chunk-level alignment and finetuned on high-quality human interpretation data, they managed to strictly improve the quality/latency trade-off through RL. However, due to complex interactions between the numerous rewards they introduced, they encountered stability issues, reward hacking and had to rely on two different stages of RL training, using only outcome rewards at first before adding process rewards. On the other hand, Hibiki-Zero uses a single and straightforward reward system based on BLEU score (Papineni et al., 2002) coupled with GRPO (Shao et al., 2024) without KL regularization as previously done by Rastogi et al. (2025) to reduce memory requirements during training. Most importantly, it does not rely on any human interpretation or annotated data to finetune the model before reinforcement. On multilingual simultaneous ST tasks, Hibiki-Zero achieves state-of-the-art translation quality, latency, naturalness and speaker identity preservation. Hibiki-Zero is even able to adapt to a new input language after a light finetuning.

## 3 Method

We consider an utterance in a source language represented as a monophonic waveform , sampled at a frame rate , of duration . Similarly, its translation is given in a target language, denoted . We assume is padded to ensure both have the same duration. Our objective is to model . Contrary to Labiausse et al. (2025), we do not constrain the modeling of knowing to be entirely causal in our training data. Thanks to the diversity of causality and latency arrangements in the dataset, it is still possible to learn a base translation model. Its behavior is then adjusted by an online reinforcement learning strategy that rewards correct and simultaneous translations.

### 3.1 Modeling

We build on the framework introduced by Défossez et al. (2024) for the joint modeling of multiple sequences of tokens and used by Labiausse et al. (2025) to perform simultaneous S2TT and S2ST with high fidelity.

#### 3.1.1 Neural audio codec

We use the pre-trained causal and streaming Mimi codec (Défossez et al., 2024) to encode and into low framerate sequences of discrete tokens. Mimi consists of an encoder and decoder from and to the waveform domain, and of an information bottleneck using Residual Vector Quantization (RVQ) (Zeghidour et al., 2022).

For language modeling, we are interested in the discrete
indices of codebook entries which Mimi latents are projected to. We denote those where is the codec framerate, is the number of audio residual quantization levels varying up to 32 and the codebooks size. Following Zhang et al. (2024b); Défossez et al. (2024), the output
of the first quantization level is trained to replicate semantic information
obtained from a WavLM self-supervised audio model (Chen et al., 2022). We refer to as *semantic* tokens, and as *acoustic* tokens with the latter arranged in a coarse to fine manner. We keep only acoustic levels which is sufficient to ensure high quality speech.

#### 3.1.2 Joint modeling of discrete audio tokens

Following Yang et al. (2023); Labiausse et al. (2025), we leverage a RQ-Transformer (Lee et al., 2022b) as shown in Figure 1 to model both over the time and quantizer axes as audio streams cannot be reasonably merged into a single discrete sequence.
It consists in a large *Temporal* Transformer (Vaswani et al., 2017) of latent dimension , operating at the same framerate as the codec, and being fed all the tokens generated so far, e.g.
for all ,

| (1) |

is defined as a deterministic token indicating the start of the generation.
Then, a smaller scale *Depth* Transformer models auto-regressively
the tokens over the quantizer axis, e.g. for all and ,

| (2) |

with also a special token, and with the goal of having,

Following (Copet et al., 2023; Défossez et al., 2024), we introduce an acoustic delay shifting acoustic tokens of 2 time steps in the future compared to the semantic stream. The streams are realigned before decoding the audio with the codec. As this delay is always applied, we don’t introduce new notations for readability and refer to directly.

#### 3.1.3 Translation as multistream modeling

Using the RQ-Transformer given by Eq. (1) and (2) to jointly model multiple discrete streams of tokens, we can perform the task of joint simultaneous S2TT and S2ST as illustrated in Figure 2. Following (Défossez et al., 2024), we use an Inner Monologue by introducing a stream of padded text tokens whose content is the aligned text transcription of the audio modeled in . This text stream is concatenated with the audio tokens from the source interpretation along the -axis such that it comes before the semantic level. Then, we concatenate the target tokens and source tokens along the -axis. At inference time, predictions of tokens are skipped and actual tokens of the input audio are used instead.

#### 3.1.4 Architectural details

At time-step , tokens from the previous step, e.g. ,
, and , are fed into dedicated embedding tables and contributions are summed with a BOS token used for the first time step .
The RQ-Transformer uses standard Transformer layers (Vaswani et al., 2017), with gated SiLU activation (Shazeer, 2020; Hendrycks and Gimpel, 2016).
A linear layer maps output of the *Temporal* Transformer to logits for the text token .
The *Depth* Transformer then operates for steps to estimate the logits for the output stream and for additional steps for the input stream.
Each depth step takes as input summed with a learned embedding of the previous audio token , or for .
We provide architectural hyper-parameters in Section 4.1.

### 3.2 Coarse alignment of speech translation data

We have assumed training pairs to not be entirely causal at the interpretation level. We now detail the specific method used to build such coarse translation alignments.

#### 3.2.1 Sentence-level alignment

We start from an unaligned speech translation pair which only verifies a sentence mapping constraint meaning that both and contain the same number of sentences and such that the sentence in is a translation of the sentence in . Inspired by Labiausse et al. (2025), we rely on the insertion of artificial silence in to delay its content with respect to . For each sentence of index , we introduce silence in to shift its sentence by an amount after the start of the sentence in where is sampled independently for each sentence, is the duration of the sentence in and is an hyperparameter. Then, using punctuation characters such as commas or colons in a precomputed transcript of , we insert silences whose durations follow at the corresponding instants in with being a hyperparameter.

#### 3.2.2 Natural pauses TTS

Using the method described in Section 3.2.1, we might break the natural flow of speech by inserting silence on punctuations which is also subject to imprecisions of the transcript timestamps. Following Zeghidour et al. (2025), we train a TTS with synced audio and text streams as output, providing a control on the emission timestamp of each word to synthesize. Moreover, we train the TTS to perform voice transfer from a short audio conditioning of maximum 10 seconds. We can then generate an audio using the original transcript of and naturally insert the pauses described in 3.2.1 while conditioned on the speaker from . This results in new training pairs (, ) where targets contain smoother transitions between speech and silences than .

### 3.3 Translation policy reinforcement

Assuming that we dispose of a simultaneous translation model as presented in Section 3.1, we now introduce a reinforcement learning procedure using process rewards based on BLEU scores to improve the translation policy of the model as illustrated in Figure 3. We adapt GRPO from Shao et al. (2024) to be our RL algorithm. We denote by the translation model to optimize and an older version of it acting as a regularizer. Given an input speech utterance with a known sentence-level text translation , we use to generate different speech translations , each of duration seconds where is the model frame rate and a fixed number of frames.

#### 3.3.1 Process rewards

Let be the number of sentences in and the frame indexes such that the sentence of index in starts at frame and ends at frame . We introduce as the sentence index at frame in i.e. for and for . For a frame index , we denote as the text concatenation of translated input sentences until the one of index included. Given a generation , we define as the partial text transcript until frame given by the model’s output text stream. We now introduce the hyperparameter and define the process reward for generation at frame as:

| (3) |

#### 3.3.2 Optimization objective

Using the modeling of and as tokens and , …, , we define the probability ratios between and for each output , codebook index and frame index as:

| (4) |

Given a set of frame indexes , we compute process rewards as defined in Section 3.3.1 for each output, namely for and . We then normalize rewards per frame index across group elements:

| (5) |

In practice, early experiments showed that using a regular frame indexes pattern along the input speech content performed better. Thus we introduce and use the end timestamp of every words in the input to set .

Then, we introduce the advantage of an output at step as the sum of normalized rewards from the following steps:

| (6) |

We compute the per-codebook objectives using the standard clipping function between and as:

| (7) |

In the end, we seek to maximize the following objective with fixed weights for each depth :

| (8) |

where denotes our input speech distribution and is a fixed version of the translation model that is replaced by every fixed number of updates .

## 4 Experiments

### 4.1 Architectural hyper-parameters

The backbone *Temporal* Transformer of Hibiki-Zero has a latent dimension of 2048 (8192 for the SiLU gating), 28 layers, 16 heads and local attention over 3000 tokens, *i.e.*, 2B parameters and a 4min context. The *Depth* Transformer has a latent dimension of 1024 (4096 for the gating), 6 layers per codebook and 16 heads. It models audio codebooks for the output stream and the same for the input stream but only at training. We reduce the size of the model before RL by distillation into a smaller one using weight sharing among the codebooks of the *Depth* Transformer. Our final model architecture contains 3B parameters.

| Short-form | Long-form | |||||||||||
| ASR | ASR | Speaker | End | ASR | ASR | Speaker | End | |||||
| BLEU () | BLEU () | COMET () | Sim. () | Offset () | LAAL () | BLEU () | BLEU () | COMET () | Sim. () | Offset () | LAAL () | |
| Seamless | ||||||||||||
| French | 33.8 | 32.8 | 76.6 | 19.1 | 2.4 | 2.8 | 27.8 | 23.9 | 33.7 | 44.4 | 3.2 | 6.2 |
| Spanish | 34.4 | 33.6 | 79.1 | 21.9 | 2.6 | 2.7 | 29.9 | 25.2 | 36.1 | 42.6 | 2.8 | 6.5 |
| Portuguese | 34.1 | 33.6 | 78.9 | 23.9 | 2.8 | 3.1 | 29.0 | 25.6 | 35.0 | 35.7 | 3.2 | 6.6 |
| German | 27.8 | 27.3 | 82.3 | 20.6 | 2.4 | 3.0 | 27.8 | 24.0 | 40.6 | 47.8 | 2.5 | 7.3 |
| Hibiki | ||||||||||||
| French | 32.4 | 31.8 | 81.5 | 35.7 | 2.5 | 3.5 | 29.5 | 26.4 | 42.0 | 52.8 | 2.6 | 6.8 |
| Hibiki-Zero | ||||||||||||
| French | 35.0 | 34.6 | 80.3 | 49.5 | 2.1 | 2.8 | 30.6 | 28.7 | 43.7 | 61.3 | 2.3 | 6.1 |
| Spanish | 33.8 | 33.9 | 80.3 | 57.0 | 2.3 | 3.1 | 32.3 | 31.5 | 42.3 | 64.6 | 2.6 | 5.6 |
| Portuguese | 33.6 | 33.6 | 78.9 | 51.4 | 2.4 | 3.0 | 33.2 | 31.3 | 42.6 | 62.1 | 2.3 | 6.3 |
| German | 28.7 | 28.6 | 82.0 | 51.5 | 1.9 | 2.8 | 29.1 | 28.3 | 42.3 | 66.0 | 2.0 | 5.9 |

### 4.2 Training protocol

We train a multilingual-to-English speech translation system through the following steps, each with a cosine learning rate schedule and AdamW (Loshchilov and Hutter, 2019), with weight decay of 0.1, and momentum parameters (0.9, 0.95).

#### 4.2.1 Text backbone initialization

We initialize the *Temporal* Transformer with Helium-1333huggingface.co/kyutai/helium-1-2b (Kyutai, 2025) weights, an open-source base text LLM with 2B parameters trained using filtered Common Crawl444commoncrawl.org data.

#### 4.2.2 Audio pretraining

Starting from the pretrained text backbone, weights of the *Depth* Transformer are added to the architecture as well as audio tokens projection layers. We perform an audio pretraining with single stream audio as done by Labiausse et al. (2025) but on multilingual speech. Our data mixture comprises approximately 12% of audio in each input language, 50% of English and less than 2% of Italian. We train for 1K steps with a batch size of 144 and a learning rate of . After this pretraining stage, we duplicate the weights of the *Depth* Transformer to allow for future multistream training.

#### 4.2.3 Coarse speech translation training

We construct a large-scale multilingual-to-English speech translation dataset comprising hours for each source language (French, Spanish, Portuguese, and German). Starting from a massive collection of multilingual audio, we extract 4 million single-speaker utterances, whose durations are between 30 and 75 seconds, and transcribe them using Whisper large-v3 (Radford et al., 2023). Transcripts are partitioned into sentences via Spacy’s core_news_sm and individually translated using MADLAD-3B (Kudugunta et al., 2023), after which we synthesize the target speech using the TTS system described in Section 3.2.2 with 10-second speaker conditioning. To ensure coarse translation alignments, we apply the silence insertion technique from Section 3.2.1 using and . Scaling our training budget following Labiausse et al. (2025), we perform gradient steps with a batch size of 96 and a learning rate of , computing the loss on both source and target streams with source noise augmentation. Finally, sequence termination is explicitly modeled by inserting a special input EOS token immediately following the source utterance and a separate EOS token in the text stream to demarcate the end of generation. Appendix Table 4 compares the performance of multilingual and monolingual models.

#### 4.2.4 Speech translation fine-tuning

We use the synthetic data generation method with natural pauses introduced in Section 3.2.2 to build a synthetic multilingual speech translation dataset of less than 200h in total. We fine-tune for 1K steps with a batch size of 16, a learning rate of and other configurations being similar to the previous phase described in 4.2.3. We then distill the model into a light copy of itself with codebooks weight sharing using the same dataset and 20K gradient updates.

#### 4.2.5 Reinforcement learning

Starting from the light fine-tuned translation model, we use data from the speech translation training introduced in Section 4.2.3 and run our reinforcement learning process as described in Section 3.3. We train with a batch size of 32, a group size of 4, learning rate of and perform 2000 updates with . Sequences of length frames are generated using a temperature of 0.8 and top-k of 250 for both text and audio streams. Process rewards are computed every input words and we set and . We use and for to balance loss between text and audio streams. The model is evaluated every updates on a valid set and we define Hibiki-Zero as the checkpoint with the best quality/latency trade-off according to objective evaluation metrics. Appendix Table 5 compares our base and fine-tuned models to Hibiki-Zero.

### 4.3 Evaluation datasets

##### Long-form data.

We build Audio-NTREX-4L, a multilingual long-form ST dataset using text translations from the NTREX (Aepli et al., 2023) corpus. We select 300 examples for each source language and synthesize them using the following high-quality TTS from the industry: ElevenLabs555elevenlabs.io/text-to-speech (“eleven-multilingual-v2 TTS”), Cartesia666cartesia.ai/sonic (“sonic-v2 TTS”) and Gradium777gradium.ai/#models (“default TTS”). We condition generations using voices from the multilingual dataset CML-TTS (de Oliveira et al., 2023). Audio-NTREX-4L contains around 15h of speech per TTS with an average duration of 45 seconds per sample and is split in balanced valid and test sets.

##### Short-form data.

We filter data from Europarl-ST (Iranzo-Sánchez et al., 2020) and retain samples with realistic transcripts and duration between 2 and 20 seconds. We build valid and test sets, each with 1024 samples per source language for a total of 10h hours per set.

### 4.4 Evaluation metrics

##### Translation quality.

We evaluate translation quality by transcribing generated speech using Whisper medium (Radford et al., 2023) and computing BLEU (Post, 2018) and COMET (Rei et al., 2020) scores with respect to a reference translation, referred to as ASR-BLEU and ASR-COMET. To reduce the impact of ASR errors, hypothesis and reference texts are normalized888github.com/openai/whisper/blob/main/whisper/normalizers before computing BLEU scores. Since Seamless and Hibiki-Zero perform speech-to-text translation in parallel, we also compute BLEU and COMET scores using their text outputs. We use the XCOMET-XL model.999github.com/Unbabel/COMET.

##### Translation Latency.

We rely on two common latency metrics known as End Offset and LAAL (Length-Adaptive Average Lagging). End Offset is defined as the time difference (in seconds) between the end of the last generated word and the end of the last word from the source. We compute LAAL following the method described by Papi et al. (2022) which defines it as an approximation of the average time (in seconds) between a source word and its translation. We use word-level emission timestamps produced by Whisper for words in the generated speech. We define where is the duration of the source speech and the number of words in the reference translation. The LAAL score is then computed as where .

| Input | Model | Audio | Speaker | Speech |
|---|---|---|---|---|
| language | Quality | Similarity | Naturalness | |
| French | Seamless | 11.4 3.1 | 21.1 4.9 | 21.2 3.8 |
| Hibiki | 62.9 4.8 | 44.7 5.1 | 57.0 4.2 | |
| Hibiki-Zero | 64.5 4.2 | 70.0 5.1 | 67.2 4.1 | |
| Spanish | Seamless | 10.7 2.6 | 21.2 4.5 | 26.5 4.4 |
| Hibiki-Zero | 66.8 3.9 | 69.0 3.9 | 66.2 4.9 | |
| Portuguese | Seamless | 11.8 3.1 | 32.5 6.0 | 22.8 3.9 |
| Hibiki-Zero | 62.0 4.1 | 60.7 4.2 | 75.6 3.4 | |
| German | Seamless | 15.6 2.7 | 25.2 4.9 | 26.4 4.8 |
| Hibiki-Zero | 73.5 3.4 | 65.3 4.3 | 69.9 3.9 |

| BLEU | ASR | Speaker | End | LAAL | |
|---|---|---|---|---|---|
| () | BLEU () | Sim. () | Offset () | () | |
| Seamless | 32.5 | 32.0 | 22.2 | 3.0 | 3.5 |
| Ours | |||||
| Base | 14.3 | 14.3 | 50.6 | 3.9 | 4.3 |
| Finetuned | 31.4 | 31.0 | 55.2 | 3.7 | 4.5 |
| Finetuned + RL | 32.1 | 31.9 | 54.2 | 3.0 | 3.5 |

##### Cross-lingual speaker similarity.

For objective voice transfer evaluation, we use a standard model for speaker verification101010github.com/microsoft/UniSpeech (“WavLM Large”) based on WavLM (Chen et al., 2022) and report the cosine similarity between the embeddings of the source and the generated speech.

##### Audio quality and naturalness.

We rely on human raters to evaluate audio quality, speech naturalness and additional cross-lingual speaker similarity of generated audios. We conduct evaluations per input language using 50 samples and 20 raters for each model with 5 comparisons per rater.

### 4.5 Inference configuration

We encode audio with the streaming codec and feed the tokens to Hibiki-Zero while decoding the output tokens to obtain a streaming translation. At the end of the input, we force EOS tokens to our model input audio streams, and keep sampling until it produces its own text stream EOS. We use temperature of 0.8 and top-k of 250 for all tokens.

### 4.6 Results

##### Objective evaluations.

Table 1 compares Hibiki-Zero against the best available baselines for simultaneous S2ST namely Seamless (Barrault et al., 2023) and Hibiki (Labiausse et al., 2025) with the latter only supporting French as input. Our model outperforms both baselines on long-form speech translation with more than 5pts of ASR BLEU, 20pts of speaker similarity and lower latency compared to Seamless. In the short-form setting, our approach outperforms Hibiki by 3pts of ASR BLEU while being faster and is on par with Seamless on the quality/latency trade-off but surpasses it on speaker similarity by more than 30pts.

##### Audio fidelity and speech expressivity.

Human evaluations reported in Table 2 confirm the clear advantage of Hibiki-Zero compared to Seamless on speaker identity transfer but also show that it produces higher quality audio with better speech naturalness. Compared to Hibiki on a French-to-English task, our model reaches equivalent audio quality while being more natural with a better speaker similarity.

##### New language adaptation.

Following our method from Section 4.2.3, we build a small coarse-aligned Italian-to-English ST dataset containing less than 1000 hours in each language. Starting from the base translation model obtained after the training stage described in Section 4.2.3, we fine-tune and apply our RL method for the Italian-to-English translation task only. Results are presented in Table 3 and show that we attain the same translation quality/latency trade-off as Seamless with better speaker similarity on an extension to Italian of our short-form evaluation data. As shown in Appendix Table 6, our model adapted to Italian also retains most of its capabilities on the original languages.

### 4.7 Ablations

We present ablation results in figures 4, 6 and 7 using exponential moving average smoothing for readability. Performance during RL is represented using BLEU and text LAAL metrics. They are computed every updates on a validation set using the output text stream of the model which is synchronized with the output audio. As observed by Labiausse et al. (2025), we also notice very high BLEU scores (around 60) compared to evaluation scores (around 30). Indeed, our train and valid sets were obtained with the same data generation process described in Section 4.2.3 thus following the same translation style as MADLAD-3B (Kudugunta et al., 2023) that our ST models learn to replicate.

##### Ablation: Quality/Latency control during RL.

We benchmark the effect of parameter introduced in Section 3.3.1 which balances total and intermediate BLEU scores in the definition of process rewards. As illustrated in Figure 4, performing RL with high values of leads to a higher translation latency but better overall translation quality as expected. On the contrary, lower values of reduce latency further at the cost of a limited quality decrease.

##### Ablation: Process rewards computation frequency.

##### Ablation: Alternative configurations.

In Figure 7, we compare alternative configurations that could be used for model development instead of our main setup referred to as Reference experiment. We keep and fixed.

Experiment (A) performs RL using the full translation of the reference input text instead of sentence-level prefixes to compute intermediate BLEU scores. This amounts to modify Equation 3 so it becomes . We observe better quality performance but at the cost of latency. According to us, this comes from intermediate BLEU scores being much noisier as translated references are too optimistic, making it harder to discriminate between sequences to optimize latency during RL.

Experiment (B) performs RL starting from a base model trained with full sentence delays between input and output speech meaning that for each sentence index using notations from Section 3.2.1. Therefore, latency is much higher when starting RL and is reduced to around 6 seconds which remains far worse than the reference experiment. We also observe this behavior in preliminary experiments where RL was unable to teach the base model to start translation of an input sentence before it ends as the base model was never trained in that manner during supervised training. This justifies the use of when building coarse alignments so RL can benefit from exploration.

Experiment (C) performs RL starting from a base model trained with coarse alignments using sentence-level silences only (). We observe a degradation both in terms of quality and latency compared to the reference experiment. The loss of quality is expected when decreasing as we don’t delay as much the output with respect to the input. The cause of higher latency is illustrated in Figure 5 where waveform B () is a speech translation where silences are located between sentences only. This results in a higher average latency than waveform A () which presents a better distribution of speech along time.

### 4.8 Limitations

This work proposes an efficient method to perform multilingual speech translation and shows promising results on new input language adaptation. However, while our model exhibits state-of-the-art speaker identity preservation, there is no way to control the intensity of the accent from the input language in the generated speech. Such control could be added by providing accent-annotated samples during supervised training and using conditioning at inference.

## 5 Conclusion

We present Hibiki-Zero, a multilingual model for simultaneous and expressive speech and text translation without requiring word-level alignment of translation data for training. Our method leverages coarse sentence-level alignments to train a base model that is further refined through Reinforcement Learning using process rewards based on BLEU score only. Hibiki-Zero outperforms the state-of-the-art across multiple languages with better quality/latency trade-offs, speaker identity transfer and speech naturalness. Moreover, we demonstrate new language adaptation with our method using less than 1000 hours of speech data. We release Hibiki-Zero weights as well as our multilingual long-form evaluation dataset to benefit the research community.

## References

- A benchmark for evaluating machine translation metrics on dialects without standard orthography. In Proceedings of the Eighth Conference on Machine Translation, WMT 2023, P. Koehn, B. Haddon, T. Kocmi, and C. Monz (Eds.), pp. 1045–1065. External Links: Cited by: §4.3.
- Seamless: multilingual expressive and streaming speech translation. CoRR abs/2312.05187. External Links: Cited by: §2.1, §4.6, Table 1, Table 1.
- WavLM: large-scale self-supervised pre-training for full stack speech processing. IEEE J. Sel. Top. Signal Process.. Cited by: §3.1.1, §4.4.
- Seed liveinterpret 2.0: end-to-end simultaneous speech-to-speech translation with your voice. CoRR abs/2507.17527. External Links: Cited by: §2.1, §2.2.
- Simple and controllable music generation. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine (Eds.), Cited by: §3.1.2.
- CML-TTS: A multilingual dataset for speech synthesis in low-resource languages. In Text, Speech, and Dialogue - 26th International Conference, TSD 2023, Pilsen, Czech Republic, September 4-6, 2023, Proceedings, pp. 188–199. External Links: Cited by: §4.3.
- Moshi: a speech-text foundation model for real-time dialogue. CoRR abs/2410.00037. External Links: Cited by: §1, Figure 1, Figure 1, §3.1.1, §3.1.1, §3.1.2, §3.1.3, §3.1.
- Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415. Cited by: §3.1.4.
- Europarl-st: A multilingual corpus for speech translation of parliamentary debates. In 2020 IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP 2020, Barcelona, Spain, May 4-8, 2020, pp. 8229–8233. External Links: Cited by: §4.3.
- Translatotron 2: high-quality direct speech-to-speech translation with voice preservation. In Proceedings of the 39th International Conference on Machine Learning, K. Chaudhuri, S. Jegelka, L. Song, C. Szepesvari, G. Niu, and S. Sabato (Eds.), Proceedings of Machine Learning Research, Vol. 162, pp. 10120–10134. Cited by: §2.1.
- Direct Speech-to-Speech Translation with a Sequence-to-Sequence Model. In Proc. Interspeech 2019, pp. 1123–1127. External Links: Cited by: §2.1.
- MADLAD-400: A multilingual and document-level large audited dataset. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine (Eds.), Cited by: §4.2.3, §4.7.
- Helium 1: a modular and multilingual llm. Note: Kyutai External Links: Cited by: §4.2.1.
- High-fidelity simultaneous speech-to-speech translation. In Forty-second International Conference on Machine Learning, ICML 2025, Vancouver, BC, Canada, July 13-19, 2025, External Links: Cited by: §1, §1, §2.1, Figure 2, Figure 2, §3.1.2, §3.1, §3.2.1, §3, §4.2.2, §4.2.3, §4.6, §4.7, Table 1, Table 1.
- Direct speech-to-speech translation with discrete units. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), S. Muresan, P. Nakov, and A. Villavicencio (Eds.), Dublin, Ireland, pp. 3327–3339. External Links: Cited by: §2.1.
- Autoregressive image generation using residual quantization. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022, pp. 11513–11522. External Links: Cited by: §3.1.2.
- Decoupled weight decay regularization. In 7th International Conference on Learning Representations, ICLR 2019, Cited by: §4.2.
- Real-time speech-to-speech translation. Note: Google Research External Links: Cited by: §2.1.
- The ATR multilingual speech-to-speech translation system. IEEE Transactions on Audio, Speech, and Language Processing. Cited by: §2.1.
- Over-generation cannot be rewarded: length-adaptive average lagging for simultaneous speech translation. CoRR abs/2206.05807. External Links: Cited by: §4.4.
- Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics, July 6-12, 2002, Philadelphia, PA, USA, pp. 311–318. External Links: Cited by: §1, §2.2.
- A call for clarity in reporting BLEU scores. In Proceedings of the Third Conference on Machine Translation: Research Papers, Brussels, Belgium, pp. 186–191. External Links: Cited by: §4.4.
- Robust speech recognition via large-scale weak supervision. In International Conference on Machine Learning, ICML 2023, A. Krause, E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, and J. Scarlett (Eds.), Proceedings of Machine Learning Research, Vol. 202, pp. 28492–28518. Cited by: §4.2.3, §4.4.
- Magistral. CoRR abs/2506.10910. External Links: Cited by: §2.2.
- COMET: A neural framework for MT evaluation. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, EMNLP 2020, pp. 2685–2702. External Links: Cited by: §4.4.
- AudioPaLM: A large language model that can speak and listen. CoRR abs/2306.12925. External Links: Cited by: §2.1.
- Proximal policy optimization algorithms. CoRR abs/1707.06347. External Links: Cited by: §2.2.
- DeepSeekMath: pushing the limits of mathematical reasoning in open language models. CoRR abs/2402.03300. External Links: Cited by: §1, §2.2, §3.3.
- Glu variants improve transformer. arXiv preprint arXiv:2002.05202. Cited by: §3.1.4.
- Attention is all you need. In Advances in Neural Information Processing Systems (NeurIPS), pp. 5998–6008. Cited by: §3.1.2, §3.1.4.
- Verbmobil: foundations of speech-to-speech translation. Springer. Cited by: §2.1.
- SeqPO-simt: sequential policy optimization for simultaneous machine translation. In Findings of the Association for Computational Linguistics, ACL 2025, Vienna, Austria, July 27 - August 1, 2025, pp. 16107–16123. External Links: Cited by: §2.2.
- Uniaudio: an audio foundation model toward universal audio generation. arXiv preprint arXiv:2310.00704. Cited by: §3.1.2.
- SimulPL: aligning human preferences in simultaneous machine translation. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025, External Links: Cited by: §2.2.
- Streaming sequence-to-sequence learning with delayed streams modeling. CoRR abs/2509.08753. External Links: Cited by: §3.2.2.
- SoundStream: an end-to-end neural audio codec. IEEE ACM Trans. Audio Speech Lang. Process. 30, pp. 495–507. External Links: Cited by: §3.1.1.
- StreamSpeech: simultaneous speech-to-speech translation with multi-task learning. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2024, Bangkok, Thailand, August 11-16, 2024, L. Ku, A. Martins, and V. Srikumar (Eds.), pp. 8964–8986. External Links: Cited by: §2.1.
- SpeechTokenizer: unified speech tokenizer for speech language models. In The Twelfth International Conference on Learning Representations, Cited by: §3.1.1.

## Appendix

| Short-form | Long-form | |||||||||
| Model | ASR | Speaker | End | ASR | Speaker | End | ||||
| BLEU () | BLEU () | Sim. () | Offset () | LAAL () | BLEU () | BLEU () | Sim. () | Offset () | LAAL () | |
| French | ||||||||||
| Base@100K | 31.8 | 31.3 | 52.1 | 3.1 | 3.6 | 28.7 | 27.7 | 67.6 | 3.4 | 6.0 |
| Base@400K | 34.5 | 34.1 | 53.1 | 3.0 | 3.7 | 31.1 | 28.8 | 67.4 | 3.4 | 6.6 |
| Base-FR@100K | 34.2 | 33.4 | 52.5 | 3.0 | 3.6 | 29.8 | 28.6 | 67.3 | 3.2 | 6.1 |
| Base-ES@100K | 5.3 | 5.1 | 39.6 | 4.4 | 5.1 | 5.9 | 5.5 | 60.1 | 7.7 | 11.2 |
| Base-PT@100K | 6.7 | 6.6 | 41.2 | 4.6 | 5.0 | 7.9 | 8.0 | 62.0 | 5.8 | 8.5 |
| Base-DE@100K | 1.6 | 1.6 | 41.6 | 4.3 | 5.6 | 2.0 | 1.7 | 64.1 | 8.3 | 10.8 |
| Spanish | ||||||||||
| Base@100K | 31.5 | 31.4 | 59.2 | 3.4 | 4.1 | 31.1 | 30.9 | 69.5 | 3.9 | 6.6 |
| Base@400K | 33.8 | 33.6 | 60.3 | 3.4 | 4.1 | 33.3 | 32.6 | 69.4 | 3.8 | 6.6 |
| Base-FR@100K | 8.9 | 8.8 | 48.2 | 3.8 | 4.1 | 11.4 | 11.4 | 63.6 | 4.6 | 7.3 |
| Base-ES@100K | 33.2 | 33.0 | 59.9 | 3.5 | 4.2 | 33.5 | 32.7 | 69.8 | 4.1 | 6.6 |
| Base-PT@100K | 22.3 | 22.2 | 50.2 | 3.5 | 4.1 | 24.5 | 24.4 | 63.8 | 4.2 | 7.0 |
| Base-DE@100K | 1.3 | 1.1 | 38.9 | 4.2 | 5.1 | 2.1 | 1.7 | 57.1 | 9.6 | 12.3 |
| Portuguese | ||||||||||
| Base@100K | 31.7 | 31.4 | 53.5 | 3.7 | 4.2 | 31.7 | 30.7 | 66.9 | 3.3 | 6.6 |
| Base@400K | 33.9 | 33.7 | 54.5 | 3.6 | 4.1 | 33.5 | 32.5 | 67.4 | 3.1 | 6.5 |
| Base-FR@100K | 2.5 | 2.4 | 43.2 | 4.2 | 4.7 | 9.0 | 8.6 | 53.6 | 4.9 | 7.8 |
| Base-ES@100K | 12.8 | 12.8 | 47.3 | 4.2 | 4.9 | 23.9 | 23.0 | 56.9 | 3.8 | 8.0 |
| Base-PT@100K | 32.5 | 32.2 | 54.4 | 3.9 | 4.4 | 32.2 | 30.8 | 67.8 | 3.4 | 7.0 |
| Base-DE@100K | 0.6 | 0.7 | 42.2 | 3.9 | 5.1 | 1.3 | 0.9 | 52.0 | 10.8 | 11.7 |
| German | ||||||||||
| Base@100K | 25.9 | 25.7 | 53.6 | 2.7 | 3.5 | 28.3 | 28.0 | 70.6 | 3.4 | 6.4 |
| Base@400K | 28.3 | 28.0 | 54.6 | 2.7 | 3.6 | 31.1 | 30.5 | 70.5 | 3.3 | 5.8 |
| Base-FR@100K | 0.7 | 0.7 | 35.6 | 4.4 | 4.6 | 1.4 | 1.2 | 54.5 | 12.1 | 10.0 |
| Base-ES@100K | 0.9 | 0.8 | 35.4 | 4.6 | 5.2 | 1.8 | 1.1 | 50.9 | 14.5 | 16.0 |
| Base-PT@100K | 0.7 | 0.7 | 33.3 | 4.1 | 4.6 | 1.6 | 1.4 | 50.2 | 8.1 | 9.6 |
| Base-DE@100K | 28.6 | 28.3 | 55.6 | 2.9 | 3.8 | 30.8 | 29.9 | 71.4 | 3.4 | 6.2 |

| Short-form | Long-form | |||||||||||
| Model | ASR | ASR | Speaker | End | ASR | ASR | Speaker | End | ||||
| BLEU () | BLEU () | COMET () | Sim. () | Offset () | LAAL () | BLEU () | BLEU () | COMET () | Sim. () | Offset () | LAAL () | |
| French | ||||||||||||
| Base | 34.4 | 34.0 | 79.2 | 53.1 | 3.0 | 3.6 | 31.1 | 29.7 | 43.1 | 67.5 | 3.5 | 6.2 |
| Finetuned | 34.3 | 33.9 | 78.7 | 52.9 | 3.5 | 4.0 | 31.0 | 29.0 | 41.6 | 66.7 | 4.8 | 6.9 |
| Hibiki-Zero | 35.0 | 34.6 | 80.3 | 49.5 | 2.1 | 2.8 | 30.6 | 28.7 | 43.7 | 61.3 | 2.3 | 6.1 |
| Spanish | ||||||||||||
| Base | 34.0 | 33.7 | 80.1 | 60.2 | 3.3 | 4.1 | 33.8 | 32.8 | 45.3 | 69.8 | 3.8 | 6.3 |
| Finetuned | 33.9 | 33.7 | 80.3 | 60.2 | 3.6 | 4.2 | 32.9 | 32.2 | 43.5 | 69.2 | 4.7 | 6.7 |
| Hibiki-Zero | 33.8 | 33.9 | 80.3 | 57.0 | 2.3 | 3.1 | 32.3 | 31.5 | 42.3 | 64.6 | 2.6 | 5.6 |
| Portuguese | ||||||||||||
| Base | 33.5 | 33.2 | 78.8 | 54.2 | 3.6 | 4.2 | 34.0 | 32.6 | 42.2 | 67.0 | 3.3 | 6.6 |
| Finetuned | 34.0 | 33.9 | 78.8 | 55.5 | 3.9 | 4.3 | 33.7 | 31.7 | 42.2 | 67.5 | 4.6 | 7.3 |
| Hibiki-Zero | 33.6 | 33.6 | 78.9 | 51.4 | 2.4 | 3.0 | 33.2 | 31.3 | 42.6 | 62.1 | 2.3 | 6.3 |
| German | ||||||||||||
| Base | 28.6 | 28.4 | 83.0 | 54.8 | 2.7 | 3.6 | 30.6 | 30.0 | 44.7 | 70.6 | 3.3 | 6.0 |
| Finetuned | 28.1 | 27.9 | 82.4 | 54.6 | 2.9 | 3.7 | 30.6 | 29.4 | 44.8 | 70.3 | 4.5 | 7.0 |
| Hibiki-Zero | 28.7 | 28.6 | 82.0 | 51.5 | 1.9 | 2.8 | 29.1 | 28.3 | 42.3 | 66.0 | 2.0 | 5.9 |

| Model | BLEU | ASR BLEU | Speaker Sim. | End Offset | LAAL |
|---|---|---|---|---|---|
| () | () | () | () | () | |
| French | |||||
| Hibiki-Zero | 30.6 | 28.7 | 61.3 | 2.3 | 6.1 |
| Italian Finetuned + RL | 30.6 | 29.1 | 59.8 | 3.0 | 6.2 |
| Spanish | |||||
| Hibiki-Zero | 32.3 | 31.5 | 64.6 | 2.6 | 5.6 |
| Italian Finetuned + RL | 31.1 | 30.3 | 62.8 | 2.6 | 6.3 |
| Portuguese | |||||
| Hibiki-Zero | 33.2 | 31.3 | 62.1 | 2.3 | 6.3 |
| Italian Finetuned + RL | 32.9 | 31.3 | 56.4 | 2.7 | 6.5 |
| German | |||||
| Hibiki-Zero | 29.1 | 28.3 | 66.0 | 2.0 | 5.9 |
| Italian Finetuned + RL | 30.5 | 28.7 | 64.2 | 2.7 | 6.4 |
