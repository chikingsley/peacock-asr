# One TTS Alignment To Rule Them All

###### Abstract

Speech-to-text alignment is a critical component of neural text-to-speech (TTS) models. Autoregressive TTS models typically use an attention mechanism to learn these alignments on-line. However, these alignments tend to be brittle and often fail to generalize to long utterances and out-of-domain text, leading to missing or repeating words. Most non-autoregressive end-to-end TTS models rely on durations extracted from external sources. In this paper we leverage the alignment mechanism proposed in RAD-TTS as a generic alignment learning framework, easily applicable to a variety of neural TTS models. The framework combines forward-sum algorithm, the Viterbi algorithm, and a simple and efficient static prior. In our experiments, the alignment learning framework improves all tested TTS architectures, both autoregressive (Flowtron, Tacotron 2) and non-autoregressive (FastPitch, FastSpeech 2, RAD-TTS). Specifically, it improves alignment convergence speed of existing attention-based mechanisms, simplifies the training pipeline, and makes the models more robust to errors on long utterances. Most importantly, the framework improves the perceived speech synthesis quality, as judged by human evaluators.

Index Terms: neural speech synthesis, speech text alignments

## 1 Introduction

Neural text-to-speech (TTS) models, especially autoregressive TTS models, produce naturally sounding speech for in-domain text [1, 2, 3]. However, these models can suffer from pronunciation issues such as missing and repeated words for out-of-domain text, especially in long utterances. A typical neural TTS model consists of an encoder that maps text inputs to hidden states, a decoder that generates mel-spectograms or waveforms from the hidden states, and an alignment mechanism or a duration source that maps the encoder states to decoder inputs [1, 2, 3, 4, 5, 6, 7]. Autoregressive TTS models rely on the attention mechanism [8, 9] to align text and speech, typically using content based attention mechanism [1, 3]. Although recent works have improved alignments by using both content and location sensitive attention [2], such models still suffer from alignment problems on long utterances [6].

In contrast, parallel (non-autoregressive) TTS models factor out durations from the decoding process, thereby requiring durations as input for each token. These models generally rely on external aligners [4] like the Montreal Forced Aligner (MFA) [10], or on durations extracted from a pre-trained autoregressive model (or forced aligner) [7, 11, 5] like Tacotron 2 [2]. In addition to the dependency on external alignments, these models can suffer from poor training efficiency, require carefully engineered training schedules to prevent unstable learning, and may be difficult to extend to languages either because pre-existing aligners are either unavailable or their output does not exactly fit the desired format. Ideally, we would like the alignment to be trained end-to-end as part of the TTS model to significantly simplify the training pipeline. We would also like the alignments to converge and stabilize rapidly as the rest of the TTS pipeline is dependent on it. Most importantly the output quality should be no worse (and hopefully better) than if we were to train on alignments provided by external sources.

This work leverages the alignment framework proposed in [12] to simplify alignment learning in several TTS models. We demonstrate its ability to convert all TTS models to a simpler end-to-end pipeline with better convergence rates and improved robustness to long utterances. We improve prior work on alignments in autoregressive TTS systems [1, 2, 3] by adding a constraint that directly maximizes the likelihood of text given speech mel-spectrograms. We demonstrate that this approach can also be used to learn alignments online in parallel TTS models [7, 4, 12], again eliminating the need for external aligners or alignments obtained from a pre-trained TTS models. In addition, we further examine the effect of a simple, static alignment prior for guiding alignment attention learning [13, 12].
We demonstrate in our experiments that our framework can improve *both* autoregressive and parallel models with respect to convergence rate of speech text alignments, closeness to hand-annotated durations, and speech quality. In summary, our resultsSamples available at https://nv-adlr.github.io/one-tts-alignment show that TTS models trained with our alignment learning framework have fewer repeated and missing words during inference, improved stability on long sequence synthesis, and improved overall speech quality based on human evaluation.

## 2 Alignment Learning Framework

We extend the alignment learning approach proposed in RAD-TTS [12] to be more broadly applicable to various text to speech models especially autoregressive models. Our alignment framework is presented in Figure 1. It takes the encoded text input and aligns it to mel-spectrograms where is number of mel frames and is the text length. In this section, we introduce the alignment learning objective and its application to autoregressive and parallel models.

### 2.1 Unsupervised alignment learning objective

To learn the alignment between mel-spectrograms and text , we use the alignment learning objective proposed in RAD-TTS [12]. This objective maximizes the likelihood of text given mel-spectrograms using the forward-sum algorithm used in Hidden Markov Models (HMMs) [14]. In our formulation, we constrain the alignment between text and speech to be monotonic, in order to avoid missing or repeating tokens. The following equation summarizes the likelihood of text given mels [12]:

| (1) |

where is a specific alignment between mel-spectrograms and text (eg: ), is the set of all possible valid monotonic alignments, is the likelihood of a specific text token aligned for mel frame at timestep . It is important to note that the above formulation of the alignment learning objective does not depend on how the likelihood is obtained. Hence, it can be applied to both autoregressive and parallel models. We define the forward sum objective that maximizes (1) as . Following RAD-TTS, we use an efficient, off-shelf CTC [15] implementation to compute this objective (details in appendix of RAD-TTS [12]).

### 2.2 Autoregressive TTS Models

Autoregressive TTS models typically use a sequential formulation of attention to learn online alignments. TTS models such as Tacotron [1] and Flowtron [3] use a content based attention mechanism that relies only on decoder inputs and the current attention hidden state to compute an attention map between encoder and decoder steps. Other autoregressive models use a location relative attention mechanism [16] to promote forward movement of alignments [2]. Although alignment learning in these autoregressive models is tightly coupled with the decoder and can be learned with the mel-spectrogram reconstruction objective, it has been observed that the likelihood of a misstep in the alignment increases with the length of the utterance. This results in catastrophic failure on long sequences and out-of-domain text [17]. The application of the unsupervised objective described in Sec 2.1 improves both convergence speed during training and robustness during inference.

Our autoregressive setup uses the standard stateful content based attention mechanism for Flowtron [3] and a hybrid attention mechanism that uses both content and location based features for Tacotron2 [2]. The location sensitive term (Eq. 4) uses features computed from attention weights at previous decoder timesteps. We use a Tacotron2 encoder to obtain the sequence of encoded text representations and an attention RNN to produce a sequence of states . A simple architecture is used to compute the alignment energies for text token at timestep for mel using the tanh attention [9]. The attention weights are computed with softmax over the text domain using the alignment energies. The following equations summarize the attention mechanism:

| (2) |

| (3) |

| (4) |

| (5) |

| (6) |

where is the location relative term for location sensitive attention (cumulative attention from [2] using a concatenation of the attention weights from the previous timestep and the cumulative attention weights). The attention weights model the distribution , which is exactly the right-most term in Equation (1), and we incorporate it as the alignment loss:

| (7) |

### 2.3 Parallel TTS Models

As parallel TTS models have durations factored out from the decoder, the alignment learning module can be decoupled from the mel decoder as a standalone aligner. This provides a lot of flexibility in choosing the architecture to formulate the distribution , where is a random variable for a text token aligned at timestep for mel frame . Similar to GlowTTS [6] and RAD-TTS [12], we compute the soft alignment distribution based on the learned pairwise affinity between all text tokens and mel frames, which is normalized with softmax across the text domain

| (8) | |||
| (9) |

We use two simple convolutional encoders from RAD-TTS [12] for encoding text as and mel-spectograms as with 2 and 3 1D convolution layers respectively. In Section 3, we demonstrate that the same architecture works well with different parallel TTS models such as FastPitch and FastSpeech 2. Parallel models require alignments to be specified beforehand, typically in the form of the number of output samples for every input phoneme, equivalent to a binary alignment map. However, attention models produce soft alignment maps, constituting a train-test domain gap. Following [12, 6], we use the Viterbi algorithm to find the most likely monotonic path through the soft alignment map in order to convert soft alignments () to hard alignments (). We further close the gap between soft and hard alignments by forcing to match as much as possible by minimizing their KL-divergence. This is used in both Glow-TTS and RAD-TTS, formulated as :

| (10) |

| (11) |

where is Hadamard product, is final alignment loss.

### 2.4 Alignment Acceleration

Faster convergence of alignments means faster training for the full TTS model, as the decoder needs a stable alignment representation to build upon. During training, since the length of mel-spectrograms is known, we use a static 2D prior [12], that is wider near the center and narrower near the corners to accelerate the alignment. This idea has been previously explored by Tachibana et al [18] where they introduce a new loss promoting near-diagonal alignments. Although our formulation with the 2D static prior is slightly different than Tachibana et al [18], but we believe both should yield similar results. The 2D prior substantially accelerates the alignment learning by making far-off-diagonal elements less probable, although other priors can also be used for this goal. We apply this prior over the alignment to obtain the following posterior:

| (12) |

| (13) |

for , where , are hyperparameters of beta function , is number of tokens and is scaling factor controlling width of prior: lower the , wider the width.

## 3 Experiments

We evaluate the effectiveness of the alignment learning framework by comparing its performance in terms of convergence speed, distance from human annotated ground truth durations, and speech quality. For autoregressive models like Flowtron and Tacotron 2, we compare with the baseline alignment methods therein. For FastPitch, we compare with an alignment method that relies on an external TTS model (Tacotron2) to obtain token durations. For the parallel models: FastSpeech 2 and RAD-TTS, we compare against an alignment method that obtains durations from the MFA aligner. We use the LJ Speech dataset (LJ) [19] for all our experiments.

### 3.1 Convergence Rate

In order to compare the convergence rate of different alignment methods, we use the mean mel-cepstral distance (MCD) [20, 17]. MCD compares the distance between synthesized and ground truth mel-spectrograms aligned temporarily with dynamic time warping (DTW). We observe in Figure 2 that using the static prior described in Section 2.4 significantly improves the convergence rate of Tacotron2. Parallel models such as RAD-TTS, FastPitch, and FastSpeech2 with the alignment framework (no dependency on external aligners) converge at the same rate as their baseline models using a forced aligner. The model that benefits the most from using the alignment framework is Flowtron. It has two autoregressive flows running in opposing directions, each with their own learned alignment. Notably, the second autoregressive flow is performed on top of the autoregressive outputs of the previous flow. This means that if the alignment in the first flow fails, so will the second. Training is very slow as the second flow can only be added after the first has converged. Prior attempts to train both flows simultaneously have resulted in poor minima where neither flow has learned to align. By using just the attention prior, we are now able to train at least two flows simultaneously, with further improvements with adding the unsupervised alignment learning objective described in Section 2.1. This significantly reduces training time and improves convergence of Flowtron.

### 3.2 Alignment Sharpness

We visually inspect alignment matrices for a specific validation sample in Figure 3. The alignment objective consistently makes the attention distribution sharper with more connected alignment paths. This suggests that models with produce more confident and continuous alignments, and by extension, continuous speech without repeating or missing words.

### 3.3 Duration Analysis

In order to observe the influence of the unsupervised alignment loss on the quality of alignments, we compare phoneme durations extracted from model alignments to manually annotated phoneme durations from 10 samples of the LJ test set. For autoregressive models, we extract the binarized alignments from soft alignments using a monotonic argmax, iterating through phonemes and identifying the phoneme with maximum attention weights among the current and next phonemes. We use this binarized alignment to extract durations for each phoneme. Figure 4 shows the average distance between durations extracted from the models with respect to ground truth annotated durations. By using our alignment framework we obtain a faster convergence rate than the baseline and alignments closer to the ground truth.

### 3.4 Pairwise Opinion Scores

We crowd-sourced pairwise preference scores to subjectively compare models trained with our alignment learning framework against baseline. Listeners were pre-screened with a hearing test based on sinusoid counting. During the pairwise ranking, raters were repeatedly given two synthesized utterances of the same text, picked at random from 100 LJ test samples. Both were synthesized with the same architecture: one being the baseline, and other using our alignment framework. The listeners were shown the text and asked to select samples with the best overall quality, defined by accuracy of text, its pleasantness and naturalness. Approximately 200 scores per model were collected. Table 1 shows pairwise preference scores of models trained with alignment framework over baseline. It shows that the alignment framework consistently improves over all baselines.

| Model | Alignment Framework vs Baseline |
|---|---|
| Tacotron 2 | |
| Flowtron | |
| RAD-TTS | |
| FastPitch | |
| FastSpeech2 |

### 3.5 Robustness to Errors on Long Utterances

We measure character error rate (CER) between synthesized and input texts using an external speech recognition model to evaluate the robustness of the alignments on long utterances. We use full sentences from the LibriTTS dataset [21]. We synthesize speech with models trained on LJ Speech, and recognize it with Jasper [22]. Figure 5 shows that autoregressive models with have a lower CER, providing evidence that the alignment objective results in more robust speech for long utterances. Parallel models such as RAD-TTS use a duration predictor and do not suffer from alignment issues, and hence have a much lower CER than autoregressive models.

## 4 Conclusion

We present an alignment framework that is broadly applicable to various TTS architectures, both autoregressive and parallel. By combining proper guidance in the form of forward-sum, Viterbi and diagonal priors, attention-based online alignment learning can be made stable and fast-converging. The alignment learning framework eliminates the need for forced aligners which are expensive to use and often not readily available for certain languages. Our experiments demonstrate improvements in overall speech quality based on human pairwise comparisons, reduced alignment failures, faster convergence, as well as robustness to errors in synthesis of long text sequences.

## References

-
[1]
Y. Wang, R. J. Skerry-Ryan, D. Stanton, Y. Wu, R. J. Weiss, N. Jaitly,
Z. Yang, Y. Xiao, Z. Chen, S. Bengio, Q. V. Le, Y. Agiomyrgiannakis,
R. Clark, and R. A. Saurous, “Tacotron: A fully end-to-end text-to-speech
synthesis model,”
*CoRR*, vol. abs/1703.10135, 2017. [Online]. Available: http://arxiv.org/abs/1703.10135 -
[2]
J. Shen, R. Pang, R. J. Weiss, M. Schuster, N. Jaitly, Z. Yang, Z. Chen,
Y. Zhang, Y. Wang, R. J. Skerry-Ryan, R. A. Saurous, Y. Agiomyrgiannakis,
and Y. Wu, “Natural TTS synthesis by conditioning wavenet on mel
spectrogram predictions,”
*CoRR*, vol. abs/1712.05884, 2017. [Online]. Available: http://arxiv.org/abs/1712.05884 - [3] R. Valle, K. Shih, R. Prenger, and B. Catanzaro, “Flowtron: an autoregressive flow-based generative network for text-to-speech synthesis,” 2020.
-
[4]
Y. Ren, C. Hu, T. Qin, S. Zhao, Z. Zhao, and T.-Y. Liu, “Fastspeech 2: Fast
and high-quality end-to-end text-to-speech,”
*arXiv preprint arXiv:2006.04558*, 2020. -
[5]
Y. Ren, Y. Ruan, X. Tan, T. Qin, S. Zhao, Z. Zhao, and T.-Y. Liu, “Fastspeech:
Fast, robust and controllable text to speech,” in
*Advances in Neural Information Processing Systems*, H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett, Eds., vol. 32. Curran Associates, Inc., 2019, pp. 3171–3180. - [6] J. Kim, S. Kim, J. Kong, and S. Yoon, “Glow-tts: A generative flow for text-to-speech via monotonic alignment search,” 2020.
- [7] A. Łańcucki, “Fastpitch: Parallel text-to-speech with pitch prediction,” 2020.
-
[8]
A. Graves, “Generating sequences with recurrent neural networks,”
*CoRR*, vol. abs/1308.0850, 2013. [Online]. Available: http://arxiv.org/abs/1308.0850 - [9] D. Bahdanau, K. Cho, and Y. Bengio, “Neural machine translation by jointly learning to align and translate,” 2014, cite arxiv:1409.0473Comment: Accepted at ICLR 2015 as oral presentation. [Online]. Available: http://arxiv.org/abs/1409.0473
-
[10]
M. McAuliffe, M. Socolof, S. Mihuc, M. Wagner, and M. Sonderegger, “Montreal
forced aligner: Trainable text-speech alignment using kaldi,” in
*INTERSPEECH*, 2017. -
[11]
K. Peng, W. Ping, Z. Song, and K. Zhao, “Non-autoregressive neural
text-to-speech,” in
*International Conference on Machine Learning*. PMLR, 2020, pp. 7586–7598. -
[12]
K. J. Shih, R. Valle, R. Badlani, A. Lancucki, W. Ping, and B. Catanzaro,
“RAD-TTS: Parallel flow-based TTS with robust alignment learning and
diverse synthesis,” in
*ICML Workshop on Invertible Neural Networks, Normalizing Flows, and Explicit Likelihood Models*, 2021. [Online]. Available: https://openreview.net/forum?id=0NQwnnwAORi -
[13]
H. Tachibana, K. Uenoyama, and S. Aihara, “Efficiently trainable
text-to-speech system based on deep convolutional networks with guided
attention,” in
*2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*. IEEE, 2018, pp. 4784–4788. -
[14]
L. R. Rabiner, “A tutorial on hidden Markov models and selected applications
in speech recognition,”
*Proceedings of the IEEE*, vol. 77, no. 2, pp. 257–286, Feb. 1989. - [15] A. Graves, S. Fernández, F. Gomez, and J. Schmidhuber, “Connectionist temporal classification: Labelling unsegmented sequence data with recurrent neural networks,” ser. ICML ’06. New York, NY, USA: Association for Computing Machinery, 2006, p. 369–376.
-
[16]
J. Chorowski, D. Bahdanau, D. Serdyuk, K. Cho, and Y. Bengio, “Attention-based
models for speech recognition,”
*CoRR*, vol. abs/1506.07503, 2015. [Online]. Available: http://arxiv.org/abs/1506.07503 -
[17]
E. Battenberg, R. J. Skerry-Ryan, S. Mariooryad, D. Stanton, D. Kao,
M. Shannon, and T. Bagby, “Location-relative attention mechanisms for
robust long-form speech synthesis,” in
*ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2020, pp. 6194–6198. -
[18]
H. Tachibana, K. Uenoyama, and S. Aihara, “Efficiently trainable
text-to-speech system based on deep convolutional networks with guided
attention,”
*CoRR*, vol. abs/1710.08969, 2017. [Online]. Available: http://arxiv.org/abs/1710.08969 - [19] K. Ito and L. Johnson, “The lj speech dataset,” https://keithito.com/LJ-Speech-Dataset/, 2017.
-
[20]
R. Kubichek, “Mel-cepstral distance measure for objective speech quality
assessment,” in
*Proceedings of IEEE Pacific Rim Conference on Communications Computers and Signal Processing*, vol. 1, 1993, pp. 125–128 vol.1. -
[21]
H. Zen, R. Clark, R. J. Weiss, V. Dang, Y. Jia, Y. Wu, Y. Zhang, and Z. Chen,
“Libritts: A corpus derived from librispeech for text-to-speech,” in
*Interspeech*, 2019. [Online]. Available: https://arxiv.org/abs/1904.02882 - [22] J. Li, V. Lavrukhin, B. Ginsburg, R. Leary, O. Kuchaiev, J. M. Cohen, H. Nguyen, and R. T. Gadde, “Jasper: An end-to-end convolutional neural acoustic model,” 2019.
