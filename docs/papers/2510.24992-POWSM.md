url: https://arxiv.org/html/2510.24992v2
title: "POWSM: A Phonetic Open Whisper-Style Speech Foundation Model"
content: |-
  Chin-Jou Li∗1, Kalvin Chang∗2, Shikhar Bharadwaj 1, Eunjung Yeo 3, Kwanghee Choi 3, 

  Jian Zhu 4, David R. Mortensen 1, Shinji Watanabe 1, 

  1 Carnegie Mellon University, 2 University of California, Berkeley, 

  3 University of Texas, Austin, 4 University of British Columbia

  ###### Abstract

  Recent advances in spoken language processing have led to substantial progress in phonetic tasks such as automatic speech recognition (ASR), phone recognition (PR), grapheme-to-phoneme conversion (G2P), and phoneme-to-grapheme conversion (P2G). Despite their conceptual similarity, these tasks have largely been studied in isolation, each relying on task-specific architectures and datasets. In this paper, we introduce POWSM (Phonetic Open Whisper-style Speech Model), the first unified framework capable of jointly performing multiple phone-related tasks. POWSM enables seamless conversion between audio, text (graphemes), and phones, opening up new possibilities for universal and low-resource speech processing. Our model outperforms or matches specialized PR models of similar size (Wav2Vec2Phoneme and ZIPA) while jointly supporting G2P, P2G, and ASR. Our training data, code 1 1 1 https://github.com/espnet and model 2 2 2 https://huggingface.co/espnet/powsm are released to foster open science.

  POWSM: A Phonetic Open Whisper-Style Speech Foundation Model

  1 Introduction
  --------------

  Figure 1: POWSM is the first phonetic foundation model that can perform four phone-related tasks: Phone Recognition (PR), Automatic Speech Recognition (ASR), audio-guided grapheme-to-phoneme conversion (G2P), and audio-guided phoneme-to-grapheme conversion (P2G). 

  Phones are the smallest units of sound in speech. Unlike graphemes, phones are shared across languages and usually represented using the International Phonetic Alphabet (IPA) (International Phonetic Association, 1999), a unified transcription standard for all languages. By providing a consistent representation of speech across languages, phone-level modeling allows fine-grained analysis and cross-lingual generalization, enabling tasks like atypical speech analysis (e.g., L2 speech Li et al. (2016); Inceoglu et al. (2023) and pathological speech Choi et al. (2025); Li et al. (2025)), endangered language documentation He et al. (2024), code-switched text-to-speech Zhou et al. (2020), and cross-lingual transfer in speech-to-text Pratap et al. (2024); Magoshi et al. (2025).

  Four key phone-related tasks underpin phonetic spoken language processing: automatic speech recognition (ASR), phone recognition (PR), grapheme-to-phoneme conversion (G2P), and phoneme-to-grapheme conversion (P2G). ASR learns implicit phonetic representations Belinkov and Glass (2017), while PR offers explicit phone-level supervision. G2P and P2G bridge orthographic and phonetic spaces. Collectively, these tasks interact through shared phonetic representations, each addressing a different aspect of the relationship between audio, phones, phonemes, and graphemes.

  Despite their conceptual similarity, these tasks have traditionally been developed in isolation, using task-specific architectures and datasets. Such systems are optimized for specific input-output mappings and cannot be easily extended to other phonetic tasks. This fragmentation has hindered the development of general-purpose models for phonetic processing, necessitating a unified phonetic foundation model that can perform multiple phone-related tasks within a single, general framework for speech processing.

  To bridge this gap, we propose POWSM, a phonetic foundation model capable of performing four core phone-related tasks — PR, ASR, audio-guided G2P, and audio-guided P2G — within one unified architecture (Figure˜1). To construct this framework, we reformulate standard ASR datasets Zhu et al. (2025) into four task-specific formats, allowing the model to learn consistent mappings across audio, phoneme, and grapheme representations. In addition, POWSM adopts an attention-based encoder-decoder (AED) architecture, following the design of large-scale speech foundation models such as Whisper (Radford et al., 2023) and OWSM (Peng et al., 2023).

  Empirically, POWSM outperforms previous PR models on both in-domain data and out-of-domain languages and achieves low-resource ASR performance comparable to web-scale multilingual foundation models. Moreover, it performs speech-grounded P2G and G2P across more than 70 languages. POWSM offers a new unified paradigm for phone-level modeling, paving the way for inclusive and globally accessible speech technologies that transcend language boundaries and resource disparities.

  To summarize, our main contributions are:

  *   •We release POWSM, a large-scale foundation model that achieves state-of-the-art PR performance, and is capable of performing multiple fundamental phone-related tasks. Our model enables seamless conversion between speech, text (graphemes/orthography), and phones. 
  *   •We thoroughly analyze POWSM to understand the interaction between multiple tasks, architecture components, and losses. 
  *   •We fully open-source all our data preparation and evaluation scripts, model checkpoint and code to foster open science. 

  2 Related Work
  --------------

  ##### Speech foundation models

  Recent speech foundation models such as Whisper (Radford et al., 2023) and OWSM (Peng et al., 2023, 2024) have driven progress in large-scale multilingual ASR and speech translation, but they do not explicitly address phoneme recognition or articulatory-level supervision. Subsequent work (Yusuyin et al., 2025; Fu et al., 2025) showed that incorporating phoneme-level objectives improves ASR for low-resource and long-tailed settings, while outputting phonemes as an intermediate benefited speech translation (gállego2025speechtotexttranslationphonemeaugmentedcot). POWSM extends this line of work by being the first open foundation model jointly trained on phone recognition and related tasks, integrating multilinguality, phonetic supervision, and multi-task scalability within one framework.

  ##### Phone recognition

  Prior work in multilingual phone recognition can broadly be categorized into (1) language-specific models (Gao et al., 2021) that rely on explicit phoneme (Xu et al., 2022) or allophone inventories (Li et al., 2020) and (2) language-agnostic approaches that aim to generalize across languages without such resources (Taguchi et al., 2023; Glocker et al., 2023; Li et al., 2021; Zhu et al., 2025). POWSM follows the latter paradigm as a fully data-driven multilingual model that learns phone representations without predefined phoneme mappings.

  WhisperPPT (Samir et al., 2025) improved Whisper (Radford et al., 2023)’s performance through data cleaning but remained limited in data coverage and task diversity. However, Whisper is trained on a closed corpus and could display harmful biases for PR which cannot be fully removed by fine-tuning. POWSM is trained from scratch on open datasets.

  ZIPA (Zhu et al., 2025) scaled PR to 17,000+ hours of data and 88 languages using a Zipformer Yao et al. (2024) encoder and noisy-student training on 4,000+ languages, achieving state-of-the-art results. To construct its training corpus, ZIPA employed a G2P system to convert large-scale ASR transcriptions into phoneme sequences, effectively repurposing ASR datasets for PR. Building on this idea, POWSM leverages both the grapheme and the G2P-generated phoneme transcriptions, reformulating them into four task-specific forms: ASR, PR, G2P, and P2G.

  G2P & P2G POWSM is the first model capable of both audio-guided G2P and audio-guided P2G. G2P conversion, sometimes called phonemization in the text-to-speech literature, can be accomplished with pronunciation dictionaries (Rudnicky, 1993), rules (Mortensen et al., 2018), WFSTs (Black and Lenzo, 2001), seq2seq models (Zhu et al., 2022), or LLMs (Qharabagh et al., 2025). Text-based G2P, however, still cannot handle phonetic variation, enforcing a one-to-one mapping between orthography and transcription. In contrast, audio-guided G2P can learn to map the different acoustic realizations of a phoneme across varieties of a language to a phone representation Route et al. (2019). In particular, Mak et al. (2025) observed a performance improvement by using audio-guided G2P versus text-based G2P alone for Cantonese. Gao et al. (2024) similarly showed that joint learning of G2P, phone recognition, and forced alignment outperform a G2P teacher model. Similarly, Sun and Richmond (2024) jointly learned G2P and TTS. Compared to G2P, P2G conversion is less studied but can be performed with a seq2seq model (Lauc, 2024) or a finetuned LLM (Ma et al., 2025).

  3 Methodology
  -------------

  ### 3.1 Data preparation

  We use IPAPack++ Zhu et al. (2025) for training. It is an open source corpus of roughly 17,000 hours of multilingual speech with paired orthographic and phonemic transcriptions. We will release all data processing scripts to make POWSM fully reproducible.

  G2P-generated transcriptions have been manually inspected and cleaned. Utterances longer than 300 phones are filtered out. IPA sequences are normalized to Unicode NFD (Canonical Decomposition); English G2P sequences are further refined with rule-based corrections to fix voice-onset time issues (see Appendix §A.1).

  To prevent IPA tokens from being confused with graphemes, sequences are split into tokens with diacritics and modifiers attached through greedy trie search of PanPhon phone entries Mortensen et al. (2016) and enclosed in slashes (e.g., \tipaencoding/p h Os@m/ →\rightarrow\tipaencoding/p h/ /O/ /s/ /@/ /m/).

  ### 3.2 Multitask data format

  Our model is trained on four tasks: PR, ASR, and audio-guided G2P and P2G. Each utterance is used once per task, with task-specific formatting as illustrated in Figure˜1, including a text prompt, language token, task token, and target output. We leave the text prompt blank (token <na>) for PR and ASR, and provide graphemes and phones as prompts for G2P and P2G. For example, for an utterance saying who is that, the G2P text prompt is "who is that", and the target output is "<eng><g2p><notimestamps> /h//u//\tipaencoding I//z//ð//æ//t/".

  ### 3.3 Training details

  POWSM adopts an attention-based encoder-decoder (AED) architecture, which flexibly models output sequences and allows the integration of additional tasks. Specifically, we follow the OWSM v3.1 architecture Peng et al. (2024), which employs an E-Branchformer encoder and a Transformer decoder, consistent with the general encoder-decoder structure of Whisper Radford et al. (2023). The model is trained from scratch using ESPnet Watanabe et al. (2018) with a hybrid CTC/attention loss as in Equation 1 Watanabe et al. (2017), where we set the ratio α ctc\alpha_{\text{ctc}} to 0.3.

  ℒ=α ctc​ℒ ctc+(1−α ctc)​ℒ attention\mathcal{L}=\alpha_{\text{ctc}}\mathcal{L}_{\text{ctc}}+(1-\alpha_{\text{ctc}})\mathcal{L}_{\text{attention}}(1)

  The encoder operates at the stride size of 40 ms. Training uses a global batch size of 256. Speech inputs are 16kHz and padded to 20 seconds. The vocabulary consists of 40k tokens, including around 6k phone tokens, language and timestamp tokens, and BPE tokens from orthography. The model has approximately 350M parameters with 9 layers for both the encoder and decoder and was trained for around 200 GPUs hours on H100s. Using a CTC loss (Graves, 2006), we align the encoder outputs with a simplified version of the phone token sequences. Unlike the decoder outputs, the phones in these sequences are stripped of break (\tipaencoding/./, \tipaencoding/*͡/) and length diacritics (\tipaencoding/e\textlengthmark/, /e\texthalflength/, /ě/) to accelerate convergence. Additional details and analyses are provided in §6.1. The decoder is an autoregressive language model conditioned on a text prompt and attends to the encoder output via cross-attention.

  4 Experimental Setup
  --------------------

  ##### Evaluation metric

  We report Phonetic Feature Error Rate (PFER), an edit distance using articulatory features from PanPhon (Mortensen et al., 2016), averaged over the number of phones and computed as in Equation 2 for PR. Each feature contributes 1 24\frac{1}{24} distance unit, while insertion and deletion cost 1 unit. The edit distance D D grows linearly with the sequence length and has no upper bound.

  PFER=1#phone​∑i=1 N D​(feat​(hyp i),feat​(ref i))\text{PFER}=\frac{1}{\text{\#phone}}\sum_{i=1}^{N}D(\text{feat}(\text{hyp}_{i}),\text{feat}(\text{ref}_{i}))(2)

  Unlike Phone Error Rate (PER), which considers only exact phone matches, or Phone Token Error Rate (PTER), which treats diacritics and modifiers as separate tokens, PFER computes the edit distance in terms of articulatory features—interpretable subphone attributes (e.g. voicing)—capturing phonetic similarity in a fine-grained fashion. Previous studies Taguchi et al. (2023); Zhu et al. (2025) define PFER as the mean articulatory feature edit distance over the evaluation set. In contrast, we normalize it by the number of phones in the reference transcription to measure the proportion of feature errors per phone.

  ##### Decoding hyperparameters

  We use a CTC weight (denoted as ctc) of 0.3 and a beam size (denoted as beam) of 3 during decoding for all reported numbers unless specified. Further details on the choice of hyperparameters are discussed in §6.1.

  ##### Evaluation datasets

  For unseen languages, we evaluate on three datasets: DoReCo Paschen et al. (2020), VoxAngeles Chodroff et al. (2024), and Tusom2021 Mortensen et al. (2021). DoReCo is a dataset of 50+ languages (with broad transcriptions) intended for documentation of small or endangered languages; we use a 45-language subset.3 3 3 We use the same DoReCo subset as Zhu et al. (2025, 2024), listed in §A.6. They removed languages mostly due to licensing issues, while others were not accessible during dataset creation.  VoxAngeles (Chodroff et al., 2024) is a postprocessed version of the UCLA Phonetics Lab Archive (Ladefoged et al., 2009) containing 95 languages. Tusom is a low-data Tangkhulic language of India not included in the training data. Tusom2021 (Mortensen et al., 2021) consists of narrow phonetic transcriptions (unlike the broad transcriptions from G2P on which POWSM was trained) of individual Tusom words. We removed the tones.

  We also test on five datasets on varieties of English: the Buckeye Corpus Pitt et al. (2005) and DoReCo South-England represent dialectal variation, while L2-ARCTIC Zhao et al. (2018), EpaDB Vidal et al. (2019), and SpeechOcean762 Zhang et al. (2021) contain L2 speakers. For L2-ARCTIC, we used the manually annotated phoneme transcriptions (which Zhu et al. (2025) termed L2-Perceived) rather than G2P dictionary-based transcriptions. The manual transcriptions reflect what the speaker actually said, whereas the dictionary-based version enforces a single pronunciation variant.4 4 4 For instance, “crayon” in American English can be pronounced as \tipaencoding/\textprimstress k⁢ræn/, \tipaencoding/\textprimstress k⁢rej.On/, or \tipaencoding/\textprimstress k⁢rej.6n/ (Vaux and Golder, 2003) (among others), but the CMU Pronouncing Dictionary (Rudnicky, 1993) only lists one. Manual inspection by a trained phonologist further showed the L2-ARCTIC transcriptions to be of extremely poor quality. For the five aforementioned datasets, we use preprocessed datasets from Zhu et al. (2025)5 5 5 https://huggingface.co/anyspeech and Koel Labs 6 6 6 texttthttps://huggingface.co/KoelLabs for better transcription quality.

  We then evaluated our model on in-domain data from IPAPack++, the dataset seen during training. We followed Zhu et al. (2025) in using LibriSpeech for English, AISHELL for Mandarin, and MLS for European languages, and additionally evaluated on IISc-MILE Tamil A et al. (2022) for Tamil and KSC Khassanov et al. (2021) for Kazakh.

  For ASR and P2G, we evaluate with FLEURS.

  See Table 1 for more details about our evaluation datasets.

  PR (In-domain)
  eng deu nld fra ita spa
  10.58 14.27 12.76 10.07 5.27 10.00
  por pol tam kaz cmn
  3.74 2.14 16.58 7.07 10.02
  PR (Out-of-domain: Unseen languages)
  DoReCo VoxA.Tusom.
  19.18 1.58 1.16
  PR (Out-of-domain: Language variation)
  Buckeye DRC-SE L2-ARC EpaDB SO762
  7.88 0.77 3.66 2.74 2.32
  ASR (FLEURS)
  afr orm aze pan tgk mkd
  0.66 0.13 2.37 1.48 1.96 2.45
  bos slv
  2.45 1.76

  Table 1: Duration of the test sets for different tasks (in hours). Abbreviated datasets (in order): VoxAngeles, Tusom2021, DoReCo South-England, L2-ARCTIC, SpeechOcean762.

  ##### Baselines

  We evaluate all PR baselines without further training with IPAPack++. See Appendix §A.2 for more details about training data and language coverage. Allosaurus Li et al. (2020, 2021) uses a phone-level CTC to train a language-agnostic model and applies language-specific allophone-to-phoneme mappings. Wav2Vec2Phoneme (Xu et al., 2022), MultIPA Taguchi et al. (2023) and Allophant Glocker et al. (2023) fine-tune XLS-R (Babu et al., 2022) with different objectives: Wav2Vec2Phoneme maps unseen phonemes using articulatory features, MultIPA leverages high-quality G2P data from seven languages, while Allophant decomposes phones into articulatory features and applies CTC losses for each. ZIPA Zhu et al. (2025) trains ZipFormer (Yao et al., 2024) from scratch on IPAPack++ using CR-CTC and also provides a variant trained with additional pseudo-labeled data (“ZIPA-CR-NS-Large”).

  For ASR, we compare POWSM with two series of models: OWSM Peng et al. (2025) and OWLS Chen et al. (2025). We select OWSM-CTC v4 because it is the best-performing model in the series, featuring an encoder-CTC architecture that supports ASR, ST, and LID. For OWLS, we include models with comparable parameter sizes.

  5 Results
  ---------

  We find that POWSM achieves comparable or better performance on PR and ASR than competitive baselines, particularly in low-resource settings; audio-G2P and P2G are discussed in §6.2. These results suggest that including PR as a pre-training task improves representation generality and reduces the data required to map acoustics to text tokens. In contrast, we find no clear evidence that ASR benefits PR under the current setup (§A.4, §A.5).

  ### 5.1 Multi-task performance

  Results on the in-domain test sets are presented in Table 2 and Table 3. We provide further discussion of G2P and P2G in §6.2.

  ##### POWSM excels at in-domain phone recognition

  From Table 2, we see that POWSM achieves the lowest average PFER in phone recognition, due to the strong language modeling capability of the decoder. We hypothesize that our English data cleaning (Appendix §A.1) may have negatively affected the PFER for Germanic languages due to a mismatch between training and test data. Nevertheless, our approach fills this gap by achieving strong performance on other languages, outperforming models trained on larger datasets.

  Table 2: PFER (↓\downarrow) on the in-domain dataset, IPAPack++. Languages not supported by Allophant are left blank. Some languages were not seen by MultiIPA. Bold indicates the best performance.

  ##### POWSM is comparable with web-scale ASR models on low-resource languages

  We hypothesize that pre-training with phone recognition benefits low-resource ASR Yusuyin et al. (2025). To choose low-resource languages, we selected languages in IPAPack++ with less than 8 hours of speech in FLEURS to serve as the test set. See §A.3 for details on the amount of data used by different models. For a fair comparison with other multilingual ASR baselines without language-specific components, we use the same decoding hyperparameters ctc=0.0, beam=1.

  As shown in Table 3, POWSM (POWSM 0.35B, ASR) is often comparable to models of similar size trained on web-scale data for ASR (OWLS 0.5B). Incorporating phones obtained from PR as text prompts (PR-P2G) significantly decreases WER, making it comparable to or even better than these models. When using gold phone labels for P2G (see analysis in §6.2), POWSM outperforms other ASR models by a large margin in most cases.

  Table 3: WER (↓\downarrow) of ASR and PR-P2G on low-resource languages. PR-P2G uses phones predicted by PR as text prompts instead of gold phones. Bold indicates the best performance, and underline indicates the second-best.

  ### 5.2 POWSM generalizes well to unseen languages

  Table 4 reports PFER on datasets with unseen languages and language variation. Results indicate that POWSM achieves strong performance across these datasets, and handles both dialectal and L2 variation effectively. Notably, our method outperforms ZIPA trained on the same data and even exceeds ZIPA trained with extra pseudo-labeled data, achieving the best results on unseen languages while performing three additional tasks. While POWSM lags behind Wav2Vec2Phoneme on socio-phonetic variations, we attribute this to its self-supervised learning with over 60k hours of speech (from wav2vec 2.0 (Baevski et al., 2020)) prior to the supervised learning stage.

  Table 4: PFER (↓\downarrow) on out-of-domain data. “DRC-SE” stands for DoReCo South-England; “L2-ARC” stands for L2-ARCTIC; “SO762” stands for SpeechOcean762. Unseen language datasets include languages not supported by Allophant; therefore, we do not report results for these datasets.

  6 Analysis
  ----------

  In this section, we analyze how POWSM works, focusing on the phonetic-aware encoder and task- and language-specific tokens, which are the defining features of the model.

  ### 6.1 Behavior of the speech encoder

  ##### The CTC encoder prefers fine-grained phones without suprasegmentals

  We observed that mixing phones and orthography as encoder targets hindered training, because the same speech input would have different encoder CTC targets for different tasks. Therefore, we used phones as encoder targets, encouraging general representations of sounds to be shared across languages.

  To determine the most effective unit for the CTC encoder, we fix the decoder vocabulary to PanPhon phones and compared four encoder targets: (1) Unicode code points vs. PanPhon, and (2) sequences with vs. without suprasegmentals (length and break marks). Unicode code points offer simplicity and a smaller vocabulary but split phones into unnatural units (e.g. \tipaencoding/p h/) and increase sequence length, while PanPhon represents each phone-diacritic combination as a unit (e.g. \tipaencoding/p h/), yielding a more natural monotonic sequence at the expense of sparsity and potential out-of-vocabulary issues. Suprasegmentals such as \tipaencoding/\textlengthmark/, though phonemic in many languages, confuse PR models (Zhu et al., 2025).

  We run small-scale experiments on a 1k-hour subset of the multi-task data (250 hours of speech repeated across four tasks). We use the validation CER of the encoder-CTC output as a proxy for training efficiency. An earlier drop indicates that the encoder is learning a useful alignment early, which improves representations fed into the decoder and accelerates overall convergence. In Figure 2, PanPhon tokenization without suprasegmentals shows the earliest drop, suggesting that alignment with decoder units aids training, while collapsing suprasegmental distinctions for CTC reduces confusion.

  Figure 2: Validation PER of encoder-CTC during training. Removing suprasegmentals from the training target of encoder accelerates convergence. 

  ##### Increased encoder weights benefit PR on out-of-domain data

  As in other encoder-decoder models Gong et al. (2023); Radford et al. (2023), we expect the encoder of POWSM to capture more general acoustic patterns, while the decoder handles language and task-specific output formats. Therefore, we investigate whether emphasizing the encoder more during different stages of model development affects performance. To balance data diversity with inference compute costs, we selected two smaller datasets from each category with distinct characteristics.

  As shown in Table 5, higher CTC decoding weights improve PR performance on out-of-domain data but degrade it on in-domain data, as expected. This echoes Zhu et al. (2025)’s finding that RNN-T (Graves and Jaitly, 2014), an encoder-only speech-to-text model with an autoregressive text prediction network, hurts generalization to unseen patterns of phones. We hypothesize that the decoder is performing implicit phonotactic language modeling and “smooths” phonetic variation towards more probabilistically likely phone patterns, as Zhu et al. (2025) described.

  Next, we examine whether focusing more on the CTC loss through training widens this gap in performance between in-domain and out-of-domain data. We find that fine-tuning with a higher CTC loss weight α ctc\alpha_{\text{ctc}} after convergence does not improve out-of-domain performance and can even degrade it. Randomly varying α ctc\alpha_{\text{ctc}} for each batch also shows no improvement. In contrast, training with a higher α ctc\alpha_{\text{ctc}} from the start benefits the out-of-domain distribution, achieving the lowest PFER on unseen languages with greedy decoding, while the PFER on in-domain data is comparatively higher. These results suggest that assigning a higher weight to the encoder during training and inference improves PR, highlighting a common trade-off between in-domain performance and generalization.

  Table 5: PFER (↓\downarrow) for different CTC weight settings. “Ft” denotes fine-tuning for 5 epochs from the checkpoint above. VoxAngeles and Tusom2021 are abbreviated. Pre-training and fine-tuning rows use ctc=0.3. All setups use beam=1. 

  Task Buckeye Example
  ASR Transcription any holidays at all they just kind of ignore
  Phonetic transcription\tipaencoding/EnihAl2deIsERAl soU DeIdZ2stkAr̃2vIgnO⁢r/
  PR 12.63\tipaencoding/ẼnihAl@deIzætOl soU ðeItS2stk h æ̃n@vIgnO⁢r/
  G2P (speech)12.71\tipaencoding/ẼnihAl@deIzætOl soU ðeItS2stk h æ̃n@vIgnO⁢r/
  G2P (both)16.38\tipaencoding/ẼnihAl@deIzætOlðeItS2stk h Ĩ nd@vIgnO⁢r/
  G2P (text prompt)23.44\tipaencoding/aIhoU\textltilde d a IzætO\textltilde ðeItSIst h Ĩ nd@vIgn\textrhookrevepsilon/

  Table 6: Comparing G2P with different available modalities with PFER (↓\downarrow). Blue for correctly capturing mispronounced parts (\tipaencoding/soU/), orange for error compared to other examples.

  ### 6.2 Inspecting Task and Language Tokens

  ##### Speech-guided G2P preserves phonetic variation; text prompts normalize it

  To better understand how POWSM integrates speech and text prompts, we analyze the relative influence of speech and text prompts in its G2P behavior. We vary the G2P conditions from purely speech-based to purely text-based, as shown in Table 6, and evaluate the model on the Buckeye dataset. When only speech is provided, the performance is comparable to the PR setting, which differs only in the task token. Adding both speech and text prompts (the standard G2P setup) leads to degraded performance, with output showing standardized pronunciations. When the model relies solely on the text prompt, performance drops sharply and pronunciations become highly standardized as expected (just as Zhu et al. (2025) reported).

  In other words, POWSM G2P responds to speech and text signals to controllably mediate between narrow and broad transcription. In the multi-task setup, this effect may be stronger because the model is trained with G2P, which could bias it toward more standardized forms.

  ##### Audio-P2G effectively handles low-resource languages

  We compare several P2G setups on the same set of low-resource languages from FLEURS, listed in Table 7. P2G significantly outperforms ASR, suggesting that it effectively leverages the provided phone context. However, since P2G uses gold phone labels, this comparison is not entirely fair. We therefore tested PR followed by P2G (PR-P2G), and found that performance improved for some languages but not for others. Error propagation does not explain this variation in performance, as PFER trends from PR differ from the observed performance drops. Yet the PFER pattern aligns closely with ASR results, suggesting that phonotactic similarity to high-resource languages may play a role.

  To test this, we run P2G with the language code set to English and post-process the output to match Cyrillic or Gurmukhi transcriptions with online conversion tools for certain languages.7 7 7 https://www.lexilogos.com for Macedonian and Tajik; https://punjabi.indiatyping.com for Panjabi. This approach often outperforms ASR and sometimes approaches P2G’s performance, indicating that P2G also relies heavily on speech input. Languages with either comparibly low or high PFER did not benefit from this transliteration approach, possibly because the model already handled them well or had not yet learned them sufficiently. This finding suggests a direction for further investigation in low-resource ASR.

  Table 7: WER (↓\downarrow) of different P2G settings on low-resource languages “Best” stands for lowest WER in Table 3 from ASR models. ∗ indicates post-processed languages.

  ##### The language token captures phonotactics

  The language identification (LID) performance of POWSM on seen languages in FLEURS reaches 92.3% accuracy, as shown in Figure 3. To see if the model implicitly learns phonotactic patterns and associates them with the language token, we evaluate PR on unseen languages by manipulating the language token at inference time. For VoxAngeles and Tusom2021, the three most frequently assigned languages are Bashkir (42.6%, 25.1%), English (30.2%, 67.7%), and Kinyarwanda (14.5%, 2.3%), which are all relatively high-resource languages in IPAPack++. Table 8 shows that assigning the detected language token yields better performance than always using English, while setting the language as unknown performs best. This indicates that the language token influences PR by shifting the output distribution toward the assigned language.

  Table 8: PFER (↓\downarrow) of PR performance with different language token. The detected language in the example is <bak>. Blue for correct, orange for error compared to other examples.

  7 Conclusion
  ------------

  We train a fully open-source phonetic speech foundation model POWSM using our scalable multi-task framework. Our model achieves state-of-the-art performance on PR while also supporting ASR across more than 70 languages. Beyond PR and ASR, the model’s ability to perform audio-guided G2P and P2G enables applications that require fine-grained linguistic analysis such as atypical speech assessment. Our analysis reveals that POWSM’s encoder benefits from phoneme-level CTC supervision and stronger encoder weighting, enhancing cross-lingual generalization. Additionally, the model demonstrates interpretable multimodal and language-aware behaviors, effectively mediating between phonetic detail and standardized phonological patterns. To conclude, POWSM not only provides a strong phone recognition foundation model for high-resource languages, but also acts as a versatile resource for unseen languages and socio-phonetic variation.

  8 Future Work
  -------------

  POWSM’s current decoder serves as a large phoneme-level phonotactic language model on which linguists could investigate hypotheses about phonetic universals (Chodroff et al., 2024; Chodroff, 2025) and phonotactics (Shim et al., 2024; Pimentel et al., 2020). In the future, we seek to adapt to socio-phonetic variation either through (unsupervised) test-time adaptation (Lin et al., 2022), in-context learning (Roll et al., 2025; Wang et al., 2024), or mechanistic interpretability (Tang et al., 2024). Furthermore, since Shim et al. (2025) found that earlier encoder layers in Whisper preserve more phonetic detail, early exiting may mitigate the decoder’s tendencies to normalize socio-phonetic variation.

  Limitations
  -----------

  POWSM has several limitations that we aim to address in future work. First, the model is neither strictly phonemic nor phonetic: its training data consist of cleaned and filtered phonemic transcriptions from multiple languages, which are not fully faithful to the phonetic or phonemic structure of the audio. Although phonemic transcriptions share similarities across languages, adding auxiliary tasks and language tokens may have reinforced language-specific biases. We also currently lack sufficient allophone-level data, which would provide more language-independent information.

  Second, the model still favors high-resource languages. Since we include a decoder for language modeling and language tokens, both of which function effectively, the model would inherently bias toward the seen distribution.

  Finally, the current AED architecture, although effective for multitasking, imposes certain engineering limitations. Inference is significantly slower than with encoder-only models, and the architecture does not easily support tone modeling, limiting its application to tonal languages.

  #### Ethics Statement

  All of our data is ethically sourced, either through permissive licensing or through proper consent. We are aware of the implicit prescriptivism and representational harms (Crawford, 2017) that normalizing socio-phonetic variation in ASR or PR models can create. This may threaten linguistic diversity instead of preserving it. We also acknowledge that accurate modeling of socio-phonetic variation can enable demographic inference, as demographics and phonetic variation are deeply intertwined (Labov, 1963). We stress that uses of POWSM must align with our vision: a future where advances in spoken language processing and NLP do not leave low-resource varieties behind.

  #### The Use of LLMs

  We acknowledge the use of large language models (LLMs) to assist with grammar correction and clarity improvements in writing this paper. All conceptual, methodological, and experimental contributions were developed independently by the authors.

  Acknowledgments
  ---------------

  This work used Bridges2 in the PSC and Delta NCSA computing systems through allocation CIS210027 from the Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support (ACCESS) program, supported by National Science Foundation grants. We appreciate the contributions of Kevin Glocker, Shih-heng Wang, Farhan Samir, and Aaricia Herygers during earlier iterations of this work. We are also grateful for feedback from David Harwath, Brendon Boldt, Sanjay Subramanian, Rudy Corona, Anya Ji, Seun Eisape, and Kayo Yin.

  References
  ----------

  *   Subword dictionary learning and segmentation techniques for automatic speech recognition in tamil and kannada. External Links: 2207.13331, Link Cited by: §4. 
  *   M. Avanzi, M. Béguelin, G. Corminboeuf, F. Diémoz, and L. A. Johnsen (2022)French (Swiss) DoReCo dataset. In Language Documentation Reference Corpus (DoReCo) 1.2, F. Seifart, L. Paschen, and M. Stave (Eds.), External Links: Link, Document Cited by: §A.6. 
  *   J. Aznar (2022)Nisvai DoReCo dataset. In Language Documentation Reference Corpus (DoReCo) 1.2, F. Seifart, L. Paschen, and M. Stave (Eds.), External Links: Link, Document Cited by: §A.6. 
  *   A. Babu, C. Wang, A. Tjandra, K. Lakhotia, Q. Xu, N. Goyal, K. Singh, P. von Platen, Y. Saraf, J. Pino, A. Baevski, A. Conneau, and M. Auli (2022)XLS-R: Self-supervised Cross-lingual Speech Representation Learning at Scale. In Interspeech 2022,  pp.2278–2282. External Links: Document, ISSN 2958-1796 Cited by: §4. 
  *   A. Baevski, Y. Zhou, A. Mohamed, and M. Auli (2020)Wav2vec 2.0: a framework for self-supervised learning of speech representations. Advances in neural information processing systems 33,  pp.12449–12460. Cited by: §5.2. 
  *   H. Bartels and M. Szczepański (2022)Lower Sorbian DoReCo dataset. In Language Documentation Reference Corpus (DoReCo) 1.2, F. Seifart, L. Paschen, and M. Stave (Eds.), External Links: Link, Document Cited by: §A.6. 
  *   Y. Belinkov and J. Glass (2017)Analyzing hidden representations in end-to-end automatic speech recognition systems. Advances in Neural Information Processing Systems 30. Cited by: §1. 
  *   A. W. Black and K. A. Lenzo (2001)Flite: a small fast run-time synthesis engine.. In SSW,  pp.204. Cited by: §2. 
  *   N. Bogomolova, D. Ganenkov, and N. N. Schiborr (2022)Tabasaran DoReCo dataset. In Language Documentation Reference Corpus (DoReCo) 1.2, F. Seifart, L. Paschen, and M. Stave (Eds.), External Links: Link, Document Cited by: §A.6. 
  *   N. Burenhult (2022)Jahai DoReCo dataset. In Language Documentation Reference Corpus (DoReCo) 1.2, F. Seifart, L. Paschen, and M. Stave (Eds.), External Links: Link, Document Cited by: §A.6. 
  *   W. Chen, J. Tian, Y. Peng, B. Yan, C. H. Yang, and S. Watanabe (2025)OWLS: scaling laws for multilingual speech recognition and translation models. In Forty-second International Conference on Machine Learning, External Links: Link Cited by: §A.3, Table 10, Table 9, §4. 
  *   E. Chodroff, B. Pažon, A. Baker, and S. Moran (2024)Phonetic segmentation of the ucla phonetics lab archive. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024),  pp.12724–12733. Cited by: §4, §8. 
  *   E. Chodroff (2025)Phonetic universals. Annual Review of Linguistics 11. Cited by: §8. 
  *   K. Choi, E. Yeo, K. Chang, S. Watanabe, and D. R. Mortensen (2025)Leveraging allophony in self-supervised speech models for atypical pronunciation assessment. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), L. Chiruzzo, A. Ritter, and L. Wang (Eds.), Albuquerque, New Mexico,  pp.2613–2628. External Links: Link, Document, ISBN 979-8-89176-189-6 Cited by: §1. 
  *   A. Y. Cobbinah (2022)Baïnounk Gubëeher DoReCo dataset. In Language Documentation Reference Corpus (DoReCo) 1.2, F. Seifart, L. Paschen, and M. Stave (Eds.), External Links: Link, Document Cited by: §A.6. 
  *   A. Cowell (2022)Arapaho DoReCo dataset. In Language Documentation Reference Corpus (DoReCo) 1.2, F. Seifart, L. Paschen, and M. Stave (Eds.), External Links: Link, Document Cited by: §A.6. 
  *   K. Crawford (2017)The trouble with bias. Note: Conference on Neural Information Processing Systems Cited by: Ethics Statement. 
  *   C. L. Däbritz, N. Kudryakova, E. Stapert, and A. Arkhipov (2022)Dolgan DoReCo dataset. In Language Documentation Reference Corpus (DoReCo) 1.2, F. Seifart, L. Paschen, and M. Stave (Eds.), External Links: Link, Document Cited by: §A.6. 
  *   C. Döhler (2022)Komnzo DoReCo dataset. In Language Documentation Reference Corpus (DoReCo) 1.2, F. Seifart, L. Paschen, and M. Stave (Eds.), External Links: Link, Document Cited by: §A.6. 
  *   D. Forker and N. N. Schiborr (2022)Sanzhi Dargwa DoReCo dataset. In Language Documentation Reference Corpus (DoReCo) 1.2, F. Seifart, L. Paschen, and M. Stave (Eds.), External Links: Link, Document Cited by: §A.6. 
  *   M. Franjieh (2022)Fanbyak DoReCo dataset. In Language Documentation Reference Corpus (DoReCo) 1.2, F. Seifart, L. Paschen, and M. Stave (Eds.), External Links: Link, Document Cited by: §A.6. 
  *   L. Fu, Y. Xin, S. Zeng, L. Fan, Y. Wu, and X. He (2025)PAC: pronunciation-aware contextualized large language model-based automatic speech recognition. arXiv preprint arXiv:2509.12647. Cited by: §2. 
  *   H. Gao, M. Hasegawa-Johnson, and C. D. Yoo (2024)G2pu: grapheme-to-phoneme transducer with speech units. In ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP),  pp.10061–10065. Cited by: §2. 
  *   H. Gao, J. Ni, Y. Zhang, K. Qian, S. Chang, and M. Hasegawa-Johnson (2021)Zero-shot cross-lingual phonetic recognition with external language embedding. In Interspeech 2021,  pp.1304–1308. External Links: Document, ISSN 2958-1796 Cited by: §2. 
  *   A. Garcia-Laguia (2022)Northern Alta DoReCo dataset. In Language Documentation Reference Corpus (DoReCo) 1.2, F. Seifart, L. Paschen, and M. Stave (Eds.), External Links: Link, Document Cited by: §A.6. 
  *   S. Gipper and J. Ballivián Torrico (2022)Yurakaré DoReCo dataset. In Language Documentation Reference Corpus (DoReCo) 1.2, F. Seifart, L. Paschen, and M. Stave (Eds.), External Links: Link, Document Cited by: §A.6. 
  *   J. Gippert (2022)Svan DoReCo dataset. In Language Documentation Reference Corpus (DoReCo) 1.2, F. Seifart, L. Paschen, and M. Stave (Eds.), External Links: Link, Document Cited by: §A.6. 
  *   K. Glocker, A. Herygers, and M. Georges (2023)Allophant: cross-lingual phoneme recognition with articulatory attributes. In Interspeech 2023,  pp.2258–2262. External Links: Document, ISSN 2958-1796 Cited by: Table 9, §2, §4. 
  *   Y. Gong, S. Khurana, L. Karlinsky, and J. Glass (2023)Whisper-at: noise-robust automatic speech recognizers are also strong general audio event taggers. In Interspeech 2023,  pp.2798–2802. External Links: Document, ISSN 2958-1796 Cited by: §6.1. 
  *   A. Graves (2006)Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks. In Proc. Int. Conf. on Machine Learning, 2006,  pp.369–376. Cited by: §3.3. 
  *   A. Graves and N. Jaitly (2014)Towards end-to-end speech recognition with recurrent neural networks. In International conference on machine learning,  pp.1764–1772. Cited by: §6.1. 
  *   R. Griscom (2022)Asimjeeg Datooga DoReCo dataset. In Language Documentation Reference Corpus (DoReCo) 1.2, F. Seifart, L. Paschen, and M. Stave (Eds.), External Links: Link, Document Cited by: §A.6. 
  *   T. Güldemann, M. Ernszt, S. Siegmund, and A. Witzlack-Makarevich (2022)N||||ng DoReCo dataset. In Language Documentation Reference Corpus (DoReCo) 1.2, F. Seifart, L. Paschen, and M. Stave (Eds.), External Links: Link, Document Cited by: §A.6. 
  *   V. Gusev, T. Klooster, B. Wagner-Nagy, and A. Arkhipov (2022)Kamas DoReCo dataset. In Language Documentation Reference Corpus (DoReCo) 1.2, F. Seifart, L. Paschen, and M. Stave (Eds.), External Links: Link, Document Cited by: §A.6. 
  *   G. Haig, M. Vollmer, and H. Thiele (2022)Northern Kurdish (Kurmanji) DoReCo dataset. In Language Documentation Reference Corpus (DoReCo) 1.2, F. Seifart, L. Paschen, and M. Stave (Eds.), External Links: Link, Document Cited by: §A.6. 
  *   I. Hartmann (2022)Hooca̧k DoReCo dataset. In Language Documentation Reference Corpus (DoReCo) 1.2, F. Seifart, L. Paschen, and M. Stave (Eds.), External Links: Link, Document Cited by: §A.6. 
  *   A. Harvey (2022)Gorwaa DoReCo dataset. In Language Documentation Reference Corpus (DoReCo) 1.2, F. Seifart, L. Paschen, and M. Stave (Eds.), External Links: Link, Document Cited by: §A.6. 
  *   K. Haude (2022)Movima DoReCo dataset. In Language Documentation Reference Corpus (DoReCo) 1.2, F. Seifart, L. Paschen, and M. Stave (Eds.), External Links: Link, Document Cited by: §A.6. 
  *   T. He, K. Choi, L. Tjuatja, N. Robinson, J. Shi, S. Watanabe, G. Neubig, D. Mortensen, and L. Levin (2024)Wav2Gloss: generating interlinear glossed text from speech. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), L. Ku, A. Martins, and V. Srikumar (Eds.), Bangkok, Thailand,  pp.568–582. External Links: Link, Document Cited by: §1. 
  *   B. Hellwig, G. Schneider-Blum, and K. B. K. Ismail (2022)Tabaq (Karko) DoReCo dataset. In Language Documentation Reference Corpus (DoReCo) 1.2, F. Seifart, L. Paschen, and M. Stave (Eds.), External Links: Link, Document Cited by: §A.6. 
  *   B. Hellwig (2022)Goemai DoReCo dataset. In Language Documentation Reference Corpus (DoReCo) 1.2, F. Seifart, L. Paschen, and M. Stave (Eds.), External Links: Link, Document Cited by: §A.6. 
  *   S. Inceoglu, W. Chen, and H. Lim (2023)Assessment of l2 intelligibility: comparing l1 listeners and automatic speech recognition. ReCALL: the Journal of EUROCALL 35 (1),  pp.89–104. Cited by: §1. 
  *   International Phonetic Association (1999)Handbook of the international phonetic association: a guide to the use of the international phonetic alphabet. Cambridge University Press. Cited by: §1. 
  *   O. Kazakevich and E. Klyachko (2022)Evenki DoReCo dataset. In Language Documentation Reference Corpus (DoReCo) 1.2, F. Seifart, L. Paschen, and M. Stave (Eds.), External Links: Link, Document Cited by: §A.6. 
  *   Y. Khassanov, S. Mussakhojayeva, A. Mirzakhmetov, A. Adiyev, M. Nurpeiissov, and H. A. Varol (2021)A crowdsourced open-source Kazakh speech corpus and initial speech recognition baseline. In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, P. Merlo, J. Tiedemann, and R. Tsarfaty (Eds.), Online,  pp.697–706. External Links: Link, Document Cited by: §4. 
  *   S. Kim (2022)Jejuan DoReCo dataset. In Language Documentation Reference Corpus (DoReCo) 1.2, F. Seifart, L. Paschen, and M. Stave (Eds.), External Links: Link, Document Cited by: §A.6. 
  *   M. Krifka (2022)Daakie DoReCo dataset. In Language Documentation Reference Corpus (DoReCo) 1.2, F. Seifart, L. Paschen, and M. Stave (Eds.), External Links: Link, Document Cited by: §A.6. 
  *   W. Labov (1963)The social motivation of a sound change. Word 19 (3),  pp.273–309. Cited by: Ethics Statement. 
  *   P. Ladefoged, B. Blankenship, R. G. Schuh, P. Jones, N. Gfroerer, E. Griffiths, L. Harrington, C. Hipp, M. Kaneko, C. Moore-Cantwell, G. Oh, K. Pfister, K. Vaughan, R. Videc, S. Weismuller, S. Weiss, J. White, S. Conlon, W. J. Lee, and R. Toribio (2009)The UCLA Phonetics Lab Archive. External Links: Link Cited by: §4. 
  *   D. Lauc (2024)PolyIPA–multilingual phoneme-to-grapheme conversion model. arXiv preprint arXiv:2412.09102. Cited by: §2. 
  *   C. Li, E. Yeo, K. Choi, P. A. Pérez-Toro, M. Someki, R. K. Das, Z. Yue, J. R. Orozco-Arroyave, E. Nöth, and D. R. Mortensen (2025)Towards inclusive asr: investigating voice conversion for dysarthric speech recognition in low-resource languages. arXiv preprint arXiv:2505.14874. Cited by: §1. 
  *   K. Li, X. Qian, and H. Meng (2016)Mispronunciation detection and diagnosis in l2 english speech using multidistribution deep neural networks. IEEE/ACM Transactions on Audio, Speech, and Language Processing 25 (1),  pp.193–207. Cited by: §1. 
  *   X. Li, S. Dalmia, J. Li, M. Lee, P. Littell, J. Yao, A. Anastasopoulos, D. R. Mortensen, G. Neubig, A. W. Black, et al. (2020)Universal phone recognition with a multilingual allophone system. In ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP),  pp.8249–8253. Cited by: Table 9, §2, §4. 
  *   X. Li, J. Li, F. Metze, and A. W. Black (2021)Hierarchical phone recognition with compositional phonetics.. In Interspeech,  pp.2461–2465. Cited by: Table 9, §2, §4. 
  *   G. Lin, S. Li, and H. Lee (2022)Listen, Adapt, Better WER: Source-free Single-utterance Test-time Adaptation for Automatic Speech Recognition. In Interspeech 2022,  pp.2198–2202. External Links: Document, ISSN 2958-1796 Cited by: §8. 
  *   T. Ma, M. Bi, S. Yusuyin, H. Huang, and Z. Ou (2025)LLM-based phoneme-to-grapheme for phoneme-based speech recognition. arXiv preprint arXiv:2506.04711. Cited by: §2. 
  *   R. Magoshi, S. Sakai, J. Lee, and T. Kawahara (2025)Multi-lingual and Zero-Shot Speech Recognition by Incorporating Classification of Language-Independent Articulatory Features. In Interspeech 2025,  pp.91–95. External Links: Document, ISSN 2958-1796 Cited by: §1. 
  *   T. S. H. Mak, K. Y. Suen, and A. Lam (2025)Speech-guided grapheme-to-phoneme conversion for cantonese text-to-speech. In Proc. Interspeech 2025,  pp.2535–2539. Cited by: §2. 
  *   A. Michaud (2022)Yongning Na DoReCo dataset. In Language Documentation Reference Corpus (DoReCo) 1.2, F. Seifart, L. Paschen, and M. Stave (Eds.), External Links: Link, Document Cited by: §A.6. 
  *   D. R. Mortensen, S. Dalmia, and P. Littell (2018)Epitran: precision g2p for many languages. In Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018), Cited by: §2. 
  *   D. R. Mortensen, J. Picone, X. Li, and K. Siminyu (2021)Tusom2021: a phonetically transcribed speech dataset from an endangered language for universal phone recognition experiments. In Proc. Interspeech 2021,  pp.3660–3664. Cited by: §4. 
  *   D. R. Mortensen, P. Littell, A. Bharadwaj, K. Goyal, C. Dyer, and L. S. Levin (2016)PanPhon: A resource for mapping IPA segments to articulatory feature vectors. In Proceedings of COLING 2016, the 26th International Conference on Computational Linguistics: Technical Papers,  pp.3475–3484. Cited by: §3.1, §4. 
  *   U. Mosel (2022)Teop DoReCo dataset. In Language Documentation Reference Corpus (DoReCo) 1.2, F. Seifart, L. Paschen, and M. Stave (Eds.), External Links: Link, Document Cited by: §A.6. 
  *   C. O’Shannessy (2022a)Light Warlpiri DoReCo dataset. In Language Documentation Reference Corpus (DoReCo) 1.2, F. Seifart, L. Paschen, and M. Stave (Eds.), External Links: Link, Document Cited by: §A.6. 
  *   C. O’Shannessy (2022b)Warlpiri DoReCo dataset. In Language Documentation Reference Corpus (DoReCo) 1.2, F. Seifart, L. Paschen, and M. Stave (Eds.), External Links: Link, Document Cited by: §A.6. 
  *   P. Ozerov (2022)Anal DoReCo dataset. In Language Documentation Reference Corpus (DoReCo) 1.2, F. Seifart, L. Paschen, and M. Stave (Eds.), External Links: Link, Document Cited by: §A.6. 
  *   L. Paschen, F. Delafontaine, C. Draxler, S. Fuchs, M. Stave, and F. Seifart (2020)Building a time-aligned cross-linguistic reference corpus from language documentation data (doreco). In Proceedings of the 12th Conference on Language Resources and Evaluation (LREC 2020), Cited by: §A.6, §4. 
  *   Y. Peng, M. Shakeel, Y. Sudo, W. Chen, J. Tian, C. Lin, and S. Watanabe (2025)OWSM v4: Improving Open Whisper-Style Speech Models via Data Scaling and Cleaning. In Interspeech 2025,  pp.2225–2229. External Links: Document, ISSN 2958-1796 Cited by: §A.4, Table 9, §4. 
  *   Y. Peng, J. Tian, W. Chen, S. Arora, B. Yan, Y. Sudo, M. Shakeel, K. Choi, J. Shi, X. Chang, J. Jung, and S. Watanabe (2024)OWSM v3.1: Better and Faster Open Whisper-Style Speech Models based on E-Branchformer. In Interspeech 2024,  pp.352–356. External Links: Document, ISSN 2958-1796 Cited by: §2, §3.3. 
  *   Y. Peng, J. Tian, B. Yan, D. Berrebbi, X. Chang, X. Li, J. Shi, S. Arora, W. Chen, R. Sharma, et al. (2023)Reproducing whisper-style training using an open-source toolkit and publicly available data. In 2023 IEEE ASRU,  pp.1–8. Cited by: §1, §2. 
  *   T. Pimentel, B. Roark, and R. Cotterell (2020)Phonotactic complexity and its trade-offs. Transactions of the Association for Computational Linguistics 8,  pp.1–18. Cited by: §8. 
  *   M. A. Pitt, K. Johnson, E. Hume, S. Kiesling, and W. Raymond (2005)The buckeye corpus of conversational speech: labeling conventions and a test of transcriber reliability. Speech Communication 45 (1),  pp.89–95. Cited by: §4. 
  *   M. Ponsonnet (2022)Dalabon DoReCo dataset. In Language Documentation Reference Corpus (DoReCo) 1.2, F. Seifart, L. Paschen, and M. Stave (Eds.), External Links: Link, Document Cited by: §A.6. 
  *   V. Pratap, A. Tjandra, B. Shi, P. Tomasello, A. Babu, S. Kundu, A. Elkahky, Z. Ni, A. Vyas, M. Fazel-Zarandi, et al. (2024)Scaling speech technology to 1,000+ languages. Journal of Machine Learning Research 25 (97),  pp.1–52. Cited by: §1. 
  *   M. F. Qharabagh, Z. Dehghanian, and H. R. Rabiee (2025)LLM-powered grapheme-to-phoneme conversion: benchmark and case study. In ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP),  pp.1–5. Cited by: §2. 
  *   J. D. Quesada, S. Skopeteas, C. Pasamonik, C. Brokmann, and F. Fischer (2022)Cabécar DoReCo dataset. In Language Documentation Reference Corpus (DoReCo) 1.2, F. Seifart, L. Paschen, and M. Stave (Eds.), External Links: Link, Document Cited by: §A.6. 
  *   A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever (2023)Robust speech recognition via large-scale weak supervision. In International conference on machine learning,  pp.28492–28518. Cited by: §A.4, §1, §2, §2, §3.3, §6.1. 
  *   S. Reiter (2022)Cashinahua DoReCo dataset. In Language Documentation Reference Corpus (DoReCo) 1.2, F. Seifart, L. Paschen, and M. Stave (Eds.), External Links: Link, Document Cited by: §A.6. 
  *   S. Riesberg (2022)Yali (Apahapsili) DoReCo dataset. In Language Documentation Reference Corpus (DoReCo) 1.2, F. Seifart, L. Paschen, and M. Stave (Eds.), External Links: Link, Document Cited by: §A.6. 
  *   H. Ring (2022)Pnar DoReCo dataset. In Language Documentation Reference Corpus (DoReCo) 1.2, F. Seifart, L. Paschen, and M. Stave (Eds.), External Links: Link, Document Cited by: §A.6. 
  *   N. Roll, C. Graham, Y. Tatsumi, K. T. Nguyen, M. Sumner, and D. Jurafsky (2025)In-context learning boosts speech recognition via human-like adaptation to speakers and language varieties. arXiv preprint arXiv:2505.14887. Cited by: §8. 
  *   F. Rose (2022)Mojeño Trinitario DoReCo dataset. In Language Documentation Reference Corpus (DoReCo) 1.2, F. Seifart, L. Paschen, and M. Stave (Eds.), External Links: Link, Document Cited by: §A.6. 
  *   J. Route, S. Hillis, I. Czeresnia Etinger, H. Zhang, and A. W. Black (2019)Multimodal, multilingual grapheme-to-phoneme conversion for low-resource languages. In Proceedings of the 2nd Workshop on Deep Learning Approaches for Low-Resource NLP (DeepLo 2019), C. Cherry, G. Durrett, G. Foster, R. Haffari, S. Khadivi, N. Peng, X. Ren, and S. Swayamdipta (Eds.), Hong Kong, China,  pp.192–201. External Links: Link, Document Cited by: §2. 
  *   A. Rudnicky (1993)The cmu pronouncing dictionary. Note: Accessed on October 2, 2025.External Links: Link Cited by: §2, footnote 4. 
  *   F. Samir, E. P. Ahn, S. Prakash, M. Soskuthy, V. Shwartz, and J. Zhu (2025)A comparative approach for auditing multilingual phonetic transcript archives. Transactions of the Association for Computational Linguistics 13,  pp.595–612. External Links: Link, Document Cited by: §2. 
  *   N. N. Schiborr (2022)English (Southern England) DoReCo dataset. In Language Documentation Reference Corpus (DoReCo) 1.2, F. Seifart, L. Paschen, and M. Stave (Eds.), External Links: Link, Document Cited by: §A.6. 
  *   S. Schnell (2022)Vera’a DoReCo dataset. In Language Documentation Reference Corpus (DoReCo) 1.2, F. Seifart, L. Paschen, and M. Stave (Eds.), External Links: Link, Document Cited by: §A.6. 
  *   F. Seifart (2022a)Bora DoReCo dataset. In Language Documentation Reference Corpus (DoReCo) 1.2, F. Seifart, L. Paschen, and M. Stave (Eds.), External Links: Link, Document Cited by: §A.6. 
  *   F. Seifart (2022b)Resígaro DoReCo dataset. In Language Documentation Reference Corpus (DoReCo) 1.2, F. Seifart, L. Paschen, and M. Stave (Eds.), External Links: Link, Document Cited by: §A.6. 
  *   R. S. Shim, K. Chang, and D. R. Mortensen (2024)Phonotactic complexity across dialects. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), N. Calzolari, M. Kan, V. Hoste, A. Lenci, S. Sakti, and N. Xue (Eds.), Torino, Italia,  pp.12734–12748. External Links: Link Cited by: §8. 
  *   R. S. Shim, D. De Cristofaro, C. M. Hu, A. Vietti, and B. Plank (2025)Languages in multilingual speech foundation models align both phonetically and semantically. arXiv preprint arXiv:2505.19606. Cited by: §8. 
  *   S. Skopeteas, V. Moisidi, N. Tsetereli, J. Lorenz, and S. Schröter (2022)Urum DoReCo dataset. In Language Documentation Reference Corpus (DoReCo) 1.2, F. Seifart, L. Paschen, and M. Stave (Eds.), External Links: Link, Document Cited by: §A.6. 
  *   S. Skopeteas (2022)Yucatec Maya DoReCo dataset. In Language Documentation Reference Corpus (DoReCo) 1.2, F. Seifart, L. Paschen, and M. Stave (Eds.), External Links: Link, Document Cited by: §A.6. 
  *   S. Sun and K. Richmond (2024)Acquiring pronunciation knowledge from transcribed speech audio via multi-task learning. arXiv preprint arXiv:2409.09891. Cited by: §2. 
  *   C. Taguchi, Y. Sakai, P. Haghani, and D. Chiang (2023)Universal automatic phonetic transcription into the international phonetic alphabet. In Interspeech 2023,  pp.2548–2552. External Links: Document, ISSN 2958-1796 Cited by: Table 9, §2, §4, §4. 
  *   T. Tang, W. Luo, H. Huang, D. Zhang, X. Wang, X. Zhao, F. Wei, and J. Wen (2024)Language-specific neurons: the key to multilingual capabilities in large language models. arXiv preprint arXiv:2402.16438. Cited by: §8. 
  *   A. Teo (2022)Sümi DoReCo dataset. In Language Documentation Reference Corpus (DoReCo) 1.2, F. Seifart, L. Paschen, and M. Stave (Eds.), External Links: Link, Document Cited by: §A.6. 
  *   N. Thieberger (2022)Nafsan (South Efate) DoReCo dataset. In Language Documentation Reference Corpus (DoReCo) 1.2, F. Seifart, L. Paschen, and M. Stave (Eds.), External Links: Link, Document Cited by: §A.6. 
  *   M. Vanhove (2022)Beja DoReCo dataset. In Language Documentation Reference Corpus (DoReCo) 1.2, F. Seifart, L. Paschen, and M. Stave (Eds.), External Links: Link, Document Cited by: §A.6. 
  *   B. Vaux and S. Golder (2003)The Harvard Dialect Survey. External Links: Link Cited by: footnote 4. 
  *   J. Vidal, L. Ferrer, and L. Brambilla (2019)EpaDB: a database for development of pronunciation assessment systems.. In INTERSPEECH,  pp.589–593. Cited by: §4. 
  *   A. Vydrina (2022)Kakabe DoReCo dataset. In Language Documentation Reference Corpus (DoReCo) 1.2, F. Seifart, L. Paschen, and M. Stave (Eds.), External Links: Link, Document Cited by: §A.6. 
  *   S. Wang, C. Yang, J. Wu, and C. Zhang (2024)Can whisper perform speech-based in-context learning?. In ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP),  pp.13421–13425. Cited by: §8. 
  *   S. Watanabe, T. Hori, S. Karita, T. Hayashi, J. Nishitoba, Y. Unno, N. E. Y. Soplin, J. Heymann, M. Wiesner, N. Chen, et al. (2018)ESPnet: end-to-end speech processing toolkit. Interspeech 2018. Cited by: §3.3. 
  *   S. Watanabe, T. Hori, S. Kim, J. R. Hershey, and T. Hayashi (2017)Hybrid ctc/attention architecture for end-to-end speech recognition. IEEE Journal of Selected Topics in Signal Processing 11 (8),  pp.1240–1253. External Links: Document Cited by: §3.3. 
  *   C. Wegener (2022)Savosavo DoReCo dataset. In Language Documentation Reference Corpus (DoReCo) 1.2, F. Seifart, L. Paschen, and M. Stave (Eds.), External Links: Link, Document Cited by: §A.6. 
  *   S. Wichmann (2022)Texistepec Popoluca DoReCo dataset. In Language Documentation Reference Corpus (DoReCo) 1.2, F. Seifart, L. Paschen, and M. Stave (Eds.), External Links: Link, Document Cited by: §A.6. 
  *   A. Witzlack-Makarevich, S. Namyalo, A. Kiriggwajjo, and Z. Molochieva (2022)Ruuli DoReCo dataset. In Language Documentation Reference Corpus (DoReCo) 1.2, F. Seifart, L. Paschen, and M. Stave (Eds.), External Links: Link, Document Cited by: §A.6. 
  *   Q. Xu, A. Baevski, and M. Auli (2022)Simple and effective zero-shot cross-lingual phoneme recognition. In Interspeech 2022,  pp.2113–2117. External Links: Document, ISSN 2958-1796 Cited by: Table 9, §2, §4. 
  *   X. Xu and B. Bai (2022)Sadu DoReCo dataset. In Language Documentation Reference Corpus (DoReCo) 1.2, F. Seifart, L. Paschen, and M. Stave (Eds.), External Links: Link, Document Cited by: §A.6. 
  *   Z. Yao, L. Guo, X. Yang, W. Kang, F. Kuang, Y. Yang, Z. Jin, L. Lin, and D. Povey (2024)Zipformer: a faster and better encoder for automatic speech recognition. International Conference on Learning Representations. Cited by: §2, §4. 
  *   S. Yusuyin, T. Ma, H. Huang, W. Zhao, and Z. Ou (2025)Whistle: data-efficient multilingual and crosslingual speech recognition via weakly phonetic supervision. IEEE Transactions on Audio, Speech and Language Processing. Cited by: §2, §5.1. 
  *   J. Zhang, Z. Zhang, Y. Wang, Z. Yan, Q. Song, Y. Huang, K. Li, D. Povey, and Y. Wang (2021)Speechocean762: an open-source non-native english speech corpus for pronunciation assessment. In Proc. Interspeech 2021,  pp.3710–3714. Cited by: §4. 
  *   G. Zhao, S. Sonsaat, A. Silpachai, I. Lucic, E. Chukharev-Hudilainen, J. Levis, and R. Gutierrez-Osuna (2018)L2-arctic: a non-native english speech corpus. In Proc. Interspeech 2018,  pp.2783–2787. Cited by: §4. 
  *   X. Zhou, X. Tian, G. Lee, R. K. Das, and H. Li (2020)End-to-end code-switching tts with cross-lingual language model. In ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Vol. ,  pp.7614–7618. External Links: Document Cited by: §1. 
  *   J. Zhu, F. Samir, E. Chodroff, and D. R. Mortensen (2025)ZIPA: a family of efficient models for multilingual phone recognition. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), W. Che, J. Nabende, E. Shutova, and M. T. Pilehvar (Eds.), Vienna, Austria,  pp.19568–19585. External Links: Link, Document, ISBN 979-8-89176-251-0 Cited by: §A.3, Table 10, Table 9, §1, §2, §2, §3.1, §4, §4, §4, §4, §6.1, §6.1, §6.2, footnote 3. 
  *   J. Zhu, C. Yang, F. Samir, and J. Islam (2024)The taste of IPA: towards open-vocabulary keyword spotting and forced alignment in any language. In Proc. NAACL, K. Duh, H. Gomez, and S. Bethard (Eds.), Mexico City, Mexico,  pp.750–772. Cited by: footnote 3. 
  *   J. Zhu, C. Zhang, and D. Jurgens (2022)ByT5 model for massively multilingual grapheme-to-phoneme conversion. In Interspeech 2022,  pp.446–450. External Links: Document, ISSN 2958-1796 Cited by: §2. 

  Appendix A Appendix
  -------------------

  ### A.1 Refining English G2P

  We observed confusion in plosive voice-onset times on unseen languages in preliminary experiments, which is likely from English G2P data. For instance, broad phonemic transcription in English typically uses /b/ to transcribe the /b/ in /bat/, but its voice onset timing is actually voiceless in Mainstream American English and is closer to [p]. To mitigate this, we apply rule-based refinements to English G2P transcriptions, adjusting plosive voicing and aspiration, lateral velarization, and vowel nasalization.

  The rules are listed below: 1) word-initial voiceless plosives (\tipaencoding/p/, /t/, /k/) are aspirated, 2) word-initial voiced plosives (\tipaencoding/b/, /d/, /g/) are voiceless, 3) lateral \tipaencoding/l/ is velarized at the end of syllables, and 4) vowel nasalization before nasal consonants.

  ### A.2 Baseline Implementation

  We provide the baselines’ training data source, number of languages covered in the data, and links to model checkpoints or repository in Table 9.

  Model Training Data Sources Language Coverage Model checkpoint / GitHub
  PR baselines
  Allosaurus VoxForge, Japanese CSJ, Hkust 12 xinjli/allosaurus
  Li et al. (2020, 2021)Tedlium, Switchboard etc
  Allophant Common Voice 10.0 34 kgnlp/allophant
  Glocker et al. (2023)
  Wav2Vec2Phoneme MLS, Common Voice,40+facebook/wav2vec2-xlsr-53-espeak-cv-ft
  Xu et al. (2022)Babel
  MultIPA Common Voice 11.0 7 ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns
  Taguchi et al. (2023)
  ZIPA IPAPack++88 lingjzhu/zipa
  Zhu et al. (2025)MMS ulab v2., VoxLingua-107 (Pseudo-label)anyspeech/zipa-large-crctc-ns-800k
  ASR baselines
  OWSM-CTC v4 OWSM v3.2, YODAS 100+espnet/owsm_ctc_v4_1B
  Peng et al. (2025)
  OWLS OWSM v3.2, YODAS 150 espnet/owls-scaling-laws-for-speech-recognition-and-translation
  Chen et al. (2025)

  Table 9: Overview of the baselines for our work.

  ### A.3 FLEURS language selection for ASR

  We first filter out languages with more than 8 hours of training data in IPAPack++ Zhu et al. (2025), keeping only those that are also present in FLEURS. Then, following the training data amounts reported in Chen et al. (2025), we further identify the 50 lowest-resource languages to exclude any that may have other substantial sources not included in IPAPack++. This process leaves us with nine languages. We finally exclude ell, as it is comparatively higher-resource and because there are already three other Balto-Slavic languages. Note that other models use strictly more data than ours—not only in terms of dataset count but also because IPAPack++ applies additional data-quality filtering. Table 10 lists the amount of ASR training data for baselines.

  Table 10: Amount of ASR training data for languages included in ASR comparison (in hours), according to Zhu et al. (2025) and Chen et al. (2025).

  Figure 3: Confusion matrix of LID on FLEURS.

  ### A.4 ASR Performance on In-Domain Data

  We run greedy decoding (ctc=0.0, beam=1) on the test sets of in-domain data. Table 11 shows that ASR performance is reasonably competitive with previous models Radford et al. (2023); Peng et al. (2025) of similar size and architecture despite being trained on much fewer data. The performance gap is smaller for languages that constitute a larger portion of the training data, and larger for mid-frequency languages. Errors often involve substitutions of phonetically similar words or phrases (e.g., “mostly rapscallions as fur” to “mostly wrapped skaggins as fer”).

  Table 11: WER (↓\downarrow) on test sets; cmn reports CER (↓\downarrow). POWSM-fix is discussed in §A.7. Whisper (Whisper-small) and OWSM (OWSM v4 small) are models of similar size and architecture, trained with 10 times more data. tam and kaz are omitted due to normalization issues; Whisper outputs numerals for cmn, inflating its CER. 

  ### A.5 Multi-tasking at Different Scales

  Multi-tasking may improve performance by tying acoustic signals to well-defined symbolic representations, yet it may distract the model if the relationships are not learned effectively. We train POWSM with different data and model scales to examine how multitask learning interacts with the setup, and use beam=1 during decoding to speed up inference.

  Table 12 shows that there is no clear trend regarding whether multitasking benefits PR performance. PR performance degrades when the model has excessive capacity relative to the available data (too little data), or when it is limited by size (too much data).

  Further evidence is needed before concluding that phoneme recognition benefits less from scaling, as we currently lack sufficient data and large model capacity to test this thoroughly. Nevertheless, the model demonstrates the ability to multitask, which represents a promising direction for future work.

  Table 12: Comparison of PFER (↓\downarrow) on different seting of tasks and data. 1 task refers to PR, 2 tasks refer to PR+ASR, and 4 tasks include PR, ASR, P2G, and G2P.

  ### A.6 DoReCo (Paschen et al., 2020)

  DoReCo consists of the following constituent datasets: (Ozerov, 2022; Riesberg, 2022; Cowell, 2022; Cobbinah, 2022; Vanhove, 2022; Seifart, 2022a; Quesada et al., 2022; Reiter, 2022; Däbritz et al., 2022; Kazakevich and Klyachko, 2022; Hellwig, 2022; Harvey, 2022; Hartmann, 2022; Burenhult, 2022; Kim, 2022; Vydrina, 2022; Gusev et al., 2022; Hellwig et al., 2022; Döhler, 2022; O’Shannessy, 2022a; Bartels and Szczepański, 2022; Haude, 2022; Ponsonnet, 2022; Aznar, 2022; Güldemann et al., 2022; Haig et al., 2022; Garcia-Laguia, 2022; Franjieh, 2022; Ring, 2022; Krifka, 2022; Seifart, 2022b; Witzlack-Makarevich et al., 2022; Xu and Bai, 2022; Forker and Schiborr, 2022; Wegener, 2022; Thieberger, 2022; Schiborr, 2022; Avanzi et al., 2022; Teo, 2022; Gippert, 2022; Bogomolova et al., 2022; Mosel, 2022; Wichmann, 2022; Rose, 2022; Griscom, 2022; Skopeteas et al., 2022; Schnell, 2022; O’Shannessy, 2022b; Michaud, 2022; Skopeteas, 2022; Gipper and Ballivián Torrico, 2022).

  ### A.7 Fixing ASR Text Normalization

  After submission, we discovered a text normalization issue specific to Librispeech that degraded its ASR performance. Continuing training with the corrected data for 20 GPUs hours on H100s mostly solved the problem (WER=14.2).

  Additionally, we trained a model from scratch with updated data, denoted as POWSM-fix, where ASR transcripts are lowercased and punctuation is removed except for apostrophes and hyphens. As shown in Table 11, ASR performance is similar or improved, while Table 13 shows comparable PR performance on out-of-domain data when decoding with ctc=0.3, beam=1, enabling faster inference. Both checkpoints are publicly available on HuggingFace for reproducibility.

  Table 13: PFER (↓\downarrow) comparison on out-of-domain data.
