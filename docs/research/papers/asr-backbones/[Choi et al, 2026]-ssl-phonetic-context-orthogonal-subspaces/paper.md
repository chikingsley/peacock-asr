---
arxiv: 2603.12642
title: "Self-Supervised Speech Models Encode Phonetic Context via Position-dependent Orthogonal Subspaces"
authors:
  - "Kwanghee Choi"
  - "Eunjung Yeo"
  - "Cheol Jun Cho"
  - "David R. Mortensen"
  - "David Harwath"
citation_author: "Choi et al"
year: 2026
venue: "arXiv preprint"
category: asr-backbones
tags: [ssl, phonetic-context, representation-analysis, orthogonal-subspaces, interpretability, layer-analysis]
pages: 10
source_pdf: "paper.pdf"
source_latex: "latex-source/template.tex"
code_url: "https://github.com/juice500ml/phonetic-arithmetic"
extraction_method: "Manual rewrite against the local PDF and LaTeX source, with section-by-section citation mapping from the paper's bibliography."
extracted_at: "2026-03-23"
llm_friendly: true
---

## Metadata

- Authors: Kwanghee Choi, Eunjung Yeo, Cheol Jun Cho, David R. Mortensen, David Harwath
- Citation author: Choi et al
- Year: 2026
- Venue/status: arXiv preprint (`arXiv:2603.12642v1`), formatted with the Interspeech class
- Pages: 10
- Source PDF: `paper.pdf`
- Source LaTeX: `latex-source/template.tex`
- Code URL from the paper: `https://github.com/juice500ml/phonetic-arithmetic`

## TL;DR

This paper asks how a single frame in a transformer-based self-supervised speech model can simultaneously represent the current phone and its phonetic context. The main claim is that frame-level representations superpose phonological vectors from multiple relative phone positions, and that those position-specific components live in approximately orthogonal subspaces.

The paper supports that claim with four linked findings: center frames are already compositionally phonological, those frames encode immediate neighbors, phonological vectors for different relative positions are approximately orthogonal, and the resulting position-sensitive structure appears to shift at phonetic boundaries rather than at a fixed symmetric window. For pronunciation work, the practical value is not a new backbone but a sharper picture of what context-sensitive phonetic information later SSL layers already contain.

## Abstract

The paper extends earlier "phonological vector arithmetic" work from isolated phones to contextualized frame-level representations. Its hypothesis is that a single S3M frame can linearly combine phonological vectors for the previous, current, and next phones, rather than only the aligned phone. The authors then test whether this contextual encoding is position-sensitive, approximately orthogonal across relative positions, and aligned with phonetic boundaries.

## 1. Introduction

The introduction frames transformer-based self-supervised speech models as highly effective but still poorly understood internally. Prior work had already shown that different layers contain different mixtures of acoustic and linguistic information, that convolutional encoders are local and relative rather than spectrogram-like, that deeper transformer layers improve probing performance, and that discretized SSL units often align with phonetic categories and boundaries. The authors argue that these observations suggest the context network is doing more than smoothing local acoustics.

The paper then extends the earlier phonological-vector view of SSL speech representations. Instead of treating a frame as encoding only the aligned phone, it proposes that each frame-level representation contains a linear combination of phonological vectors for a short sequence of phones. From that hypothesis it derives four explicit predictions: frame-level compositionality, neighboring-phone encoding, positional orthogonality, and phonetic-boundary-sensitive segmentation.

Citations in this section: `[baevski2020wav2vec]`, `[hsu2021hubert]`, `[chen2022wavlm]`, `[schneider2019wav2vec]`, `[pasad2021layer]`, `[pasad2023comparative]`, `[liu2023self]`, `[meng25b_interspeech]`, `[choi2022opening]`, `[choi2024self]`, `[choi2026self]`, `[choi2025leveraging]`, `[sicherman2023analysing]`, `[wells22_interspeech]`, `[shih2024interface]`, `[chang2024exploring]`, `[de2024human]`, `[abdullah2023information]`, `[li2023dissecting]`, `[pasad2024self]`, `[baade2025syllablelm]`, `[cho2025sylber]`, `[cho2026sylber]`, `[visser2026zerosyl]`, `[choi2025device]`

## 2. Settings

### 2.1 Self-supervised Speech Models (S3Ms)

The analysis uses three English monolingual large checkpoints: wav2vec 2.0 Large LV-60, HuBERT Large LL-60k, and WavLM Large. The authors also include log-mel spectrograms and MFCCs as non-SSL baselines. For phone-level analyses, they follow prior work and pool frame representations within phone segments, comparing ordinary mean pooling with center pooling depending on the experiment.

Citations in this section: `[baevski2020wav2vec]`, `[hsu2021hubert]`, `[chen2022wavlm]`, `[choi2026self]`, `[mcfee2015librosa]`, `[pasad2021layer]`, `[pasad2023comparative]`

### 2.2 S3M Analysis via Phonological Analogies

The core probe is phonological analogy success rate. Given a quadruplet of phones that instantiates a shared phonological contrast, the method checks whether vector arithmetic in the representation space reconstructs the target phone better than a different-phone baseline and worse than a same-phone upper bound. Analogy construction is driven by `PanPhon`, and the success-rate computation follows the earlier phonological-vector paper, with 1,000 bootstrap-style random samples and 10 replications for a 99% confidence interval.

Citations in this section: `[chaabouni2017learning]`, `[zouhar2024pwesuite]`, `[choi2026self]`, `[mortensen2016panphon]`

### 2.3 Datasets

The experiments use TIMIT and VoxAngeles, both with phonetic transcriptions and manual segmentation. TIMIT supplies read English sentence speech; VoxAngeles supplies word-level read speech from 95 languages unseen during S3M training. The paper then filters phones to those with `PanPhon` mappings and later removes very rare phones for reliable analogy estimation.

One detail in the text appears to contain a typo: it says the TIMIT inventory is reduced "from 47 to 47 to 44 phones." The operative filtered counts used afterward are 44 phones for TIMIT and 468 for VoxAngeles, with 43 and 57 phones retained for the analogy experiments after frequency filtering.

Citations in this section: `[garofolo1993darpa]`, `[chodroff2024voxangeles]`, `[choi2026self]`

## 3. Experiments

### 3.1 Frame-level compositionality

This section tests whether a single center frame carries the same kind of phonological-vector structure that earlier work demonstrated on mean-pooled phone representations. The comparison is mean pooling versus center pooling over each phone segment.

The result is that center pooling is usually comparable to, and often stronger than, mean pooling across datasets and models. Both beat spectral baselines in the layers where the SSL models become most phonologically useful. The paper takes this as direct evidence that phonological compositionality is present at the frame level, not only after temporal averaging.

Citations in this section: `[pasad2021layer]`, `[pasad2023comparative]`, `[choi2026self]`, `[pasad2024self]`, `[choi2024self]`, `[choi2025leveraging]`

### 3.2 Contextual phonological vectors

This section asks whether a frame centered on the current phone also contains phonological information about neighboring phones.

#### 3.2.1 How much neighboring context does the center frame encode?

The setup fixes a five-phone context window `[p^-2, p^-1, p^0, p^+1, p^+2]` and extracts one center-pooled representation from `p^0`. That single representation is then used to probe phonological analogies at each relative phone position.

The main result is that later S3M layers support analogies not only for the current phone `p^0`, but also for immediate neighbors `p^-1` and `p^+1`. The effect is strongest in later transformer layers and weak for more distant phones, which the authors interpret as evidence that frame-level representations encode contextual phonological information instead of only the aligned phone.

#### 3.2.2 Effective window size for phonological analogies

To localize where that contextual information is strongest, the paper switches from center pooling to random pooling over frames inside each phone from `p^-2` through `p^+2`, bins frames by relative position inside each segment, and measures analogy success for the center phone `p^0`. The chosen layers are those that worked best in the previous subsection: layer 24 for HuBERT and WavLM, and layer 9 for wav2vec 2.0.

The key empirical pattern is a trapezoid, clearest in WavLM: high success inside the center phone, lower but nonzero success for immediate neighbors, and near-zero success for phones at distance 2. HuBERT shows a similar pattern on TIMIT but less clearly on VoxAngeles, while wav2vec 2.0 is weaker on both datasets. Spectral baselines only work near position 0. The section closes by connecting this to coarticulation and previewing the later mask-filling and segmentation analyses.

Direct citations in section 3.2: none in the LaTeX text itself.

### 3.3 Positional orthogonality

This section addresses the ambiguity created when current and neighboring phones share the same phonological property. The authors hypothesize that the model resolves that ambiguity by encoding phonological information for different relative positions in approximately orthogonal subspaces.

They estimate phonological vectors by difference-of-means, following the earlier phonological-vector paper, for eight features: four vowel features (`high`, `low`, `back`, `round`) and four consonantal features (`nasal`, `sonorant`, `strident`, `voicing`). The primary analysis uses center-pooled frame representations from the final WavLM-Large layer, with phonological vectors estimated on the training split.

Three results matter here. First, the within-position phonological-vector geometry looks like the earlier mean-pooled result: opposing vowel features are strongly negative, related consonantal features are positively aligned, and vowel and consonant feature groups are roughly orthogonal. Second, vectors from different relative positions have much lower cosine similarity than vectors from the same position, suggesting approximate positional orthogonality. Third, vector norms decay with distance from the center phone, following `0 > +/-1 > +/-2`, even though the more distant positions are still visible in the orthogonality analysis. The authors argue this means simple success-rate tests understate how much distant context is present.

Citations in this section: `[choi2026self]`

### 3.4 Phonetic segmentation

If relative-position subspaces are genuinely tied to "previous/current/next phone" roles, the model should track phonetic boundaries rather than a fixed symmetric time window. This section tests that by taking windows of 11 frame-level representations around phone boundaries and measuring cosine similarity to phonological vectors for previous, current, and next positions.

The strongest reported pattern is that the cosine-similarity curves for competing position-specific vectors cross very close to the annotated TIMIT phone boundaries, both at onset boundaries and at offset boundaries. The authors interpret this as evidence that the model's contextual frame representations are organized with respect to phonetic segments, not just absolute temporal distance.

Direct citations in section 3.4: none in the LaTeX text itself.

## 4. Discussions

### 4.1 Qualitative example of phonological vectors

This subsection scales up the earlier toy illustration to a full TIMIT utterance: "She had your dark suit in greasy wash water all year." The visualization tracks cosine similarity between frame-level representations and phonological vectors for positions `-2` through `+2`.

The authors report a staircase-like pattern that lines up with phonetic boundaries rather than a fixed sliding window. They highlight `+low` and `+high` behavior around the vowel `[E]` as an intuitive example. The subsection is mainly qualitative but reinforces the argument that section 3.2 underestimates the strength of the positional structure.

Direct citations in section 4.1: none in the LaTeX text itself.

### 4.2 Layerwise mask-filling behavior

This section asks whether the observed contextual phonological structure could be related to the models' masked prediction objectives. The experiment compares representations extracted from original audio with representations extracted from a masked version of the same audio, measuring cosine similarity on the masked region after ZCA whitening.

HuBERT and WavLM show steadily increasing original-versus-masked similarity across layers, while wav2vec 2.0 peaks earlier and then degrades. The authors suggest that these layerwise differences may help explain why HuBERT and especially WavLM carry stronger contextual phonological information in the earlier experiments. The interpretation stays cautious: mask-filling pressure may encourage encoding of neighboring phonological information because coarticulation makes surrounding phones useful for reconstructing the masked content.

Citations in this section: `[schneider2019wav2vec]`, `[baevski2020wav2vec]`, `[hsu2021hubert]`, `[chen2022wavlm]`, `[choi2025leveraging]`, `[ethayarajh2019HowCA]`, `[krizhevsky2009learning]`, `[pasad2021layer]`, `[pasad2023comparative]`, `[huo2025iterative]`

### 4.3 Connection with Observations from Previous Works

This subsection is a structured comparison against earlier strands of work.

- Interpretability: prior interpretability work mostly asked what information each layer encodes; this paper instead asks how phonetic information is geometrically organized inside representations.
- Phonological vectors: the paper positions itself as an extension of the earlier phonological-vector account from isolated phones to contextualized phone sequences.
- Context-dependent triphone HMMs: it argues that S3Ms may be rediscovering a context-dependent phonetic structure similar in spirit to older triphone-based ASR.
- Effective context window: it reframes prior "how wide is the context window?" analyses as a phonetic-boundary-aware question rather than a purely temporal one.
- Unsupervised segmentation: it argues that the observed position-sensitive subspaces offer a concrete mechanism for why unsupervised phone- and syllable-boundary signals can emerge in S3Ms.

Citations in this section: `[pasad2021layer]`, `[pasad2023comparative]`, `[choi2022opening]`, `[choi2024understanding]`, `[martin2023probing]`, `[choi2024self]`, `[cho2023sslart]`, `[cho2024ssluniart]`, `[baade2025syllablelm]`, `[cho2026sylber]`, `[cho2025sylber]`, `[peng2022word]`, `[pasad2024self]`, `[shen2023wave]`, `[choi2026self]`, `[chang2024exploring]`, `[baevski2020wav2vec]`, `[hsu2021hubert]`, `[wells22_interspeech]`, `[abdullah2023information]`, `[sicherman2023analysing]`, `[choi2025leveraging]`, `[schwartz1985context]`, `[young1994tree]`, `[meng25b_interspeech]`, `[choi2025device]`, `[visser2026zerosyl]`, `[li2023dissecting]`

### 4.4 Implications for Future Work

The future-work section draws three practical consequences.

- Interpretability: phonological, speaker, and possibly other sources of information may all be represented as structured subspaces, suggesting a more general disentanglement story for SSL speech models.
- Discrete speech units: the authors speculate that S3M-derived discrete units may outperform codec-style acoustic tokens partly because S3Ms encode phonetic context in a more structured way.
- Interpretable speech representations: the staircase-like phonological-vector traces could serve as linguistically meaningful intermediate representations for downstream systems such as vocoders or TTS.

Citations in this section: `[liu2023self]`, `[feng2022silence]`, `[kamper2025linearvc]`, `[chang2024exploring]`, `[mousavi2025discrete]`, `[borsos2023audiolm]`, `[defossez2024moshi]`, `[zhang2024speechtokenizer]`, `[huang2026kanade]`, `[hazen2009query]`, `[morrison2024fine]`, `[cernak2015phonological]`, `[cho2024coding]`

## 5. Conclusion

The paper concludes that contextualized frame-level S3M representations encode phonological information compositionally, across multiple neighboring phone positions, in approximately orthogonal subspaces, and in a way that is aligned to phonetic boundaries. The conclusion is less about proposing a new model than about giving a unified explanatory account of how transformer context networks organize phonetic context internally.

Direct citations in this section: none in the LaTeX text itself.

## 6. Acknowledgments

The camera-ready text thanks Stephen McIntosh and Ian Shih for feedback.

Direct citations in this section: none.

## 7. Generative AI Use Disclosure

The authors state that generative AI tools were used for code auto-completion and minor wording or grammar edits, but that scientific content and code implementations were independently developed, verified, and finalized by the authors.

Direct citations in this section: none.

## Relevance To Peacock

This paper is directly useful for Peacock's SSL-backbone work because it gives a mechanistic explanation for why later layers and layer combinations can matter in phonetic tasks. The central takeaway is that context is not just "more information"; it is structured by relative phone position and appears to be segmented by phonetic boundaries. That is relevant to any GOP-style scoring, frame selection, layer mixing, or context-sensitive pronunciation feature design.

It is less useful as an off-the-shelf modeling recipe than as a representation-analysis paper. The practical value is in how it sharpens hypotheses for backbone probing, layer fusion, and phonological-feature extraction.

## References Cited In The Paper

- `[baevski2020wav2vec]` Baevski et al. (2020). "wav2vec 2.0: A framework for self-supervised learning of speech representations." NeurIPS.
- `[hsu2021hubert]` Hsu et al. (2021). "HuBERT: Self-supervised speech representation learning by masked prediction of hidden units." IEEE/ACM TASLP.
- `[chen2022wavlm]` Chen et al. (2022). "WavLM: Large-scale self-supervised pre-training for full stack speech processing." IEEE Journal of Selected Topics in Signal Processing.
- `[schneider2019wav2vec]` Schneider et al. (2019). "wav2vec: Unsupervised Pre-Training for Speech Recognition." Interspeech.
- `[pasad2021layer]` Pasad et al. (2021). "Layer-wise analysis of a self-supervised speech representation model." ASRU.
- `[pasad2023comparative]` Pasad et al. (2023). "Comparative layer-wise analysis of self-supervised speech models." ICASSP.
- `[liu2023self]` Liu et al. (2023). "Self-supervised Predictive Coding Models Encode Speaker and Phonetic Information in Orthogonal Subspaces." Interspeech.
- `[meng25b_interspeech]` Meng et al. (2025). "Effective Context in Neural Speech Models." Interspeech.
- `[choi2022opening]` Choi et al. (2022). "Opening the black box of wav2vec feature encoder." arXiv preprint arXiv:2210.15386.
- `[choi2024self]` Choi et al. (2024). "Self-Supervised Speech Representations are More Phonetic than Semantic." Interspeech.
- `[choi2026self]` Choi et al. (2026). "[b]=[d]-[t]+[p]: Self-supervised Speech Models Discover Phonological Vector Arithmetic." arXiv preprint arXiv:2602.18899.
- `[choi2025leveraging]` Choi et al. (2025). "Leveraging allophony in self-supervised speech models for atypical pronunciation assessment." NAACL.
- `[sicherman2023analysing]` Sicherman et al. (2023). "Analysing discrete self supervised speech representation for spoken language modeling." ICASSP.
- `[wells22_interspeech]` Wells et al. (2022). "Phonetic Analysis of Self-supervised Representations of English Speech." Interspeech.
- `[shih2024interface]` Shih et al. (2024). "Interface Design for Self-Supervised Speech Models." Interspeech.
- `[chang2024exploring]` Chang et al. (2024). "Exploring speech recognition, translation, and understanding with discrete speech units: A comparative study." ICASSP.
- `[de2024human]` de Heer Kloots et al. (2024). "Human-like Linguistic Biases in Neural Speech Models: Phonetic Categorization and Phonotactic Constraints in Wav2Vec2.0." Interspeech.
- `[abdullah2023information]` Abdullah et al. (2023). "An Information-Theoretic Analysis of Self-supervised Discrete Representations of Speech." Interspeech.
- `[li2023dissecting]` Li et al. (2023). "Dissecting neural computations in the human auditory pathway using deep neural networks for speech." Nature Neuroscience.
- `[pasad2024self]` Pasad et al. (2024). "What do self-supervised speech models know about words?" TACL.
- `[baade2025syllablelm]` Baade et al. (2025). "SyllableLM: Learning Coarse Semantic Units for Speech Language Models." ICLR.
- `[cho2025sylber]` Cho et al. (2025). "Sylber: Syllabic Embedding Representation of Speech from Raw Audio." ICLR.
- `[cho2026sylber]` Cho et al. (2026). "Sylber 2.0: A Universal Syllable Embedding." arXiv preprint arXiv:2601.22306.
- `[visser2026zerosyl]` Visser et al. (2026). "ZeroSyl: Simple Zero-Resource Syllable Tokenization for Spoken Language Modeling." arXiv preprint arXiv:2210.15386.
- `[choi2025device]` Choi et al. (2025). "On-device Streaming Discrete Speech Units." Interspeech.
- `[mcfee2015librosa]` McFee et al. (2015). "librosa: Audio and music signal analysis in python." SciPy.
- `[chaabouni2017learning]` Chaabouni et al. (2017). "Learning Weakly Supervised Multimodal Phoneme Embeddings." Interspeech.
- `[zouhar2024pwesuite]` Zouhar et al. (2024). "PWESuite: Phonetic word embeddings and tasks they facilitate." LREC-COLING.
- `[mortensen2016panphon]` Mortensen et al. (2016). "Panphon: A resource for mapping IPA segments to articulatory feature vectors." COLING.
- `[garofolo1993darpa]` Garofolo et al. (1993). "DARPA TIMIT: Acoustic-Phonetic Continuous Speech Corpus CD-ROM, NIST Speech Disc 1-1.1."
- `[chodroff2024voxangeles]` Chodroff et al. (2024). "Phonetic Segmentation of the UCLA Phonetics Lab Archive." LREC-COLING.
- `[ethayarajh2019HowCA]` Ethayarajh (2019). "How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings." EMNLP.
- `[krizhevsky2009learning]` Krizhevsky (2009). "Learning Multiple Layers of Features from Tiny Images." Master's thesis, University of Toronto.
- `[huo2025iterative]` Huo et al. (2025). "Iterative Refinement, Not Training Objective, Makes HuBERT Behave Differently from wav2vec 2.0." Interspeech.
- `[choi2024understanding]` Choi et al. (2024). "Understanding probe behaviors through variational bounds of mutual information." ICASSP.
- `[martin2023probing]` Martin et al. (2023). "Probing Self-supervised Speech Models for Phonetic and Phonemic Information: A Case Study in Aspiration." Interspeech.
- `[cho2023sslart]` Cho et al. (2023). "Evidence of vocal tract articulation in self-supervised learning of speech." ICASSP.
- `[cho2024ssluniart]` Cho et al. (2024). "Self-supervised models of speech infer universal articulatory kinematics." ICASSP.
- `[peng2022word]` Peng et al. (2022). "Word Discovery in Visually Grounded, Self-Supervised Speech Models." Interspeech.
- `[shen2023wave]` Shen et al. (2023). "Wave to Syntax: Probing spoken language models for syntax." Interspeech.
- `[schwartz1985context]` Schwartz et al. (1985). "Context-dependent modeling for acoustic-phonetic recognition of continuous speech." ICASSP.
- `[young1994tree]` Young et al. (1994). "Tree-based state tying for high accuracy modelling." Human Language Technology workshop proceedings.
- `[feng2022silence]` Feng et al. (2022). "Silence is sweeter than speech: Self-supervised model using silence to store speaker information." arXiv preprint arXiv:2205.03759.
- `[kamper2025linearvc]` Kamper et al. (2025). "LinearVC: Linear Transformations of Self-Supervised Features Through the Lens of Voice Conversion." Interspeech.
- `[mousavi2025discrete]` Mousavi et al. (2025). "Discrete Audio Tokens: More Than a Survey!" TMLR.
- `[borsos2023audiolm]` Borsos et al. (2023). "AudioLM: a language modeling approach to audio generation." IEEE/ACM TASLP.
- `[defossez2024moshi]` Defossez et al. (2024). "Moshi: a speech-text foundation model for real-time dialogue." arXiv preprint arXiv:2410.00037.
- `[zhang2024speechtokenizer]` Zhang et al. (2024). "SpeechTokenizer: Unified Speech Tokenizer for Speech Language Models." ICLR.
- `[huang2026kanade]` Huang et al. (2026). "Kanade: A Simple Disentangled Tokenizer for Spoken Language Modeling." arXiv preprint arXiv:2602.00594.
- `[hazen2009query]` Hazen et al. (2009). "Query-by-example spoken term detection using phonetic posteriorgram templates." ASRU.
- `[morrison2024fine]` Morrison et al. (2024). "Fine-Grained and Interpretable Neural Speech Editing." Interspeech.
- `[cernak2015phonological]` Cernak et al. (2015). "Phonological vocoding using artificial neural networks." ICASSP.
- `[cho2024coding]` Cho et al. (2024). "Coding speech through vocal tract kinematics." IEEE Journal of Selected Topics in Signal Processing.
