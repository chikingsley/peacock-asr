# From-Scratch Phoneme CTC: Is Conformer Training Competitive with Fine-Tuned SSL Encoders?

## Abstract

*TODO: Write after Phase 3 experiments complete.*

We investigate whether a Conformer encoder trained from scratch on phoneme-labeled speech
can produce CTC posteriors of sufficient quality for pronunciation scoring, compared to
fine-tuning a pretrained SSL encoder (wav2vec2-BERT 2.0). Our Track 05 baseline
(fine-tuned w2v-BERT 2.0, segmentation-free GOP features, GOPT scorer) achieves PCC 0.648
on SpeechOcean762. We train Conformer models on LibriSpeech 960h with ARPABET phone labels,
extract GOP posteriors through the same pipeline, and evaluate end-task PCC. This directly
answers: how much does pretraining matter for phone posterior quality, and what is the
minimum compute budget for competitive from-scratch training?

## 1. Introduction

Automatic pronunciation assessment (APA) systems typically rest on a pretrained acoustic
model that provides phoneme-level posteriors, which are then converted to goodness of
pronunciation (GOP) features and passed to a downstream scorer. The dominant approach
uses SSL encoders (wav2vec2, HuBERT, wav2vec2-BERT) fine-tuned for phoneme-level CTC
[@cao2026segmentation_free_gop; @gong2022gopt_transformer_pronunciation_assessment].
These models inherit millions of hours of self-supervised pretraining and need only a few
hundred hours of phoneme-labeled data to fine-tune.

An alternative is to train the acoustic model from scratch on phoneme-labeled data.
This is what ZIPA [@zhu2025zipa] did: a Zipformer [@icefall] trained on 17K hours of
IPAPack++ data achieves 2.71 Phone Feature Error Rate (PFER) across 88 languages.
POWSM [@bigi2025powsm] takes a multi-task approach, training a 350M-parameter
Whisper-style model on phoneme recognition, ASR, G2P, and P2G jointly.
PRiSM [@prism2025] benchmarks multiple architectures and finds encoder-CTC to be
the most stable approach for phone-level output.

However, these from-scratch models were evaluated on phone recognition accuracy,
not on their utility as posterior sources for downstream pronunciation scoring.
It is not clear whether lower phone error rate translates to better GOP features
for scoring. A model trained on broad phonetic diversity (88 languages, IPA)
may produce different posterior distributions than a model trained specifically
on the target language (English, ARPABET).

This paper addresses that gap through a controlled comparison:

1. We train Conformer models from scratch on LibriSpeech 960h with ARPABET phone labels
   (Phase 2 of the ablation plan).
2. We extract GOP posteriors through the identical pipeline used for our fine-tuned
   w2v-BERT 2.0 baseline.
3. We evaluate end-task PCC on SpeechOcean762 under identical scoring conditions (Phase 3).
4. If results are promising, we compare Conformer vs Zipformer architectures (Phase 4).

## 2. Related Work

### 2.1 From-Scratch Phoneme Recognizers

ZIPA [@zhu2025zipa] trains a Zipformer from scratch on IPAPack++ (17K hours, 88 languages,
127 IPA characters). The model achieves state-of-the-art multilingual phone recognition
but uses an IPA character-level vocabulary that is incompatible with ARPABET-based GOP
scoring: diphthongs (AW, AY, EY, OW, OY) and affricates (CH, JH) are represented as
multiple IPA characters rather than single tokens, making direct posterior extraction
for GOP impossible.

POWSM [@bigi2025powsm] takes a multi-task approach based on the Whisper architecture
(350M parameters). By training jointly on phone recognition, ASR, G2P, and P2G, it
outperforms ZIPA and wav2vec2-phoneme on phone recognition benchmarks. The encoder-decoder
architecture differs from our CTC-based posterior extraction approach.

PRiSM [@prism2025] benchmarks multiple approaches for phone recognition and finds
encoder-CTC to be the most stable architecture. This supports our choice of CTC as
the primary training objective.

### 2.2 Architecture: Conformer and Zipformer

The Conformer [@gulati2020conformer] combines Multi-Head Self-Attention for global
context and depth-wise convolution for local acoustic patterns within each encoder block.
It became the standard modern ASR architecture after its introduction in 2020.

Zipformer [@icefall] is k2-fsa's improved Conformer variant with better parameter
efficiency and training stability. ZIPA adopted Zipformer for its from-scratch multilingual
training. In icefall benchmarks, Zipformer consistently outperforms Conformer on
LibriSpeech word-level ASR tasks.

### 2.3 Fine-Tuned SSL Encoders as Baselines

Conneau et al. [@conneau2021zeroshotphoneme] demonstrated that wav2vec2-xlsr, fine-tuned
with only small amounts of labeled data, achieves strong cross-lingual phone recognition.
MMS [@pratap2023mms] scales this to 1000+ languages. Our Track 05 work uses wav2vec2-BERT 2.0
fine-tuned on LibriSpeech and achieves PCC 0.648 on SpeechOcean762 — the primary
comparison point for this track.

### 2.4 Pronunciation Assessment and GOP

*Inherit from Track 05 manuscript, Sections 2.1-2.3.*

### 2.5 Data Labeling for Phoneme Training

Large-scale training from scratch requires phoneme-level labels. For English,
gilkeyio/librispeech-alignments provides forced-alignment ARPABET labels for all
960 hours of LibriSpeech. For multilingual data, ByT5-based G2P [@gutkin2022byt5g2p]
enables labeling of text-only corpora.

Phoneme similarity modeling [@yao2025phoneme_similarity] and soft-alignment CTC
[@lcsctc2025] are potential extensions if base results are promising.

## 3. Method

### 3.1 Training Setup

We adapt the icefall [@icefall] Conformer LibriSpeech recipe for phoneme-level CTC:

- Replace BPE tokenizer with ARPABET phone set (~40 phones)
- Use forced-alignment phone labels from gilkeyio/librispeech-alignments
- CTC loss on phone sequence output
- Architecture: Conformer-M (matching the medium-scale icefall recipe)

*TODO after Phase 1 (icefall infrastructure validated): fill in exact model dimensions,
layers, attention heads, FFN size, batch size, learning rate schedule.*

### 3.2 Posterior Extraction

*TODO after Phase 2: describe how CTC output layer activations are extracted and
passed to the GOP pipeline. Must match the format of `gop.py`.*

### 3.3 Evaluation Pipeline

Identical to Track 05:

- GOP-SF feature extraction from CTC posteriors (LPP + LPR, 42-dim)
- GOPT scorer (3-layer transformer) trained on SpeechOcean762
- Evaluation: phone-level PCC on SpeechOcean762 test set
- Minimum 3 seeds

## 4. Experimental Setup

### 4.1 Datasets

Training: LibriSpeech 960h [@librispeech] with ARPABET phone labels
(gilkeyio/librispeech-alignments).

Evaluation: SpeechOcean762 [@speechocean762], 2500 train / 2500 test, pinned revision.

### 4.2 Baselines

- **Track 05 fine-tuned**: w2v-BERT 2.0 fine-tuned on LibriSpeech, PCC 0.648
- **TIMIT TDNN-LSTM** (Phase 1 only): icefall reference, PER 17.66%

### 4.3 Implementation

*TODO after Phase 1.*

## 5. Results

### 5.1 Phase 1: TIMIT TDNN-LSTM (Infrastructure Validation)

*TODO: PER on TIMIT. Match target: within 1% relative of 17.66% paper result.*

### 5.2 Phase 2: Conformer on LibriSpeech

*TODO: Table with PER by model size and training data size.*

| Run | Architecture | Training Data | PER (TIMIT) | Notes |
|-----|-------------|--------------|-------------|-------|
| P2-A | Conformer-S | LS-100h | TBD | |
| P2-B | Conformer-M | LS-960h | TBD | |

### 5.3 Phase 3: From-Scratch vs Fine-Tuned (Main Result)

*TODO: Table comparing from-scratch Conformer vs fine-tuned w2v-BERT 2.0 on SpeechOcean762.*

| System | Acoustic Model | PCC (SpeechOcean762) | PER | GPU-hours |
|--------|---------------|---------------------|-----|-----------|
| Track 05 baseline | w2v-BERT 2.0 (fine-tuned) | 0.648 | -- | ~4 |
| P3-B | Conformer-M (from scratch) | TBD | TBD | TBD |
| P3-C | Conformer-L (from scratch) | TBD | TBD | TBD |

### 5.4 Phase 4: Conformer vs Zipformer (If Applicable)

*TODO if Phase 3 shows from-scratch is competitive.*

## 6. Discussion

*TODO after Phase 3.*

Key questions to address:

- Does lower phone PER translate to better GOP posterior quality for scoring?
- What is the break-even compute point (from-scratch GPU-hours vs fine-tuning GPU-hours)?
- Is the IPA vocabulary mismatch the main reason ZIPA cannot be used directly?
- Can Zipformer with ARPABET output close the gap with fine-tuned SSL models?

## 7. Conclusion

*TODO after experiments.*

## References
