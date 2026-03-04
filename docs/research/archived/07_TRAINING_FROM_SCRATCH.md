# 07: Training a Phoneme Model from Scratch

Status: stub -- architectures identified, no work started.

This track covers training a full phoneme recognition model from
scratch (not fine-tuning a pretrained SSL encoder). Requires
significantly more data and compute than track 1 (05).

Last updated: 2026-03-02

## The Idea

Instead of taking a pretrained encoder and adding a small CTC head
(track 1), train the entire model end-to-end on phoneme-labeled
speech data. This is what ZIPA did (Zipformer on 17K hours).

Key differences from track 1 (05):

- needs thousands of hours of labeled data (vs hundreds for fine-tuning)
- needs days of GPU time (vs hours)
- architecture choice matters (Conformer vs Zipformer vs FastConformer)
- potentially better if you have enough data and the right architecture
- harder to iterate on -- each experiment is expensive

## Architecture Options

### TDNN-LSTM

- old-school, the icefall TIMIT recipe uses this
- trains from scratch, weakest results (17.66% PER on TIMIT)
- useful only as a learning sandbox for CTC training mechanics
- recipe: github.com/k2-fsa/icefall egs/timit/ASR

### Conformer

- CNN + Transformer hybrid (Gulati et al., 2020)
- the standard modern ASR architecture
- icefall has Conformer recipes for LibriSpeech but only BPE/word-level
- would need to adapt recipe for phoneme-level output
- paper: 2005.08100 (in grapheme-to-phoneme/)

### Zipformer

- k2-fsa's improved Conformer variant
- ZIPA uses this internally (trained on IPAPack++ 17K hours)
- icefall has Zipformer recipes but again only BPE
- best results in icefall benchmarks
- no phoneme-level Zipformer recipe exists anywhere

### FastConformer (NVIDIA)

- NVIDIA's Conformer variant in NeMo framework
- Parakeet-CTC uses this architecture
- 80ms temporal resolution (4x coarser than standard 20ms)
- the coarser resolution may hurt GOP scoring -- untested
- paper: 2509.14128 (in streaming_realtime/)

## What ZIPA Did (Our Best Reference)

ZIPA trained a Zipformer from scratch:

- 17K hours of IPAPack++ data (88 languages)
- 127 IPA character output vocabulary
- achieved 2.71 PFER on seen languages, 0.66 on English
- acoustically excellent but character-level vocab breaks GOP
  (diphthongs AW/AY/CH/EY/JH/OW/OY can't be single tokens)
- paper: 2505.23170 (root docs/papers/)
- code: github.com/lingjzhu/zipa

## What POWSM Did

POWSM trained a Whisper-style model (350M params) on multiple tasks:

- phoneme recognition + ASR + G2P + P2G (multi-task)
- outperforms ZIPA and wav2vec2-phoneme on phone recognition
- ESPnet-based
- paper: 2510.24992 (root docs/papers/)

## Papers We Have

### Architecture papers

    2005.08100 (grapheme-to-phoneme/)
    "Conformer: Convolution-augmented Transformer for Speech Recognition"
    The original Conformer paper. CNN+Transformer hybrid.

    2509.14128 (streaming_realtime/)
    "Canary-1B-v2 & Parakeet-TDT"
    NVIDIA FastConformer models. Includes Parakeet-CTC at 80ms.

### Models trained from scratch on phonemes

    2505.23170-ZIPA (root)
    "ZIPA: Efficient Multilingual Phone Recognition"
    Zipformer trained from scratch on 17K hours. Our reference
    for what from-scratch training looks like.

    2510.24992-POWSM (root)
    "POWSM: A Phonetic Open Whisper-Style Speech Foundation Model"
    Multi-task phonetic model. Alternative architecture to CTC-only.

    2601.14046-PRiSM (root)
    "PRiSM: Benchmarking Phone Realization in Speech Models"
    Benchmark comparing encoder-CTC vs other approaches for phone
    recognition. Finding: encoder-CTC is most stable.

### Phoneme recognition and evaluation

    2507.14346 (core_segmentation_free/)
    "Towards Accurate Phonetic Error Detection Through Phoneme
    Similarity Modeling"
    Phoneme similarity for error detection. Relevant if training
    a model that needs to distinguish similar phones.

    2508.03937 (core_segmentation_free/)
    "LCS-CTC: Leveraging Soft Alignments"
    CTC variant with soft alignments. Could improve training.

### G2P (needed for labeling large datasets)

    2204.03067 (grapheme-to-phoneme/)
    "ByT5 model for massively multilingual G2P"
    Multilingual G2P across 100 languages. Needed if training
    on data that only has text transcripts, not phone labels.

    rezackova21 (grapheme-to-phoneme/)
    "T5G2P: T5 for Grapheme-to-Phoneme Conversion"

    2105.13626 (grapheme-to-phoneme/)
    "ByT5: Token-Free Byte-to-Byte Models"
    Foundation for byte-level G2P models.

    2108.10447 (grapheme-to-phoneme/)
    "One TTS Alignment To Rule Them All"
    Alignment learning framework. Relevant for forced alignment
    during data preparation.

## Data Requirements

From-scratch training needs much more data than fine-tuning:

    ZIPA used:     17K hours (IPAPack++, 88 languages)
    POWSM used:    multi-task data (size not specified in our notes)
    Minimum:       probably 1K+ hours for competitive results
    Comparison:    fine-tuning w2v-BERT 2.0 needs ~100-960h

Labeling options for large datasets:

- gilkeyio/librispeech-alignments: 960h, pre-labeled ARPABET (ready)
- CommonVoice: 1000+ hours per language, needs G2P labeling
- IPAPack++: ZIPA's dataset, 17K hours, G2P-labeled via CharsiuG2P

## What We Don't Have Yet

- no from-scratch training infrastructure set up
- no experience with icefall training pipeline
- no phoneme-level Conformer/Zipformer recipe (would need to adapt
  existing BPE recipes)
- no cost/time estimates for full training runs
- unclear whether from-scratch beats fine-tuning for our use case

## Open Questions

- does from-scratch training produce better posteriors than fine-tuning
  a pretrained SSL encoder? ZIPA's acoustic quality is excellent (2.71
  PFER) but the comparison isn't fair because of the vocab mismatch.
- is the icefall training pipeline learnable in a reasonable timeframe?
- what's the minimum viable dataset size for competitive phone posteriors?
- Conformer vs Zipformer vs FastConformer -- which architecture is best
  for phoneme-level CTC?
- can we just use ZIPA's encoder with a new ARPABET head? (hybrid
  approach -- see 05 Section 7.2 Q2)

## Code References

- github.com/k2-fsa/icefall -- 43 recipes, TIMIT is only phoneme CTC
- github.com/lingjzhu/zipa -- ZIPA training code
- NVIDIA NeMo -- FastConformer/Parakeet training
- ESPnet -- POWSM training (not validated)
