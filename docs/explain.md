# How Pronunciation Scoring Works

Quick reference. No research citations, no architecture diagrams.
Just what you need, what it is, and where to get it.

## The 3 Pieces You Need (per language/dialect)

To score pronunciation for any language, you need exactly 3 things:

1. vocab.json -- the phoneme inventory
2. training data -- audio recordings in that language
3. phoneme labels -- which phones appear in each recording

These 3 things must match each other. The vocab defines what phones
exist, the labels use those phones, and the audio is people speaking
that language.

## What is vocab.json

A tiny JSON file mapping each phone to an integer. This becomes
the output alphabet of the model -- what it can predict.

American English example (39 ARPABET phones + 2 special tokens):

    {
      "AA": 0, "AE": 1, "AH": 2, "AO": 3, "AW": 4, "AY": 5,
      "B": 6, "CH": 7, "D": 8, "DH": 9, "EH": 10, "ER": 11,
      "EY": 12, "F": 13, "G": 14, "HH": 15, "IH": 16, "IY": 17,
      "JH": 18, "K": 19, "L": 20, "M": 21, "N": 22, "NG": 23,
      "OW": 24, "OY": 25, "P": 26, "R": 27, "S": 28, "SH": 29,
      "T": 30, "TH": 31, "UH": 32, "UW": 33, "V": 34, "W": 35,
      "Y": 36, "Z": 37, "ZH": 38,
      "[PAD]": 39, "[UNK]": 40
    }

This changes per language. Arabic has pharyngeals and emphatics.
Mandarin has tones. French has nasalized vowels. Each language
has its own phone inventory.

For multilingual, you use IPA instead of ARPABET (xlsr-espeak
uses 387 IPA symbols covering all languages in one vocab).

## What are phoneme labels

For each audio recording, a list of which phones were spoken in
what order. CTC training does NOT need timestamps -- just the
sequence.

Example: audio of someone saying "hello"
  label: HH AH L OW

Two ways to get labels:

a) pre-labeled dataset -- someone already did the work
b) G2P (grapheme-to-phoneme) -- convert text transcripts to phones
   automatically using a model or dictionary

## What is G2P

Grapheme-to-phoneme converts written text to phone sequences.

    "hello" → HH AH L OW
    "cat"   → K AE T

For English, the CMU pronunciation dictionary covers ~130K words.
For words not in the dictionary (or other languages), you use a
G2P model that predicts the phones from spelling.

G2P tools:

- CMU dict (English, lookup table): built into nltk
- NeMo G2P (multilingual, neural): docs.nvidia.com/nemo-framework
- CharsiuG2P (multilingual): github.com/lingjzhu/CharsiuG2P
- espeak-ng (rule-based, 100+ languages): command-line tool

## Per Language: What Exists

### American English -- ready to go

vocab: 39 ARPABET phones (the standard, shown above)

training data + labels:

- gilkeyio/librispeech-alignments (HuggingFace)
  960h, pre-labeled ARPABET with timestamps from Montreal Forced Aligner
  this is the one to use -- no G2P step needed
- TIMIT: 5.4h, hand-labeled phones (gold standard but tiny)
- SpeechOcean762: 5h, ARPABET labels with pronunciation scores
  too small for training the CTC head -- this is our eval dataset

### British English -- needs work

vocab: same 39 ARPABET phones work (shared phoneme inventory),
  though some phones differ in distribution (non-rhotic R, BATH vowel)

training data: no pre-labeled British English phoneme dataset exists
  on HuggingFace. would need to take a British English audio corpus
  (e.g. CommonVoice en-GB subset) and label it via G2P.

### Arabic -- needs work

vocab: different phone set entirely (~30-40 phones depending on
  dialect, includes pharyngeals, emphatics, uvulars)

training data: CommonVoice has Arabic audio.
  would need Arabic G2P to label it.
  or use MMS/xlsr-espeak which already have IPA labels.

### Other languages -- same pattern

1. define the phone inventory for that language → vocab.json
2. find or record audio in that language
3. label it with G2P or use a pre-labeled dataset

## What You Train

You take a pretrained speech encoder (a model that already
understands audio but outputs the wrong things) and add a small
new output layer that predicts your phones.

    pretrained encoder (understands audio)
         ↓
    new linear layer (encoder_dim → vocab_size)
         ↓
    CTC loss (learns to align audio frames to phone sequence)

The encoder stays mostly frozen. The new layer learns which of
your phones each audio frame sounds like.

Candidates for the encoder:

- w2v-BERT 2.0 (580M params, 4.5M hours pretraining, HuggingFace)
- omniASR (325M-6.5B params, 1600 languages, fairseq2)
- WavLM-large (317M params, best phonetic probing scores)
- icefall Zipformer (TIMIT recipe exists for learning CTC mechanics)

Training takes ~6-12h on an A100 GPU for 960h of data.

## What You Get Out

Frame-level phoneme posteriors: for each 20ms audio frame, a
probability distribution over all phones in your vocab.

    frame 0: AA=0.01  AE=0.02  AH=0.85  AO=0.01 ...
    frame 1: AA=0.01  AE=0.01  AH=0.90  AO=0.01 ...
    frame 2: AA=0.02  AE=0.03  AH=0.05  L=0.82  ...
    ...

These posteriors go into GOP (goodness of pronunciation) scoring,
which compares "how likely is the canonical pronunciation" vs
"how likely is any other pronunciation" across all possible
alignments. That produces a per-phone score.

The per-phone GOP scores then go into GOPT (a small transformer)
which produces final pronunciation scores at phone, word, and
sentence level.

    posteriors → GOP scoring → per-phone features → GOPT → scores

## Current Status

American English pipeline is working end-to-end:

- using wav2vec2-large encoder with existing CTC head
- GOP-SF scoring (segmentation-free, no forced alignment needed)
- GOPT transformer for final scores
- PCC 0.662 on SpeechOcean762 (matches paper benchmarks)

next step: train our own CTC head on w2v-BERT 2.0 using
librispeech-alignments to get better posteriors → better scores
