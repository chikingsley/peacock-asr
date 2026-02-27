# Phoneme CTC Heads: Research Backing & Implementation Plan

From BPE/character CTC to IPA phoneme posteriors for GOP-based pronunciation scoring.

This document is structured as a research narrative: each component of the pipeline
is backed by prior work, with clear markers where we follow established methods
(**GROUNDED**) versus where we must experiment (**OPEN QUESTION**).

---

## 1. The Thesis

**Claim**: Replacing the output head of a modern pretrained speech encoder with an
IPA phoneme CTC head, then feeding those posteriors into segmentation-free GOP
scoring, should produce pronunciation scores that match or exceed the current
state of the art — with no forced alignment, no Kaldi, and deployable as a single
ONNX model.

**Why we believe this**:

1. CTC head replacement/addition on SSL encoders is standard practice with known
   training recipes (Section 3)
2. IPA phoneme recognition has been demonstrated at high accuracy by multiple
   independent systems (Section 4)
3. GOP scoring from CTC posteriors has been proven to work without forced alignment
   (Section 5)
4. Each piece exists in isolation; the combination is straightforward engineering
   with one identified research gap (Section 7)

---

## 2. Background: What Exists Today

### 2.1 Our Current Benchmark Results

Three backends evaluated on SpeechOcean762 with GOP-SF-SD-Norm scoring:

```
Backend              PCC     95% CI              Phones   Status
─────────────────────────────────────────────────────────────────
xlsr-espeak          0.320   [0.312, 0.328]      39/39    Best current
original (w2v2)      0.310   [0.302, 0.319]      39/39    Paper baseline
zipa                 0.075   [0.066, 0.084]      32/39    Vocab mismatch
```

**Key finding**: ZIPA's poor score is NOT due to poor acoustic modeling (it has
2.71 PFER vs xlsr-espeak's 11.88). It's a structural incompatibility: ZIPA uses
127 IPA characters, and 7 English diphthongs/affricates (AW, AY, CH, EY, JH, OW,
OY) cannot be expressed as single tokens.

**Implication**: A phoneme-level head (not character-level) is required. This is
the core motivation for this work.

### 2.2 The Pipeline We're Building

```
Audio (16kHz)
    │
    ▼
┌─────────────────────────────────┐
│  Pretrained Speech Encoder      │  ◄── Section 3: well-established
│  (w2v-BERT 2.0 / omniASR /     │      SSL models, frozen or fine-tuned
│   WavLM-large)                  │
└─────────────┬───────────────────┘
              │  hidden states [T × d_model]
              ▼
┌─────────────────────────────────┐
│  IPA Phoneme CTC Head           │  ◄── Section 4: this is the new piece
│  Linear(d_model → N_phones)     │      but the technique is standard
│  + CTC loss                     │
└─────────────┬───────────────────┘
              │  posteriors [T × N_phones]
              ▼
┌─────────────────────────────────┐
│  GOP-SF Scoring                 │  ◄── Section 5: paper algorithm,
│  (Segmentation-Free,            │      already implemented in this repo
│   CTC forward-backward)         │
└─────────────┬───────────────────┘
              │  GOP feature vectors
              ▼
┌─────────────────────────────────┐
│  GOPT Transformer               │  ◄── Section 6: downstream scorer
│  (or polynomial regression)     │      (small, fast to train)
└─────────────┬───────────────────┘
              │
              ▼
        Pronunciation Scores
        (phoneme / word / sentence)
```

---

## 3. Component 1: Pretrained Speech Encoders

### Status: **GROUNDED** — extensive prior work, multiple options

### 3.1 The SSL Pretraining Paradigm

Self-supervised speech models learn rich acoustic representations from unlabeled
audio via masked prediction tasks. The key insight: the encoder's hidden states
already encode phonetic information — we just need to add a head that reads it out.

**Foundational papers**:

| Model | Year | Params | Pretraining Data | Key Innovation |
|-------|------|--------|------------------|----------------|
| wav2vec 2.0 | 2020 | 95-317M | 960h LibriSpeech | Contrastive + quantized targets |
| HuBERT | 2021 | 95-317M | 960h LibriSpeech | Offline clustering targets |
| WavLM | 2022 | 95-317M | 94K hrs mixed | Denoising + speech structure |
| XLS-R | 2022 | 300M-2B | 436K hrs, 128 langs | Multilingual wav2vec 2.0 |
| w2v-BERT 2.0 | 2024 | 580M | 4.5M hrs, 143 langs | Contrastive + MLM combined |
| data2vec 2.0 | 2024 | 95-317M | 960h+ | Teacher-student, multi-modal |

**What the literature shows**:
- These models produce representations where phoneme identity is linearly
  separable from intermediate layers (Pasad et al., 2021; Shah et al., 2021)
- Layer 7-12 of a 24-layer model typically carries the most phonetic information
- Fine-tuning with CTC on as little as 10 minutes of labeled data produces
  usable ASR (Baevski et al., 2020)

**References**:
- wav2vec 2.0: Baevski et al., "wav2vec 2.0: A Framework for Self-Supervised
  Learning of Speech Representations" (NeurIPS 2020)
- HuBERT: Hsu et al., "HuBERT: Self-Supervised Speech Representation Learning
  by Masked Prediction of Hidden Units" (IEEE/ACM TASLP 2021)
- WavLM: Chen et al., "WavLM: Large-Scale Self-Supervised Pre-Training for
  Full Stack Speech Processing" (IEEE JSTSP 2022)
- XLS-R: Babu et al., "XLS-R: Self-supervised Cross-lingual Speech
  Representation Learning at Scale" (INTERSPEECH 2022)
- w2v-BERT 2.0: Released as part of Seamless Communication (Meta, 2024)

### 3.2 Candidate Encoders for This Work

Based on our OPTIONS.md analysis:

| Encoder | Params | Pretraining | Temporal Res | Advantage | Framework |
|---------|--------|-------------|--------------|-----------|-----------|
| **w2v-BERT 2.0** | 580M | 4.5M hrs / 143 langs | 20ms | Most pretrain data, HF native | HuggingFace |
| **omniASR** | 325M-6.5B | 4.3M hrs / 1600 langs | 20ms | Most languages, already has CTC | fairseq2 |
| **WavLM-large** | 317M | 94K hrs | 20ms | Best phonetic probing results | HuggingFace |
| Parakeet CTC | 600M | 64K hrs EN | **80ms** | Best English WER | NeMo |

**OPEN QUESTION**: Parakeet's 80ms temporal resolution (4x coarser than the others)
may hurt GOP scoring. GOP-SF relies on frame-level posteriors at each timestep;
coarser resolution means fewer frames per phoneme and potentially less precise
alignment-free scoring. The paper (2507.16838) used 20ms resolution. This needs
empirical validation.

### 3.3 Encoder Selection Recommendation

**Primary**: w2v-BERT 2.0 (facebook/w2v-bert-2.0)
- Most pretraining data (4.5M hours) → best representations
- HuggingFace `Wav2Vec2BertForCTC` class exists with training recipe
- Complete blog tutorial: "Fine-Tune W2V2-Bert for low-resource ASR with
  Transformers" (HuggingFace, Yoach Lacombe, Jan 2024)
- MIT license

**Secondary**: omniASR CTC 1B (facebook/omniASR-CTC-1B)
- Already has a CTC head (9,812 chars) — swap final_proj layer
- 1,600 languages → strongest multilingual signal
- Apache 2.0 license
- Caveat: fairseq2 ecosystem (not HuggingFace)

---

## 4. Component 2: IPA Phoneme CTC Head

### Status: **GROUNDED** — this is exactly how wav2vec2-xlsr-espeak-cv-ft was built

### 4.1 The Core Technique

Adding a CTC head to an SSL encoder is a single linear layer:

```python
# This is literally what Wav2Vec2ForCTC does
class PhonemeHead(nn.Module):
    def __init__(self, encoder_dim, n_phonemes):
        super().__init__()
        self.proj = nn.Linear(encoder_dim, n_phonemes)  # That's it

    def forward(self, hidden_states):
        logits = self.proj(hidden_states)  # [B, T, N_phones]
        return logits
```

Training uses CTC loss between the predicted frame-level logits and the target
phoneme sequence. CTC handles the alignment problem — we don't need to know which
frame maps to which phoneme.

**This is not novel.** Every CTC-based ASR model does this. The only difference
is the vocabulary: instead of characters or BPE tokens, we use IPA phonemes.

### 4.2 Prior Art: Models That Already Did This

| Model | Backbone | Vocab | Training Data | PFER | Code |
|-------|----------|-------|---------------|------|------|
| **wav2vec2-xlsr-53-espeak-cv-ft** | XLS-R 300M | 387 IPA | CommonVoice (56K hrs SSL + CTC tune) | 11.88 seen, 5.45 EN | HuggingFace ✓ |
| **wav2vec2-lv-60-espeak-cv-ft** | wav2vec2-large | 387 IPA | LibriVox 60K + CommonVoice | Similar | HuggingFace ✓ |
| **ZIPA-CR** | Zipformer (icefall) | 127 IPA chars | IPAPack++ 17K hrs | 2.71 seen, 0.66 EN | GitHub + ONNX ✓ |
| **POWSM** | Whisper-style (350M) | IPA phones | Multi-task (PR+ASR+G2P+P2G) | Outperforms ZIPA & W2V2Phoneme | ESPnet/HuggingFace ✓ |
| **Wav2Vec2Phoneme** | XLS-R | IPA | CommonVoice + BABEL | — | HuggingFace ✓ |
| **MMS** (Meta) | wav2vec2 | IPA per-lang | 1100+ langs, 491K hrs | — | HuggingFace ✓ |
| **Allosaurus** | Custom + allophone layer | Universal IPA | Multilingual | 22.33 | GitHub (pip) ✓ |

**Key observation**: All of these except ZIPA use a phoneme-level vocabulary.
ZIPA uses character-level IPA (single Unicode codepoints), which is why it fails
on English diphthongs (Section 2.1). This distinction — phoneme vs character —
is critical and well-understood.

### 4.3 The HuggingFace Training Recipe

The canonical reference for CTC head training on a pretrained encoder:

**Blog**: "Fine-Tune W2V2-Bert for low-resource ASR with Transformers"
(saved in `docs/fine-tune-w2v2-bert.md`)

**Key steps** (all standard, nothing invented):

1. **Define vocabulary**: Create `vocab.json` mapping each target token to an index.
   For our case: 39 ARPABET phones + blank + pad = 41 tokens (English),
   or ~100-400 IPA phones for multilingual.

2. **Create tokenizer**: `Wav2Vec2CTCTokenizer.from_pretrained("./", ...)`

3. **Load model with new head**:
   ```python
   model = Wav2Vec2BertForCTC.from_pretrained(
       "facebook/w2v-bert-2.0",
       ctc_loss_reduction="mean",
       pad_token_id=processor.tokenizer.pad_token_id,
       vocab_size=len(processor.tokenizer),  # NEW HEAD SIZE
       # Disable dropout for small-data fine-tuning:
       attention_dropout=0.0,
       hidden_dropout=0.0,
       feat_proj_dropout=0.0,
       mask_time_prob=0.0,
       layerdrop=0.0,
       add_adapter=True,  # optional adapter layers
   )
   ```

4. **Train with HuggingFace Trainer**:
   ```python
   TrainingArguments(
       per_device_train_batch_size=16,
       gradient_accumulation_steps=2,
       num_train_epochs=10,
       learning_rate=5e-5,
       warmup_steps=500,
       fp16=True,      # or bf16=True on A100
       gradient_checkpointing=True,
   )
   ```

5. **Result**: A model that outputs frame-level posteriors over IPA phonemes.

**Important w2v-BERT detail**: Unlike wav2vec2, w2v-BERT 2.0 needs a
**convolutional adapter layer** (`add_adapter=True`) to sub-sample encoder outputs
for proper CTC token duration alignment. This is handled automatically by
`Wav2Vec2BertForCTC` when you set `add_adapter=True`.

**The blog achieves 32.4% WER on Mongolian Cyrillic** with 14h of data on a
V100 — comparable to Whisper-large-v3 fine-tuned on the same data.

For our purpose: we don't need WER. We need good phoneme posteriors. And the
literature shows that phoneme posteriors from CTC fine-tuned SSL models are
already good enough for GOP (our xlsr-espeak backend achieves PCC 0.320).

### 4.4 Vocabulary Design

**GROUNDED for English (ARPABET)**:

The CTC-based-GOP paper (2507.16838) uses 39 ARPABET phones. This is the standard
CMU pronunciation dictionary phone set. SpeechOcean762 annotations use this set.

```
39 ARPABET phones (stress stripped):
AA AE AH AO AW AY B CH D DH EH ER EY F G HH IH IY JH K L M N
NG OW OY P R S SH T TH UH UW V W Y Z ZH
+ blank (CTC) + pad = 41 total
```

**OPEN QUESTION for multilingual (IPA)**:

Two schools:
- **Large vocab (387 IPA)**: wav2vec2-xlsr-53-espeak-cv-ft approach. Covers all
  languages but many phones have little training data.
- **Core IPA (~100-200)**: Focus on phonemically contrastive segments. ZIPA uses
  127. POWSM uses a similar range.
- **Per-language small vocab**: MMS approach — separate adapter per language with
  small phone set. Highest accuracy per language but doesn't scale.

For English MVP, ARPABET is the clear choice (matches evaluation dataset exactly).
For multilingual, this is a design decision we'll need to make empirically.

### 4.5 Training Data for the Phoneme Head

**GROUNDED**:

| Dataset | Hours | Phones | Use Case |
|---------|-------|--------|----------|
| **LibriSpeech 960h** | 960 | ARPABET via G2P | English phoneme CTC training |
| **CommonVoice** | 1000+ per lang | IPA via espeak/G2P | Multilingual CTC training |
| **TIMIT** | 5.4 | Hand-labeled phones | Small but gold-standard phone labels |
| **SpeechOcean762** | ~5 | ARPABET (annotated) | NOT for CTC training — too small. For GOP eval only |

**The approach used by existing models**:
- wav2vec2-xlsr-espeak: CommonVoice, phonemized via espeak-ng
- ZIPA: IPAPack++ (17K hrs, G2P labels from CharsiuG2P)
- POWSM: Multi-task training on mixed datasets
- CTC-based-GOP baseline: LibriSpeech, ARPABET labels from CMU dict

**Recommendation**: Start with **LibriSpeech + CMU pronunciation dictionary**
for English (960h, ARPABET). This is exactly what the CTC-based-GOP paper's
baseline model was trained on, so we have a direct comparison.

---

## 5. Component 3: GOP-SF Scoring

### Status: **GROUNDED** — paper algorithm, already implemented

### 5.1 The Algorithm

GOP-SF (Goodness of Pronunciation, Segmentation-Free) from Cao et al. (2507.16838):

For each phoneme position `p` in canonical transcription `W = (p₁, ..., pₖ)`:

```
GOP-SF(p) = [log P_CTC(W | X) - log P_CTC(W_p^* | X)] / E[d_p]
```

Where:
- `P_CTC(W | X)` = CTC forward probability of canonical transcription (numerator)
- `P_CTC(W_p^* | X)` = CTC forward probability with position p replaced by any
  phone (denominator)
- `E[d_p]` = expected duration of phone p (from CTC forward algorithm)
- The normalization by expected duration gives "GOP-SF-SD-Norm" — the best variant

**Why this works**: Instead of forcing a single alignment (which introduces
segmentation errors), GOP-SF marginalizes over ALL possible alignments weighted
by their CTC probability. A well-pronounced phone will have high canonical
probability relative to any-phone probability across all plausible alignments.

### 5.2 Implementation Status

**Already implemented** in `src/gopt_bench/gop.py` (323 lines):

- `compute_gop()` — main entry point
- `_ctc_forward()` — CTC forward algorithm (numerator)
- `_ctc_forward_denom()` — CTC forward with arbitrary position (denominator)
- Returns `GOPResult` with per-phone scores and occupancies

The implementation is a direct port of the paper's reference code from
`references/CTC-based-GOP/taslpro26/gop_sf_sd_norm.py`.

### 5.3 What GOP-SF Requires from the Phoneme Head

**Input**: Frame-level log posteriors `[T × N_phones]` where:
- `T` = number of frames (typically audio_length / 20ms)
- `N_phones` = vocabulary size (39 for ARPABET, more for IPA)
- Values are log-softmax over the phone dimension

**Critical requirement**: The vocabulary must include ALL phones in the canonical
transcription as single tokens. This is why ZIPA failed — diphthongs like /aʊ/
are two characters in ZIPA's vocab but one phone in ARPABET.

**GROUNDED**: Our GOP implementation already works with the xlsr-espeak backend
(PCC 0.320) and the original wav2vec2 backend (PCC 0.310). Any new phoneme head
that outputs posteriors in the same format will plug in directly.

### 5.4 Emerging: Logit-Based GOP

Recent work (arXiv: 2506.12067, 2025) shows that using pre-softmax logits
instead of posteriors for GOP computation can outperform traditional
probability-based GOP for mispronunciation detection. Maximum-logit GOP shows
strongest alignment with human perception. A hybrid approach combining logit and
probability features gives the best overall performance.

**OPEN QUESTION**: Our current GOP-SF implementation uses log-softmax posteriors.
Extending it to also compute logit-based scores is straightforward (skip the
softmax) but the interaction with CTC forward-backward needs validation. This is
a low-effort, potentially high-reward experiment to run alongside the main work.

---

## 6. Component 4: Score Prediction

### Status: **GROUNDED** — multiple approaches in literature

### 6.1 Polynomial Regression (Current)

Our current evaluation uses per-phone polynomial regression (degree 2) on GOP
scores, following the protocol from the CTC-based-GOP paper. This is a simple
baseline that maps raw GOP scores to human ratings.

**Result**: PCC 0.310-0.320 depending on backend.

### 6.2 GOPT Transformer (Established)

The GOPT model (Gong et al., this repo's origin) is a small transformer that takes
GOP feature vectors and predicts multi-aspect pronunciation scores:
- Input: `[batch, max_phones, feat_dim]` GOP features + canonical phone embeddings
- Output: phoneme / word / utterance level scores
- Published results: PCC 0.612 phone, 0.742 sentence on SpeechOcean762

**Reference**: Gong et al., "Transformer-based Multi-Aspect Multi-Granularity
Non-native English Speaker Pronunciation Assessment" (ICASSP 2022)

### 6.3 The PCC Gap: Scalar GOP vs Feature Vectors

**Critical context**: Our current PCC of 0.320 uses **scalar GOP scores** with
polynomial regression. The paper (2507.16838) achieves PCC **0.581** with
polynomial regression and **0.648** with SVR when using **GOP feature vectors**
(41 dimensions: LPP + LPR + expected count). The GOPT transformer achieves
PCC **0.612** on its Kaldi-extracted features.

This means our current evaluation is not an apples-to-apples comparison with the
paper. We're using scalar scores; the paper uses feature vectors. The next step
to close this gap (independent of encoder choice) is to extract full GOP feature
vectors and use SVR or GOPT for scoring.

### 6.4 Expected Improvement from Better Encoder

The GOPT paper's PCC 0.612 used Kaldi-extracted GOP features from a
TDNN-HMM model. Replacing Kaldi with a modern SSL encoder + CTC should improve
feature quality. The CTC-based-GOP paper (2507.16838) already shows that CTC-based
GOP features achieve AUC 0.914 on CMU Kids (vs 0.796 for the traditional approach).

**OPEN QUESTION**: How much of the GOPT improvement comes from better features
vs better scoring head? We'll answer this by:
1. Running GOPT on our new CTC-based GOP features (apples-to-apples on features)
2. Comparing polynomial regression vs GOPT on the same features

---

## 7. Identified Gaps and Open Questions

### 7.1 **GROUNDED**: Things We Know Work

| Component | Evidence | Reference |
|-----------|----------|-----------|
| SSL encoder produces phonetic representations | Layer probing studies | Pasad et al. 2021 |
| CTC head on SSL encoder produces good posteriors | Multiple deployed models | wav2vec2-espeak, MMS, etc. |
| CTC head training recipe | Complete tutorial | HF blog (fine-tune-w2v2-bert) |
| GOP-SF from CTC posteriors | Paper + our implementation | 2507.16838, our gop.py |
| Phoneme-level vocab for English GOP | Working baseline | Our xlsr-espeak backend (PCC 0.320) |
| GOPT scoring from GOP features | Published results | Gong et al. ICASSP 2022 |

### 7.2 **OPEN QUESTIONS**: Things We Need to Validate Empirically

#### Q1: Does w2v-BERT 2.0 produce better posteriors than XLS-R for GOP?

**Hypothesis**: Yes, because w2v-BERT 2.0 has 80x more pretraining data
(4.5M hrs vs 56K hrs). Better representations → better posteriors → better GOP.

**How to test**: Fine-tune w2v-BERT 2.0 with ARPABET CTC head on LibriSpeech,
compute GOP on SpeechOcean762, compare PCC with xlsr-espeak baseline (0.320).

**Risk**: Low. The fine-tuning recipe is identical (same code, different
`from_pretrained` string). If it doesn't beat xlsr-espeak, we still have the
baseline.

**Expected effort**: ~4-8h training on A100 + minutes of evaluation.

#### Q2: Can we fine-tune ZIPA's encoder with a phoneme-level head?

**Hypothesis**: ZIPA's encoder (Zipformer, trained on 17K hrs of 88-language
phone data) may produce excellent phoneme posteriors if we replace its 127-char
head with a 39-phone ARPABET head.

**The gap**: ZIPA is in the icefall/k2 ecosystem, not HuggingFace. Fine-tuning
requires icefall training infrastructure. Alternatively, we could use ZIPA's
ONNX encoder as a frozen feature extractor and train only a linear head on top.

**Risk**: Medium. We'd need to either:
(a) Learn icefall training, or
(b) Extract encoder features and train a separate head (loses end-to-end CTC).

**OPEN QUESTION**: Is approach (b) — frozen ZIPA encoder + separate phoneme
classifier — viable? This loses the CTC alignment marginalization, so we'd need
forced alignment or a different scoring method.

#### Q3: Does temporal resolution matter for GOP-SF?

**Context**: Parakeet-CTC operates at 80ms resolution (4x coarser than the
20ms used by all other models and by the GOP-SF paper).

**Hypothesis**: Coarser resolution may hurt GOP-SF scoring because:
- Fewer frames per phoneme → less precise occupancy estimates
- Short phones (stops, flaps) may span only 1-2 frames at 80ms
- The CTC forward-backward algorithm has fewer timesteps to work with

**How to test**: Downsample xlsr-espeak posteriors by 4x, recompute GOP, compare.
This gives us an answer without training a new model.

**Risk**: Medium. If 80ms doesn't work, Parakeet is off the table for GOP
(still useful as word-level ASR sidecar).

#### Q4: How much training data do we need for the phoneme head?

**Prior art**:
- CTC-based-GOP baseline: LibriSpeech 960h (full)
- wav2vec2-xlsr-espeak: CommonVoice (varies by language)
- w2v-BERT blog demo: 14h Mongolian achieved competitive WER

**Hypothesis**: For phoneme posteriors (not ASR transcription), we may need less
data because: (a) the phone vocabulary is smaller (39 vs thousands), and
(b) we only need posterior quality, not sequence-level decoding accuracy.

**How to test**: Train on LibriSpeech subsets (100h, 460h, 960h), compare GOP PCC
on SpeechOcean762.

#### Q5: ARPABET-only or IPA for the phoneme head?

**Trade-off**:
- ARPABET (39 phones): Exact match to SpeechOcean762 labels. Simpler. English only.
- IPA (~100-400 phones): Multilingual from the start. Requires mapping IPA → ARPABET
  for evaluation (we already do this in xlsr-espeak backend).

**Recommendation**: Start ARPABET for English MVP (direct comparison with
baselines), then expand to IPA. The mapping code already exists in
`src/gopt_bench/backends/xlsr_espeak.py`.

### 7.3 **KNOWN HARD PROBLEM**: The ZIPA Diphthong Gap

**Status**: Identified, understood, solution clear

The 7 unmappable ARPABET phones in ZIPA:

```
ARPABET   IPA       ZIPA tokens     Problem
────────────────────────────────────────────
AW        /aʊ/      a + ʊ           2 characters, no single token
AY        /aɪ/      a + ɪ           2 characters, no single token
CH        /tʃ/      t + ʃ           2 characters (affricate)
EY        /eɪ/      e + ɪ           2 characters, no single token
JH        /dʒ/      d + ʒ           2 characters (affricate)
OW        /oʊ/      o + ʊ           2 characters, no single token
OY        /ɔɪ/      ɔ + ɪ           2 characters, no single token
```

**Solution options**:
1. **Train new head** (recommended): Replace ZIPA's 127-char head with 39-phone
   ARPABET head. Each diphthong becomes a single output class.
2. **Merge adjacent tokens**: Sum/max-pool adjacent ZIPA character predictions for
   diphthongs. Hacky, loses CTC alignment properties.
3. **Multi-token GOP**: Extend GOP-SF to handle multi-token phones. Requires
   algorithmic changes to the CTC forward-backward computation.

Option 1 is the standard approach. Options 2-3 are inventive.

---

## 8. The Execution Plan

### Phase 0: Preparation (Current)
- [x] Implement GOP-SF algorithm (scalar scores)
- [x] Build backend architecture with pluggable phoneme models
- [x] Benchmark three existing backends
- [x] Identify the ZIPA gap
- [x] Document research landscape (this document)
- [ ] Extract full GOP feature vectors (41-dim: LPP + LPR + expected count)
- [ ] Evaluate with SVR/GOPT on feature vectors (expect PCC ~0.58-0.65)
- [ ] Build phoneme vocabulary for English (ARPABET 39 + blank + pad)
- [ ] Set up training infrastructure (RunPod A100)

### Phase 1: w2v-BERT 2.0 + ARPABET Head (Low Risk)

**What**: Fine-tune w2v-BERT 2.0 with a 41-class (39 phone + blank + pad) CTC
head on LibriSpeech.

**Why this first**: Lowest risk path. The HuggingFace recipe is complete. We just
change the vocabulary from Mongolian Cyrillic to ARPABET.

**Script outline** (derived from HuggingFace blog):

```python
# 1. Create ARPABET vocabulary
vocab = {phone: i for i, phone in enumerate(ARPABET_39)}
vocab["[PAD]"] = len(vocab)  # blank/CTC token
# Save as vocab.json

# 2. Create tokenizer + processor
tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", pad_token="[PAD]")
feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
processor = Wav2Vec2BertProcessor(feature_extractor, tokenizer)

# 3. Prepare LibriSpeech with phoneme labels
# For each utterance: text → ARPABET via CMU dict or G2P
# This is the same pipeline as CTC-based-GOP/is24/ctc-ASR-training/

# 4. Load model
model = Wav2Vec2BertForCTC.from_pretrained(
    "facebook/w2v-bert-2.0",
    vocab_size=len(vocab),
    pad_token_id=vocab["[PAD]"],
    ctc_loss_reduction="mean",
    add_adapter=True,
)

# 5. Train (A100 80GB config)
TrainingArguments(
    bf16=True,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=2,
    learning_rate=3e-5,
    warmup_ratio=0.1,
    num_train_epochs=15,
)
```

**Expected result**: PCC > 0.320 on SpeechOcean762 (should beat xlsr-espeak
because of vastly more pretraining data).

**Time**: ~6-12h on A100 80GB.
**Cost**: ~$8-17 on RunPod.

**GROUNDED**: Every step here follows the HuggingFace blog exactly, substituting
vocabulary only.

### Phase 2: omniASR Head Swap (Low Risk)

**What**: Replace omniASR-CTC-1B's `final_proj` layer (9,812 chars → 41 phones)
and fine-tune.

**Why**: omniASR has 4.3M hours of pretraining across 1,600 languages.
Its encoder representations may be the strongest available.

**The approach**:
```python
# omniASR CTC is fairseq2-based
# The final_proj is just a Linear layer
model.final_proj = nn.Linear(model.encoder_dim, 41)  # swap head
# Fine-tune with CTC loss on LibriSpeech
```

**GROUNDED**: Head swapping on pretrained CTC models is standard practice.
The omniASR paper itself describes the CTC head as a simple linear projection.

**OPEN QUESTION**: fairseq2 training infrastructure is less documented than
HuggingFace. May need to port the encoder to HuggingFace for easier training.

### Phase 3: Evaluate with GOP-SF

**What**: Plug new model(s) into existing GOP-SF pipeline, compare with baselines.

**Steps**:
1. Create new backend class (e.g., `w2v_bert_phoneme.py`)
2. Load fine-tuned model, produce posteriors
3. Run `gopt-bench run --backend w2v-bert-phoneme`
4. Compare PCC with existing backends

**GROUNDED**: The backend architecture already supports this. See
`src/gopt_bench/backends/base.py` for the protocol.

### Phase 4: GOPT Retraining (Low Risk)

**What**: Train GOPT transformer on new GOP-SF features, compare with polynomial
regression.

**GROUNDED**: The GOPT model is a small transformer (~1M params), trains in minutes,
and has a complete training pipeline in this repo.

### Phase 5: Ablations & Analysis (Research)

**Questions to answer**:
1. Per-phone improvement analysis: Which phones benefit most from the new encoder?
2. Data efficiency: 100h vs 460h vs 960h LibriSpeech — where's the plateau?
3. Encoder comparison: w2v-BERT 2.0 vs omniASR vs WavLM-large
4. Temporal resolution: Does 80ms (Parakeet) work for GOP?
5. Vocabulary: ARPABET-only vs full IPA with mapping

---

## 9. What We Follow vs What We Invent

### Things We Follow (References Available, Recipes Exist)

| What | Reference | Risk |
|------|-----------|------|
| SSL pretraining | wav2vec 2.0, HuBERT, WavLM papers | None — use pretrained |
| CTC head addition | HuggingFace blog (fine-tune-w2v2-bert) | None — tutorial exists |
| ARPABET vocabulary | CMU Pronunciation Dictionary | None — standard |
| CTC training on LibriSpeech | CTC-based-GOP paper, Section 3 | None — recipe exists |
| GOP-SF scoring | Cao et al. 2507.16838 + our gop.py | None — implemented |
| GOPT transformer | Gong et al. ICASSP 2022 + this repo | None — code exists |
| SpeechOcean762 evaluation | Paper protocol + our evaluate.py | None — implemented |

### Things We Explore (Grounded Hypotheses, Need Empirical Validation)

| What | Hypothesis | Baseline Comparison | Risk |
|------|-----------|---------------------|------|
| w2v-BERT 2.0 produces better posteriors | More pretraining → better features | xlsr-espeak PCC 0.320 | Low |
| omniASR encoder is strong for GOP | Most multilingual pretraining | xlsr-espeak PCC 0.320 | Low-Med |
| Training data efficiency for phoneme heads | <100h may suffice for posteriors | 960h full LibriSpeech | Low |

### Things That Require Invention or Validation (Limited Prior Art)

| What | Why It's Novel | Approach | Risk |
|------|---------------|----------|------|
| Validating 80ms resolution for GOP-SF | No one has done GOP-SF at 80ms | Downsample experiment | Low |
| Logit-based GOP-SF | Logit GOP shown for traditional GOP, not yet for GOP-SF | Use pre-softmax values in CTC forward | Low-Med |
| ZIPA encoder + ARPABET head | ZIPA is icefall-only, no HF recipe | Either icefall training or frozen+linear | Medium |
| Multi-token GOP for ZIPA characters | GOP-SF assumes single-token phones | Extend CTC forward-backward | High |
| End-to-end phoneme-posterior-to-score | Skip GOPT, predict directly | Research idea, not planned yet | High |

---

## 10. Reference List

### Core Papers (In Our Collection)

1. **Cao et al.** (2026). "Segmentation-Free Goodness of Pronunciation Scoring."
   IEEE/ACM TASLP. arXiv: 2507.16838. [Our main algorithm]

2. **Zhu et al.** (2025). "ZIPA: Efficient Multilingual Phone Recognition."
   ACL 2025. arXiv: 2505.23170. [ZIPA model & IPAPack++ dataset]

3. **Li et al.** (2025). "POWSM: A Phonetic Open Whisper-Style Speech Foundation
   Model." arXiv: 2510.24992. [Multi-task phonetic model]

4. **Bharadwaj et al.** (2026). "PRiSM: Benchmarking Phone Realization in Speech
   Models." arXiv: 2601.14046. [PR benchmark, encoder-CTC = most stable]

5. **Gong et al.** (2022). "Transformer-based Multi-Aspect Multi-Granularity
   Non-native English Speaker Pronunciation Assessment." ICASSP 2022.
   [GOPT model]

### Foundation Model Papers

6. **Baevski et al.** (2020). "wav2vec 2.0: A Framework for Self-Supervised
   Learning of Speech Representations." NeurIPS 2020.

7. **Hsu et al.** (2021). "HuBERT: Self-Supervised Speech Representation Learning
   by Masked Prediction of Hidden Units." IEEE/ACM TASLP.

8. **Chen et al.** (2022). "WavLM: Large-Scale Self-Supervised Pre-Training for
   Full Stack Speech Processing." IEEE JSTSP.

9. **Babu et al.** (2022). "XLS-R: Self-supervised Cross-lingual Speech
   Representation Learning at Scale." INTERSPEECH 2022.

10. **Seamless Communication Team** (2024). "Seamless: Multilingual Expressive
    and Streaming Speech Translation." [Includes w2v-BERT 2.0]

### CTC & Phoneme Recognition Papers

11. **Graves et al.** (2006). "Connectionist Temporal Classification: Labelling
    Unsegmented Sequence Data with Recurrent Neural Networks." ICML 2006.

12. **Conneau et al.** (2021). "Unsupervised Cross-lingual Representation Learning
    for Speech Recognition." INTERSPEECH.

### CTC Head & Phoneme Recognition Papers

11. **Xu, Baevski, Auli** (2022). "Simple and Effective Zero-shot Cross-lingual
    Phoneme Recognition." INTERSPEECH 2022. arXiv: 2109.11680.
    [How wav2vec2-xlsr-53-espeak-cv-ft was built — our xlsr-espeak baseline]

12. **Pratap et al.** (2023). "Scaling Speech Technology to 1,000+ Languages."
    arXiv: 2305.13516. [MMS — per-language adapter approach]

13. **Li, X.** (2020). "Universal Phone Recognition with a Multilingual Allophone
    System." ICASSP 2020. arXiv: 2002.11800. [Allosaurus]

### GOP Scoring Papers

14. **Parikh et al.** (2025). "Enhancing GOP in CTC-Based MDD with Phonological
    Knowledge." arXiv: 2506.02080. [Restricted phoneme substitutions in GOP]

15. **Logit-based GOP** (2025). "Evaluating Logit-Based GOP Scores."
    arXiv: 2506.12067. [Logit-based GOP can outperform probability-based GOP;
    hybrid methods combining logit + probability features give best results]

### Key Blog Posts & Tutorials

16. **Lacombe, Y.** (2024). "Fine-Tune W2V2-Bert for low-resource ASR with
    Transformers." HuggingFace Blog. [Our primary training recipe]

17. **von Platen, P.** (2021). "Fine-Tune XLSR-Wav2Vec2 for Multi-Lingual ASR
    with Transformers." HuggingFace Blog. [Original CTC fine-tuning tutorial]

18. **von Platen, P.** (2021). "Fine-Tune Wav2Vec2 for English ASR with
    Transformers." HuggingFace Blog. [Wav2Vec2 CTC on TIMIT]

### Datasets

19. **SpeechOcean762**: Zhang et al. (2021). "speechocean762: An Open-Source
    Non-native English Speech Corpus For Pronunciation Assessment."
    arXiv: 2104.01378.

20. **LibriSpeech**: Panayotov et al. (2015). "LibriSpeech: An ASR corpus based
    on public domain audio books." ICASSP 2015.

21. **CMU Pronunciation Dictionary**: Carnegie Mellon University.
    http://www.speech.cs.cmu.edu/cgi-bin/cmudict

### Code Repositories

22. **CTC-based-GOP**: github.com/frank613/CTC-based-GOP [Reference GOP-SF impl]
23. **GOPT**: github.com/YuanGongND/gopt [This repo's origin]
24. **ZIPA**: github.com/lingjzhu/zipa [ZIPA phone recognizer]
25. **Multilingual-PR**: github.com/ASR-project/Multilingual-PR
    [Comparison of wav2vec2/HuBERT/WavLM for phoneme recognition with CTC]
26. **IPA-Wav2Vec2**: github.com/Srinath-N-R/IPA-Wav2Vec2-Phoneme-Recognition
    [End-to-end IPA phoneme recognition pipeline]
27. **XLSR Phoneme Recognition**: github.com/kosuke-kitahara/xlsr-wav2vec2-phoneme-recognition
    [XLSR-Wav2Vec2 fine-tuned on TIMIT for IPA phoneme recognition]

---

## 11. Summary: The Story

**Where we are**: We have a working pronunciation scoring pipeline (GOP-SF) with
three phoneme backends. The best (xlsr-espeak) achieves PCC 0.320 on
SpeechOcean762. We identified that ZIPA fails due to character-level vocabulary,
not acoustic quality.

**What we're doing**: Replacing the phoneme head with one we control — trained
on IPA/ARPABET phones at the phoneme level, on top of the best available
pretrained encoder (w2v-BERT 2.0, 4.5M hrs pretraining).

**Why it should work**: Every piece of the pipeline has been demonstrated
independently. CTC head training is a solved problem with complete tutorials.
GOP-SF scoring from CTC posteriors is proven. The only new thing is the
specific combination — a better encoder with a purpose-built phoneme vocabulary,
evaluated through GOP scoring.

**Where we might need to get creative**: (a) If the w2v-BERT 2.0 phoneme head
doesn't beat xlsr-espeak despite more pretraining, we'll investigate why
(perhaps the CommonVoice CTC fine-tuning in xlsr-espeak covers phoneme patterns
better than LibriSpeech). (b) If we want to use ZIPA's encoder, we need to
bridge the icefall↔HuggingFace gap. (c) If we want 80ms models (Parakeet) for
GOP, we need to validate temporal resolution experimentally.

**The expected outcome**: A pronunciation scoring system where we control every
component, with a clear upgrade path from English MVP to multilingual, using
the most modern speech encoder available, with no Kaldi dependency.
